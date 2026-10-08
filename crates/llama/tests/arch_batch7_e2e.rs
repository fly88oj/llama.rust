//! arch_batch7_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-26: **deepseek4** — hyper-connections (4 residual streams, the
//! fused `ggml_dsv4_hc_{pre,comb,post}` ops) + the compressed DSV4 KV cache
//! (`llama_kv_cache_dsv4`: the iswa raw pair + the csa/hca/lid compressed
//! caches + the three compressor states, llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-6 (PARITY.md): no local deepseek4 GGUF exists,
//! so the file is synthetic — `tokenizer.*` KV copied verbatim from the llama
//! SPM vocab fixture, the arch's own KV, and exactly the tensor names + shapes
//! `load_arch_tensors` asks for, all F32. The proportions are shrunk but real:
//! n_embd 128 / 4 heads / q_lora 32 / key_length 64 (= indexer head 64, so
//! every cache gets the Hadamard `k_rot`) / n_rot 16 / hc_mult 4 /
//! o_groups 2 / o_lora 16 / 4 experts (2 used) / window 64.
//!
//! The four layers pin the three compression ratios of deepseek4.cpp:
//!   * layer 0 — ratio 0 (raw SWA only) **and** the hash layer
//!     (`ffn_gate_tid2eid` token-id routing, deepseek4.cpp:157-159);
//!   * layer 1 — ratio 4 (CSA overlap compression + the lightning-indexer
//!     top-k mask, the `build_csa_lid_attention` path);
//!   * layer 2 — ratio 128 (HCA block compression, `build_hca_attention`);
//!   * layer 3 — ratio 4 again (a second CSA layer so the compressor state
//!     carries two planes of the same kind).
//!
//! The default-run tests drive `DecodeContext::new_with` itself; the parity
//! runs drive the release CLI (`ARCH_BATCH7=1 ./parity/arch_batch_parity.sh`,
//! batch-1 protocol, 16 greedy tokens + a `-long` >64-token prompt cell that
//! pushes positions past the 64-token window).

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::{AttnParams, ForwardResult};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch7";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;
const N_EXPERT_SHARED: i64 = 1;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    key_length: i64,
    value_length: i64,
    q_lora_rank: i64,
    n_rot: i64,
    n_ff_exp: i64,
    indexer_head_size: i64,
    indexer_n_head: i64,
    indexer_top_k: i64,
    hc_mult: i64,
    o_group_count: i64,
    o_lora_rank: i64,
    n_swa: i64,
    compress_rope_base: f32,
    hc_sinkhorn_iters: u32,
    hc_eps: f32,
    hash_layer_count: u32,
    /// per-layer DSV4_CSA_RATIO / DSV4_HCA_RATIO / 0
    ratios: Vec<u32>,
    swiglu_clamp_exp: f32,
    swiglu_clamp_shexp: f32,
    n_ctx: u32,
}

impl SynthSpec {
    fn hc_mix_dim(&self) -> i64 {
        (2 + self.hc_mult) * self.hc_mult
    }
    fn hc_dim(&self) -> i64 {
        self.hc_mult * self.n_embd
    }
    fn n_ff_shexp(&self) -> i64 {
        self.n_ff_exp // n_expert_shared = 1
    }
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut s = self.clone();
        f(&mut s);
        s
    }
}

fn spec_deepseek4() -> SynthSpec {
    SynthSpec {
        arch: "deepseek4",
        suffix: "",
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 1,
        key_length: 64,
        value_length: 64,
        q_lora_rank: 32,
        n_rot: 16,
        n_ff_exp: 24,
        // key_length == indexer_head_size switches attn_rot_k on for every
        // cache (llama-kv-cache.cpp:321-332)
        indexer_head_size: 64,
        indexer_n_head: 2,
        indexer_top_k: 8,
        hc_mult: 4,
        o_group_count: 2,
        o_lora_rank: 16,
        n_swa: 64,
        compress_rope_base: 10000.0,
        hc_sinkhorn_iters: 3,
        hc_eps: 1e-3,
        hash_layer_count: 1,
        ratios: vec![0, 4, 128, 4],
        swiglu_clamp_exp: 7.0,
        swiglu_clamp_shexp: 0.05,
        n_ctx: 512,
    }
}

// ---------------------------------------------------------------------------
// the tensor table (exactly what load_arch_tensors asks for)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Bias,
    Proj,
    Router,
    /// small-magnitude (ape / hc scale-base rows)
    Small,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(String, Vec<i64>, Role)> {
    let mut v: Vec<(String, Vec<i64>, Role)> = Vec::new();
    let n_embd = spec.n_embd;
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $role))
        };
    }
    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    push!("output_norm.weight", vec![n_embd], Role::Norm);
    push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);

    // the model-level hyper-connection head (deepseek4.cpp:105-107)
    push!(
        "output_hc_fn.weight",
        vec![spec.hc_dim(), spec.hc_mult],
        Role::Proj
    );
    push!("output_hc_base.weight", vec![spec.hc_mult], Role::Small);
    push!("output_hc_scale.weight", vec![1], Role::Small);

    for i in 0..spec.n_layer as i32 {
        let ratio = spec.ratios[i as usize];
        let coff: i64 = if ratio == 4 { 2 } else { 1 };

        push!(
            format!("blk.{i}.attn_norm.weight"),
            vec![n_embd],
            Role::Norm
        );
        push!(
            format!("blk.{i}.attn_sinks.weight"),
            vec![spec.n_head],
            Role::Small
        );
        push!(
            format!("blk.{i}.attn_q_a.weight"),
            vec![n_embd, spec.q_lora_rank],
            Role::Proj
        );
        push!(
            format!("blk.{i}.attn_q_a_norm.weight"),
            vec![spec.q_lora_rank],
            Role::Norm
        );
        push!(
            format!("blk.{i}.attn_q_b.weight"),
            vec![spec.q_lora_rank, spec.n_head * spec.key_length],
            Role::Proj
        );
        push!(
            format!("blk.{i}.attn_kv.weight"),
            vec![n_embd, spec.key_length],
            Role::Proj
        );
        push!(
            format!("blk.{i}.attn_kv_a_norm.weight"),
            vec![spec.key_length],
            Role::Norm
        );
        // the file stores the 2-D form; the loader reshapes to 3-D
        // (TENSOR_ALLOW_RESHAPE, deepseek4.cpp:120-122)
        push!(
            format!("blk.{i}.attn_output_a.weight"),
            vec![
                spec.n_head * spec.key_length / spec.o_group_count,
                spec.o_lora_rank * spec.o_group_count
            ],
            Role::Proj
        );
        push!(
            format!("blk.{i}.attn_output_b.weight"),
            vec![spec.o_group_count * spec.o_lora_rank, n_embd],
            Role::Proj
        );

        let hc_mix = spec.hc_mix_dim();
        push!(
            format!("blk.{i}.hc_attn_fn.weight"),
            vec![spec.hc_dim(), hc_mix],
            Role::Proj
        );
        push!(
            format!("blk.{i}.hc_attn_base.weight"),
            vec![hc_mix],
            Role::Small
        );
        push!(
            format!("blk.{i}.hc_attn_scale.weight"),
            vec![3],
            Role::Small
        );
        push!(
            format!("blk.{i}.hc_ffn_fn.weight"),
            vec![spec.hc_dim(), hc_mix],
            Role::Proj
        );
        push!(
            format!("blk.{i}.hc_ffn_base.weight"),
            vec![hc_mix],
            Role::Small
        );
        push!(format!("blk.{i}.hc_ffn_scale.weight"), vec![3], Role::Small);

        if ratio != 0 {
            push!(
                format!("blk.{i}.attn_compressor_kv.weight"),
                vec![n_embd, coff * spec.key_length],
                Role::Proj
            );
            push!(
                format!("blk.{i}.attn_compressor_gate.weight"),
                vec![n_embd, coff * spec.key_length],
                Role::Proj
            );
            push!(
                format!("blk.{i}.attn_compressor_ape.weight"),
                vec![coff * spec.key_length, ratio as i64],
                Role::Small
            );
            push!(
                format!("blk.{i}.attn_compressor_norm.weight"),
                vec![spec.key_length],
                Role::Norm
            );

            if ratio == 4 {
                push!(
                    format!("blk.{i}.indexer.proj.weight"),
                    vec![n_embd, spec.indexer_n_head],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer.attn_q_b.weight"),
                    vec![
                        spec.q_lora_rank,
                        spec.indexer_n_head * spec.indexer_head_size
                    ],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer_compressor_kv.weight"),
                    vec![n_embd, 2 * spec.indexer_head_size],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer_compressor_gate.weight"),
                    vec![n_embd, 2 * spec.indexer_head_size],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer_compressor_ape.weight"),
                    vec![2 * spec.indexer_head_size, ratio as i64],
                    Role::Small
                );
                push!(
                    format!("blk.{i}.indexer_compressor_norm.weight"),
                    vec![spec.indexer_head_size],
                    Role::Norm
                );
            }
        }

        push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
        push!(
            format!("blk.{i}.ffn_gate_inp.weight"),
            vec![n_embd, N_EXPERT],
            Role::Router
        );
        if (i as u32) < spec.hash_layer_count {
            push!(
                format!("blk.{i}.ffn_gate_tid2eid.weight"),
                vec![N_EXPERT_USED, N_VOCAB],
                Role::Proj
            );
        } else {
            push!(
                format!("blk.{i}.exp_probs_b.bias"),
                vec![N_EXPERT],
                Role::Bias
            );
        }
        push!(
            format!("blk.{i}.ffn_gate_exps.weight"),
            vec![n_embd, spec.n_ff_exp, N_EXPERT],
            Role::Proj
        );
        push!(
            format!("blk.{i}.ffn_down_exps.weight"),
            vec![spec.n_ff_exp, n_embd, N_EXPERT],
            Role::Proj
        );
        push!(
            format!("blk.{i}.ffn_up_exps.weight"),
            vec![n_embd, spec.n_ff_exp, N_EXPERT],
            Role::Proj
        );
        push!(
            format!("blk.{i}.ffn_gate_shexp.weight"),
            vec![n_embd, spec.n_ff_shexp()],
            Role::Proj
        );
        push!(
            format!("blk.{i}.ffn_down_shexp.weight"),
            vec![spec.n_ff_shexp(), n_embd],
            Role::Proj
        );
        push!(
            format!("blk.{i}.ffn_up_shexp.weight"),
            vec![n_embd, spec.n_ff_shexp()],
            Role::Proj
        );
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch6_e2e.rs)
// ---------------------------------------------------------------------------

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

fn scale_of(role: Role, n_embd: i64) -> f32 {
    match role {
        Role::Norm => 1.0,
        Role::Bias => 0.02,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        Role::Router => 1.0 / (n_embd as f32).sqrt(),
        Role::Small => 0.02,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn build_file(spec: &SynthSpec) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch7");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = spec.arch;
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!(
        "general.name",
        Value::String(format!("llama-rust-synth-{a}"))
    );
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(spec.n_ctx));
    kv!(
        format!("{a}.embedding_length"),
        Value::U32(spec.n_embd as u32)
    );
    kv!(format!("{a}.block_count"), Value::U32(spec.n_layer as u32));
    kv!(
        format!("{a}.attention.head_count"),
        Value::U32(spec.n_head as u32)
    );
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::U32(spec.n_head_kv as u32)
    );
    kv!(
        format!("{a}.attention.key_length"),
        Value::U32(spec.key_length as u32)
    );
    kv!(
        format!("{a}.attention.value_length"),
        Value::U32(spec.value_length as u32)
    );
    kv!(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5)
    );
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.n_rot as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    kv!(
        format!("{a}.attention.sliding_window"),
        Value::U32(spec.n_swa as u32)
    );
    kv!(
        format!("{a}.attention.q_lora_rank"),
        Value::U32(spec.q_lora_rank as u32)
    );
    // MoE
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(
        format!("{a}.expert_used_count"),
        Value::U32(N_EXPERT_USED as u32)
    );
    kv!(
        format!("{a}.expert_shared_count"),
        Value::U32(N_EXPERT_SHARED as u32)
    );
    kv!(
        format!("{a}.expert_feed_forward_length"),
        Value::Array(
            GgufType::Uint32,
            vec![Value::U32(spec.n_ff_exp as u32); spec.n_layer]
        )
    );
    kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
    // deepseek4.cpp:34 — required (get_key), unlike deepseek32's optional read
    kv!(format!("{a}.expert_weights_scale"), Value::F32(2.5));
    // deepseek4.cpp:63-66 — SQRT_SOFTPLUS is required
    kv!(format!("{a}.expert_gating_func"), Value::U32(4));
    // the swiglu clamps (f32 arrays broadcast per layer)
    kv!(
        format!("{a}.swiglu_clamp_exp"),
        Value::Array(
            GgufType::Float32,
            vec![Value::F32(spec.swiglu_clamp_exp); spec.n_layer]
        )
    );
    kv!(
        format!("{a}.swiglu_clamp_shexp"),
        Value::Array(
            GgufType::Float32,
            vec![Value::F32(spec.swiglu_clamp_shexp); spec.n_layer]
        )
    );
    // the indexer
    kv!(
        format!("{a}.attention.indexer.head_count"),
        Value::U32(spec.indexer_n_head as u32)
    );
    kv!(
        format!("{a}.attention.indexer.key_length"),
        Value::U32(spec.indexer_head_size as u32)
    );
    kv!(
        format!("{a}.attention.indexer.top_k"),
        Value::U32(spec.indexer_top_k as u32)
    );
    // the dsv4 geometry
    kv!(
        format!("{a}.attention.output_group_count"),
        Value::U32(spec.o_group_count as u32)
    );
    kv!(
        format!("{a}.attention.output_lora_rank"),
        Value::U32(spec.o_lora_rank as u32)
    );
    kv!(
        format!("{a}.attention.compress_rope_freq_base"),
        Value::F32(spec.compress_rope_base)
    );
    kv!(
        format!("{a}.hyper_connection.count"),
        Value::U32(spec.hc_mult as u32)
    );
    kv!(
        format!("{a}.hyper_connection.sinkhorn_iterations"),
        Value::U32(spec.hc_sinkhorn_iters)
    );
    kv!(
        format!("{a}.hyper_connection.epsilon"),
        Value::F32(spec.hc_eps)
    );
    kv!(
        format!("{a}.hash_layer_count"),
        Value::U32(spec.hash_layer_count)
    );
    kv!(
        format!("{a}.attention.compress_ratios"),
        Value::Array(
            GgufType::Uint32,
            spec.ratios.iter().map(|&r| Value::U32(r)).collect()
        )
    );

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for (name, ne, role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role, spec.n_embd);
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        // the hash table stores expert *indices* — I32, exactly what
        // ggml_get_rows must return for the MoE selected-experts input
        if name.ends_with("ffn_gate_tid2eid.weight") {
            let vals: Vec<i32> = (0..n).map(|i| (i % N_EXPERT_USED as i64) as i32).collect();
            let mut bytes = Vec::with_capacity(n as usize * 4);
            for x in &vals {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            w.add_tensor(name, GgmlType::I32, ne4);
            data.push(bytes);
            continue;
        }
        let vals: Vec<f32> = match role {
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            _ => (0..n).map(|_| s * rng.next()).collect(),
        };
        w.add_tensor(name, GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }

    let path = spec.path();
    let f = std::fs::File::create(&path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    (table.len(), std::fs::metadata(&path).unwrap().len())
}

fn load_synth(spec: &SynthSpec) -> LlamaModel {
    build_file(spec);
    open_model(&spec.path())
}

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let mut want: Vec<String> = tensors_for(spec).into_iter().map(|(n, _, _)| n).collect();
    want.sort();
    want.dedup();
    let got = {
        let mut v: Vec<String> = m.tensors.keys().cloned().collect();
        v.sort();
        v
    };
    assert_eq!(got, want, "{}: created tensor set mismatch", spec.arch);
}

// ---------------------------------------------------------------------------
// weights + params assembly (the same derivations llama-cli's forward_weights
// performs — kept in sync with crates/tools/llama-cli/src/main.rs)
// ---------------------------------------------------------------------------

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
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
        use_flash_attn: fa,
    }
}

fn deepseek4_weights(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Deepseek4ModelWeights {
    use llama::graph_arch as ga;
    let layers = m.layers[..n_trunk]
        .iter()
        .map(|l| ga::Deepseek4LayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            attn_sinks: l.attn_sinks.unwrap(),
            wq_a: l.wq_a.unwrap(),
            attn_q_a_norm: l.attn_q_a_norm.unwrap(),
            wq_b: l.wq_b.unwrap(),
            wkv: l.wkv_a_mqa.unwrap(),
            attn_kv_norm: l.attn_kv_a_norm.unwrap(),
            wo_a: l.wo_a.unwrap(),
            wo_b: l.wo_b_dsv4.unwrap(),
            hc_attn_fn: l.hc_attn_fn.unwrap(),
            hc_attn_base: l.hc_attn_base.unwrap(),
            hc_attn_scale: l.hc_attn_scale.unwrap(),
            hc_ffn_fn: l.hc_ffn_fn.unwrap(),
            hc_ffn_base: l.hc_ffn_base.unwrap(),
            hc_ffn_scale: l.hc_ffn_scale.unwrap(),
            attn_comp_wkv: l.attn_comp_wkv,
            attn_comp_wgate: l.attn_comp_wgate,
            attn_comp_ape: l.attn_comp_ape,
            attn_comp_norm: l.attn_comp_norm,
            indexer_proj: l.indexer_proj,
            indexer_attn_q_b: l.indexer_attn_q_b,
            indexer_comp_wkv: l.indexer_comp_wkv,
            indexer_comp_wgate: l.indexer_comp_wgate,
            indexer_comp_ape: l.indexer_comp_ape,
            indexer_comp_norm: l.indexer_comp_norm,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate_inp: l.ffn_gate_inp.unwrap(),
            ffn_gate_tid2eid: l.ffn_gate_tid2eid,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_exp_probs_b_vl: l.ffn_exp_probs_b_vl,
            ffn_gate_exps: l.ffn_gate_exps.unwrap(),
            ffn_down_exps: l.ffn_down_exps.unwrap(),
            ffn_up_exps: l.ffn_up_exps.unwrap(),
            ffn_gate_shexp: l.ffn_gate_shexp.unwrap(),
            ffn_down_shexp: l.ffn_down_shexp.unwrap(),
            ffn_up_shexp: l.ffn_up_shexp.unwrap(),
        })
        .collect();
    ga::Deepseek4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.unwrap(),
        hc_head_base: m.hc_head_base.unwrap(),
        hc_head_scale: m.hc_head_scale.unwrap(),
        layers,
    }
}

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa);
    let n_trunk = hp.n_layer() as usize;
    (
        ForwardWeights::Deepseek4(
            deepseek4_weights(m, n_trunk),
            llama::graph_arch::Deepseek4Params {
                attn,
                n_embd: hp.n_embd as i64,
                hc_mult: hp.dsv4_hc_mult as i64,
                hc_eps: hp.dsv4_hc_eps,
                hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters as i32,
                o_group_count: hp.dsv4_o_group_count as i64,
                o_lora_rank: hp.dsv4_o_lora_rank as i64,
                compress_rope_base: hp.dsv4_compress_rope_base,
                indexer_n_head: hp.indexer_n_head as i64,
                indexer_head_size: hp.indexer_head_size as i64,
                indexer_top_k: hp.indexer_top_k as i64,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
                swiglu_clamp_exp: hp.swiglu_clamp_exp[..n_trunk].to_vec(),
                swiglu_clamp_shexp: hp.swiglu_clamp_shexp[..n_trunk].to_vec(),
                ratios: hp.dsv4_compress_ratios[..n_trunk].to_vec(),
                hash_layer_count: hp.dsv4_hash_layer_count,
                n_swa: hp.n_swa,
                f_attn_temp_scale: hp.f_attn_temp_scale,
            },
        ),
        attn,
    )
}

fn driver_for(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    // n_ctx 512 like the parity runs; n_batch covers the long prompts
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
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

/// 6-token prefill + decode steps past one CSA block boundary (ratio 4) and
/// towards the HCA one, plus a bit-identical repeat on a cleared cache —
/// through both FA modes so wiring issues surface here rather than in the
/// parity runs.
fn smoke_forward(spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut m = load_synth(spec);
    let mut d = driver_for(&mut m, fa);

    let toks = [3i32, 17, 42, 9, 21, 5];
    let pos: Vec<i32> = (0..6).collect();
    let a = d.decode(&toks, &pos).expect("decode").to_vec();
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{}{}: non-finite logits (fa={fa})",
        spec.arch,
        spec.suffix
    );
    let spread = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - a.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}{}: logits degenerate (spread {spread}, fa={fa})",
        spec.arch,
        spec.suffix
    );

    // step past a block boundary (pos 7 completes CSA block 1)
    let mut cur = a.clone();
    for p in 6..10 {
        let next = argmax(&cur);
        cur = d.decode(&[next], &[p]).expect("decode step").to_vec();
        assert!(cur.iter().all(|v| v.is_finite()));
    }
    assert_eq!(d.kv.used_cells(), 10);

    // fresh cache → bit-identical first decode
    d.reset_sequence();
    let a2 = d.decode(&toks, &pos).expect("decode repeat").to_vec();
    assert_eq!(
        a, a2,
        "{}{}: repeat mismatch (fa={fa})",
        spec.arch, spec.suffix
    );
    a
}

// ---------------------------------------------------------------------------
// default-run tests (no reference needed)
// ---------------------------------------------------------------------------

#[test]
fn synth_deepseek4_loader_and_forward() {
    let spec = spec_deepseek4();
    let (n, bytes) = build_file(&spec);
    println!(
        "deepseek4 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_lora_q, 32);
    assert_eq!(hp.n_embd_head_k(0), 64);
    assert_eq!(hp.n_embd_head_v(0), 64);
    assert_eq!(hp.n_head_kv(0), 1, "MQA");
    assert_eq!(hp.n_rot(0), 16);
    assert_eq!(hp.n_swa, 64);
    assert_eq!(hp.dsv4_hc_mult, 4);
    assert_eq!(hp.dsv4_o_group_count, 2);
    assert_eq!(hp.dsv4_o_lora_rank, 16);
    assert_eq!(hp.dsv4_hash_layer_count, 1);
    assert_eq!(hp.dsv4_compress_ratios, vec![0, 4, 128, 4]);
    assert_eq!(hp.swiglu_clamp_exp[0], 7.0);
    assert_eq!(hp.swiglu_clamp_shexp[0], 0.05);
    assert_eq!(hp.expert_gating_func, 4, "SQRT_SOFTPLUS (required)");
    assert_eq!(hp.n_embd_out(), 4 * 128, "n_embd_out = hc_mult * n_embd");
    assert_eq!(hp.indexer_head_size, 64);
    assert_eq!(hp.indexer_top_k, 8);
    assert_eq!(
        hp.non_causal_type,
        llama::hparams::LlamaNonCausalType::SWA_FULL
    );
    assert_eq!(hp.swa_type, llama::hparams::LlamaSwaType::STANDARD);
    assert!(hp.is_swa(0) && hp.is_swa(3), "all trunk layers SWA");
    // the wo_a reshape (TENSOR_ALLOW_RESHAPE)
    assert_eq!(*m.ctx.ne(m.layers[0].wo_a.unwrap()), [128, 16, 2, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].wo_b_dsv4.unwrap()), [32, 128, 1, 1]);
    // hash layer 0 carries tid2eid, not the router bias
    assert!(m.layers[0].ffn_gate_tid2eid.is_some());
    assert!(m.layers[0].ffn_exp_probs_b.is_none());
    assert!(m.layers[1].ffn_gate_tid2eid.is_none());
    assert!(m.layers[1].ffn_exp_probs_b.is_some());
    // the compressors exist exactly at their ratios
    assert!(m.layers[0].attn_comp_wkv.is_none());
    assert!(m.layers[1].attn_comp_wkv.is_some());
    assert!(m.layers[2].attn_comp_wkv.is_some());
    assert!(m.layers[1].indexer_comp_wkv.is_some());
    assert!(m.layers[2].indexer_comp_wkv.is_none());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek4 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// The dsv4 cache geometry (llama_kv_cache_dsv4's constructor,
/// llama-kv-cache-dsv4.cpp:1292-1329): comp caches at
/// PAD(ceil(512/ratio), 256) = 256 cells, states 8/128 rows wide, the raw
/// iswa pair full-size.
#[test]
fn dsv4_cache_geometry() {
    let spec = spec_deepseek4().with(|s| s.suffix = "-geom");
    build_file(&spec);
    let mut m = open_model(&spec.path());
    let d = driver_for(&mut m, false);
    let kv = &d.kv;
    let dsv4 = kv.dsv4.as_ref().unwrap();
    assert_eq!(dsv4.csa.size, 256, "PAD(ceil(512/4), 256)");
    assert_eq!(dsv4.hca.size, 256, "PAD(ceil(512/128), 256)");
    assert_eq!(dsv4.lid.size, 256);
    assert_eq!(dsv4.csa.n_embd_head, 64);
    assert_eq!(dsv4.lid.n_embd_head, 64, "indexer head");
    assert_eq!(dsv4.csa.layer_ids, vec![1, 3]);
    assert_eq!(dsv4.hca.layer_ids, vec![2]);
    assert_eq!(dsv4.csa_state.state_size, 8, "2*CSA_RATIO");
    assert_eq!(dsv4.hca_state.state_size, 128, "HCA_RATIO");
    assert_eq!(dsv4.csa_state.n_embd_state, 128, "2*n_embd_head_k");
    assert_eq!(dsv4.hca_state.n_embd_state, 64, "n_embd_head_k");
    assert_eq!(dsv4.lid_state.n_embd_state, 128, "2*indexer_head");
    assert!(dsv4.raw_k_rot, "key_length == indexer_head_size");
    assert!(kv.has_swa() && kv.is_swa.iter().all(|&s| s));
    // the K row width of the raw cache is the MLA-style single head
    assert_eq!(kv.n_embd_k_gqa, 64);
}

/// Dsv4Plan::build — the single-sequence slice of dsv4_build_comp_plan:
/// block completion, dummy blocks, the 256 padding and the persist ordering.
#[test]
fn dsv4_plan_builder() {
    use llama::kv_cache::{Dsv4Plan, DSV4_CSA_RATIO, DSV4_HCA_RATIO};

    // a 6-token ubatch at ratio 4: block 0 completes at pos 3, the reserve
    // plan wants ceil(6/4) = 2 blocks → one dummy write at cell kv_size-1
    let p = Dsv4Plan::build(&[0, 1, 2, 3, 4, 5], DSV4_CSA_RATIO, true, 8, 256);
    assert_eq!(p.n_visible, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(p.state_pos, vec![0, 1, 2, 3, 0, 1]);
    assert_eq!(p.state_write_idxs, vec![0, 255]);
    assert_eq!(p.state_write_pos, vec![0, 0]);
    assert_eq!(p.n_kv, 256);
    // overlap reads are [all blocks' prev | all blocks' cur]
    // (llama-kv-cache-dsv4.cpp:629-636): block 0's prev window (positions
    // -4..0) maps past the scratch to the appended zero row (state_rows(8) +
    // n_tokens(6) = 14); its cur window is the ubatch's own tokens 0..4
    // (scratch rows 8..12); the dummy block repeats the first token's
    // scratch row in both windows
    assert_eq!(p.state_read_idxs.len(), 16);
    assert_eq!(p.state_read_idxs[..4], vec![14, 14, 14, 14]);
    assert_eq!(p.state_read_idxs[4..8], vec![8, 8, 8, 8]);
    assert_eq!(p.state_read_idxs[8..12], vec![8, 9, 10, 11]);
    assert_eq!(p.state_read_idxs[12..16], vec![8, 8, 8, 8]);
    // persist: one row per ring residue (state_size 8 > 6 tokens → all
    // distinct here), dst-ordered
    assert_eq!(p.state_persist_dst_idxs, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(p.state_persist_src_idxs, vec![0, 1, 2, 3, 4, 5]);

    // a single decode token that completes block 1 (pos 7): the prev window
    // (positions 0..4) is not in this ubatch → the state plane (pos % 8);
    // the cur window (4..8) reads plane rows 4,5,6 and the current token's
    // scratch row 8 (pos 7 is token 0 of this ubatch)
    let p = Dsv4Plan::build(&[7], DSV4_CSA_RATIO, true, 8, 256);
    assert_eq!(p.n_visible, vec![2]);
    assert_eq!(p.state_write_idxs, vec![1]);
    assert_eq!(p.state_write_pos, vec![4]);
    assert_eq!(p.state_read_idxs, vec![0, 1, 2, 3, 4, 5, 6, 8]);
    assert_eq!(p.state_persist_dst_idxs, vec![7]);
    assert_eq!(p.state_persist_src_idxs, vec![0]);

    // a single decode token that completes no block: dummy-only
    let p = Dsv4Plan::build(&[6], DSV4_CSA_RATIO, true, 8, 256);
    assert_eq!(p.n_visible, vec![1]);
    assert_eq!(p.state_write_idxs, vec![255]);
    assert_eq!(p.state_read_idxs, vec![8, 8, 8, 8, 8, 8, 8, 8]);
    assert_eq!(p.state_persist_dst_idxs, vec![6]);
    assert_eq!(p.state_persist_src_idxs, vec![0]);

    // the HCA dummy (no block can complete in a short ubatch)
    let p = Dsv4Plan::build(&[0, 1, 2], DSV4_HCA_RATIO, false, 128, 256);
    assert_eq!(p.state_write_idxs, vec![255]);
    assert_eq!(p.state_read_idxs.len(), 128);

    // n_kv padding: 129 visible blocks → PAD(129, 256) = 256
    let pos: Vec<i32> = (0..520).collect();
    let p = Dsv4Plan::build(&pos, DSV4_HCA_RATIO, false, 128, 256);
    assert_eq!(p.n_kv, 256);
    let p = Dsv4Plan::build(&pos, DSV4_CSA_RATIO, true, 8, 256);
    assert_eq!(p.n_visible.last(), Some(&130));
    assert_eq!(p.n_kv, 256);
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_deepseek4().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("deepseek4"));
    assert_eq!(g.get_u32("deepseek4.attention.output_group_count"), Some(2));
    assert_eq!(g.get_u32("deepseek4.attention.output_lora_rank"), Some(16));
    assert_eq!(g.get_u32("deepseek4.hyper_connection.count"), Some(4));
    assert_eq!(g.get_u32("deepseek4.hash_layer_count"), Some(1));
    assert_eq!(g.get_u32("deepseek4.expert_gating_func"), Some(4));
    assert_eq!(g.get_u32("deepseek4.attention.sliding_window"), Some(64));
}

/// `Dsv4Plan::build_seq` — the multi-stream + rollback slices of
/// `dsv4_build_comp_plan` (:442-716): the per-sequence stream offsets of the
/// writes/persists, and the :653-716 restore/snapshot index vectors.
#[test]
fn dsv4_plan_builder_multi_seq_rollback() {
    use llama::kv_cache::{Dsv4Plan, DSV4_CSA_RATIO};

    // two streams (n_seq_max 2), a 4-token ubatch of sequence 1: block 0
    // completes at pos 3 with the cell at stream_off(kv_size) + pos/ratio =
    // 256 + 0, the persist rows land in stream 1's half
    // (state_size + pos%state_size)
    let p = Dsv4Plan::build_seq(
        &[0, 1, 2, 3],
        &[1, 1, 1, 1],
        DSV4_CSA_RATIO,
        true,
        8,
        256,
        2,
        0,
        &[],
    );
    assert_eq!(p.state_write_idxs, vec![256]);
    assert_eq!(p.state_write_pos, vec![0]);
    assert_eq!(p.state_persist_dst_idxs, vec![8, 9, 10, 11]);
    // the block's read window (positions -4..0 → the zero row; 0..4 → the
    // ubatch scratch at scratch_off = 8*2*(1+0) = 16)
    assert_eq!(p.state_read_idxs[..4], vec![20, 20, 20, 20]);
    assert_eq!(p.state_read_idxs[4..8], vec![16, 17, 18, 19]);

    // the rollback index vectors at n_rs_seq 2, a pending rollback of 1
    // (:653-716): restore src = plane 1 rows (1*state_rows + r), dst = the
    // live rows; snapshot dst = plane d rows, src = the live row unless one
    // of the first (n_seq_tokens - d) tokens wrote it (single token →
    // prefix 0 for both d → the live row)
    let p = Dsv4Plan::build_seq(&[9], &[0], DSV4_CSA_RATIO, true, 8, 256, 1, 2, &[1]);
    assert_eq!(p.state_restore_src_idxs, vec![8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(p.state_restore_dst_idxs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        p.state_snapshot_src_idxs,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        p.state_snapshot_dst_idxs,
        vec![8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    );

    // a two-token step with n_rs_seq 1: snapshot 1 = the state after the
    // first token (its fresh scratch row where the residue matches,
    // the pre-step live row elsewhere) — the literal :692-714 rule
    let p = Dsv4Plan::build_seq(&[9, 10], &[0, 0], DSV4_CSA_RATIO, true, 8, 256, 1, 1, &[0]);
    // scratch_off = 8*1*(1+1) = 16; token 0 at pos 9 → residue 1 → row 1
    // from scratch, rows 0/2..7 from the live plane
    assert_eq!(p.state_snapshot_src_idxs, vec![0, 16, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        p.state_snapshot_dst_idxs,
        vec![8, 9, 10, 11, 12, 13, 14, 15]
    );
}

/// `load_synth` must not race the writer (build_file re-creates the shared
/// synth path on every call — a concurrent open can hit a truncating write,
/// the "Truncated(magic)" SIGBUS flake the other batches guard against)
fn synth_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// the deepseek4 driver variant with the dsv4 layout knobs
/// (`DecodeContext::new_with_dsv4`): n_seq_max streams + n_rs_seq rollback
/// planes.
fn driver_for_dsv4(m: &mut LlamaModel, fa: bool, n_seq_max: u32, n_rs_seq: u32) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with_dsv4(gctx, weights, attn, 512, 8, 512, n_seq_max, n_rs_seq)
}

/// multi-sequence decode with a dsv4 model (llama_kv_cache_dsv4's per-seq
/// streams, llama-kv-cache-dsv4.cpp:1287): a two-sequence batch through
/// `decode_batch` must equal two fresh single-sequence runs — the port's
/// documented bit-identity criterion (the same one the qwen2.5 multi-seq
/// anchor uses). Each sequence's tokens differ, so a stream mix-up (or a
/// missed compressor-plane separation) shifts the logits immediately.
#[test]
fn dsv4_multi_sequence_bit_identity() {
    let spec = spec_deepseek4().with(|s| s.suffix = "-mseq");
    build_file(&spec);

    let toks0 = [3i32, 17, 42, 9, 21, 5, 8, 30];
    let toks1 = [7i32, 11, 29, 2, 16, 4, 13, 25];
    // positions past one CSA block boundary (ratio 4) and past the 8-row
    // state ring, so both compressors recycle residues while interleaved
    let pos: Vec<i32> = (0..toks0.len() as i32).collect();

    let mk_batch = |toks: &[i32], seq: i32| {
        let mut b = llama::batch::LlamaBatch::default();
        for (i, &t) in toks.iter().enumerate() {
            b.add(t, i as i32, &[seq], true);
        }
        b
    };

    // (a) the two-sequence batch: one context, both sequences in one call
    let mut both = {
        let _g = synth_lock();
        let mut m = load_synth(&spec);
        driver_for_dsv4(&mut m, false, 2, 0)
    };
    let mut batch = llama::batch::LlamaBatch::default();
    for (b, seq) in [(&toks0, 0i32), (&toks1, 1i32)] {
        for (i, &t) in b.iter().enumerate() {
            batch.add(t, i as i32, &[seq], true);
        }
    }
    let out = both.decode_batch(&batch).expect("two-sequence dsv4 batch");
    let n_vocab = out.n_vocab;
    assert_eq!(out.n_outputs, 16);
    let rows: Vec<Vec<f32>> = (0..16)
        .map(|r| out.logits[r * n_vocab..(r + 1) * n_vocab].to_vec())
        .collect();

    // (b) two fresh single-sequence runs on the same geometry (n_stream 2 —
    // identical plane layout, only one stream live)
    let single = |toks: &[i32], seq: i32| -> Vec<Vec<f32>> {
        let _g = synth_lock();
        let mut m = load_synth(&spec);
        let mut d = driver_for_dsv4(&mut m, false, 2, 0);
        let b = mk_batch(toks, seq);
        let o = d.decode_batch(&b).expect("single-sequence dsv4 batch");
        (0..toks.len())
            .map(|r| o.logits[r * n_vocab..(r + 1) * n_vocab].to_vec())
            .collect()
    };
    let s0 = single(&toks0, 0);
    let s1 = single(&toks1, 1);

    for (i, ((r, e0), e1)) in rows.iter().zip(&s0).zip(&s1).enumerate() {
        let expect = if i < 8 { e0 } else { e1 };
        assert_eq!(
            r, expect,
            "two-sequence row {i} != the fresh single-sequence run (bit-identity)"
        );
    }
}

/// seq_cp on the dsv4 cache (llama-kv-cache-dsv4.cpp:1512-1527): after
/// copying sequence 0 onto sequence 1, decoding sequence 1's own continuation
/// must equal a context that saw sequence 0's tokens then the same
/// continuation — i.e. the copied compressed streams (K rows + state planes)
/// carry sequence 0's history.
#[test]
fn dsv4_seq_cp_compressed_streams() {
    let spec = spec_deepseek4().with(|s| s.suffix = "-seqcp");
    build_file(&spec);

    let toks = [3i32, 17, 42, 9, 21, 5, 8, 30];
    let more = [14i32, 27, 6, 19];

    // (a) prompt seq 0, seq_cp(0 → 1), continue on seq 1
    let mut m = {
        let _g = synth_lock();
        load_synth(&spec)
    };
    let mut d = driver_for_dsv4(&mut m, false, 2, 0);
    {
        let mut b = llama::batch::LlamaBatch::default();
        for (i, &t) in toks.iter().enumerate() {
            b.add(t, i as i32, &[0], true);
        }
        d.decode_batch(&b).expect("prompt seq 0");
    }
    d.seq_cp(0, 1, -1, -1).expect("dsv4 seq_cp");
    let mut cont_a = Vec::new();
    {
        let mut pos = toks.len() as i32;
        for &t in more.iter() {
            let mut b = llama::batch::LlamaBatch::default();
            b.add(t, pos, &[1], true);
            let o = d.decode_batch(&b).expect("seq 1 step");
            let row = &o.logits[..o.n_vocab];
            cont_a.push(argmax(row));
            pos += 1;
        }
    }

    // (b) the same tokens on one sequence of a fresh context
    let mut m2 = {
        let _g = synth_lock();
        load_synth(&spec)
    };
    let mut d2 = driver_for_dsv4(&mut m2, false, 2, 0);
    {
        let mut b = llama::batch::LlamaBatch::default();
        for (i, &t) in toks.iter().chain(more.iter()).enumerate() {
            b.add(t, i as i32, &[0], true);
        }
        let o = d2.decode_batch(&b).expect("prompt + more");
        let mut cont_b = Vec::new();
        for r in toks.len()..toks.len() + more.len() {
            cont_b.push(argmax(&o.logits[r * o.n_vocab..(r + 1) * o.n_vocab]));
        }
        assert_eq!(
            cont_a, cont_b,
            "the copied streams must carry seq 0's history"
        );
    }
}

/// the dsv4 rollback planes (`n_rs_seq > 0`, llama-kv-cache-dsv4.cpp:653-716
/// + :1481-1500): decode → `seq_rm` back to a snapshot → decode the same
/// tokens again must reproduce the first pass's post-snapshot outputs
/// bit-identically (the compressor state planes restored from the snapshot
/// group, the raw cells rewound).
#[test]
fn dsv4_rollback_reproduces_post_snapshot_tokens() {
    let spec = spec_deepseek4().with(|s| s.suffix = "-rsrb");
    build_file(&spec);

    let n_rs_seq = 4u32;
    let mut m = {
        let _g = synth_lock();
        load_synth(&spec)
    };
    let mut d = driver_for_dsv4(&mut m, false, 1, n_rs_seq);

    // first pass: greedy single-token decodes through 12 positions — past
    // three CSA block boundaries and one full state-ring recycle
    let first = 3i32;
    let n_tokens = 12usize;
    let mut logits_at: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    let mut toks = vec![first];
    let mut cur = first;
    for p in 0..n_tokens as i32 {
        let o = d.decode(&[cur], &[p]).expect("first-pass step");
        let row = o.to_vec();
        logits_at.push(row.clone());
        cur = argmax(&row);
        toks.push(cur);
    }

    // rollback 4: keep positions [0, 8) (the p0 = 8 removal drops the last
    // four tokens; rollback = 11 - 7 = 4 == n_rs_seq)
    let p0 = 8i32;
    d.seq_rm(0, p0, -1);
    assert_eq!(d.seq_pos_max(0), 7, "the raw cells rewound to position 7");

    // second pass: re-decode the recorded tokens from the snapshot — the
    // outputs must reproduce the first pass's bit-identically
    for (i, &t) in toks[p0 as usize..n_tokens].iter().enumerate() {
        let p = p0 + i as i32;
        let o = d.decode(&[t], &[p]).expect("second-pass step");
        assert_eq!(
            o,
            &logits_at[p as usize][..],
            "position {p}: the rolled-back decode must reproduce the first pass"
        );
    }
}

// ---------------------------------------------------------------------------
// generator (#[ignore]) — the parity runs drive the release llama-cli itself
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~150 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch7_write_synth() {
    for spec in [spec_deepseek4()] {
        let (n, bytes) = build_file(&spec);
        println!(
            "{:>14}: {:4} tensors, {:>10} bytes -> {}",
            format!("{}{}", spec.arch, spec.suffix),
            n,
            bytes,
            spec.path()
        );
    }
    let gguf = Gguf::open(spec_deepseek4().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!("\nparity: ARCH_BATCH7=1 ./parity/arch_batch_parity.sh deepseek4 deepseek4-long");
}

// ---------------------------------------------------------------------------
// node-dump mirror (parity/ref_decode_dump.c protocol) — env-driven, ignored:
//   DSV4_DUMP_OUT=/tmp/port_dsv4.bin DSV4_FA_OFF=1 \
//     cargo test --release -p llama --test arch_batch7_e2e -- --ignored \
//       --nocapture dsv4_prefill_node_dump
// then: python3 parity/decode_dump_cmp.py /tmp/ref_dsv4.bin /tmp/port_dsv4.bin
// ---------------------------------------------------------------------------

mod dump {
    use ggml::compute::{set_eval_callback, EvalNode};
    use ggml::types::GgmlType;
    use std::io::Write as _;
    use std::sync::{Mutex, OnceLock};

    pub const ELEM_CAP: u64 = 1 << 19;

    pub struct DumpState {
        pub out: Vec<u8>,
        pub nodes: u32,
    }

    pub static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();

    fn put_str(buf: &mut Vec<u8>, s: &str) {
        let len = s.len().min(255);
        buf.push(len as u8);
        buf.extend_from_slice(&s.as_bytes()[..len]);
    }

    /// ggml_op_desc(t) — including the UNARY / GLU specialisations
    /// (ggml.c:1380-1389) so the op-class keys pair with the reference stream
    fn op_desc(op: ggml::GgmlOp, params: &[i32]) -> &'static str {
        use ggml::GgmlOp::*;
        match op {
            None => "NONE",
            Dup => "DUP",
            Add => "ADD",
            Mul => "MUL",
            Div => "DIV",
            Sub => "SUB",
            Norm => {
                if params.len() > 1 && params[1] == 1 {
                    "RMS_NORM"
                } else {
                    "NORM"
                }
            }
            MulMat => "MUL_MAT",
            Scale => "SCALE",
            Cpy => "CPY",
            Reshape => "RESHAPE",
            View => "VIEW",
            Permute => "PERMUTE",
            Transpose => "TRANSPOSE",
            GetRows => "GET_ROWS",
            DiagMaskInf => "DIAG_MASK_INF",
            SoftMax => "SOFT_MAX",
            RoPE => "ROPE",
            RoPEBack => "rope_back(x)",
            MulMatId => "MUL_MAT_ID",
            Argsort => "ARGSORT",
            ArgMax => "ARGMAX",
            Repeat => "REPEAT",
            Concat => "CONCAT",
            // UNARY ops are described by their variant name (ggml.c:1382)
            Silu => match params.first().copied().unwrap_or(10) {
                4 => "TANH",
                7 => "SIGMOID",
                8 => "GELU",
                15 => "SOFTPLUS",
                _ => "SILU",
            },
            SumRows => "SUM_ROWS",
            SetRows => "SET_ROWS",
            FlashAttnExt => "FLASH_ATTN_EXT",
            AddId => "ADD_ID",
            // GLU ops are described by their variant name (ggml.c:1385-1389)
            Glu => match params.first().copied().unwrap_or(2) {
                0 => "REGLU",
                1 => "GEGLU",
                3 => "SWIGLU_OAI",
                6 => "SWIGLU_CLAMP",
                _ => "SWIGLU",
            },
            Clamp => "CLAMP",
            Fill => "FILL",
            LightningIndexer => "lightning_indexer(q, k, weights, mask)",
            TopK => "TOP_K",
            Sqrt => "SQRT",
            Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
            Dsv4HcPre => "dsv4_hc_pre(x, weights)",
            Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
            other => {
                let _ = other;
                "OTHER"
            }
        }
    }

    fn type_desc(ty: GgmlType) -> &'static str {
        match ty {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::I64 => "i64",
            GgmlType::I32 => "i32",
            _ => "other",
        }
    }

    fn dump_cb(node: &EvalNode<'_>, ask: bool) -> bool {
        if ask {
            return true;
        }
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let Some(st) = guard.as_mut() else {
            return true;
        };
        let n: i64 = node.ne.iter().product();
        st.nodes += 1;
        put_str(&mut st.out, op_desc(node.op, &node.op_params));
        put_str(&mut st.out, node.name);
        put_str(&mut st.out, type_desc(node.ty));
        st.out
            .extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
        st.out.extend_from_slice(&(n as u64).to_le_bytes());
        let data = node.data.unwrap_or(&[]);
        if n as u64 >= ELEM_CAP {
            return true;
        }
        if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
            st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
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
            let v: f32 = if node.ty == GgmlType::F32 {
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                half::f16::from_le_bytes([data[off], data[off + 1]]).to_f32()
            };
            st.out.extend_from_slice(&v.to_le_bytes());
        }
        true
    }

    /// one `--embeddings --pooling none` prefill of the batch-7 synth file,
    /// streaming every node in the DECDMP1 format
    pub fn dump_prefill(model_path: &str, fa: bool, prompt: &str) -> Vec<i32> {
        use ggml::{Context, Gguf};
        use llama::context::{DecodeContext, ForwardWeights};
        use llama::model::load_model;

        let gguf = Gguf::open(model_path).expect("open synth");
        let f = std::fs::File::open(model_path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
        let mut model = load_model(&gguf, mmap).expect("load synth");
        let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
        let ids = vocab.tokenize(prompt, true, true);

        let (weights, attn) = super::forward_of(&mut model, fa);
        let gctx: Context = std::mem::replace(&mut model.ctx, Context::new());
        let mut dctx = DecodeContext::new_with(gctx, weights, attn, 512, 8, 512);

        DUMP.get_or_init(|| {
            Mutex::new(Some(DumpState {
                out: Vec::new(),
                nodes: 0,
            }))
        });
        {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            *guard = Some(DumpState {
                out: Vec::new(),
                nodes: 0,
            });
        }
        set_eval_callback(Some(dump_cb));
        let pos: Vec<i32> = (0..ids.len() as i32).collect();
        // plain decode (n_outputs = 1): the reference's --embeddings path
        // asserts inside its own reserve-graph reuse for dsv4
        // (parity/ref_dsv4_nodes.c DSV4_NOEMB), so both sides dump the
        // logits-only graph
        let _ = dctx.decode(&ids, &pos).expect("decode");
        set_eval_callback(None);

        let out_path =
            std::env::var("DSV4_DUMP_OUT").unwrap_or_else(|_| "/tmp/port_dsv4.bin".to_string());
        let (nodes, body) = {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            let st = guard.take().unwrap();
            (st.nodes, st.out)
        };
        let mut f = std::fs::File::create(&out_path).expect("create dump");
        f.write_all(b"DECDMP1\0").unwrap();
        f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
        for &id in &ids {
            f.write_all(&id.to_le_bytes()).unwrap();
        }
        f.write_all(&nodes.to_le_bytes()).unwrap();
        f.write_all(&body).unwrap();
        println!(
            "dsv4 prefill dump: {nodes} nodes, {} bytes -> {out_path}",
            body.len()
        );
        ids
    }
}

#[test]
#[ignore = "manual: node-dump bisection helper (parity/ref_decode_dump.c protocol)"]
fn dsv4_prefill_node_dump() {
    let fa = std::env::var("DSV4_FA_OFF").is_err();
    let spec = spec_deepseek4();
    build_file(&spec);
    let ids = dump::dump_prefill(&spec.path(), fa, "The capital of France is");
    println!("tokens: {ids:?} (fa={})", if fa { "on" } else { "off" });
}
