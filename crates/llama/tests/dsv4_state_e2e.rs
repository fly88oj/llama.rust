//! dsv4_state_e2e.rs — the DSV4 sequence-state serialization port
//! (llama-kv-cache-dsv4.cpp:1080-1158 / :1594-1673 + the raw-cache halves of
//! llama-kv-cache.cpp:2055-2628 behind `llama_state_seq_get_data`, @
//! bd4f514db1) on the batch-7 synthetic deepseek4 model — the generator below
//! is tests/arch_batch7_e2e.rs's, byte-identical by construction (same table,
//! same RNG stream).
//!
//!   * default test `dsv4_state_round_trip`: decode N tokens, serialize
//!     sequence 0, restore into a *fresh* context on the same model, continue
//!     greedy — the continuation's logits must equal the uninterrupted run's
//!     bit-for-bit (the state carries the raw iswa pair + the three
//!     compressed K caches + the three compressor/recurrent planes, so a lost
//!     or mis-rowed plane shifts the logits immediately). Both FA modes.
//!   * default test `dsv4_state_format_self_consistency`: the dummy-mode size
//!     equals the real blob's length, and the partial_only blob is a strict
//!     prefix subset relationship (fewer bytes, same header).
//!   * `#[ignore]d dsv4_state_dump_blob` writes the port blob for
//!     `parity/dsv4_state_parity.sh`, whose reference side is
//!     parity/ref_dsv4_state.c (`llama_state_seq_get_data` on the same
//!     synthetic model) — the byte-format comparison cell.

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-dsv4state";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;
const N_EXPERT_SHARED: i64 = 1;

// ---------------------------------------------------------------------------
// the synthetic model spec — tests/arch_batch7_e2e.rs's deepseek4 spec
// ---------------------------------------------------------------------------

struct SynthSpec {
    arch: &'static str,
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
        format!("{OUT_DIR}/{}-synth-state.gguf", self.arch)
    }
}

fn spec_deepseek4() -> SynthSpec {
    SynthSpec {
        arch: "deepseek4",
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 1,
        key_length: 64,
        value_length: 64,
        q_lora_rank: 32,
        n_rot: 16,
        n_ff_exp: 24,
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
// the tensor table (exactly what load_arch_tensors asks for — batch 7's)
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
// writer (the batch-7 recipe: same RNG stream in table order → the same file
// bytes as /tmp/arch-batch7/deepseek4-synth.gguf)
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

fn build_file(spec: &SynthSpec) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");

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
    kv!(format!("{a}.expert_weights_scale"), Value::F32(2.5));
    kv!(format!("{a}.expert_gating_func"), Value::U32(4));
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
}

fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
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

/// the deepseek4 driver with the dsv4 layout knobs (`DecodeContext::
/// new_with_dsv4`): n_seq_max streams + n_rs_seq rollback planes — the
/// reference's default single-sequence configuration is (1, 0).
fn driver_for_dsv4(m: &mut LlamaModel, fa: bool, n_seq_max: u32, n_rs_seq: u32) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with_dsv4(gctx, weights, attn, 512, 8, 512, n_seq_max, n_rs_seq)
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

/// the fixed token stream parity/ref_dsv4_state.c decodes
const PROMPT: [i32; 16] = [3, 17, 42, 9, 21, 5, 8, 30, 11, 29, 2, 16, 4, 13, 25, 7];

// ---------------------------------------------------------------------------
// default-run tests
// ---------------------------------------------------------------------------

/// the round-trip criterion: decode N tokens, serialize sequence 0, restore
/// into a fresh context, continue greedy — the continuation's logits equal
/// the uninterrupted run's bit-for-bit. The checkpoint sits past a CSA block
/// boundary (pos 15 → blocks 0-3 complete) so the compressed caches and the
/// compressor/recurrent planes carry real state, and the continuation rolls
/// them further. Both FA modes.
#[test]
fn dsv4_state_round_trip() {
    let _files = file_lock();
    let spec = spec_deepseek4();
    build_file(&spec);

    const N_PRE: usize = PROMPT.len(); // 16
    const N_CKPT: usize = 4; // greedy steps before the checkpoint
    const N_CONT: usize = 8; // greedy steps compared after it

    for fa in [false, true] {
        // (a) the uninterrupted run: prefill + N_CKPT + N_CONT greedy steps.
        // `ids[s]` is the token step s decodes (at position N_PRE + s);
        // `ref_logits[s - N_CKPT]` the logits of the compared tail steps
        let (ids, ref_logits) = {
            let mut m = open_model(&spec.path());
            let mut d = driver_for_dsv4(&mut m, fa, 1, 0);
            let mut logits = d
                .decode(&PROMPT, &(0..N_PRE as i32).collect::<Vec<_>>())
                .expect("prefill")
                .to_vec();
            let mut ids = Vec::new();
            let mut tail = Vec::new();
            for s in 0..N_CKPT + N_CONT {
                let id = argmax(&logits);
                let p = (N_PRE + s) as i32;
                logits = d.decode(&[id], &[p]).expect("decode").to_vec();
                ids.push(id);
                if s >= N_CKPT {
                    tail.push(logits.clone());
                }
            }
            (ids, tail)
        };

        // (b) the checkpoint run: prefill + the first N_CKPT steps, then
        // `llama_state_seq_get_data(ctx, ..., seq 0)` (the full state)
        let blob = {
            let mut m = open_model(&spec.path());
            let mut d = driver_for_dsv4(&mut m, fa, 1, 0);
            let mut logits = d
                .decode(&PROMPT, &(0..N_PRE as i32).collect::<Vec<_>>())
                .expect("prefill")
                .to_vec();
            for s in 0..N_CKPT {
                let id = argmax(&logits);
                let p = (N_PRE + s) as i32;
                logits = d.decode(&[id], &[p]).expect("decode").to_vec();
            }
            let blob = d.state_seq_get_data(0, false);
            // `llama_state_seq_get_size` agrees with the real writer
            assert_eq!(
                blob.len(),
                d.state_seq_get_size(0, false),
                "fa={fa}: size mismatch"
            );
            blob
        };

        // (c) a FRESH context on the same model (a second model load — a new
        // ggml Context, new cache tensors), restore, replay the tail ids
        let mut m2 = open_model(&spec.path());
        let mut d2 = driver_for_dsv4(&mut m2, fa, 1, 0);
        d2.state_seq_set_data(0, &blob, false).expect("restore");

        // the restored sequence holds the checkpointed positions
        assert_eq!(
            d2.kv.seq_pos_max_of(0),
            (N_PRE + N_CKPT - 1) as i32,
            "fa={fa}"
        );

        for s in 0..N_CONT {
            let p = (N_PRE + N_CKPT + s) as i32;
            let logits = d2
                .decode(&[ids[N_CKPT + s]], &[p])
                .expect("continuation decode")
                .to_vec();
            assert_eq!(
                logits, ref_logits[s],
                "fa={fa}: continuation step {s} diverged after the restore"
            );
        }

        println!(
            "dsv4 state round-trip fa={fa}: blob {} bytes, {N_CONT}/{N_CONT} continuation steps \
             bit-identical",
            blob.len()
        );
    }
}

/// self-consistency of the format itself: the dummy-mode size equals the real
/// blob's length; the partial_only blob is shorter and shares the dsv4 frame
/// header; a corrupted magic is refused.
#[test]
fn dsv4_state_format_self_consistency() {
    let _files = file_lock();
    let spec = spec_deepseek4();
    build_file(&spec);

    let mut m = open_model(&spec.path());
    let mut d = driver_for_dsv4(&mut m, true, 1, 0);
    d.decode(&PROMPT, &(0..PROMPT.len() as i32).collect::<Vec<_>>())
        .expect("prefill");

    let full = d.state_seq_get_data(0, false);
    let partial = d.state_seq_get_data(0, true);
    assert_eq!(full.len(), d.state_seq_get_size(0, false));
    assert!(
        partial.len() < full.len(),
        "partial_only must skip the compressed halves ({} vs {})",
        partial.len(),
        full.len()
    );
    // the io framing (llama-context.cpp:3142/:3166-3167) then the dsv4 frame
    // (llama-kv-cache-dsv4.cpp:1597-1603): magic / version / mode
    assert_eq!(&full[..4], &DecodeContext::STATE_SEQ_IO_MAGIC.to_le_bytes());
    assert_eq!(&full[4..8], &0i32.to_le_bytes(), "the saved seq id");
    assert_eq!(
        &full[8..12],
        &llama::kv_cache::DSV4_STATE_MAGIC.to_le_bytes()
    );
    assert_eq!(
        &full[12..16],
        &llama::kv_cache::DSV4_STATE_VERSION.to_le_bytes()
    );
    assert_eq!(
        &full[16..20],
        &llama::kv_cache::DSV4_STATE_MODE_FULL.to_le_bytes()
    );
    assert_eq!(
        &partial[16..20],
        &llama::kv_cache::DSV4_STATE_MODE_PARTIAL.to_le_bytes()
    );

    // a corrupted magic is refused
    let mut bad = full.clone();
    bad[8] ^= 0xff; // the DSV4_STATE_MAGIC byte
    let mut m2 = open_model(&spec.path());
    let mut d2 = driver_for_dsv4(&mut m2, true, 1, 0);
    assert!(d2.state_seq_set_data(0, &bad, false).is_err());

    // and a good blob restores cleanly into the fresh context
    d2.state_seq_set_data(0, &full, false).expect("restore");
}

// ---------------------------------------------------------------------------
// the #[ignore] generator for the byte-format parity run
// (parity/dsv4_state_parity.sh drives the probe + this dumper)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "writes /tmp/arch-dsv4state/port-state.bin for parity/dsv4_state_parity.sh"]
fn dsv4_state_dump_blob() {
    let _files = file_lock();
    let spec = spec_deepseek4();
    build_file(&spec);
    let out = format!("{OUT_DIR}/port-state.bin");
    let out_tail = format!("{OUT_DIR}/port-state-tail.bin");

    for (path, tail) in [(out.as_str(), 0), (out_tail.as_str(), 8)] {
        let mut m = open_model(&spec.path());
        let mut d = driver_for_dsv4(&mut m, true, 1, 0);
        // the probe's geometry: fa on (v_trans 0), n_ctx/n_ubatch 512
        d.decode(&PROMPT, &(0..PROMPT.len() as i32).collect::<Vec<_>>())
            .expect("prefill");
        for s in 0..tail {
            let p = (PROMPT.len() + s as usize) as i32;
            d.decode(&[PROMPT[s as usize % PROMPT.len()]], &[p])
                .expect("tail decode");
        }
        let blob = d.state_seq_get_data(0, false);
        let mut bytes = Vec::with_capacity(8 + blob.len());
        bytes.extend_from_slice(b"DS4S");
        bytes.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&blob);
        std::fs::write(path, bytes).unwrap();
        println!(
            "dsv4 port state (tail {tail}): {} bytes -> {path}",
            blob.len()
        );
    }
}
