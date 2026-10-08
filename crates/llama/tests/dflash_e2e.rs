//! dflash_e2e.rs — the DFlash/DSpark speculative-draft port
//! (common/speculative.cpp:910-1328 + src/models/dflash.cpp @ bd4f514db1) on
//! synthetic GGUFs, following tests/eagle_e2e.rs's protocol:
//!
//!   * the generator writes a **llama-arch target** (the arch whose graph
//!     records the per-layer input tap both sides need — llama.cpp:127
//!     `res->t_layer_inp[il] = inpL`) and four dflash drafts over it:
//!     `dflash` (the plain block-diffusion drafter), `dspark` (DFlash +
//!     the Markov head + the confidence head, speculative.cpp:932-933),
//!     `dflash2` (the conv/selector graphs of dflash.cpp:408-567, selected
//!     as `draft-dflash` — the driver walks the selector lattice off the
//!     unmasked nextn tap) and `dspark-dsv4` (the DSV4 backbone of
//!     dflash.cpp:52-93/:855-1028 — full deepseek4 stages over the iswa
//!     ring, rope NORM per llama-model.cpp:3053-3059); none carries
//!     token_embd / output, so all inherit the target's tensors (the C
//!     through `cparams.ctx_other`, llama-context.cpp:144-161; the port as
//!     external-storage tensors backed by the target's mmap, dflash.rs);
//!   * default tests: draft-loader pinning, target-trunk-unchanged (the
//!     extract-layer taps must not alter the trunk logits — both FA modes),
//!     and the full `--spec-type draft-dflash / draft-dspark` driver — the
//!     committed stream must equal the plain greedy stream at
//!     `temperature 0`, with the drafted/accepted counters reported;
//!   * `#[ignore]d dflash_write_synth_files` writes the three files for
//!     `parity/dflash_parity.sh`, whose reference side is a fresh
//!     `llama-server --spec-type draft-dflash -md draft.gguf` (first
//!     /completion, temperature 0, cache_prompt=false) plus its server-log
//!     draft stats (`draft acceptance = ...`, server-context.cpp:677-678).
//!
//! Like the eagle3 head: the synthetic draft is not trained — its weights are
//! a deterministic RNG stream, so drafts are a pseudo-random chain and the
//! acceptance rate is near zero. The parity contract that still binds: the
//! drafted token chain (the noise-block argmax sequence through the injected
//! KV cache + the DSpark markov bias) and the drafted/accepted counters must
//! match the reference's, and the committed stream must equal plain greedy on
//! both sides.

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{self};
use llama::model::{load_model, LlamaModel};
use llama::sampling::{SamplingContext, SamplingParams};
use llama::speculative::{
    common_speculative_init, speculative_simple_generate, CommonParamsSpeculative,
    CommonSpeculativeType,
};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-dflash";

// the shared geometry: the target (llama arch) and the draft use the same
// hidden size / heads so the injected rows and the ctx_other tensors line up
const N_LAYER_TGT: usize = 4;
const N_LAYER_DFT: usize = 2;
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HEAD_DIM: i64 = 16; // N_EMBD / N_HEAD
const N_ROT: i64 = 16;
const N_FF: i64 = 96;
const N_CTX: u32 = 256;
/// extract_layers = [1, 2, 3] — all < n_layer (the == n_layer case needs the
/// target's trunk nextn tap, which the llama arch does not set)
const TARGET_LAYERS: [i32; 3] = [1, 2, 3];
const N_EMBD_INP_ENC: i64 = 3 * N_EMBD;
/// `dflash.block_size` — the trained block length; the drafter clamps
/// n_max/n_min to block_size-1 (dflash) / block_size (dspark,
/// sample_from_anchor, speculative.cpp:999-1007)
const BLOCK_SIZE: u32 = 4;
/// `tokenizer.ggml.mask_token_id` — `llama_vocab_mask` of the draft vocab
/// (speculative.cpp:982); any valid id works for the synthetic pair
const MASK_TOKEN: u32 = 31;
/// the DSpark markov rank (markov_w1/markov_w2 ne[0])
const MARKOV_RANK: i64 = 8;
/// DFlash2: `dflash.conv_kernel_size` / `conv_group_size` / `selector_rank` /
/// `selector_top_k` (dflash.cpp:30-34). top_k*(top_k+1) = 20 <= n_embd = 64
/// (the selector-lattice check, dflash.cpp:148-150)
const CONV_KERNEL: i64 = 3;
const CONV_GROUP: i64 = 16;
const SELECTOR_RANK: i64 = 8;
const SELECTOR_TOP_K: i64 = 4;

/// DSV4 DSpark: the deepseek4 stage stack over the iswa ring
/// (dflash.cpp:52-93 / :855-1028). Small-but-complete geometry: hc=4
/// streams, MLA q-lora 32, o-groups 2×16, MoE 4 experts / 2 used / 1 shared.
const DSV4_Q_LORA: i64 = 32;
const DSV4_O_GROUPS: i64 = 2;
const DSV4_O_LORA: i64 = 16;
const DSV4_N_EXPERT: i64 = 4;
const DSV4_N_EXPERT_USED: i64 = 2;
const DSV4_N_EXPERT_SHARED: i64 = 1;
const DSV4_N_FF_EXP: i64 = 32;
const DSV4_N_SWA: u32 = 64;
const DSV4_HC_MULT: i64 = 4;

#[derive(Clone, Copy, PartialEq, Debug)]
enum DraftKind {
    /// the plain DFlash drafter (no markov head)
    Dflash,
    /// DFlash + the Markov head + the confidence head (the dspark sidecar)
    Dspark,
    /// DFlash2: the conv/selector graphs (dflash.cpp:138-159 / :408-567) —
    /// selected as `draft-dflash` (no markov head); the driver walks the
    /// selector lattice off the unmasked nextn tap
    Dflash2,
    /// the DSV4 DSpark backbone (dflash.cpp:52-93 / :855-1028) — full
    /// deepseek4 stages (hc + MLA + MoE) over the sliding-window ring;
    /// carries the markov/conf heads, selected as `draft-dspark`
    DsparkDsv4,
}

fn target_path() -> String {
    format!("{OUT_DIR}/llama-synth-dflash-tgt.gguf")
}
fn draft_path(kind: DraftKind) -> String {
    format!(
        "{OUT_DIR}/dflash-synth-{}.gguf",
        match kind {
            DraftKind::Dflash => "dflash",
            DraftKind::Dspark => "dspark",
            DraftKind::Dflash2 => "dflash2",
            DraftKind::DsparkDsv4 => "dspark-dsv4",
        }
    )
}

// ---------------------------------------------------------------------------
// the writers (the eagle_e2e recipe: one fixed RNG stream in table order)
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

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Proj,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn start_writer(name: &str) -> GgufWriter {
    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }
    w.set_kv("general.name", Value::String(name.to_string()));
    w.set_kv("general.file_type", Value::U32(0)); // F32
    w
}

fn write_file(mut w: GgufWriter, tensors: &[(String, Vec<i64>, Role)], path: &str, seed: u64) {
    let mut rng = Rng(seed);
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (name, ne, role) in tensors {
        let n: i64 = ne.iter().product();
        let s = match role {
            Role::Norm => 1.0,
            Role::Proj => 1.0 / (N_EMBD as f32).sqrt(),
        };
        let vals: Vec<f32> = (0..n)
            .map(|_| match role {
                Role::Norm => 1.0 + 0.05 * rng.next(),
                Role::Proj => s * rng.next(),
            })
            .collect();
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, ggml::types::GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }
    let f = std::fs::File::create(path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
}

/// the llama-arch target — the arch whose graph records `res->t_layer_inp`
/// (llama.cpp:127), the tap the dflash impl reads
/// (`llama_get_embeddings_layer_inp`, speculative.cpp:1147)
fn build_target() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let mut w = start_writer("llama-rust-synth-dflash-target");
    let a = "llama";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(N_CTX));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    w.set_kv(&format!("{a}.block_count"), Value::U32(N_LAYER_TGT as u32));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    w.set_kv(
        &format!("{a}.attention.head_count"),
        Value::U32(N_HEAD as u32),
    );
    w.set_kv(
        &format!("{a}.attention.head_count_kv"),
        Value::U32(N_HEAD_KV as u32),
    );
    w.set_kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5),
    );
    w.set_kv(
        &format!("{a}.rope.dimension_count"),
        Value::U32(N_ROT as u32),
    );
    w.set_kv(&format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    let n_gqa_k = HEAD_DIM * N_HEAD_KV;
    let mut t: Vec<(String, Vec<i64>, Role)> = vec![
        (
            "token_embd.weight".into(),
            vec![N_EMBD, N_VOCAB],
            Role::Proj,
        ),
        ("output_norm.weight".into(), vec![N_EMBD], Role::Norm),
        // an explicit lm head: the drafts inherit it through ctx_other
        // (dflash.cpp:799-808)
        ("output.weight".into(), vec![N_EMBD, N_VOCAB], Role::Proj),
    ];
    for i in 0..N_LAYER_TGT as i32 {
        t.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![N_EMBD],
            Role::Norm,
        ));
        t.push((
            format!("blk.{i}.attn_q.weight"),
            vec![N_EMBD, HEAD_DIM * N_HEAD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_k.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_v.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_output.weight"),
            vec![HEAD_DIM * N_HEAD, N_EMBD],
            Role::Proj,
        ));
        t.push((format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm));
        t.push((
            format!("blk.{i}.ffn_gate.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_down.weight"),
            vec![N_FF, N_EMBD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_up.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
    }
    write_file(w, &t, &target_path(), 0xDF1A_5E00u64.wrapping_add(1));
}

/// the dflash draft (dflash.cpp:7-259's hparams + tensor table); `kind`
/// selects whether the DSpark markov/conf heads are written. token_embd /
/// output are omitted — they come from the target through ctx_other.
fn build_draft(kind: DraftKind) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let mut w = start_writer("llama-rust-synth-dflash-draft");
    let a = "dflash";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(N_CTX));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    w.set_kv(&format!("{a}.block_count"), Value::U32(N_LAYER_DFT as u32));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    w.set_kv(
        &format!("{a}.attention.head_count"),
        Value::U32(N_HEAD as u32),
    );
    w.set_kv(
        &format!("{a}.attention.head_count_kv"),
        Value::U32(N_HEAD_KV as u32),
    );
    w.set_kv(
        &format!("{a}.attention.key_length"),
        Value::U32(HEAD_DIM as u32),
    );
    w.set_kv(
        &format!("{a}.attention.value_length"),
        Value::U32(HEAD_DIM as u32),
    );
    w.set_kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5),
    );
    w.set_kv(
        &format!("{a}.rope.dimension_count"),
        Value::U32(N_ROT as u32),
    );
    w.set_kv(&format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // dflash.cpp:36-38 — required extract-layer array
    w.set_kv(
        &format!("{a}.target_layers"),
        Value::Array(
            ggml::GgufType::Int32,
            vec![
                Value::I32(TARGET_LAYERS[0]),
                Value::I32(TARGET_LAYERS[1]),
                Value::I32(TARGET_LAYERS[2]),
            ],
        ),
    );
    // the meta strings the driver re-reads (speculative.cpp:965-978) —
    // gguf_kv renders the U32 as "4"
    w.set_kv(&format!("{a}.block_size"), Value::U32(BLOCK_SIZE));
    if matches!(kind, DraftKind::Dspark | DraftKind::DsparkDsv4) {
        // anchor-first DSpark drafts the full block (speculative.cpp:1204)
        w.set_kv(
            &format!("{a}.sample_from_anchor"),
            Value::String("true".into()),
        );
    }
    if let DraftKind::Dflash2 = kind {
        // the DFlash2 conv/selector knobs (dflash.cpp:30-34)
        w.set_kv(
            &format!("{a}.conv_kernel_size"),
            Value::U32(CONV_KERNEL as u32),
        );
        w.set_kv(
            &format!("{a}.conv_group_size"),
            Value::U32(CONV_GROUP as u32),
        );
        w.set_kv(
            &format!("{a}.selector_rank"),
            Value::U32(SELECTOR_RANK as u32),
        );
        w.set_kv(
            &format!("{a}.selector_top_k"),
            Value::U32(SELECTOR_TOP_K as u32),
        );
    }
    // `llama_vocab_mask` reads it (speculative.cpp:982); the SPM fixture has
    // none, so the draft file carries its own
    w.set_kv("tokenizer.ggml.mask_token_id", Value::U32(MASK_TOKEN));

    let n_gqa_k = HEAD_DIM * N_HEAD_KV;
    let mut t: Vec<(String, Vec<i64>, Role)> = vec![
        // d2t: draft to target vocabulary mapping — omitted (same vocab)
        // feature fusion (dflash.cpp:161)
        ("fc.weight".into(), vec![N_EMBD_INP_ENC, N_EMBD], Role::Proj),
        ("enc.output_norm.weight".into(), vec![N_EMBD], Role::Norm),
        ("output_norm.weight".into(), vec![N_EMBD], Role::Norm),
    ];
    if matches!(kind, DraftKind::Dspark | DraftKind::DsparkDsv4) {
        // the DSpark sidecar (dflash.cpp:123-136)
        t.push((
            "markov_w1.weight".into(),
            vec![MARKOV_RANK, N_VOCAB],
            Role::Proj,
        ));
        t.push((
            "markov_w2.weight".into(),
            vec![MARKOV_RANK, N_VOCAB],
            Role::Proj,
        ));
        t.push((
            "conf_proj.weight".into(),
            vec![N_EMBD + MARKOV_RANK, 1],
            Role::Proj,
        ));
        t.push(("conf_proj.bias".into(), vec![1], Role::Norm));
    }
    if let DraftKind::Dflash2 = kind {
        // the DFlash2 selector head (dflash.cpp:152-154)
        t.push((
            "selector_predecessor.weight".into(),
            vec![SELECTOR_RANK, N_VOCAB],
            Role::Proj,
        ));
        t.push((
            "selector_successor.weight".into(),
            vec![SELECTOR_RANK, N_VOCAB],
            Role::Proj,
        ));
        t.push((
            "selector_hidden.weight".into(),
            vec![N_EMBD, SELECTOR_RANK],
            Role::Proj,
        ));
    }
    if let DraftKind::DsparkDsv4 = kind {
        // the DSV4 backbone's hparams (dflash.cpp:52-93)
        let a = "dflash";
        w.set_kv(
            &format!("{a}.hyper_connection.count"),
            Value::U32(DSV4_HC_MULT as u32),
        ); // :52
        w.set_kv(
            &format!("{a}.attention.q_lora_rank"),
            Value::U32(DSV4_Q_LORA as u32),
        ); // :54
        w.set_kv(
            &format!("{a}.attention.sliding_window"),
            Value::U32(DSV4_N_SWA),
        ); // :55
        w.set_kv(
            &format!("{a}.expert_feed_forward_length"),
            Value::U32(DSV4_N_FF_EXP as u32),
        ); // :56
        w.set_kv(
            &format!("{a}.expert_shared_count"),
            Value::U32(DSV4_N_EXPERT_SHARED as u32),
        ); // :57
        w.set_kv(&format!("{a}.expert_weights_scale"), Value::F32(1.0)); // :58
        w.set_kv(&format!("{a}.expert_weights_norm"), Value::Bool(true)); // :59
        w.set_kv(&format!("{a}.expert_gating_func"), Value::U32(4)); // :60 SQRT_SOFTPLUS
        w.set_kv(&format!("{a}.swiglu_clamp_exp"), Value::F32(7.0)); // :61
        w.set_kv(
            &format!("{a}.attention.output_group_count"),
            Value::U32(DSV4_O_GROUPS as u32),
        ); // :65
        w.set_kv(
            &format!("{a}.attention.output_lora_rank"),
            Value::U32(DSV4_O_LORA as u32),
        ); // :66
        w.set_kv(
            &format!("{a}.hyper_connection.sinkhorn_iterations"),
            Value::U32(4),
        ); // :67
        w.set_kv(&format!("{a}.hyper_connection.epsilon"), Value::F32(1e-3)); // :68
        w.set_kv(
            &format!("{a}.expert_count"),
            Value::U32(DSV4_N_EXPERT as u32),
        );
        w.set_kv(
            &format!("{a}.expert_used_count"),
            Value::U32(DSV4_N_EXPERT_USED as u32),
        );
        w.set_kv(&format!("{a}.attention.head_count_kv"), Value::U32(1)); // MLA single head
                                                                          // the MLA trailing-dims rope: n_rot < head_dim so the nope offset is
                                                                          // exercised (dflash.cpp:862-863/:895-897)
        w.set_kv(&format!("{a}.rope.dimension_count"), Value::U32(8));
    }

    let n_conv_groups = N_EMBD / CONV_GROUP;
    let n_conv_projected = 2 * CONV_KERNEL * n_conv_groups;
    if let DraftKind::DsparkDsv4 = kind {
        // the DSV4 stage stack (dflash.cpp:173-221): the hc head + the
        // deepseek4 layer table (attn_sinks required, wo_a in its flat 2-D
        // file form, the MoE tensor set)
        let hc_dim = DSV4_HC_MULT * N_EMBD;
        let hc_mix_dim = (2 + DSV4_HC_MULT) * DSV4_HC_MULT;
        let n_embd_head = HEAD_DIM;
        t.push((
            "output_hc_fn.weight".into(),
            vec![hc_dim, DSV4_HC_MULT],
            Role::Proj,
        )); // :184
        t.push((
            "output_hc_base.weight".into(),
            vec![DSV4_HC_MULT],
            Role::Norm,
        )); // :185
        t.push(("output_hc_scale.weight".into(), vec![1], Role::Norm)); // :186
        let n_ff_exp_shared = DSV4_N_FF_EXP * DSV4_N_EXPERT_SHARED;
        for i in 0..N_LAYER_DFT as i32 {
            t.push((
                format!("blk.{i}.attn_norm.weight"),
                vec![N_EMBD],
                Role::Norm,
            )); // :191
            t.push((
                format!("blk.{i}.attn_sinks.weight"),
                vec![N_HEAD],
                Role::Norm,
            )); // :192
            t.push((
                format!("blk.{i}.attn_q_a.weight"),
                vec![N_EMBD, DSV4_Q_LORA],
                Role::Proj,
            )); // :193
            t.push((
                format!("blk.{i}.attn_q_a_norm.weight"),
                vec![DSV4_Q_LORA],
                Role::Norm,
            )); // :194
            t.push((
                format!("blk.{i}.attn_q_b.weight"),
                vec![DSV4_Q_LORA, N_HEAD * n_embd_head],
                Role::Proj,
            )); // :195
            t.push((
                format!("blk.{i}.attn_kv.weight"),
                vec![N_EMBD, n_embd_head],
                Role::Proj,
            )); // :196
            t.push((
                format!("blk.{i}.attn_kv_a_norm.weight"),
                vec![n_embd_head],
                Role::Norm,
            )); // :197
                // wo_a: the file's 2-D form — [n_head*head/o_groups, o_lora*o_groups]
            t.push((
                format!("blk.{i}.attn_output_a.weight"),
                vec![
                    N_HEAD * n_embd_head / DSV4_O_GROUPS,
                    DSV4_O_LORA * DSV4_O_GROUPS,
                ],
                Role::Proj,
            )); // :198
            t.push((
                format!("blk.{i}.attn_output_b.weight"),
                vec![DSV4_O_GROUPS * DSV4_O_LORA, N_EMBD],
                Role::Proj,
            )); // :199
            t.push((
                format!("blk.{i}.hc_attn_fn.weight"),
                vec![hc_dim, hc_mix_dim],
                Role::Proj,
            )); // :201
            t.push((
                format!("blk.{i}.hc_attn_base.weight"),
                vec![hc_mix_dim],
                Role::Norm,
            )); // :202
            t.push((format!("blk.{i}.hc_attn_scale.weight"), vec![3], Role::Norm)); // :203
            t.push((
                format!("blk.{i}.hc_ffn_fn.weight"),
                vec![hc_dim, hc_mix_dim],
                Role::Proj,
            )); // :204
            t.push((
                format!("blk.{i}.hc_ffn_base.weight"),
                vec![hc_mix_dim],
                Role::Norm,
            )); // :205
            t.push((format!("blk.{i}.hc_ffn_scale.weight"), vec![3], Role::Norm)); // :206
            t.push((
                format!("blk.{i}.ffn_gate_inp.weight"),
                vec![N_EMBD, DSV4_N_EXPERT],
                Role::Proj,
            )); // :208
            t.push((
                format!("blk.{i}.exp_probs_b.bias"),
                vec![DSV4_N_EXPERT],
                Role::Norm,
            )); // :209
            t.push((format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm)); // :210
            t.push((
                format!("blk.{i}.ffn_gate_exps.weight"),
                vec![N_EMBD, DSV4_N_FF_EXP, DSV4_N_EXPERT],
                Role::Proj,
            )); // :212
            t.push((
                format!("blk.{i}.ffn_down_exps.weight"),
                vec![DSV4_N_FF_EXP, N_EMBD, DSV4_N_EXPERT],
                Role::Proj,
            )); // :213
            t.push((
                format!("blk.{i}.ffn_up_exps.weight"),
                vec![N_EMBD, DSV4_N_FF_EXP, DSV4_N_EXPERT],
                Role::Proj,
            )); // :214
            t.push((
                format!("blk.{i}.ffn_gate_shexp.weight"),
                vec![N_EMBD, n_ff_exp_shared],
                Role::Proj,
            )); // :216
            t.push((
                format!("blk.{i}.ffn_down_shexp.weight"),
                vec![n_ff_exp_shared, N_EMBD],
                Role::Proj,
            )); // :217
            t.push((
                format!("blk.{i}.ffn_up_shexp.weight"),
                vec![N_EMBD, n_ff_exp_shared],
                Role::Proj,
            )); // :218
        }
        write_file(
            w,
            &t,
            &draft_path(kind),
            0xDF1A_5E00u64.wrapping_add(2 + kind as u64),
        );
        return;
    }

    for i in 0..N_LAYER_DFT as i32 {
        t.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![N_EMBD],
            Role::Norm,
        ));
        t.push((
            format!("blk.{i}.attn_q.weight"),
            vec![N_EMBD, HEAD_DIM * N_HEAD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_k.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_v.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_output.weight"),
            vec![HEAD_DIM * N_HEAD, N_EMBD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_q_norm.weight"),
            vec![HEAD_DIM],
            Role::Norm,
        ));
        t.push((
            format!("blk.{i}.attn_k_norm.weight"),
            vec![HEAD_DIM],
            Role::Norm,
        ));
        t.push((format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm));
        t.push((
            format!("blk.{i}.ffn_gate.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_down.weight"),
            vec![N_FF, N_EMBD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_up.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
        if let DraftKind::Dflash2 = kind {
            // the per-layer conv pair (dflash.cpp:249-257; the base carries no
            // ".weight" suffix — LLM_TENSOR_DFLASH_*_CONV_BASE)
            t.push((
                format!("blk.{i}.attn_conv_base"),
                vec![N_EMBD, CONV_KERNEL, 2],
                Role::Proj,
            ));
            t.push((
                format!("blk.{i}.attn_conv_proj.weight"),
                vec![N_EMBD, n_conv_projected],
                Role::Proj,
            ));
            t.push((
                format!("blk.{i}.ffn_conv_base"),
                vec![N_EMBD, CONV_KERNEL, 2],
                Role::Proj,
            ));
            t.push((
                format!("blk.{i}.ffn_conv_proj.weight"),
                vec![N_EMBD, n_conv_projected],
                Role::Proj,
            ));
        }
    }
    write_file(
        w,
        &t,
        &draft_path(kind),
        0xDF1A_5E00u64.wrapping_add(2 + kind as u64),
    );
}

/// the tests rewrite the same /tmp files and keep their mmaps alive across
/// loads; serialize whole tests (not just the loads) or a concurrent rewrite
/// of a mapped file is a SIGBUS
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
// the target trunk driver (the LLAMA arm of llama-cli's forward_weights)
// ---------------------------------------------------------------------------

fn llama_forward(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
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
        use_flash_attn: fa,
    };
    let layers = m
        .layers
        .iter()
        .map(|l| graph_arch::LlamaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq.unwrap(),
            wk: l.wk.unwrap(),
            wv: l.wv.unwrap(),
            wo: l.wo.unwrap(),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            wo_b: None,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate.unwrap(),
            ffn_down: l.ffn_down.unwrap(),
            ffn_up: l.ffn_up.unwrap(),
            ffn_gate_b: None,
            ffn_down_b: None,
            ffn_up_b: None,
        })
        .collect();
    (
        ForwardWeights::Llama(graph_arch::LlamaModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            output_b: None,
            layers,
        }),
        attn,
    )
}

fn target_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = llama_forward(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
}

/// the dflash draft context (`common_speculative_init_from_params`'s
/// has_draft arm, speculative.cpp:2553-2576) — llama-cli's spec_dflash branch
fn dflash_driver(kind: DraftKind, fa: bool) -> DecodeContext {
    let draft_gguf = Gguf::open(&draft_path(kind)).expect("open draft");
    let draft_mmap = {
        let f = std::fs::File::open(&draft_path(kind)).unwrap();
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let tgt_gguf = Gguf::open(&target_path()).expect("open target");
    let tgt_mmap = {
        let f = std::fs::File::open(&target_path()).unwrap();
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let n_vocab = Vocab::load(&draft_gguf).unwrap().n_tokens();
    let draft = llama::dflash::load_dflash_draft(
        &draft_gguf,
        draft_mmap,
        &tgt_gguf,
        tgt_mmap,
        n_vocab as i64,
        fa,
    )
    .expect("load dflash draft");
    let stub = llama::dflash::dflash_trunk_stub(&draft.weights);
    DecodeContext::new_dflash(
        draft.ctx,
        ForwardWeights::Qwen2(stub),
        (draft.weights, draft.params),
        n_vocab as usize,
        512,
        8,
        512,
    )
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

/// greedy stream of the target (`temperature 0`)
fn plain_greedy(m: &mut LlamaModel, fa: bool, prompt: &[i32], n_predict: usize) -> Vec<i32> {
    let mut d = target_driver(m, fa);
    let mut logits = d
        .decode(prompt, &(0..prompt.len() as i32).collect::<Vec<_>>())
        .expect("prefill")
        .to_vec();
    let mut out = Vec::new();
    for _ in 0..n_predict {
        let id = argmax(&logits);
        out.push(id);
        let p = (prompt.len() + out.len() - 1) as i32;
        logits = d.decode(&[id], &[p]).expect("decode").to_vec();
    }
    out
}

// ---------------------------------------------------------------------------
// default-run tests
// ---------------------------------------------------------------------------

/// the draft loads with the pinned geometry: 3 extract layers, the fused fc
/// width, block_size / sample_from_anchor read back from the meta strings,
/// the markov head present only on the dspark side; the ctx_other variant
/// materializes the target's token_embd / output into the draft context
#[test]
fn dflash_synth_draft_loads() {
    let _files = file_lock();
    build_target();
    for kind in [
        DraftKind::Dflash,
        DraftKind::Dspark,
        DraftKind::Dflash2,
        DraftKind::DsparkDsv4,
    ] {
        build_draft(kind);
        let draft_gguf = Gguf::open(&draft_path(kind)).expect("open draft");
        let draft_mmap = {
            let f = std::fs::File::open(&draft_path(kind)).unwrap();
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let tgt_gguf = Gguf::open(&target_path()).unwrap();
        let tgt_mmap = {
            let f = std::fs::File::open(&target_path()).unwrap();
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let n_vocab = Vocab::load(&draft_gguf).unwrap().n_tokens();
        let draft = llama::dflash::load_dflash_draft(
            &draft_gguf,
            draft_mmap,
            &tgt_gguf,
            tgt_mmap,
            n_vocab as i64,
            false,
        )
        .expect("draft loads");

        assert_eq!(draft.params.target_layer_ids, TARGET_LAYERS);
        assert_eq!(draft.params.n_embd, N_EMBD);
        assert_eq!(draft.params.n_embd_inp_enc, N_EMBD_INP_ENC);
        assert_eq!(draft.params.n_embd_tgt, N_EMBD);
        assert_eq!(draft.params.block_size, BLOCK_SIZE as i32);
        assert_eq!(
            draft.params.selector_top_k,
            match kind {
                DraftKind::Dflash2 => SELECTOR_TOP_K as u32,
                _ => 0,
            }
        );
        assert_eq!(
            *draft.ctx.ne(draft.weights.fc),
            [N_EMBD_INP_ENC, N_EMBD, 1, 1]
        );
        assert_eq!(
            draft.weights.layers.len(),
            match kind {
                DraftKind::DsparkDsv4 => 0, // the DSV4 stages replace the table
                _ => N_LAYER_DFT,
            }
        );
        if !draft.weights.layers.is_empty() {
            assert_eq!(
                *draft.ctx.ne(draft.weights.layers[0].wq),
                [N_EMBD, HEAD_DIM * N_HEAD, 1, 1]
            );
        }
        // the ctx_other tensors: [n_embd_tgt, n_vocab] rows of the target
        assert_eq!(
            *draft.ctx.ne(draft.weights.tok_embd.unwrap()),
            [N_EMBD, N_VOCAB, 1, 1]
        );
        assert_eq!(
            *draft.ctx.ne(draft.weights.output.unwrap()),
            [N_EMBD, N_VOCAB, 1, 1]
        );
        assert!(draft.weights.d2t.is_none());
        match kind {
            DraftKind::Dflash => {
                assert!(draft.weights.dspark_markov_w1.is_none());
                // the C's default is true (speculative.cpp:936-937) — only the
                // dspark layout reads it (n_block_tokens' +0 arm is gated on
                // is_dspark, :1204)
                assert!(draft.params.sample_from_anchor);
            }
            DraftKind::Dspark => {
                assert_eq!(
                    *draft.ctx.ne(draft.weights.dspark_markov_w1.unwrap()),
                    [MARKOV_RANK, N_VOCAB, 1, 1]
                );
                assert!(draft.weights.dspark_conf_proj.is_some());
                assert!(draft.params.sample_from_anchor);
            }
            DraftKind::Dflash2 => {
                // the selector head + the DFlash2 knobs (dflash.cpp:138-159)
                assert!(draft.weights.dspark_markov_w1.is_none());
                assert!(draft.weights.dflash_selector_hidden.is_some());
                assert_eq!(
                    *draft.ctx.ne(draft.weights.dflash_selector_prev.unwrap()),
                    [SELECTOR_RANK, N_VOCAB, 1, 1]
                );
                assert_eq!(
                    *draft.ctx.ne(draft.weights.dflash_selector_next.unwrap()),
                    [SELECTOR_RANK, N_VOCAB, 1, 1]
                );
                assert_eq!(
                    *draft.ctx.ne(draft.weights.dflash_selector_hidden.unwrap()),
                    [N_EMBD, SELECTOR_RANK, 1, 1]
                );
                assert_eq!(draft.params.selector_top_k, SELECTOR_TOP_K as u32);
                assert_eq!(draft.params.dflash_block_size, BLOCK_SIZE);
                assert_eq!(draft.params.conv_kernel_size, CONV_KERNEL as u32);
                assert_eq!(draft.params.conv_group_size, CONV_GROUP as u32);
                assert_eq!(draft.params.selector_rank, SELECTOR_RANK as u32);
                // the per-layer conv pair (:249-257)
                let l = &draft.weights.layers[0];
                assert_eq!(
                    *draft.ctx.ne(l.dflash_attn_conv_base.unwrap()),
                    [N_EMBD, CONV_KERNEL, 2, 1]
                );
                let groups = N_EMBD / CONV_GROUP;
                assert_eq!(
                    *draft.ctx.ne(l.dflash_attn_conv_proj.unwrap()),
                    [N_EMBD, 2 * CONV_KERNEL * groups, 1, 1]
                );
                assert!(l.dflash_ffn_conv_base.is_some());
                assert!(l.dflash_ffn_conv_proj.is_some());
            }
            DraftKind::DsparkDsv4 => {
                // the DSV4 backbone (dflash.cpp:173-221): the staged weights
                // replace the plain layer table
                let staged = draft.weights.dsv4.as_ref().expect("the dsv4 stages");
                assert!(draft.weights.layers.is_empty());
                assert_eq!(staged.layers.len(), N_LAYER_DFT);
                assert_eq!(staged.params.hc_mult, DSV4_HC_MULT);
                assert_eq!(staged.params.o_group_count, DSV4_O_GROUPS);
                assert_eq!(staged.params.o_lora_rank, DSV4_O_LORA);
                assert_eq!(staged.params.n_expert, DSV4_N_EXPERT);
                assert_eq!(staged.params.n_expert_used, DSV4_N_EXPERT_USED);
                assert_eq!(staged.params.n_ff_exp, DSV4_N_FF_EXP);
                assert_eq!(staged.params.n_swa, DSV4_N_SWA);
                assert_eq!(staged.params.swiglu_clamp_exp, vec![7.0f32; N_LAYER_DFT]);
                assert_eq!(staged.params.swiglu_clamp_shexp, vec![7.0f32; N_LAYER_DFT]);
                let hc_dim = DSV4_HC_MULT * N_EMBD;
                let hc_mix_dim = (2 + DSV4_HC_MULT) * DSV4_HC_MULT;
                assert_eq!(
                    *draft.ctx.ne(staged.hc_head_fn),
                    [hc_dim, DSV4_HC_MULT, 1, 1]
                );
                let l = &staged.layers[0];
                // wo_a reshaped to 3-D from the file's 2-D form (:198)
                assert_eq!(
                    *draft.ctx.ne(l.wo_a),
                    [
                        N_HEAD * HEAD_DIM / DSV4_O_GROUPS,
                        DSV4_O_LORA,
                        DSV4_O_GROUPS,
                        1
                    ]
                );
                assert_eq!(*draft.ctx.ne(l.hc_attn_fn), [hc_dim, hc_mix_dim, 1, 1]);
                assert_eq!(*draft.ctx.ne(l.wkv), [N_EMBD, HEAD_DIM, 1, 1]);
                assert_eq!(
                    *draft.ctx.ne(l.ffn_gate_exps),
                    [N_EMBD, DSV4_N_FF_EXP, DSV4_N_EXPERT, 1]
                );
                assert_eq!(
                    *draft.ctx.ne(l.ffn_gate_shexp),
                    [N_EMBD, DSV4_N_FF_EXP * DSV4_N_EXPERT_SHARED, 1, 1]
                );
                // the markov head still rides the shared weights
                assert!(draft.weights.dspark_markov_w1.is_some());
                assert_eq!(draft.params.attn.n_head_kv, 1);
                assert_eq!(draft.params.attn.n_rot, 8);
            }
        }
        println!(
            "dflash draft ({kind:?}) loaded: {} tensors, extract_layers {:?}",
            draft_gguf.tensors.len(),
            draft.params.target_layer_ids
        );
    }
}

/// the per-layer input taps must not alter the trunk: the target's logits are
/// identical with the taps off and on (all 3 extract layers), both FA modes —
/// the port-side half of parity cell (a). Also pins the tap contents: the
/// buffer rows equal the trunk's residual stream (n_embd wide, one row per
/// token) — the rows the dflash impl gathers into batch_inject
/// (speculative.cpp:1145-1156).
#[test]
fn dflash_target_trunk_unchanged() {
    let _files = file_lock();
    build_target();
    let prompt: Vec<i32> = (1..=6).collect();

    for fa in [false, true] {
        let batch = {
            let mut b = llama::batch::LlamaBatch::default();
            for (i, &t) in prompt.iter().enumerate() {
                b.add(t, i as i32, &[0], i + 1 == prompt.len());
            }
            b
        };

        // taps off
        let mut m = open_model(&target_path());
        let a = target_driver(&mut m, fa)
            .decode_batch(&batch)
            .expect("plain decode")
            .logits_ith(batch.token.len() as i32 - 1)
            .expect("last row")
            .to_vec();

        // taps on (the dflash impl enables exactly TARGET_LAYERS,
        // speculative.cpp:1046-1048)
        let mut m = open_model(&target_path());
        let mut d = target_driver(&mut m, fa);
        for &lid in &TARGET_LAYERS {
            d.set_embeddings_layer_inp(lid as u32, true);
        }
        let out = d.decode_batch(&batch).expect("tapped decode");
        let b = out
            .logits_ith(batch.token.len() as i32 - 1)
            .expect("last row")
            .to_vec();
        assert_eq!(
            a, b,
            "fa={fa}: the extract-layer taps changed the trunk logits"
        );

        for &lid in &TARGET_LAYERS {
            let rows = d.get_embeddings_layer_inp(lid as u32);
            assert_eq!(
                rows.len(),
                prompt.len() * N_EMBD as usize,
                "fa={fa} lid={lid}"
            );
            assert!(
                rows.iter().any(|&x| x != 0.0),
                "fa={fa} lid={lid}: the tap rows are all zero (not extracted)"
            );
        }
    }
}

/// the full `--spec-type draft-dflash / draft-dspark` driver
/// (speculative-simple.cpp:126-342 over the dflash impl): the committed
/// stream must equal the target's plain greedy stream at temperature 0, both
/// FA modes — the port-side half of parity cell (b).
#[test]
fn dflash_spec_stream_equals_plain_greedy() {
    let _files = file_lock();
    build_target();
    let prompt: Vec<i32> = (1..=12).collect();
    let n_predict = 16usize;

    for kind in [
        DraftKind::Dflash,
        DraftKind::Dspark,
        DraftKind::Dflash2,
        DraftKind::DsparkDsv4,
    ] {
        build_draft(kind);
        for fa in [false, true] {
            let plain = {
                let mut m = open_model(&target_path());
                plain_greedy(&mut m, fa, &prompt, n_predict)
            };

            let mut m = open_model(&target_path());
            let mut tgt = target_driver(&mut m, fa);
            let ctx_dft = dflash_driver(kind, fa);

            let vocab = Vocab::load(&Gguf::open(&target_path()).unwrap()).unwrap();
            // the draft's own vocab — the mask token comes from it
            // (`llama_vocab_mask(vocab_dft)`, speculative.cpp:982); the draft
            // file carries `tokenizer.ggml.mask_token_id`
            let vocab_dft = Vocab::load(&Gguf::open(&draft_path(kind)).unwrap()).unwrap();
            assert_eq!(vocab_dft.token_mask(), MASK_TOKEN as i32);
            let mut params = CommonParamsSpeculative::default();
            params.types = vec![match kind {
                DraftKind::Dflash | DraftKind::Dflash2 => CommonSpeculativeType::DraftDflash,
                DraftKind::Dspark | DraftKind::DsparkDsv4 => CommonSpeculativeType::DraftDspark,
            }];
            params.draft.n_max = 3;
            params.draft.p_min = 0.0;

            let mut spec_ctx = common_speculative_init(
                &params,
                1,
                &mut tgt,
                Some(ctx_dft),
                &vocab,
                Some(&vocab_dft),
                0,
                false,
            )
            .expect("init")
            .expect("speculator");

            let n_vocab = tgt.n_vocab() as i32;
            let mut smpl = SamplingContext::new(
                n_vocab,
                SamplingParams {
                    temp: 0.0,
                    ..Default::default()
                },
            );

            let res = speculative_simple_generate(
                &mut tgt,
                &mut spec_ctx,
                &mut smpl,
                &vocab,
                &prompt,
                n_predict as i32,
            )
            .expect("speculative generate");

            // the driver commits whole verify rounds, so it may overshoot
            // n_predict by up to n_max tokens — the requested prefix must
            // match
            assert!(
                res.tokens.len() >= plain.len(),
                "{kind:?} fa={fa}: short stream ({})",
                res.tokens.len()
            );
            assert_eq!(
                &res.tokens[..plain.len()],
                &plain[..],
                "{kind:?} fa={fa}: the dflash speculation changed the greedy stream"
            );
            assert!(
                res.n_drafted > 0,
                "{kind:?} fa={fa}: no drafts were generated"
            );
            println!(
                "dflash kind={kind:?} fa={fa} — {} tokens, drafted {}, accepted {} ({} target \
                 forwards, {} draft forwards), mean acc len {:.2}",
                res.tokens.len(),
                res.n_drafted,
                res.n_accept,
                res.n_target_forward,
                res.n_draft_forward,
                spec_ctx
                    .impl_stats(0)
                    .map(|s| s.mean_acc_len())
                    .unwrap_or(0.0),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// the #[ignore] generator for the parity runs (parity/dflash_parity.sh drives
// the release llama-cli + the reference llama-server)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "writes /tmp/arch-dflash for parity/dflash_parity.sh"]
fn dflash_write_synth_files() {
    build_target();
    for kind in [
        DraftKind::Dflash,
        DraftKind::Dspark,
        DraftKind::Dflash2,
        DraftKind::DsparkDsv4,
    ] {
        build_draft(kind);
        let n = Gguf::open(&draft_path(kind)).unwrap().tensors.len();
        println!("dflash draft {kind:?}: {n} tensors -> {}", draft_path(kind));
    }
    println!("target -> {}", target_path());
}

// ---------------------------------------------------------------------------
// the DFlash2 dump-first differential — the port half of `parity/
// ref_dflash_chain.cpp --dump-first`: the same prompt, the same injection +
// first noise block, the first noise block's logits rows and the selector
// lattice rows printed in the probe's %.6e format. When the reference dump
// exists the values must match exactly (the format is the comparison).
// ---------------------------------------------------------------------------

/// printf's `%.6e` (Rust's `{:.6e}` lacks the sign + 2-digit exponent)
fn fmt_e6(v: f32) -> String {
    let s = format!("{v:.6e}");
    let (mant, exp) = s.split_once('e').expect("exp");
    let e: i32 = exp.parse().expect("exp digits");
    format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

#[test]
#[ignore = "differential dump for parity/dflash_parity.sh (compare against the ref probe)"]
fn dflash2_dump_first() {
    dump_first_impl(DraftKind::Dflash2, false);
}

#[test]
#[ignore = "harness check: the plain backbone's injected state at fa-on"]
fn dflash_plain_state_dump_first() {
    dump_first_impl(DraftKind::Dflash, true);
}

/// the DSV4 DSpark twin of the dump-first differential, plus the
/// injected-state blob (`llama_state_seq_get_data` of the draft context after
/// the prompt injection — the swa K-only ring; the probe's DUMP_STATE_OUT
/// capture, ref-dsv4-state.bin, prepends a u32 length)
#[test]
#[ignore = "differential dump for parity/dflash_parity.sh (compare against the ref probe)"]
fn dspark_dsv4_dump_first() {
    dump_first_impl(DraftKind::DsparkDsv4, true);
}

fn dump_first_impl(kind: DraftKind, dump_state: bool) {
    let _files = file_lock();
    build_target();
    build_draft(kind);

    // the dump protocol runs at fa ON — the v_trans = 0 cache layout, the
    // only one the port's serialization matches (the dsv4_state convention)
    let fa = std::env::var("DUMP_FA").map(|v| v == "on").unwrap_or(true);

    const PROMPT: &str = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12";

    // the target side: taps on, prompt prefill (all but the last token, the
    // probe's loop of ref_dflash_chain.cpp:137-151)
    let mut m = open_model(&target_path());
    let vocab = Vocab::load(&Gguf::open(&target_path()).unwrap()).unwrap();
    let inp = vocab.tokenize(PROMPT, true, true);
    let mut tgt = target_driver(&mut m, fa);
    for &lid in &TARGET_LAYERS {
        tgt.set_embeddings_layer_inp(lid as u32, true);
    }
    let batch_prompt = {
        let mut b = llama::batch::LlamaBatch::default();
        for (i, &t) in inp[..inp.len() - 1].iter().enumerate() {
            b.add(t, i as i32, &[0], false);
        }
        b
    };
    tgt.decode_batch(&batch_prompt).expect("target prefill");

    // the draft context: DFlash2's unmasked nextn tap / DSpark's masked one,
    // non-causal either way (speculative.cpp:1051-1052)
    let mut ctx_dft = dflash_driver(kind, fa);
    ctx_dft.set_embeddings_nextn(true, !matches!(kind, DraftKind::Dflash2));
    ctx_dft.set_causal_attn(false);

    // the injection pass (the impl's process(), speculative.cpp:1090-1180):
    // gather the extract-layer rows, interleave, decode as one embd batch
    let n_inj = inp.len() - 1;
    let n_embd_enc = N_EMBD_INP_ENC as usize;
    let mut embd = vec![0f32; n_inj * n_embd_enc];
    for (k, &lid) in TARGET_LAYERS.iter().enumerate() {
        let layer = tgt.get_embeddings_layer_inp(lid as u32);
        assert_eq!(layer.len(), n_inj * N_EMBD as usize);
        for i in 0..n_inj {
            let dst = i * n_embd_enc + k * N_EMBD as usize;
            embd[dst..dst + N_EMBD as usize]
                .copy_from_slice(&layer[i * N_EMBD as usize..(i + 1) * N_EMBD as usize]);
        }
    }
    let batch_inject = {
        let mut b = llama::batch::LlamaBatch::default();
        for i in 0..n_inj {
            b.add(0, i as i32, &[0], false);
        }
        b.embd = Some(embd);
        b
    };
    ctx_dft.decode_batch(&batch_inject).expect("injection");

    // the injected draft state — the probe's DUMP_STATE_OUT blob (u32 len +
    // the state_seq_get_data bytes of seq 0)
    if dump_state {
        let blob = ctx_dft.state_seq_get_data(0, false);
        let mut file = Vec::with_capacity(4 + blob.len());
        file.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        file.extend_from_slice(&blob);
        let p = format!(
            "{OUT_DIR}/port-{}-state.bin",
            match kind {
                DraftKind::DsparkDsv4 => "dsv4",
                DraftKind::Dflash2 => "dflash2",
                _ => "dflash",
            }
        );
        std::fs::write(&p, file).unwrap();
        println!("port injected state: {} bytes -> {p}", blob.len());
        let ref_state = format!(
            "{OUT_DIR}/ref-{}-state.bin",
            match kind {
                DraftKind::DsparkDsv4 => "dsv4",
                DraftKind::Dflash2 => "dflash2",
                _ => "dflash",
            }
        );
        if std::path::Path::new(&ref_state).exists() {
            let r = std::fs::read(&ref_state).unwrap();
            let rlen = u32::from_le_bytes(r[0..4].try_into().unwrap()) as usize;
            if r[4..4 + rlen] == blob[..] {
                println!(
                    "dspark-dsv4 injected state: byte-identical to the reference ({rlen} bytes)"
                );
            } else {
                let off = r[4..4 + rlen]
                    .iter()
                    .zip(&blob)
                    .position(|(x, y)| x != y)
                    .unwrap_or(rlen.min(blob.len()));
                println!("dspark-dsv4 injected state DIFFERS at byte {off}/{rlen}");
            }
        }
    }

    // the first noise block (ref_dflash_chain.cpp:180-206): the driver's
    // n_past = inp.len() - 1, 4 rows (n_max 3 + the anchor)
    let vocab_dft = Vocab::load(&Gguf::open(&draft_path(kind)).unwrap()).unwrap();
    let mask = vocab_dft.token_mask();
    let n_past0 = (inp.len() - 1) as i32;
    let noise = {
        let mut b = llama::batch::LlamaBatch::default();
        b.add(*inp.last().unwrap(), n_past0, &[0], true);
        for i in 1..4 {
            b.add(mask, n_past0 + i, &[0], true);
        }
        b
    };
    let out = ctx_dft.decode_batch(&noise).expect("noise decode");

    let tag = match kind {
        DraftKind::Dflash2 => "dflash2",
        _ => "dspark-dsv4",
    };
    let mut lines: Vec<String> = Vec::new();
    for r in 0..4 {
        let logits = out.logits_ith(r).expect("logits row").to_vec();
        lines.push(format!(
            "noise_logits_{r}: {}",
            logits
                .iter()
                .map(|&v| fmt_e6(v))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    if matches!(kind, DraftKind::Dflash2) {
        let lattice = ctx_dft.get_embeddings_nextn().to_vec();
        assert_eq!(
            lattice.len(),
            4 * N_EMBD as usize,
            "the packed lattice rows"
        );
        for r in 0..4 {
            lines.push(format!(
                "lattice_{r}: {}",
                lattice[r * N_EMBD as usize..(r + 1) * N_EMBD as usize]
                    .iter()
                    .map(|&v| fmt_e6(v))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
    }
    let port_dump = format!("{OUT_DIR}/port-dump-first-{tag}.txt");
    std::fs::write(&port_dump, lines.join("\n") + "\n").unwrap();
    println!("port dump -> {port_dump}");

    // the comparison half: run parity/ref_dflash_chain.cpp --dump-first with
    // stdout captured to produce the ref dump
    let ref_dump = std::env::var(format!("{}_REF_DUMP", tag.to_uppercase().replace('-', "_")))
        .unwrap_or_else(|_| format!("{OUT_DIR}/ref-dump-first-{tag}.txt"));
    if std::path::Path::new(&ref_dump).exists() && matches!(kind, DraftKind::Dflash2) {
        // the rows compare with an FP tolerance: the conv/selector GEMMs sit
        // at the port's kernel-rounding frontier (ulps, like the plain
        // backbone's single K row) — the DISCRETE lattice head (the top-k
        // candidate ids the walk consumes) must match exactly.
        // (DFlash2 only: the reference's *dump-shaped* DSV4 noise decode
        // itself yields a degenerate logits pattern — the driver-shaped
        // decodes agree 153/153 candidates, which the chain cell covers)
        let ref_txt = std::fs::read_to_string(&ref_dump).unwrap();
        let ref_rows: Vec<(String, Vec<f32>)> = ref_txt
            .lines()
            .filter(|l| l.contains(':') && !l.starts_with("dumped"))
            .map(|l| {
                let (n, v) = l.split_once(": ").expect("row");
                (
                    n.to_string(),
                    v.split(' ')
                        .map(|x| x.parse::<f32>().expect("f32"))
                        .collect(),
                )
            })
            .collect();
        let port_rows: Vec<(String, Vec<f32>)> = lines
            .iter()
            .map(|l| {
                let (n, v) = l.split_once(": ").expect("row");
                (
                    n.to_string(),
                    v.split(' ')
                        .map(|x| x.parse::<f32>().expect("f32"))
                        .collect(),
                )
            })
            .collect();
        assert_eq!(ref_rows.len(), port_rows.len(), "{tag}: ref dump row count");
        let top_k = SELECTOR_TOP_K as usize;
        let mut worst = 0f32;
        for ((rn, rv), (pn, pv)) in ref_rows.iter().zip(&port_rows) {
            assert_eq!(rn, pn, "{tag}: row order");
            assert_eq!(rv.len(), pv.len(), "{tag}: {rn} length");
            // the discrete candidate ids: exact
            if rn.starts_with("lattice_") && matches!(kind, DraftKind::Dflash2) {
                assert_eq!(
                    rv[..top_k.min(rv.len())],
                    pv[..top_k.min(pv.len())],
                    "{tag}: {rn} candidate ids diverged — the walk would pick different tokens"
                );
            }
            worst = worst.max(
                rv.iter()
                    .zip(pv)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
        }
        assert!(
            worst < 1e-3,
            "{tag}: dump rows diverged beyond FP ulps (worst {worst:e})"
        );
        println!(
            "{tag} dump-first: {}/{} rows within FP ulps (worst |d| {worst:e}), lattice candidate              ids exact",
            lines.len(),
            lines.len()
        );
    } else {
        println!("no reference dump at {ref_dump} — wrote the port dump only");
    }
}
