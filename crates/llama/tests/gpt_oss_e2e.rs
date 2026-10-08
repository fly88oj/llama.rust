//! gpt_oss_e2e.rs — end-to-end verification of `build_gpt_oss_forward`
//! (src/models/openai-moe.cpp) on the real gpt-oss-20b GGUFs on this machine,
//! plus the MoE/SWA parameter derivation from the GGUF hparams.
//!
//! Scope / ownership: this file exercises the ported builder and never modifies
//! the implementation; every blocker found is reported in the agent report,
//! not worked around in lib code.
//!
//! Runs by default (metadata + mmap only, no tensor data read):
//!   * `gpt_oss_20b_hparams_and_params` — n_expert / n_expert_used / SWA
//!     pattern / rope freq_base_swa / sinks / per-expert biases for MXFP4 and
//!     Q4_K_M, and the AttnParams+GptOssParams the forward test wires from them.
//!
//! `#[ignore]`d (manual; see each test's doc comment):
//!   * `gpt_oss_20b_mxfp4_reference_parity` — 11.3 GiB MXFP4: prefill
//!     "The capital of France is" + 16 greedy tokens vs the reference server
//!   * `gpt_oss_20b_q4_k_m_reference_parity` — same protocol, Q4_K_M file
//!
//! Reference capture (PARITY.md protocol: fresh llama-server, first request on
//! the slot, `temperature=0`, `cache_prompt=false`):
//!   /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server \
//!       -m <MXFP4 file> -c 512 -t 8 --port 8840 --host 127.0.0.1
//!   curl -s http://127.0.0.1:8840/completion -H 'Content-Type: application/json' \
//!       -d '{"prompt":"The capital of France is","n_predict":16,
//!            "temperature":0,"logprobs":20,"cache_prompt":false}'
//! (FA ON — the reference default — and the port runs the FA path too:
//! `AttnParams::use_flash_attn` defaults on, see PARITY.md's gpt-oss section for
//! the fp16-VKQ/Q→f16/sinks semantics that took MXFP4 to 16/16.)
//!
//! Run both heavy tests (each loads its own model, single-threaded to bound RAM):
//!   cargo test --release -p llama --test gpt_oss_e2e -- --ignored --nocapture \
//!       --test-threads=1
//!
//! Headline (re-measured 2026-09-27 after the FA one_chunk row-shape closure —
//! flash_attn.rs's sinks `S = S*ms + vs` line unfused + split-KV merge
//! contraction, both matching the reference binary; see PARITY.md):
//!   * MXFP4: prompt tokens == the reference's ([976, 9029, 328, 10128, 382]).
//!     **Acceptance is teacher-forced, not greedy**: feeding the reference's own
//!     16 tokens keeps the context identical at every step and measures op-level
//!     fidelity directly — ref token in our top-5 16/16, top-5 id *set*
//!     identical **16/16**, worst |delta logprob| **0.001** (the FA tail that
//!     used to cost 316/20480 FA-output elements / 0.235 logprob is closed;
//!     every FA dump case is bit-exact now). The greedy run flips a
//!     0.157-logit pair at step 3 and scores 4/16; the test asserts only that
//!     any greedy divergence stays inside that band. History: the "16/16 /
//!     0.158" era held while the rope carried a compensating 1-ulp error; the
//!     "13/16 / 0.235" era was the FA sinks-line FMA divergence; the current
//!     numbers measure the faithful routing (llamafile tinyBLAS), the
//!     contracted `S = S*ms + vs` inner line, the contracted rope rotate and
//!     the unfused FA sinks line.
//!   * Q4_K_M: **8/16 greedy tokens** (same prompt/protocol, re-captured fresh
//!     server; the stored REF16_Q4KM reproduced 16/16 exactly). First divergence
//!     at **step 7**: the port picks 4705 (-2.368) over the reference's 4022
//!     (-2.403; ref's 4705 is at -2.433) — pair-wise gap 0.065 logits on the
//!     reference's own **0.030-logit** top-1 margin, i.e. a tie flip inside the
//!     residual band, not a structural break. Steps 0-6 agree on the top-5 ids.
//!   * Q4_K_M residual cause (measured, not inferred): the reference build loads
//!     with GGML_USE_CPU_REPACK + GGML_NATIVE and its `ggml_repack_get_optimal_repack_type`
//!     gives the 24 `blk.N.attn_output` Q4_K tensors the **q4_K_8x8** trait
//!     (repack.cpp:5006; the reference prints "repack tensor with q4_K_8x8"),
//!     so its production GEMM is the 8x8 outer-product gemv/gemm. This port has
//!     no Q4_K repack — its row-wise `vec_dot_q4_K_q8_K` equals the reference's
//!     *plain* path bit for bit but differs from the repacked output by
//!     ~1.3e-6 relative (parity/q4k_goss_ref.bin, crates/ggml/src/vec_dot.rs
//!     `kquant_real_tensor_tests`). The other two types new in this file are
//!     clean: Q5_0/Q8_0 are bit-exact even though the reference routes their
//!     prefill GEMMs to llamafile tinyBLAS (parity/q5_0_goss_ref.bin /
//!     q8_0_goss_ref.bin prove sgemm returns true there and our vec_dot still
//!     matches it exactly).
//!   * Performance (port, 8 threads, release, MXFP4): prefill ~1 t/s incl. the
//!     one-time 9.3 GiB lazy repack, gen ~1.1-1.3 t/s vs the reference's
//!     42-56 t/s / 28-34 t/s (repacked SIMD kernels + tuned FA; not tuned here).

use std::path::Path;
use std::sync::Arc;

use ggml::types::GgmlType;
use ggml::{Context, Gguf, TensorId};
use llama::graph::{AttnParams, DecodeInputs, ForwardResult};
use llama::graph_arch::{
    build_gpt_oss_forward, GptOssLayerWeights, GptOssModelWeights, GptOssParams,
};
use llama::kv_cache::{KvCache, SlotInfo};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// models / reference data on this machine
// ---------------------------------------------------------------------------

const GPTOSS20B_MXFP4: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";
const GPTOSS20B_Q4KM: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";

const PROMPT: &str = "The capital of France is";

/// Same protocol, Q4_K_M conversion (experts are still MXFP4, attention/embd
/// Q4_K/Q5_0/Q8_0) — captured 2026-09-24, fresh server + first request,
/// default FA. Text: ' Paris."\n\nSure! Here\'s a short story about a cat named Whiskers'
const REF16_Q4KM: [i32; 16] = [
    12650, 3692, 279, 62915, 0, 44257, 261, 4022, 4869, 1078, 261, 9059, 11484, 1656, 3295, 409,
];

/// Reference per-step top-8 (id, logprob) for the fresh Q4_K_M capture below —
/// the pair-wise comparison baseline at the first divergence (step 7).
#[rustfmt::skip]
const REF_TOP8_Q4KM: [[(i32, f32); 8]; 16] = [
    [(12650, -0.3538), (25, -3.8382), (392, -3.9202), (625, -3.9746), (354, -4.2011), (723, -4.3742), (290, -4.5329), (2381, -4.6158)],
    [(3692, -1.5639), (14396, -1.9786), (13, -2.2931), (6635, -2.4915), (558, -3.1348), (364, -3.2479), (11, -3.5235), (21161, -3.5866)],
    [(279, -2.3485), (623, -2.9031), (350, -2.9567), (1328, -3.1997), (2747, -3.2100), (306, -3.4627), (326, -3.6786), (382, -3.7092)],
    [(62915, -1.8945), (976, -2.7492), (637, -3.0582), (410, -3.1984), (12253, -3.3925), (31639, -3.4463), (40, -3.7427), (17, -3.9995)],
    [(0, -0.4484), (11, -1.1068), (4435, -4.0600), (13, -4.7159), (1703, -6.5129), (364, -7.2665), (1402, -8.1460), (279, -8.4868)],
    [(44257, -0.9976), (7306, -1.2056), (623, -3.1701), (357, -3.2790), (41021, -3.4580), (18754, -4.0340), (730, -4.1081), (2514, -4.2064)],
    [(261, -0.5064), (290, -1.9015), (448, -1.9436), (1495, -2.8327), (634, -4.4077), (1001, -4.7754), (3613, -4.8281), (1412, -5.4866)],
    [(4022, -2.4031), (4705, -2.4330), (52287, -3.1980), (26534, -3.2616), (4149, -3.4046), (21872, -3.5627), (4853, -3.8953), (82463, -3.9563)],
    [(4869, -1.4991), (11713, -2.9431), (2201, -3.1071), (41339, -3.1227), (326, -3.4275), (27853, -3.5438), (11, -3.7988), (26534, -3.9259)],
    [(1078, -1.4811), (306, -1.9696), (484, -1.9845), (4122, -2.5167), (483, -2.7114), (395, -2.7276), (1402, -3.1161), (2360, -3.3052)],
    [(261, -1.0429), (290, -1.5545), (392, -3.3470), (448, -3.5598), (484, -3.8718), (1495, -3.9573), (634, -4.0575), (12650, -4.1545)],
    [(9059, -2.7881), (8881, -2.9670), (8473, -3.1777), (5612, -3.3929), (1647, -3.6015), (873, -3.6524), (220, -3.8666), (6446, -3.9399)],
    [(11484, -0.8535), (484, -2.0049), (1218, -2.0533), (1402, -2.4458), (326, -2.7758), (306, -3.7678), (11, -3.7940), (483, -3.9935)],
    [(1656, -1.8377), (392, -2.6965), (66278, -2.7934), (391, -3.4786), (152784, -3.5621), (10093, -3.5784), (3632, -3.9508), (353, -4.0147)],
    [(3295, -0.0010), (321, -8.4490), (121422, -8.5783), (482, -8.9629), (276, -9.2464), (138214, -9.3647), (8759, -10.4473), (52421, -10.7648)],
    [(409, -0.0353), (259, -3.5725), (1402, -6.5793), (1286, -6.9186), (364, -7.6998), (52468, -7.7181), (326, -7.8542), (11, -8.0948)],
];

/// Reference greedy ids (16 tokens), fresh server + first request, default FA.
const REF16_MXFP4: [i32; 16] = [
    12650, 14396, 271, 10213, 271, 1069, 12482, 290, 13427, 198, 271, 23478, 125774, 314, 2273,
    192800,
];

/// Reference per-step top-5 (id, logprob) for the same run — used for the
/// pair-wise logprob comparison at the first divergence.
#[rustfmt::skip]
const REF_TOP5_MXFP4: [[(i32, f32); 5]; 16] = [
    [(12650, -0.462), (625, -3.573), (25, -3.677), (354, -4.019), (2381, -4.209)],
    [(14396, -1.832), (3692, -1.856), (13, -2.469), (6635, -2.510), (11, -3.286)],
    [(271, -1.514), (309, -2.357), (256, -2.453), (12, -2.659), (26178, -2.948)],
    [(10213, -2.576), (1069, -2.603), (1862, -2.712), (3696, -3.077), (606, -3.190)],
    [(271, -0.436), (1944, -1.276), (943, -3.930), (739, -4.300), (220, -4.790)],
    [(1069, -1.917), (395, -2.375), (2123, -3.502), (17125, -3.539), (2359, -3.901)],
    [(12482, -2.676), (6585, -2.871), (19496, -3.194), (10885, -3.336), (17951, -3.365)],
    [(290, -1.388), (13427, -1.600), (2454, -1.759), (40536, -3.235), (326, -3.359)],
    [(13427, -1.173), (20830, -2.685), (73662, -2.719), (2201, -2.795), (1238, -2.968)],
    [(198, -0.256), (316, -2.246), (326, -3.282), (1819, -3.820), (483, -4.316)],
    [(271, -0.000), (168394, -10.512), (26178, -11.500), (309, -12.674), (220, -13.129)],
    [(23478, -2.095), (2273, -2.302), (14191, -2.903), (395, -3.166), (4376, -3.227)],
    [(125774, -0.505), (192800, -1.701), (314, -2.960), (3537, -3.306), (9118, -4.147)],
    [(314, -0.097), (11, -2.524), (23083, -6.784), (16, -6.913), (11295, -7.324)],
    [(2273, -0.354), (29539, -2.574), (8911, -3.432), (122926, -3.517), (723, -3.955)],
    [(192800, -0.077), (9118, -3.701), (1303, -4.330), (61043, -5.032), (42518, -5.358)],
];

// ---------------------------------------------------------------------------
// helpers (same shape as arch_e2e.rs)
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    /// keeps the weight storage alive (model tensors point into it)
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    size_bytes: u64,
}

fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let size_bytes = std::fs::metadata(path).ok()?.len();
    let file = std::fs::File::open(path).expect("open model");
    // SAFETY: read-only use of a model file (same policy as the rest of the port)
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    match load_model(&gguf, mmap.clone()) {
        Ok(model) => Some(Loaded {
            model,
            gguf,
            mmap,
            size_bytes,
        }),
        Err(e) => {
            eprintln!("SKIP: load_model({path}) failed: {e}");
            None
        }
    }
}

fn mem_available_gb() -> f64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0 / 1024.0;
        }
    }
    0.0
}

fn mem_guard(label: &str, need_gb: f64) -> bool {
    let avail = mem_available_gb();
    if avail < need_gb {
        eprintln!("SKIP {label}: {avail:.1} GiB available < {need_gb:.1} GiB required");
        return false;
    }
    true
}

/// greedy: strict >, first max wins (llama_sampler_init_greedy)
fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

/// Top-k (id, logit) pairs, descending (ties: lower id first).
fn topk(v: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.into_iter().take(k).map(|i| (i as i32, v[i])).collect()
}

/// logprobs of a full row, descending (same convention as llama-server).
fn logprobs(v: &[f32]) -> Vec<(i32, f32)> {
    let all = topk(v, v.len());
    let mx = all[0].1;
    let lse = mx
        + (all
            .iter()
            .map(|&(_, x)| ((x - mx) as f64).exp())
            .sum::<f64>())
        .ln() as f32;
    all.into_iter().map(|(i, x)| (i, x - lse)).collect()
}

fn threads() -> usize {
    8
}

/// `AttnParams` + `GptOssParams` from loaded hparams — exactly what the
/// builder needs (openai-moe.cpp:3-18 + llama-model.cpp:2251 per-layer rope).
fn gpt_oss_params(m: &LlamaModel) -> (AttnParams, GptOssParams) {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    let n_layer = m.layers.len();
    let is_swa: Vec<bool> = (0..n_layer).map(|il| hp.is_swa(il)).collect();
    let n_expert_used: Vec<u32> = (0..n_layer).map(|il| hp.n_expert_used(il)).collect();
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
        // FA+sinks now wired (flash_attn_ext_sinks -> src[4]); the reference
        // default runs FA, so the parity probe uses it. Set false for the
        // non-FA + soft_max_add_sinks cross-check.
        use_flash_attn: std::env::var("GPTOSS_FA_OFF").is_err(),
    };
    let gp = GptOssParams {
        n_expert: hp.n_expert as i64,
        n_expert_used,
        // llama-graph.cpp:2288-2290
        swiglu_oai_alpha: 1.702,
        swiglu_oai_limit: 7.0,
        expert_weights_scale: hp.expert_weights_scale,
        is_swa,
        // openai-moe.cpp:14-17 (freq_base_swa key absent -> the dense value)
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
    };
    (attn, gp)
}

fn gpt_oss_weights(m: &LlamaModel) -> GptOssModelWeights {
    GptOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, x)| GptOssLayerWeights {
                attn_norm: x
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: x
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: x.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: x.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: x.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: x.wq_b,
                wk_b: x.wk_b,
                wv_b: x.wv_b,
                wo: x.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: x.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                attn_sinks: x
                    .attn_sinks
                    .unwrap_or_else(|| panic!("layer {il}: attn_sinks")),
                ffn_gate_inp: x
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_inp_b: x
                    .ffn_gate_inp_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_b")),
                ffn_up_exps: x
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_up_exps_b: x
                    .ffn_up_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps_b")),
                ffn_gate_exps: x
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_gate_exps_b: x
                    .ffn_gate_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps_b")),
                ffn_down_exps: x
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_down_exps_b: x
                    .ffn_down_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps_b")),
            })
            .collect(),
    }
}

/// Decode harness: the RealHarness input/graph protocol from arch_e2e.rs,
/// specialised to the gpt-oss builder (22 tensor handles per layer).
struct Harness {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
}

impl Harness {
    /// Decode `tokens` at `pos`; returns the last token's logits [n_vocab].
    fn decode(
        &mut self,
        w: &GptOssModelWeights,
        attn: &AttnParams,
        gp: &GptOssParams,
        tokens: &[i32],
        pos: &[i32],
    ) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        // assign first, then the (256-padded) n_kv — step_inputs order
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        // FA requires an F16 mask, the non-FA path an F32 one
        // (llama-graph.cpp:38-39); 0/-inf are exact in both.
        let mask_ty = if attn.use_flash_attn {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        let kq_mask = self.gctx.new_tensor_2d(mask_ty, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| p.copy_from_slice(pos))
            .unwrap();
        {
            let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
            self.gctx
                .data_bytes_mut(row_idx)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&idxs));
        }
        {
            // mask [n_kv, n_tokens]: element (kv s, query t) at s + t*n_kv
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            // padded (empty) cells keep pos = -1 → masked by the fills
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            match mask_ty {
                GgmlType::F16 => {
                    let mask: &mut [half::f16] = bytemuck::cast_slice_mut(mask_bytes);
                    mask.fill(half::f16::NEG_INFINITY);
                    for (iq, &qp) in pos.iter().enumerate() {
                        for (ik, &kp) in kv_pos.iter().enumerate() {
                            if 0 <= kp && kp <= qp {
                                mask[iq * n_kv as usize + ik] = half::f16::ZERO;
                            }
                        }
                    }
                }
                _ => {
                    let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
                    mask.fill(f32::NEG_INFINITY);
                    for (iq, &qp) in pos.iter().enumerate() {
                        for (ik, &kp) in kv_pos.iter().enumerate() {
                            if 0 <= kp && kp <= qp {
                                mask[iq * n_kv as usize + ik] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let result: ForwardResult = build_gpt_oss_forward(
            &mut self.gctx,
            w,
            attn,
            gp,
            &self.kv,
            &inputs,
            SlotInfo {
                s0: sinfo.s0,
                s1: sinfo.s1,
            },
            n_kv,
            n,
        );
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, threads());
        // (assign already happened before the graph build, step_inputs order)

        let n_vocab = self.gctx.ne(logits)[0] as usize;
        let all: Vec<f32> = self
            .gctx
            .data_bytes(logits)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let out = all[n_vocab * (n - 1)..n_vocab * n].to_vec();
        debug_assert!(
            out.iter().all(|v| v.is_finite()),
            "gpt-oss logits not finite"
        );
        out
    }
}

/// n_embd_k_gqa / v of the loaded model (needed before KvCache::new).
fn kv_widths(m: &LlamaModel) -> (i64, i64) {
    let hp = &m.hparams;
    (
        hp.n_embd_head_k(0) as i64 * hp.n_head_kv(0) as i64,
        hp.n_embd_head_v(0) as i64 * hp.n_head_kv(0) as i64,
    )
}

// ===========================================================================
// 1. hparams -> GptOssParams (runs by default; metadata + mmap only)
// ===========================================================================

#[test]
fn gpt_oss_20b_hparams_and_params() {
    for (path, label) in [(GPTOSS20B_MXFP4, "MXFP4"), (GPTOSS20B_Q4KM, "Q4_K_M")] {
        let Some(l) = load_real(path) else { continue };
        let hp = &l.model.hparams;
        let (attn, gp) = gpt_oss_params(&l.model);

        assert_eq!(l.model.layers.len(), 24, "{label}: n_layer");
        assert_eq!(hp.n_expert, 32, "{label}: n_expert");
        assert_eq!(
            gp.n_expert_used,
            vec![4u32; 24],
            "{label}: n_expert_used per layer"
        );
        // SWA pattern: load_swa_pattern(ml, 2) with dense_first=false ->
        // even layers sliding-window (llama-model.cpp set_swa_pattern)
        assert!(gp.is_swa[0] && !gp.is_swa[1], "{label}: swa pattern");
        assert_eq!(
            gp.is_swa.iter().filter(|&&s| s).count(),
            12,
            "{label}: 12 swa layers"
        );
        assert_eq!(hp.n_swa, 128, "{label}: sliding window");
        // this file has no rope.scaling.freq_base_swa key -> the swa copies
        // default to the dense values (openai-moe.cpp:14-17)
        assert_eq!(
            gp.rope_freq_base_swa, hp.rope_freq_base_train,
            "{label}: freq_base_swa"
        );
        assert_eq!(
            gp.rope_freq_scale_swa, hp.rope_freq_scale_train,
            "{label}: freq_scale_swa"
        );
        assert_eq!(attn.freq_base, 150000.0, "{label}: rope freq base");
        assert!(
            (attn.freq_scale - 1.0 / 32.0).abs() < 1e-9,
            "{label}: yarn freq scale"
        );
        assert_eq!(attn.rope_mode, 2, "{label}: NEOX");
        assert_eq!(attn.n_rot, 64, "{label}: n_rot");
        assert_eq!(attn.n_head, 64, "{label}: n_head");
        assert_eq!(attn.n_head_kv, 8, "{label}: n_head_kv");
        // FA is the default now; the non-FA path remains available via
        // GPTOSS_FA_OFF=1 (both produce the same ids on the parity probe).
        assert!(attn.use_flash_attn, "{label}: FA default");
        // no expert_weights_scale key -> llama-hparams.h default 0.0f, and the C
        // `w_scale != 0 && != 1` guard (llama-graph.cpp:2183-2186) skips the
        // ggml_scale node for both 0.0 and 1.0
        assert_eq!(
            gp.expert_weights_scale, 0.0,
            "{label}: expert_weights_scale"
        );
        assert_eq!((gp.swiglu_oai_alpha, gp.swiglu_oai_limit), (1.702, 7.0));

        // per-layer tensor set the builder consumes
        let (n_k, n_v) = kv_widths(&l.model);
        assert_eq!((n_k, n_v), (512, 512), "{label}: kv widths");
        for (il, lay) in l.model.layers.iter().enumerate() {
            assert!(lay.attn_sinks.is_some(), "{label} {il}: attn_sinks");
            // the reference applies the q/k/v biases whenever the file carries
            // them (build_qkv llama-graph.cpp:1710-1718); this file does
            assert!(lay.wq_b.is_some(), "{label} {il}: wq_b");
            assert!(lay.wk_b.is_some(), "{label} {il}: wk_b");
            assert!(lay.wv_b.is_some(), "{label} {il}: wv_b");
            assert!(lay.ffn_gate_inp_b.is_some(), "{label} {il}: ffn_gate_inp_b");
            assert!(lay.ffn_up_exps_b.is_some(), "{label} {il}: ffn_up_exps_b");
            assert!(
                lay.ffn_gate_exps_b.is_some(),
                "{label} {il}: ffn_gate_exps_b"
            );
            assert!(
                lay.ffn_down_exps_b.is_some(),
                "{label} {il}: ffn_down_exps_b"
            );
            assert!(lay.wo_b.is_some(), "{label} {il}: wo_b");
            assert!(lay.attn_post_norm.is_some(), "{label} {il}: attn_post_norm");
            // both conversions keep the MoE experts in MXFP4 (model.rs
            // gpt_oss_q4_k_m_same_mapping)
            assert_eq!(
                l.model.ctx.ty(lay.ffn_gate_exps.unwrap()),
                GgmlType::Mxfp4,
                "{label} {il}: expert dtype"
            );
        }
        println!(
            "[{label}] hparams ok: {} tensors, {} bytes, swa={} freq_base_swa={}",
            l.model.tensors.len(),
            l.size_bytes,
            hp.n_swa,
            gp.rope_freq_base_swa
        );
    }
}

// ===========================================================================
// 2. real-model reference parity (manual; ~11 GiB MXFP4)
// ===========================================================================

/// MXFP4 prefill + 16 greedy tokens vs the fresh reference server (default FA).
/// `#[ignore]`d: 11.3 GiB model, one prefill + 16 decode steps.
///
/// Measured history (same test, same reference): **16/16** at 2026-09-24 21:47
/// with the FA-faithful path (fp16 VKQ + Q→f16 + sinks), then **1/16** after a
/// concurrent `crates/ggml/src/flash_attn.rs` edit (mtime 22:08) — reproduced
/// twice, and NOT the repack path: `LLAMA_RUST_REPACK=0` gives the same 1/16
/// trajectory, while `GPTOSS_FA_OFF=1` still reproduces the long-standing
/// non-FA 4/16. The regression is FA-path-only and gpt-oss is the only sinks
/// user, so check the sinks/src[4] handling there first.
///
/// The regression floor below therefore stays at 4 (the reproducible non-FA
/// result) *plus* the tie-gap assertion; **raise it to 16 once the FA path is
/// restored** (the pre-regression run matched all 16 ids and the reference's
/// own margins).
///
/// Run:
///   cargo test --release -p llama --test gpt_oss_e2e -- --ignored --nocapture \
///       gpt_oss_20b_mxfp4_reference_parity
#[test]
#[ignore = "manual: 11.3 GiB MXFP4 model, prefill + 16 decode steps in release"]
fn gpt_oss_20b_mxfp4_reference_parity() {
    let Some(l) = load_real(GPTOSS20B_MXFP4) else {
        return;
    };
    // 4 GiB covers the row-wise vec_dot path; with CPU_REPACK on (the default,
    // as in the reference) the 72 MXFP4 expert tensors are additionally
    // materialized as ~10.1 GiB of 8x8-interleaved bytes — the same ~10 GiB
    // anonymous RSS the reference server shows next to its mapped file pages.
    let need_gb = if ggml::repack::repack_enabled() {
        14.0
    } else {
        4.0
    };
    if !mem_guard("gpt_oss_20b_mxfp4_reference_parity", need_gb) {
        return;
    }
    let n_ctx = 512u32;
    let (attn, gp) = gpt_oss_params(&l.model);
    let (n_k, n_v) = kv_widths(&l.model);
    let w = gpt_oss_weights(&l.model);
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let prompt_ids = vocab.tokenize(PROMPT, true, false);
    // reference (/tokenize, add_special=true): [976, 9029, 328, 10128, 382]
    assert_eq!(
        prompt_ids,
        vec![976, 9029, 328, 10128, 382],
        "prompt tokens"
    );

    // the model's own build context: the TensorIds in `w` index into it
    let n_layer = l.model.layers.len();
    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, n_ctx);
    let mut h = Harness {
        gctx,
        kv,
        watermark: 0,
    };
    h.watermark = h.gctx.mark();

    // ---- prefill ----
    let t0 = std::time::Instant::now();
    let pos: Vec<i32> = (0..prompt_ids.len() as i32).collect();
    let stats0 = ggml::repack::repack_stats();
    let ms0 = ggml::repack::repack_materialize_ms();
    let calls0 = ggml::repack::repack_gemv_calls();
    let mut logits = h.decode(&w, &attn, &gp, &prompt_ids, &pos);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let stats1 = ggml::repack::repack_stats();
    let repack_ms = ggml::repack::repack_materialize_ms() - ms0;
    println!(
        "repack          : enabled={} materialized {} tensors / {:.1} MiB in {:.0} ms \
         (one-time, lazy: the reference does this at load), gemv calls {}",
        ggml::repack::repack_enabled(),
        stats1.0 - stats0.0,
        (stats1.1 - stats0.1) as f64 / (1024.0 * 1024.0),
        repack_ms,
        ggml::repack::repack_gemv_calls() - calls0,
    );
    if repack_ms > 0.0 {
        println!(
            "prefill steady  : {:.1} ms ({:.2} t/s) excluding the one-time {:.0} ms repack",
            prefill_ms - repack_ms,
            prompt_ids.len() as f64 / ((prefill_ms - repack_ms) / 1e3),
            repack_ms
        );
    }

    let mut gen_ids = Vec::new();
    let mut gen_logprobs: Vec<Vec<(i32, f32)>> = Vec::new();
    // raw argmax rows (prefill + each generated step) for cross-run diffs, e.g.
    // LLAMA_RUST_REPACK=0 vs 1 — written when LLAMA_E2E_DUMP is set
    let mut logit_rows: Vec<Vec<f32>> = Vec::new();
    let mut gen_ms = 0f64;
    let mut next_pos = pos.last().unwrap() + 1;
    for step in 0..16 {
        let id = argmax(&logits);
        gen_ids.push(id);
        gen_logprobs.push(logprobs(&logits));
        logit_rows.push(logits.clone());
        if step + 1 < 16 {
            let t1 = std::time::Instant::now();
            logits = h.decode(&w, &attn, &gp, &[id], &[next_pos]);
            gen_ms += t1.elapsed().as_secs_f64() * 1e3;
            next_pos += 1;
        }
    }
    if let Ok(path) = std::env::var("LLAMA_E2E_DUMP") {
        let mut bytes = Vec::with_capacity(logit_rows.len() * logit_rows[0].len() * 4);
        for row in &logit_rows {
            bytes.extend_from_slice(bytemuck::cast_slice(row));
        }
        std::fs::write(&path, &bytes).expect("dump logits");
        println!(
            "logits dump     : {} rows x {} -> {path}",
            logit_rows.len(),
            logit_rows[0].len()
        );
    }

    // ---- teacher-forced pass ------------------------------------------------
    // The greedy sequence is chaotic: a step whose top-2 margin is below the
    // port's residual band (~0.01..0.12 logits on this model) can flip and
    // everything after it is a different context. This pass instead feeds the
    // *reference's* tokens, so context always agrees and the per-step deltas
    // measure op-level fidelity directly. A fresh KV is required (the greedy
    // pass left its own tokens in the cache).
    let mut tf_delta_max = 0f32;
    let mut tf_ref_in_top5 = 0usize;
    let mut tf_set_matches = 0usize;
    let mut tf_top1_gap_margin = Vec::new();
    {
        h.kv.clear();
        h.watermark = h.gctx.mark();
        let pos: Vec<i32> = (0..prompt_ids.len() as i32).collect();
        let mut tf_logits = h.decode(&w, &attn, &gp, &prompt_ids, &pos);
        let mut next_pos = pos.last().unwrap() + 1;
        for (step, &ref_id) in REF16_MXFP4.iter().enumerate() {
            let mine = logprobs(&tf_logits);
            let theirs = REF_TOP5_MXFP4[step];
            let my_set: std::collections::BTreeSet<i32> =
                mine.iter().take(5).map(|&(id, _)| id).collect();
            let ref_set: std::collections::BTreeSet<i32> =
                theirs.iter().map(|&(id, _)| id).collect();
            if my_set == ref_set {
                tf_set_matches += 1;
            } else {
                println!("tf step {step:2}: top-5 SET differs: mine {my_set:?} ref {ref_set:?}");
            }
            if mine.iter().take(5).any(|&(id, _)| id == ref_id) {
                tf_ref_in_top5 += 1;
            }
            for &(id, ref_lp) in theirs.iter() {
                if let Some(&(_, my_lp)) = mine.iter().find(|&&(i, _)| i == id) {
                    let d = (my_lp - ref_lp).abs();
                    if d > tf_delta_max {
                        tf_delta_max = d;
                    }
                }
            }
            // top-1 agreement only where the reference's own margin is not a tie
            let ref_margin = theirs[0].1 - theirs[1].1;
            if ref_margin >= 0.05 && mine[0].0 != theirs[0].0 {
                tf_top1_gap_margin.push((step, ref_margin, mine[0].0, theirs[0].0));
            }
            tf_logits = h.decode(&w, &attn, &gp, &[ref_id], &[next_pos]);
            next_pos += 1;
        }
    }
    println!(
        "teacher-forced  : ref token in our top-5 {tf_ref_in_top5}/16; top-5 set identical \
         {tf_set_matches}/16; worst |delta logprob| {tf_delta_max:.3}"
    );
    assert_eq!(
        tf_ref_in_top5, 16,
        "reference token must stay in our top-5 at every step"
    );
    // Re-measured 2026-09-26/27: after the FA one_chunk closure (flash_attn.rs)
    // the teacher-forced pass is exact for practical purposes — top-5 sets
    // 16/16, worst |delta logprob| 0.001. The earlier band (sets 13/16, worst
    // 0.235, one channel per (head, token) off by 1-2 ulp on rows with >= 4
    // unmasked lanes) was the sinks `S = S*ms + vs` line of FA one_chunk
    // computed as an FMA where the reference binary keeps it unfused (mul +
    // add), plus the unfused F32-V `ggml_vec_mad_f32` — both now reproduced
    // literally (see flash_attn.rs's contraction profile; pinned by
    // parity/fa_probe.bin). The gates below are tightened to the measured
    // values with a 10x margin.
    assert!(
        tf_set_matches == 16,
        "top-5 set matches dropped to {tf_set_matches}/16 — beyond the measured 0.001-exact band"
    );
    assert!(
        tf_delta_max < 0.01,
        "worst teacher-forced delta {tf_delta_max:.3} exceeds the measured 0.001-exact band"
    );
    assert!(
        tf_top1_gap_margin.is_empty(),
        "top-1 differs where the reference's own margin is not a tie: {tf_top1_gap_margin:?}"
    );

    let text: String = gen_ids.iter().map(|&t| vocab.token_to_piece(t)).collect();
    println!(
        "prompt ids      : {prompt_ids:?} ({} tokens)",
        prompt_ids.len()
    );
    println!("greedy ids      : {gen_ids:?}");
    println!("text            : {text:?}");
    println!(
        "perf            : prefill {:.1} ms ({:.2} t/s), gen {:.1} ms ({:.2} t/s)",
        prefill_ms,
        prompt_ids.len() as f64 / (prefill_ms / 1e3),
        gen_ms,
        15.0 / (gen_ms / 1e3)
    );

    // ---- token comparison ----
    let mut matched = 0usize;
    let mut first_diff: Option<usize> = None;
    for (i, (&got, &want)) in gen_ids.iter().zip(REF16_MXFP4.iter()).enumerate() {
        if got == want {
            matched += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    println!("MATCH           : {matched}/16 vs reference (fresh server, first request)");

    for (i, (mine, theirs)) in gen_logprobs.iter().zip(REF_TOP5_MXFP4.iter()).enumerate() {
        let mine5: Vec<(i32, f32)> = mine.iter().take(5).copied().collect();
        let same_ids: Vec<i32> = theirs.iter().map(|&(id, _)| id).collect();
        println!(
            "step {i:2}: ref top5 {same_ids:?}  mine top5 {:?}",
            mine5
                .iter()
                .map(|&(id, lp)| (id, (lp * 1000.0).round() / 1000.0))
                .collect::<Vec<_>>()
        );
        for &(id, ref_lp) in theirs.iter() {
            if let Some(&(_, my_lp)) = mine.iter().find(|&&(i, _)| i == id) {
                println!(
                    "          pair id {id}: ref {ref_lp:+.3} mine {my_lp:+.3} (d {:+.3})",
                    my_lp - ref_lp
                );
            } else {
                println!("          pair id {id}: ref {ref_lp:+.3} mine (outside top-20)");
            }
        }
    }

    // Tie analysis at the first divergence: the reference's own margin there
    // (from REF_TOP5) vs the port's margin — a flip is expected when both are
    // smaller than the documented repack/MXFP4 noise band.
    let mut tie_gap = f32::NAN;
    if let Some(k) = first_diff {
        let d = &gen_logprobs[k];
        let (my_top, my_lp) = d[0];
        let my_ref_lp = d
            .iter()
            .find(|&&(i, _)| i == REF16_MXFP4[k])
            .map(|&(_, v)| v)
            .unwrap_or(f32::NAN);
        let my_second = d
            .iter()
            .find(|&&(i, _)| i != my_top)
            .map(|&(_, v)| v)
            .unwrap_or(f32::NAN);
        let ref_lp_of_my_top = REF_TOP5_MXFP4[k]
            .iter()
            .find(|&&(i, _)| i == my_top)
            .map(|&(_, v)| v);
        let ref_margin = REF_TOP5_MXFP4[k][0].1
            - REF_TOP5_MXFP4[k]
                .get(1)
                .map(|&(_, v)| v)
                .unwrap_or(REF_TOP5_MXFP4[k][0].1);
        println!(
            "FIRST DIVERGENCE step {k}: mine {my_top} (lp {my_lp:+.3}, port margin {mm:+.3}) \
             vs ref {rk} (lp {rl:+.3}, ref margin {ref_margin:.3})",
            mm = my_lp - my_second,
            rk = REF16_MXFP4[k],
            rl = REF_TOP5_MXFP4[k][0].1,
        );
        println!(
            "  ref token {my_ref_lp:+.3} under my distribution (ref side {ref_lp:+.3}); \
             my token {my_top_lp} under the reference",
            ref_lp = REF_TOP5_MXFP4[k][0].1,
            my_top_lp = ref_lp_of_my_top
                .map(|v| format!("{v:+.3}"))
                .unwrap_or("outside top-5".into())
        );
        tie_gap = (my_lp - my_ref_lp).abs();
    }

    // Greedy-token count is *not* an acceptance criterion on this model: the
    // trajectory contains steps whose top-2 margin (reference's own margin
    // included) is ~0.02 logits, i.e. far below the documented residual band, so
    // which side of such a tie the port lands on is not a measure of fidelity —
    // the teacher-forced block above is. What must hold here is that any greedy
    // divergence stays inside the documented residual band (the FA one_chunk
    // f16-accumulator tail, 1-2 ulp per row — see the teacher-forced block's
    // 2026-09-26 note: after the rope FMA fix removed the compensating error,
    // the first greedy flip lands at step 3 with a 0.157-logit pair gap and
    // MATCH 4/16; a structural break desyncs by logits-scale gaps).
    if let Some(k) = first_diff {
        assert!(
            tie_gap < 0.25,
            "first divergence at step {k} is not inside the residual band: gap {tie_gap:.3} logits"
        );
        let ref_margin = REF_TOP5_MXFP4[k][0].1 - REF_TOP5_MXFP4[k][1].1;
        assert!(
            ref_margin < 0.25,
            "divergence at step {k} despite a {ref_margin:.3}-logit reference margin"
        );
        println!(
            "  => tie gap {tie_gap:.3} logits (ref's own margin {:.3})",
            REF_TOP5_MXFP4[k][0].1 - REF_TOP5_MXFP4[k][1].1
        );
    } else {
        println!("16/16 — no divergence");
    }
}

/// Q4_K_M variant of the same protocol (11.6 GiB, /home/jeffrey/.lmstudio/
/// models/unsloth/gpt-oss-20b-GGUF — the lmstudio-community dir only has the
/// MXFP4 file). `#[ignore]`d.
///
/// Measured 2026-09-24 (fresh reference server on 8870, first request):
/// **8/16** with the flip at step 7 (4705 vs 4022, pair-wise gap 0.065 logits
/// on the reference's 0.030 top-1 margin). The file mixes in 24 Q4_K
/// `attn_output` tensors + 61 Q5_0 + 13 Q8_0 that MXFP4 does not have; of those
/// only Q4_K takes a different path in the reference (q4_K_8x8 repack, see the
/// module header) — measurement and artifacts in crates/ggml/src/vec_dot.rs
/// `kquant_real_tensor_tests` (parity/q4k_goss_ref.bin).
///
/// Run:
///   cargo test --release -p llama --test gpt_oss_e2e -- --ignored --nocapture \
///       gpt_oss_20b_q4_k_m_reference_parity
#[test]
#[ignore = "manual: 11.8 GiB Q4_K_M model, prefill + 16 decode steps in release"]
fn gpt_oss_20b_q4_k_m_reference_parity() {
    let Some(l) = load_real(GPTOSS20B_Q4KM) else {
        return;
    };
    if !mem_guard("gpt_oss_20b_q4_k_m_reference_parity", 4.0) {
        return;
    }
    let n_ctx = 512u32;
    let (attn, gp) = gpt_oss_params(&l.model);
    let (n_k, n_v) = kv_widths(&l.model);
    let w = gpt_oss_weights(&l.model);
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let prompt_ids = vocab.tokenize(PROMPT, true, false);

    // the model's own build context: the TensorIds in `w` index into it
    let n_layer = l.model.layers.len();
    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, n_ctx);
    let mut h = Harness {
        gctx,
        kv,
        watermark: 0,
    };
    h.watermark = h.gctx.mark();

    let t0 = std::time::Instant::now();
    let pos: Vec<i32> = (0..prompt_ids.len() as i32).collect();
    let mut logits = h.decode(&w, &attn, &gp, &prompt_ids, &pos);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut gen_ids = Vec::new();
    let mut gen_logprobs: Vec<Vec<(i32, f32)>> = Vec::new();
    let mut gen_ms = 0f64;
    let mut next_pos = pos.last().unwrap() + 1;
    for step in 0..16 {
        let id = argmax(&logits);
        gen_ids.push(id);
        gen_logprobs.push(logprobs(&logits));
        if step + 1 < 16 {
            let t1 = std::time::Instant::now();
            logits = h.decode(&w, &attn, &gp, &[id], &[next_pos]);
            gen_ms += t1.elapsed().as_secs_f64() * 1e3;
            next_pos += 1;
        }
    }
    let text: String = gen_ids.iter().map(|&t| vocab.token_to_piece(t)).collect();
    println!("greedy ids: {gen_ids:?}");
    println!("text      : {text:?}");
    println!(
        "perf      : prefill {:.1} ms ({:.2} t/s), gen {:.2} t/s",
        prefill_ms,
        prompt_ids.len() as f64 / (prefill_ms / 1e3),
        15.0 / (gen_ms / 1e3)
    );
    let mut matched = 0usize;
    let mut first_diff: Option<usize> = None;
    for (i, (&got, &want)) in gen_ids.iter().zip(REF16_Q4KM.iter()).enumerate() {
        if got == want {
            matched += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    println!("MATCH     : {matched}/16 vs reference (fresh server, first request)");

    // pair-wise logprob diff = logit diff (the reference server reports
    // log_softmax values), the diagnostic that separates "tie inside a noise
    // band" from "a structural difference".
    let mut gap_at_diff = f32::NAN;
    for (i, (mine, theirs)) in gen_logprobs.iter().zip(REF_TOP8_Q4KM.iter()).enumerate() {
        let mine5: Vec<(i32, f32)> = mine.iter().take(5).copied().collect();
        println!(
            "step {i:2}: ref top5 {:?}  mine top5 {:?}",
            theirs.iter().take(5).map(|&(id, _)| id).collect::<Vec<_>>(),
            mine5
                .iter()
                .map(|&(id, lp)| (id, (lp * 1000.0).round() / 1000.0))
                .collect::<Vec<_>>()
        );
        let mut worst = 0f32;
        for &(id, ref_lp) in theirs.iter().take(5) {
            if let Some(&(_, my_lp)) = mine.iter().find(|&&(i, _)| i == id) {
                println!(
                    "          pair id {id}: ref {ref_lp:+.3} mine {my_lp:+.3} (d {:+.3})",
                    my_lp - ref_lp
                );
                if let Some(k) = first_diff {
                    if i == k {
                        worst = worst.max((my_lp - ref_lp).abs());
                    }
                }
            } else {
                println!("          pair id {id}: ref {ref_lp:+.3} mine (outside top-8)");
            }
        }
        if first_diff == Some(i) {
            gap_at_diff = worst;
            let ref_margin = theirs[0].1 - theirs[1].1;
            let my_margin = mine[0].1 - mine[1].1;
            println!(
                "  FIRST DIVERGENCE step {i}: mine {my_top} (margin {my_margin:+.3}) vs ref {ref_top} \
                 (margin {ref_margin:+.3}); pair-wise gap at the flip {gap_at_diff:+.3} logits",
                my_top = mine[0].0,
                ref_top = theirs[0].0
            );
        }
    }

    if let Some(k) = first_diff {
        println!(
            "FIRST DIVERGENCE step {k}: mine {} vs ref {}",
            gen_ids[k], REF16_Q4KM[k]
        );
        println!(
            "  => residual at the flip {gap_at_diff:+.3} logits; the reference's own margin there \
             {:.3} (a tie whose flip needs only a small residual)",
            REF_TOP8_Q4KM[k][0].1 - REF_TOP8_Q4KM[k][1].1
        );
    } else {
        println!("16/16 — no divergence");
    }

    // Floor: the MXFP4 tests reach 16/16 because in that conversion *every*
    // weight type matches the reference's production path. This file mixes in
    // 24 Q4_K tensors (blk.N.attn_output), which the reference repacks to
    // q4_K_8x8 at load (GGML_USE_CPU_REPACK, repack.cpp:5006) and computes with
    // the outer-product gemv/gemm, while this port has no Q4_K repack — its
    // row-wise `vec_dot_q4_K_q8_K` equals the reference's *plain* path bit for
    // bit (parity/q4k_goss_ref.bin, crates/ggml/src/vec_dot.rs
    // kquant_real_tensor_tests) but differs from the repacked result by
    // ~1.3e-6 relative. Q5_0/Q8_0 (the other new types here) are bit-exact
    // despite the reference routing their GEMMs through llamafile tinyBLAS.
    assert!(
        matched >= 8,
        "regression: only {matched}/16 tokens match (first diff {first_diff:?})"
    );
    if let Some(k) = first_diff {
        assert!(
            k >= 7,
            "regression: divergence at step {k} (expected the step-7 tie)"
        );
        assert!(
            gap_at_diff < 0.2,
            "first divergence at step {k}: pair-wise residual {gap_at_diff:.3} logits is beyond \
             the documented Q4_K-repack noise band"
        );
    }
}
// ===========================================================================
// 3. MXFP4 mul_mat_id on real expert weights (manual; heavy-ish)
// ===========================================================================

/// The MXFP4 path has no vec_dot in vec_dot.rs (the reference registers it in
/// ggml-cpu.c:287) — this checks the ported `vec_dot_mxfp4_q8_0` kernel against
/// the *dequantized* weights on the real gpt-oss expert tensor (the dequantizer
/// itself is bit-exact vs the C reference, PARITY.md).
///
///   cargo test --release -p llama --test gpt_oss_e2e -- --ignored --nocapture \
///       dbg_mxfp4_real_weights
#[test]
#[ignore = "manual: reads one 141 MiB MXFP4 expert tensor from the model"]
fn dbg_mxfp4_real_weights() {
    let Some(l) = load_real(GPTOSS20B_MXFP4) else {
        return;
    };
    let hp = &l.model.hparams;
    let n = hp.n_embd as usize; // 2880
    let n_ff = l.model.ctx.ne(l.model.layers[0].ffn_gate_exps.unwrap())[1] as usize;
    let _n_expert = hp.n_expert as usize;
    let a = l.model.layers[0].ffn_gate_exps.unwrap();
    let mut ctx = l.model.ctx;
    assert_eq!(ctx.ty(a), GgmlType::Mxfp4);

    // b: two columns with a deterministic pattern
    let b = ctx.new_tensor_3d(GgmlType::F32, n as i64, 1, 1);
    ctx.arena_resize_tensor(b);
    let xv: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) * 2.0 - 1.0)
        .collect();
    ctx.with_f32_mut(b, |p| p.copy_from_slice(&xv)).unwrap();
    // ids: [2, 1] -> slot0 = expert 5, slot1 = expert 0
    let ids = ctx.new_tensor_2d(GgmlType::I32, 2, 1);
    ctx.arena_resize_tensor(ids);
    ctx.with_i32_mut(ids, |p| p.copy_from_slice(&[5, 0]))
        .unwrap();

    let out = ctx.mul_mat_id(a, b, ids);
    let mut g = ggml::graph::Graph::new(8);
    g.build_forward(&ctx, out);
    ggml::compute::graph_compute(&mut ctx, &mut g, 4);
    let got: Vec<f32> = ctx
        .data_bytes(out)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // reference: dequantize the expert rows, quantize the activation row the same
    // way the kernel does (Q8_0) and dot in f64
    let abytes = ctx.data_bytes(a).unwrap();
    let rs_x = GgmlType::Mxfp4.row_size(n); // 17*90
    let mut yq = vec![0u8; GgmlType::Q8_0.row_size(n)];
    ggml::quants::quantize_row_q8_0(&xv, bytemuck::cast_slice_mut(&mut yq));
    let mut yf = vec![0f32; n];
    ggml::quants::dequantize_row(GgmlType::Q8_0, &yq, &mut yf);

    let mut worst_rel = 0f64;
    let mut worst = (0usize, 0usize, 0f64, 0f64);
    for (slot, e) in [5usize, 0].iter().enumerate() {
        let mut wrow = vec![0f32; n];
        for r in 0..n_ff {
            let off = (r + e * n_ff) * rs_x;
            ggml::quants::dequantize_row(GgmlType::Mxfp4, &abytes[off..off + rs_x], &mut wrow);
            let want: f64 = wrow
                .iter()
                .zip(&yf)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            let got_v = got[slot * n_ff + r] as f64;
            let rel = (got_v - want).abs() / want.abs().max(1e-6);
            if rel > worst_rel {
                worst_rel = rel;
                worst = (slot, r, got_v, want);
            }
        }
    }
    println!("mxfp4 real weights: worst rel {worst_rel:.3e} at (slot,row,g,want) {worst:?}");
    assert!(worst_rel < 2e-3, "mxfp4 kernel mismatch on real weights");
}

// ===========================================================================
// 4. layer-0 magnitude probe on the real weights (manual)
// ===========================================================================

fn rms_of(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn read_f32(ctx: &Context, id: TensorId) -> Vec<f32> {
    ctx.data_bytes(id)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Layer-0 magnitudes with the real weights: attention output vs MoE output vs
/// the residual, plus the kq logit spread and the sink share. No reference
/// needed — this is a sanity probe for the branch wiring.
///
///   cargo test --release -p llama --test gpt_oss_e2e -- --ignored --nocapture \
///       dbg_real_layer0_magnitudes
#[test]
#[ignore = "manual: real 11 GiB model, one layer forward"]
fn dbg_real_layer0_magnitudes() {
    let Some(l) = load_real(GPTOSS20B_MXFP4) else {
        return;
    };
    let (attn, gp) = gpt_oss_params(&l.model);
    let (n_k, n_v) = kv_widths(&l.model);
    let w = gpt_oss_weights(&l.model);
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let prompt_ids = vocab.tokenize(PROMPT, true, false);
    let n_layer = l.model.layers.len();
    let mut ctx = l.model.ctx;
    let kv = KvCache::new(&mut ctx, n_layer, n_k, n_v, 512);
    let n = prompt_ids.len();
    let tokens_t = ctx.new_tensor_1d(GgmlType::I32, n as i64);
    let pos_t = ctx.new_tensor_1d(GgmlType::I32, n as i64);
    let kq_mask = ctx.new_tensor_2d(GgmlType::F32, n as i64, n as i64);
    let row_idx = ctx.new_tensor_1d(GgmlType::I64, n as i64);
    for t in [tokens_t, pos_t, kq_mask, row_idx] {
        ctx.arena_resize_tensor(t);
    }
    ctx.with_i32_mut(tokens_t, |p| p.copy_from_slice(&prompt_ids))
        .unwrap();
    let pos: Vec<i32> = (0..n as i32).collect();
    ctx.with_i32_mut(pos_t, |p| p.copy_from_slice(&pos))
        .unwrap();
    ctx.data_bytes_mut(row_idx)
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&[0i64, 1, 2, 3, 4]));
    {
        let m: &mut [f32] = bytemuck::cast_slice_mut(ctx.data_bytes_mut(kq_mask).unwrap());
        m.fill(f32::NEG_INFINITY);
        for iq in 0..n {
            for ik in 0..n {
                if ik <= iq {
                    m[iq * n + ik] = 0.0;
                }
            }
        }
    }

    let lw = &w.layers[0];
    let mut graph = ggml::graph::Graph::new(256);
    let inp_l = ctx.get_rows(w.tok_embd, tokens_t);
    let cur = {
        let r = ctx.rms_norm(inp_l, attn.norm_eps);
        ctx.mul(r, lw.attn_norm)
    };
    let q = {
        let q = ctx.mul_mat(lw.wq, cur);
        let q = ctx.reshape_3d(q, attn.n_embd_head_k, attn.n_head, n as i64);
        ctx.rope_ext(
            q,
            pos_t,
            None,
            attn.n_rot as i32,
            attn.rope_mode,
            attn.n_ctx_orig,
            attn.freq_base,
            attn.freq_scale,
            attn.ext_factor,
            attn.attn_factor,
            attn.beta_fast,
            attn.beta_slow,
        )
    };
    let kk = {
        let k = ctx.mul_mat(lw.wk, cur);
        let k = ctx.reshape_3d(k, attn.n_embd_head_k, attn.n_head_kv, n as i64);
        ctx.rope_ext(
            k,
            pos_t,
            None,
            attn.n_rot as i32,
            attn.rope_mode,
            attn.n_ctx_orig,
            attn.freq_base,
            attn.freq_scale,
            attn.ext_factor,
            attn.attn_factor,
            attn.beta_fast,
            attn.beta_slow,
        )
    };
    let v = {
        let v = ctx.mul_mat(lw.wv, cur);
        ctx.reshape_3d(v, attn.n_embd_head_v, attn.n_head_kv, n as i64)
    };
    // cpy_k / cpy_v
    let nb2 = ctx.nb(kk)[2] as usize;
    let k_rows = ctx.view_2d(kk, attn.n_embd_head_k * attn.n_head_kv, n as i64, nb2, 0);
    let v_rows = ctx.view_2d(v, attn.n_embd_head_v * attn.n_head_kv, n as i64, nb2, 0);
    let k_dst = ctx.set_rows(kv.layers[0].k, k_rows, row_idx);
    let v_dst = ctx.set_rows(kv.layers[0].v, v_rows, row_idx);
    graph.build_forward(&ctx, k_dst);
    graph.build_forward(&ctx, v_dst);
    let k_view = kv.get_k(&mut ctx, 0, attn.n_embd_head_k, attn.n_head_kv, n as u32);
    let v_view = kv.get_v(&mut ctx, 0, attn.n_embd_head_v, attn.n_head_kv, n as u32);
    let qp = ctx.permute(q, 0, 2, 1, 3);
    let kp = ctx.permute(k_view, 0, 2, 1, 3);
    let kq = ctx.mul_mat(kp, qp);
    let kqs = ctx.soft_max_ext(kq, Some(kq_mask), 1.0 / (attn.n_rot as f32).sqrt(), 0.0);
    ctx.soft_max_add_sinks(kqs, Some(lw.attn_sinks));
    let vp = ctx.permute(v_view, 0, 2, 1, 3);
    let vt = ctx.transpose(vp);
    let vc = ctx.cont(vt);
    let kqv = ctx.mul_mat(vc, kqs);
    let kqvp = ctx.permute(kqv, 0, 2, 1, 3);
    let kqvc = ctx.cont(kqvp);
    let flat = ctx.reshape_2d(kqvc, attn.n_embd_head_v * attn.n_head, n as i64);
    let attn_out = ctx.mul_mat(lw.wo, flat);
    let attn_out = ctx.add(attn_out, lw.wo_b);
    let ffn_inp = ctx.add(attn_out, inp_l);
    let normed = {
        let r = ctx.rms_norm(ffn_inp, attn.norm_eps);
        ctx.mul(r, lw.attn_post_norm)
    };
    // MoE via the builder's own path is not reachable here; call the public
    // pieces the same way build_moe_ffn_gpt_oss does.
    let (n_embd, n_tok, k_used) = (2880i64, n as i64, gp.n_expert_used[0] as i64);
    let logits = ctx.mul_mat(lw.ffn_gate_inp, normed);
    let logits = ctx.add(logits, lw.ffn_gate_inp_b);
    let sel = ctx.argsort_top_k(logits, k_used as i32);
    let probs = ctx.reshape_3d(logits, 1, gp.n_expert, n_tok);
    let wts = ctx.get_rows(probs, sel);
    let wts = ctx.reshape_2d(wts, k_used, n_tok);
    let wts = ctx.soft_max(wts);
    let wts = ctx.reshape_3d(wts, 1, k_used, n_tok);
    let cur3 = ctx.reshape_3d(normed, n_embd, 1, n_tok);
    let up = ctx.mul_mat_id(lw.ffn_up_exps, cur3, sel);
    let up = ctx.add_id(up, lw.ffn_up_exps_b, sel);
    let gate = ctx.mul_mat_id(lw.ffn_gate_exps, cur3, sel);
    let gate = ctx.add_id(gate, lw.ffn_gate_exps_b, sel);
    let act = ctx.swiglu_oai(gate, up, gp.swiglu_oai_alpha, gp.swiglu_oai_limit);
    let down = ctx.mul_mat_id(lw.ffn_down_exps, act, sel);
    let down = ctx.add_id(down, lw.ffn_down_exps_b, sel);
    let experts = ctx.mul(down, wts);
    let nb1 = ctx.nb(experts)[1] as usize;
    let nb22 = ctx.nb(experts)[2] as usize;
    let mut moe = ctx.view_2d(experts, n_embd, n_tok, nb22, 0);
    graph.build_forward(&ctx, moe);
    for i in 1..k_used as usize {
        let vi = ctx.view_2d(experts, n_embd, n_tok, nb22, i * nb1);
        graph.build_forward(&ctx, vi);
        moe = ctx.add(moe, vi);
        graph.build_forward(&ctx, moe);
    }
    let l0 = ctx.add(moe, ffn_inp);
    for tid in [
        cur, q, kq, kqs, attn_out, ffn_inp, normed, sel, wts, act, down, moe, l0,
    ] {
        graph.build_forward(&ctx, tid);
    }
    ggml::compute::graph_compute(&mut ctx, &mut graph, 8);

    let emb = read_f32(&ctx, inp_l);
    println!(
        "inp_l      rms {:.4} max {:.4}",
        rms_of(&emb),
        emb.iter().fold(0f32, |a, b| a.max(b.abs()))
    );
    for (name, id) in [
        ("normed_in", cur),
        ("attn_out", attn_out),
        ("ffn_inp", ffn_inp),
        ("moe", moe),
        ("l0_out", l0),
    ] {
        let v = read_f32(&ctx, id);
        println!(
            "{name:<10} rms {:.4} max {:.4}",
            rms_of(&v),
            v.iter().fold(0f32, |a, b| a.max(b.abs()))
        );
    }
    let kq_raw = read_f32(&ctx, kq);
    let sinks0 = read_f32(&ctx, lw.attn_sinks);
    let n_kv = n;
    let mut extremes = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in kq_raw.iter() {
        extremes.0 = extremes.0.min(v);
        extremes.1 = extremes.1.max(v);
        extremes.2 += v as f64;
    }
    // per-head spread of (max over kv) - sink for token 0
    let mut diff = Vec::new();
    for h in 0..attn.n_head as usize {
        let mut mx = f32::NEG_INFINITY;
        for s in 0..n_kv {
            mx = mx.max(kq_raw[s + 0 * n_kv + h * n_kv * n]);
        }
        diff.push(mx - sinks0[h]);
    }
    println!(
        "kq(raw)     min {:.3} max {:.3} mean {:.4}  |  (max_kv - sink) per head: min {:.2} max {:.2} mean {:.2}",
        extremes.0,
        extremes.1,
        extremes.2 / kq_raw.len() as f64,
        diff.iter().cloned().fold(f32::INFINITY, f32::min),
        diff.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        diff.iter().sum::<f32>() / diff.len() as f32
    );
    let qv = read_f32(&ctx, q);
    let kvv = read_f32(&ctx, kk);
    println!(
        "q rms {:.4} max {:.4}; k rms {:.4} max {:.4}",
        rms_of(&qv),
        qv.iter().fold(0f32, |a, b| a.max(b.abs())),
        rms_of(&kvv),
        kvv.iter().fold(0f32, |a, b| a.max(b.abs()))
    );
    let kqv_v = read_f32(&ctx, kqs);
    println!(
        "softmax     sum-per-row {:?}",
        kqv_v
            .chunks_exact(n)
            .map(|r| r.iter().sum::<f32>())
            .collect::<Vec<f32>>()
    );
    println!(
        "softmax     max {:.4}",
        kqv_v.iter().fold(0f32, |a, b| a.max(b.abs()))
    );
    let selv = ctx.i32s(sel).unwrap().to_vec();
    let snb1 = ctx.nb(sel)[1] as usize / 4;
    println!(
        "selected experts per token: {:?}",
        (0..n)
            .map(|t| selv[t * snb1..t * snb1 + k_used as usize].to_vec())
            .collect::<Vec<_>>()
    );
    let wv = read_f32(&ctx, wts);
    println!(
        "router weights: {:?}",
        (0..n)
            .map(|t| wv[t * k_used as usize..(t + 1) * k_used as usize].to_vec())
            .collect::<Vec<_>>()
    );
    let av = read_f32(&ctx, act);
    println!("act rms {:.4}", rms_of(&av));
    let sinks = read_f32(&ctx, lw.attn_sinks);
    println!(
        "sinks min {:.3} max {:.3} mean {:.3}",
        sinks.iter().cloned().fold(f32::INFINITY, f32::min),
        sinks.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        sinks.iter().sum::<f32>() / sinks.len() as f32
    );
}
