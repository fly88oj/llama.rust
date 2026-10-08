//! lora_e2e.rs — end-to-end verification of the LoRA adapter port
//! (src/llama-adapter.cpp/.h + `llm_graph_context::build_lora_mm`,
//! llama-graph.cpp:1514-1543) on the qwen2.5-0.5b base model.
//!
//! The port has no LoRA adapter file on this machine, so this file *creates*
//! one with the port's byte-exact GGUF writer (`ggml::gguf_write`, the writer
//! pinned bit-for-bit against the reference in ggml's own tests): a synthetic
//! rank-8 adapter covering the projections of the first four layers plus
//! `token_embd.weight` (flipped A/B layout) and `output.weight`, with
//! `adapter.lora.alpha = 16` — i.e. `scale = 16/8 = 2` at adapter scale 1.
//!
//! What the reference required of that file (llama-adapter.cpp:168-241 — the
//! loader refuses everything else):
//!   * `general.type` == "adapter"                (:205-208)
//!   * `general.architecture` == "qwen2"          (:210-214, must equal the
//!     base model's arch)
//!   * `adapter.type` == "lora"                   (:216-219)
//!   * `adapter.lora.alpha` as F32                (:221, absent ⇒ 0.0 ⇒ the
//!     user scale is used unscaled)
//!   * every tensor named `<base tensor name>.lora_a` / `.lora_b`
//!     (:274-297), `.lora_a` = `[n_embd, rank]`, `.lora_b` = `[rank, n_out]`
//!     for plain weights and A/B flipped for `token_embd.weight` (:358-371)
//!   * F16 payloads are fine (the C dups the file type, :374-375)
//!
//! Runs by default: nothing here (the default-run adapter tests live in
//! `crates/llama/src/adapter.rs`: metadata/shape/scale-math + the f64
//! `lora_mm` reference).
//!
//! `#[ignore]`d (manual):
//!   * `gen_lora_adapter` — writes the synthetic adapter to `ADAPTER` and pins
//!     the file's SHA-256 against the one the reference accepted.
//!   * `lora_reference_parity` — the real forward pass: prefill + 16 greedy
//!     tokens with the adapter applied, compared against the reference
//!     capture below; then the same run with the adapter cleared (base model)
//!     and with `scale = 0`, which must reproduce the base logits bit-for-bit.
//!
//! Reference capture (PARITY.md protocol: fresh llama-server + first
//! `/completion` request, `temperature=0`, `cache_prompt=false`,
//! `logprobs=20`; the reference CLI needs a tty, the server does not):
//!   # 1. build the adapter (deterministic bytes)
//!   cargo test -p llama --release --test lora_e2e -- --ignored --nocapture gen_lora_adapter
//!   # 2. reference server with the adapter applied
//!   /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server \
//!       -m /home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!       -c 512 -t 8 -fa on --port 18130 --host 127.0.0.1 \
//!       --lora /tmp/rust-lora-qwen2-r8.gguf
//!   curl -s http://127.0.0.1:18130/health          # wait for {"status":"ok"}
//!   curl -s http://127.0.0.1:18130/completion -H 'Content-Type: application/json' \
//!       -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0,
//!            "cache_prompt":false,"logprobs":20}'
//!   # repeat on a fresh server/port with `-fa off` for the non-FA capture,
//!   # and with `--lora-scaled /tmp/rust-lora-qwen2-r8.gguf:0` for the
//!   # "adapter cleared" capture (must equal the plain-model run).
//!
//! Measured 2026-09-25 (fresh servers, first request): see PARITY.md.
//!
//! Run:
//!   cargo test -p llama --release --test lora_e2e -- --ignored --nocapture

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, Value};
use llama::adapter::{self, AdapterLora};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const MODEL: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const ADAPTER: &str = "/tmp/rust-lora-qwen2-r8.gguf";

/// sha256 of the adapter file the reference accepted (see `gen_lora_adapter`);
/// the file is deterministic, so a mismatch means the generator changed and
/// every captured number below has to be re-taken.
const ADAPTER_SHA256: &str = "df1c5636845e6a71b1795bad3a4160279876817a42e686e8f329813d7707e585";

const PROMPT: &str = "The capital of France is";
const N_PREDICT: usize = 16;
/// n_ctx / n_batch of the reference captures (llama-server `-c 512`) and of the
/// port's DecodeContext below.
const N_CTX: u32 = 512;

/// qwen2.5-0.5b: n_embd 896, n_ff 4864, n_head_kv * head_dim = 128,
/// vocab 151936 (llama-gguf dump of the base file).
const N_EMBD: i64 = 896;
const N_FF: i64 = 4864;
const N_EMBD_GQA: i64 = 128;
const N_VOCAB: i64 = 151936;
const RANK: i64 = 8;
const ALPHA: f32 = 16.0;

/// Reference greedy ids, `-fa on` + `--lora /tmp/rust-lora-qwen2-r8.gguf`
/// (text: ' Paris. It is the capital of the European Union. It is the capital of').
const REF16_FA: [i32; N_PREDICT] = [
    12095, 13, 1084, 374, 279, 6722, 315, 279, 7513, 9145, 13, 1084, 374, 279, 6722, 315,
];

/// Reference per-step top-5 (id, logprob), `-fa on` + the adapter.
#[rustfmt::skip]
const REF_TOP5_FA: [[(i32, f32); 5]; N_PREDICT] = [
    [(12095, -0.152), (32671, -3.804), (1304, -3.832), (30743, -4.315), (508, -4.519)],
    [(13, -0.555), (11, -2.114), (624, -2.183), (382, -2.560), (323, -2.768)],
    [(1084, -1.597), (15920, -1.913), (576, -2.447), (12095, -2.447), (758, -3.110)],
    [(374, -0.724), (572, -1.801), (702, -1.845), (594, -2.694), (748, -3.444)],
    [(279, -0.605), (7407, -2.352), (264, -2.522), (1083, -3.017), (30083, -3.122)],
    [(6722, -0.764), (7772, -1.528), (10723, -3.039), (1429, -3.048), (23513, -3.235)],
    [(315, -0.676), (3283, -0.862), (323, -4.008), (1576, -4.490), (11, -4.816)],
    [(279, -0.945), (9625, -1.363), (892, -2.082), (4505, -2.712), (30743, -4.170)],
    [(7513, -0.971), (3146, -1.206), (8585, -3.109), (15072, -3.598), (3639, -3.716)],
    [(9145, -0.044), (12062, -4.895), (3146, -5.024), (5537, -5.125), (11300, -6.203)],
    [(13, -0.894), (323, -1.697), (11, -1.942), (320, -2.357), (382, -2.910)],
    [(1084, -0.949), (15920, -2.459), (576, -2.842), (12095, -2.908), (758, -3.438)],
    [(374, -0.494), (702, -1.681), (572, -2.532), (594, -3.792), (1083, -4.081)],
    [(279, -0.500), (1083, -1.536), (264, -3.356), (537, -3.995), (7407, -4.048)],
    [(6722, -0.658), (7772, -1.981), (10723, -2.327), (2086, -3.637), (3283, -4.130)],
    [(315, -0.070), (3283, -3.038), (369, -5.613), (323, -5.999), (11, -6.726)],
];

/// Reference greedy ids, `-fa off` + the adapter (same 16 ids and the same text
/// as `-fa on`, exactly like the base model: the FA/non-FA split only moves the
/// logprob tail).
const REF16_NOFA: [i32; N_PREDICT] = REF16_FA;

/// Reference per-step top-5 (id, logprob), `-fa off` + the adapter.
#[rustfmt::skip]
const REF_TOP5_NOFA: [[(i32, f32); 5]; N_PREDICT] = [
    [(12095, -0.276), (32671, -3.313), (1304, -3.343), (508, -3.941), (30743, -4.005)],
    [(13, -0.570), (11, -2.039), (624, -2.226), (382, -2.615), (323, -2.703)],
    [(1084, -1.620), (15920, -1.836), (12095, -2.425), (576, -2.470), (758, -3.147)],
    [(374, -0.692), (572, -1.818), (702, -1.888), (594, -2.696), (748, -3.581)],
    [(279, -0.604), (7407, -2.351), (264, -2.569), (1083, -2.945), (30083, -3.104)],
    [(6722, -0.766), (7772, -1.515), (10723, -3.034), (1429, -3.090), (23513, -3.287)],
    [(315, -0.627), (3283, -0.906), (323, -4.171), (1576, -4.793), (11, -4.858)],
    [(279, -1.002), (9625, -1.370), (892, -1.840), (4505, -2.810), (30743, -4.119)],
    [(7513, -0.862), (3146, -1.313), (8585, -3.000), (15072, -3.731), (3639, -3.782)],
    [(9145, -0.041), (12062, -4.637), (5537, -5.345), (3146, -5.464), (11300, -6.282)],
    [(13, -0.880), (323, -1.659), (11, -1.991), (320, -2.390), (382, -2.898)],
    [(1084, -0.990), (15920, -2.451), (576, -2.849), (12095, -2.973), (758, -3.411)],
    [(374, -0.493), (702, -1.677), (572, -2.513), (594, -3.783), (1083, -4.076)],
    [(279, -0.488), (1083, -1.548), (264, -3.382), (7407, -4.070), (537, -4.078)],
    [(6722, -0.537), (7772, -2.180), (10723, -2.528), (2086, -3.554), (3283, -4.291)],
    [(315, -0.065), (3283, -3.119), (369, -5.732), (323, -6.068), (320, -6.840)],
];

// ---------------------------------------------------------------------------
// `--lora-scaled <adapter>:0.5` (reference, `-fa off`)
// ---------------------------------------------------------------------------

/// The `adapter_scale` path (`scale = adapter_scale * alpha / rank`,
/// llama-adapter.h:53-57): a *third* trajectory, distinct from the base model
/// and from scale 1.0 — and it hinges on a 0.035-logprob tie at step 5
/// (7772 vs 6722), so a scale-blind port cannot match it by luck.
const REF16_HALF: [i32; N_PREDICT] = [
    12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279,
];

#[rustfmt::skip]
const REF_TOP5_HALF: [[(i32, f32); 5]; N_PREDICT] = [
    [(12095, -0.525), (32671, -2.667), (1304, -2.787), (30743, -3.219), (510, -3.673)],
    [(13, -0.551), (11, -1.756), (624, -2.394), (382, -2.747), (323, -3.035)],
    [(1084, -1.469), (576, -2.212), (15920, -2.486), (12095, -2.657), (758, -3.082)],
    [(374, -0.633), (702, -1.926), (572, -2.061), (594, -2.645), (748, -3.388)],
    [(279, -0.844), (7407, -2.014), (264, -2.046), (30083, -3.094), (304, -3.313)],
    [(7772, -1.238), (6722, -1.273), (1429, -2.603), (10723, -2.852), (23513, -2.971)],
    [(3283, -0.185), (323, -2.292), (57406, -4.118), (11, -4.395), (315, -4.660)],
    [(304, -0.313), (315, -2.148), (323, -2.398), (553, -3.453), (320, -4.733)],
    [(4505, -0.927), (279, -1.232), (9625, -1.373), (18494, -4.386), (10867, -4.868)],
    [(323, -0.691), (13, -1.508), (11, -1.812), (553, -3.086), (448, -3.874)],
    [(279, -0.496), (825, -2.481), (702, -2.722), (374, -2.931), (264, -3.678)],
    [(2086, -1.095), (4843, -1.418), (25031, -2.855), (7772, -2.910), (1429, -3.324)],
    [(7772, -0.606), (66967, -0.960), (1429, -3.308), (8538, -3.932), (62398, -5.038)],
    [(3283, -0.533), (304, -0.968), (15662, -5.168), (1283, -5.440), (323, -5.796)],
    [(304, -0.049), (389, -4.389), (553, -4.530), (1283, -4.928), (315, -5.659)],
    [(279, -0.006), (10867, -6.469), (4505, -6.892), (9625, -7.113), (50015, -8.762)],
];

// ---------------------------------------------------------------------------
// the synthetic adapter
// ---------------------------------------------------------------------------

/// `(base tensor name, model ne)` of every tensor the adapter carries a pair
/// for. Layers 0..4 + the two embedding-side tensors: the lm head and the
/// embedding gather are the two *different* lora paths (`build_lora_mm` vs
/// `build_inp_embd`, llama-graph.cpp:2389-2405), so both must be covered.
fn adapter_plan() -> Vec<(String, [i64; 4])> {
    let mut plan: Vec<(String, [i64; 4])> = Vec::new();
    for il in 0..4 {
        let p = format!("blk.{il}.");
        plan.push((format!("{p}attn_q.weight"), [N_EMBD, N_EMBD, 1, 1]));
        plan.push((format!("{p}attn_k.weight"), [N_EMBD, N_EMBD_GQA, 1, 1]));
        plan.push((format!("{p}attn_v.weight"), [N_EMBD, N_EMBD_GQA, 1, 1]));
        plan.push((format!("{p}attn_output.weight"), [N_EMBD, N_EMBD, 1, 1]));
        plan.push((format!("{p}ffn_gate.weight"), [N_EMBD, N_FF, 1, 1]));
        plan.push((format!("{p}ffn_up.weight"), [N_EMBD, N_FF, 1, 1]));
        plan.push((format!("{p}ffn_down.weight"), [N_FF, N_EMBD, 1, 1]));
    }
    plan.push(("token_embd.weight".to_string(), [N_EMBD, N_VOCAB, 1, 1]));
    plan.push(("output.weight".to_string(), [N_EMBD, N_VOCAB, 1, 1]));
    plan
}

/// Deterministic F16 payload for one lora tensor: a tiny LCG so the values are
/// reproducible without a dependency (`y = 1103515245*x + 12345`).
fn lora_payload(n: usize, seed: u32, scale: f32) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u = ((state >> 8) & 0xffff) as f32 / 65535.0; // [0, 1]
        let v = (u - 0.5) * 2.0 * scale;
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Write the synthetic adapter; the shape rules mirror the loader's validation
/// (llama-adapter.cpp:358-371): plain weights take `a = [ne0, rank]`,
/// `b = [rank, ne1]`, `token_embd.weight` is flipped.
fn write_adapter(path: &str) -> std::io::Result<()> {
    let mut w = GgufWriter::new(32);
    w.set_kv("general.type", Value::String("adapter".into()));
    w.set_kv("general.architecture", Value::String("qwen2".into()));
    w.set_kv("adapter.type", Value::String("lora".into()));
    w.set_kv("adapter.lora.alpha", Value::F32(ALPHA));

    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut seed = 0x51ed270bu32;
    for (name, ne) in adapter_plan() {
        let is_tok_embd = name == "token_embd.weight";
        let (a_ne, b_ne) = if is_tok_embd {
            // A and B flipped (llama-adapter.cpp:359-363): b ne[1] == n_embd
            ([RANK, ne[1], 1, 1], [RANK, ne[0], 1, 1])
        } else {
            ([ne[0], RANK, 1, 1], [RANK, ne[1], 1, 1])
        };
        for (suffix, tne) in [(".lora_a", a_ne), (".lora_b", b_ne)] {
            w.add_tensor(&format!("{name}{suffix}"), GgmlType::F16, tne);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            payloads.push(lora_payload(tne[0] as usize * tne[1] as usize, seed, 0.03));
        }
    }
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let mut buf = Vec::new();
    w.write(&mut buf, &refs)?;
    std::fs::write(path, &buf)
}

/// Writes the adapter and pins it by hash — the reference capture below is only
/// valid for this exact file.
#[test]
#[ignore]
fn gen_lora_adapter() {
    write_adapter(ADAPTER).expect("write adapter");
    let bytes = std::fs::read(ADAPTER).expect("read back");
    let sum = sha256_hex(&bytes);
    println!("wrote {ADAPTER}: {} bytes, sha256 {sum}", bytes.len());
    println!("pairs: {}", adapter_plan().len());
    assert_eq!(
        sum, ADAPTER_SHA256,
        "adapter generator changed — re-capture PARITY.md"
    );
}

/// Minimal SHA-256 (no external dependency; the workspace has no hash crate).
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
}

fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).expect("open model");
    // SAFETY: read-only use of a model file (same policy as the rest of the port)
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    match load_model(&gguf, mmap.clone()) {
        Ok(model) => Some(Loaded { model, mmap }),
        Err(e) => {
            eprintln!("SKIP: load_model({path}) failed: {e}");
            None
        }
    }
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

/// Report line `step k: id port ref delta …` for the reference's own top-5 ids.
/// Returns `(worst |delta|, worst rank-5 overlap)`: the reference and the port
/// flip near-ties at ranks 3-5 (the tails sit within ~0.3 logprobs of each
/// other, exactly like the base model's documented numeric band), so the check
/// is "≥ 4 of the reference's 5 ids are in the port's top-5, and every id's
/// logprob agrees".
fn report_top5(rows: &[Vec<f32>], ref_top5: &[[(i32, f32); 5]], label: &str) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut min_overlap = 5usize;
    for (step, (row, hers)) in rows.iter().zip(ref_top5).enumerate() {
        let mine_ids = top5_ids(row);
        let overlap = hers.iter().filter(|(id, _)| mine_ids.contains(id)).count();
        min_overlap = min_overlap.min(overlap);
        assert!(
            overlap >= 4,
            "{label} step {step}: only {overlap}/5 of the reference's ids are in the port's top-5 \
             {:?} (reference {:?})",
            mine_ids,
            hers.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        );
        let mut parts = Vec::new();
        for &(id, lp) in hers.iter() {
            let mine = logprob_of(row, id);
            let d = mine - lp;
            worst = worst.max(d.abs());
            parts.push(format!("{id}:{mine:+.3}/{lp:+.3}({d:+.3})"));
        }
        println!(
            "{label} step {step:2}: overlap {overlap}/5  {}",
            parts.join(" ")
        );
    }
    (worst, min_overlap)
}

/// `AttnParams` for qwen2 — the same derivation llama-cli uses.
fn attn_params(m: &LlamaModel, fa: bool) -> AttnParams {
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

fn qwen2_weights(m: &LlamaModel) -> ModelWeights {
    let layers: Vec<LayerWeights> = m
        .layers
        .iter()
        .map(|l| LayerWeights {
            attn_norm: l.attn_norm.expect("attn_norm"),
            wq: l.wq.expect("wq"),
            wk: l.wk.expect("wk"),
            wv: l.wv.expect("wv"),
            wo: l.wo.expect("wo"),
            wq_b: l.wq_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            ffn_norm: l.ffn_norm.expect("ffn_norm"),
            ffn_gate: l.ffn_gate.expect("ffn_gate"),
            ffn_down: l.ffn_down.expect("ffn_down"),
            ffn_up: l.ffn_up.expect("ffn_up"),
        })
        .collect();
    ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers,
    }
}

/// One full run: prefill the prompt, then `N_PREDICT` greedy tokens. Returns
/// the generated ids and the full logits row each token was sampled from (the
/// top-5 comparison needs logprobs of *specific* ids, not just the top ranks —
/// the reference and the port flip near-ties at ranks 3-5).
fn run_greedy(dctx: &mut DecodeContext, tokens: &[i32]) -> (Vec<i32>, Vec<Vec<f32>>) {
    let pos: Vec<i32> = (0..tokens.len() as i32).collect();
    let mut logits = dctx.decode(tokens, &pos).expect("prefill").to_vec();
    let mut ids = Vec::with_capacity(N_PREDICT);
    let mut rows = Vec::with_capacity(N_PREDICT);
    let mut next_pos = tokens.len() as i32;
    for _ in 0..N_PREDICT {
        rows.push(logits.clone());
        let id = argmax(&logits);
        ids.push(id);
        logits = dctx.decode(&[id], &[next_pos]).expect("decode").to_vec();
        next_pos += 1;
    }
    (ids, rows)
}

/// logprob of one id in a full row (llama-server's normalized convention).
fn logprob_of(row: &[f32], id: i32) -> f32 {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = mx + (row.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>()).ln() as f32;
    row[id as usize] - lse
}

/// The ids of the port's own top-5, for the ordering-insensitive set check.
fn top5_ids(row: &[f32]) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    let mut ids: Vec<i32> = idx.into_iter().take(5).map(|i| i as i32).collect();
    ids.sort_unstable();
    ids
}

// ---------------------------------------------------------------------------
// the parity test
// ---------------------------------------------------------------------------

/// Real forward pass with the synthetic adapter, compared against the reference
/// capture in the header; then base-model-unchanged and scale-0 checks.
#[test]
#[ignore]
fn lora_reference_parity() {
    if !Path::new(ADAPTER).exists() {
        eprintln!("adapter {ADAPTER} missing — run gen_lora_adapter first; SKIP");
        return;
    }
    let Some(mut loaded) = load_real(MODEL) else {
        return;
    };
    let fa = std::env::var("LORA_FA_OFF").is_err();

    // base-model tensor dims for the loader's shape validation (:333, :358-371)
    let model_dims: HashMap<String, [i64; 4]> = loaded
        .model
        .tensors
        .iter()
        .map(|(name, &id)| (name.clone(), *loaded.model.ctx.ne(id)))
        .collect();

    let adapter: Rc<AdapterLora> = adapter::load_adapter_lora(
        &mut loaded.model.ctx,
        loaded.model.arch,
        &|name| model_dims.get(name).copied(),
        ADAPTER,
    )
    .expect("load adapter");
    println!(
        "adapter: alpha={} pairs={} n_nodes={}",
        adapter.alpha,
        adapter.ab_map.len(),
        adapter.get_n_nodes()
    );

    let vocab = {
        let path = std::fs::File::open(MODEL).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&path).unwrap() });
        let gguf = Gguf::from_bytes(mmap).unwrap();
        Vocab::load(&gguf).expect("vocab")
    };
    let tokens = vocab.tokenize(PROMPT, true, true);
    let attn = attn_params(&loaded.model, fa);
    let weights = qwen2_weights(&loaded.model);

    // `common_init_from_params` ordering: adapter tensors live in the model
    // Context *before* the DecodeContext takes it over
    let mut dctx = DecodeContext::new_with(
        loaded.model.ctx,
        ForwardWeights::Qwen2(weights),
        attn,
        N_CTX,
        8,
        512,
    );

    // ---- (a) adapter active at scale 1.0 ----
    adapter::set_adapters_lora(&dctx.gctx, &[(adapter.clone(), 1.0)]).expect("set adapter");
    let (ids, rows) = run_greedy(&mut dctx, &tokens);
    println!("port ids (lora):   {ids:?}");
    let reference = if fa { &REF16_FA[..] } else { &REF16_NOFA[..] };
    let ref_top5 = if fa {
        &REF_TOP5_FA[..]
    } else {
        &REF_TOP5_NOFA[..]
    };
    let matched = ids.iter().zip(reference).filter(|(a, b)| a == b).count();
    let first_diff = ids.iter().zip(reference).position(|(a, b)| a != b);
    println!("MATCH: {matched}/{N_PREDICT}  first_diff={first_diff:?}  (fa={fa})");

    // ---- (b) base model, adapter cleared ----
    adapter::clear_adapter_lora();
    dctx.reset_sequence();
    let (base_ids, _) = run_greedy(&mut dctx, &tokens);
    println!("port ids (base):   {base_ids:?}");
    let changed = ids.iter().zip(&base_ids).filter(|(a, b)| a != b).count();
    println!("adapter changes {changed}/{N_PREDICT} tokens vs the base model");
    assert!(
        changed > 0,
        "the adapter had no effect — the lora nodes are not in the graph"
    );

    // ---- (c) scale 0: `set_adapters_lora` drops it (:1342-1344) → must equal
    // the cleared run *bit-for-bit*, not just token-wise ----
    adapter::set_adapters_lora(&dctx.gctx, &[(adapter.clone(), 0.0)]).expect("scale 0");
    dctx.reset_sequence();
    let pos: Vec<i32> = (0..tokens.len() as i32).collect();
    let logits_scale0 = dctx.decode(&tokens, &pos).expect("prefill").to_vec();
    adapter::clear_adapter_lora();
    dctx.reset_sequence();
    let logits_base = dctx.decode(&tokens, &pos).expect("prefill").to_vec();
    assert_eq!(
        logits_scale0, logits_base,
        "scale 0 must reproduce the base model bit-for-bit"
    );
    println!(
        "scale 0 == base prefill logits: bit-for-bit ({} values)",
        logits_base.len()
    );

    // ---- (d) numeric comparison against the reference capture ----
    assert_eq!(
        matched, N_PREDICT,
        "reference mismatch: port {ids:?} vs reference {reference:?} (first_diff {first_diff:?})"
    );
    let (worst, min_overlap) = report_top5(&rows, ref_top5, "top5");
    // The reference's own FA/non-FA runs differ by up to 0.6 logprobs on the
    // tail of these top-5s; the port sits inside that band (PARITY.md).
    assert!(
        worst < 0.6,
        "worst |logprob delta| {worst} outside the numeric band"
    );
    println!(
        "worst |logprob delta| vs reference over {N_PREDICT} steps x top-5: {worst:.3}; \
         min rank-5 overlap {min_overlap}/5"
    );

    // ---- (e) `--lora-scaled …:0.5` — the adapter_scale multiplication path.
    // Captured with `-fa off` only (the FA split moves logprob tails, not ids);
    // contains a 0.035-margin tie at step 5 (7772 over 6722). ----
    if !fa {
        adapter::set_adapters_lora(&dctx.gctx, &[(adapter.clone(), 0.5)]).expect("scale 0.5");
        dctx.reset_sequence();
        let (ids_half, rows_half) = run_greedy(&mut dctx, &tokens);
        let matched_half = ids_half
            .iter()
            .zip(&REF16_HALF)
            .filter(|(a, b)| a == b)
            .count();
        let first_half = ids_half.iter().zip(&REF16_HALF).position(|(a, b)| a != b);
        println!("port ids (scale 0.5): {ids_half:?}");
        println!("MATCH (scale 0.5): {matched_half}/{N_PREDICT}  first_diff={first_half:?}");
        let (worst_half, _) = report_top5(&rows_half, &REF_TOP5_HALF, "half ");
        assert!(
            worst_half < 0.6,
            "scale-0.5 worst |logprob delta| {worst_half}"
        );
        assert_eq!(
            matched_half, N_PREDICT,
            "scale-0.5 mismatch: port {ids_half:?} vs reference {REF16_HALF:?} (first_diff {first_half:?})"
        );
    }
    adapter::clear_adapter_lora();
}
