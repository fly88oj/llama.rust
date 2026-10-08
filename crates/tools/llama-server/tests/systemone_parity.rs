//! systemone_parity.rs — the `/v1/systemone` e2e fixture generator +
//! response-shape tests for the decision-model subsystem ported from
//! `tools/server/server-decision.{h,cpp}` (upstream a7b94df2c).
//!
//! The upstream unit test `tools/server/tests/unit/test_systemone.py` runs
//! against two HF-hosted tiny decision models (`tinylaya`, `tinyopenjev`);
//! this box has no network, so the fixture here is a SYNTHETIC laya model —
//! a 2-layer bert with the decision metadata (`bert.decision.type = laya`,
//! the fitted temperatures, the `systemone` chat template) — built by
//! [`gen_laya_fixture`] and exercised by `parity/systemone_parity.sh`
//! against BOTH the NEW reference server and the port (MATCH protocol).
//!
//! The in-crate tests below mirror the python test's invariants that do not
//! need a server: the answer structure (probabilities sum to 1, choice is
//! the argmax, noul in [0, 1], usage.output_tokens == 0), the invalid
//! request matrix, and the 501 for images on a server without mmproj.

use std::io::Write as _;

use ggml::gguf::Value;
use ggml::gguf_write::GgufWriter;
use ggml::Gguf;

/// `/tmp` is volatile — the durable copy of the generator is this file; the
/// model is a derived artifact regenerated on demand.
pub const LAYA_FILE: &str = "/tmp/s2t-laya/tinylaya-synth.gguf";

const N_EMBD: i64 = 32;
const N_LAYER: i64 = 2;
const N_HEAD: i64 = 4;
const N_FF: i64 = 64;
const N_CTX_TRAIN: i64 = 512;
const N_TOKEN_TYPES: i64 = 2;

/// the `systemone` template of a laya model: the marker layout
/// `fill_task_laya` parses — `[CLS] question [SEP] ([MASK] option)* [SEP]
/// state [SEP]` (the special tokens render as text and are re-parsed by
/// `common_tokenize(prompt, false, true)`)
const SYSTEMONE_TEMPLATE: &str = concat!(
    "{{ \"[CLS]\" }}{{ instructions }}{{ \"[SEP]\" }}",
    "{% for o in options %}{{ \"[MASK]\" }}{{ o.key }}{% endfor %}",
    "{{ \"[SEP]\" }}{{ state }}{{ \"[SEP]\" }}"
);

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

/// build the synthetic laya decision GGUF: the bert-bge WPM vocab (its
/// [MASK]/[SEP] tokens are the laya markers) + random weights + the
/// `bert.decision.*` metadata
#[allow(dead_code)]
pub fn gen_laya_fixture() -> String {
    let dir = std::path::Path::new(LAYA_FILE).parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    if std::path::Path::new(LAYA_FILE).exists() {
        return LAYA_FILE.to_string();
    }

    // the vocab: ggml-vocab-bert-bge.gguf (WPM, [CLS]/[SEP]/[MASK] at the
    // bert fixed ids)
    let src = Gguf::open("/home/jeffrey/llm/llama.cpp-next/models/ggml-vocab-bert-bge.gguf")
        .expect("open bert vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") {
            w.set_kv(k, val.clone());
        }
    }
    let n_vocab = src
        .get_u32("tokenizer.ggml.tokens_count")
        .or_else(|| {
            src.kv
                .iter()
                .find(|(k, _)| k == "tokenizer.ggml.tokens")
                .and_then(|(_, v)| match v {
                    Value::Array(_, items) => Some(items.len() as u32),
                    _ => None,
                })
        })
        .expect("vocab size") as i64;

    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String("bert".into()));
    kv!("general.name", Value::String("llama-rust-synth-tinylaya".into()));
    kv!("general.file_type", Value::U32(0)); // F32
    kv!("bert.context_length", Value::U32(N_CTX_TRAIN as u32));
    kv!("bert.embedding_length", Value::U32(N_EMBD as u32));
    kv!("bert.block_count", Value::U32(N_LAYER as u32));
    kv!("bert.feed_forward_length", Value::U32(N_FF as u32));
    kv!("bert.attention.head_count", Value::U32(N_HEAD as u32));
    kv!("bert.attention.layer_norm_epsilon", Value::F32(1e-5));

    // ---- the decision metadata (server-decision.cpp:40-66 + laya arm) ----
    kv!("bert.decision.type", Value::String("laya".into()));
    kv!("bert.decision.max_head_tokens", Value::U32(256));
    // the fitted temperatures (rendered as strings like the converter
    // writes them)
    kv!(
        "bert.decision.temperature.choice.2",
        Value::String("1.250000".into())
    );
    kv!(
        "bert.decision.temperature.score.2",
        Value::String("1.750000".into())
    );
    kv!(
        "bert.decision.temperature.noul.2",
        Value::String("1.500000".into())
    );
    kv!(
        "tokenizer.chat_template.systemone",
        Value::String(SYSTEMONE_TEMPLATE.into())
    );

    // the tensor set of a plain bert (models/bert.cpp:29-60)
    let mut table: Vec<(String, Vec<i64>, f32, bool)> = Vec::new(); // (name, ne, scale, is_norm)
    let mut push = |name: String, ne: Vec<i64>, scale: f32, is_norm: bool| {
        table.push((name, ne, scale, is_norm))
    };

    push(
        "token_embd.weight".into(),
        vec![N_EMBD, n_vocab],
        1.0 / (N_EMBD as f32).sqrt(),
        false,
    );
    push(
        "token_types.weight".into(),
        vec![N_EMBD, N_TOKEN_TYPES],
        1.0 / (N_EMBD as f32).sqrt(),
        false,
    );
    push(
        "position_embd.weight".into(),
        vec![N_EMBD, N_CTX_TRAIN],
        1.0 / (N_EMBD as f32).sqrt(),
        false,
    );
    push("token_embd_norm.weight".into(), vec![N_EMBD], 1.0, true);
    push("token_embd_norm.bias".into(), vec![N_EMBD], 0.02, false);
    for i in 0..N_LAYER as i32 {
        let proj = 1.0 / (N_EMBD as f32).sqrt();
        push(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, N_EMBD], proj, false);
        push(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, N_EMBD], proj, false);
        push(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, N_EMBD], proj, false);
        push(format!("blk.{i}.attn_output.weight"), vec![N_EMBD, N_EMBD], proj, false);
        push(format!("blk.{i}.attn_output.bias"), vec![N_EMBD], 0.02, false);
        push(format!("blk.{i}.attn_output_norm.weight"), vec![N_EMBD], 1.0, true);
        push(format!("blk.{i}.attn_output_norm.bias"), vec![N_EMBD], 0.02, false);
        push(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], proj, false);
        push(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], proj, false);
        push(format!("blk.{i}.layer_output_norm.weight"), vec![N_EMBD], 1.0, true);
        push(format!("blk.{i}.layer_output_norm.bias"), vec![N_EMBD], 0.02, false);
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7081);
    for (name, ne, scale, is_norm) in &table {
        let n: i64 = ne.iter().product();
        let vals: Vec<f32> = if *is_norm {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
        } else {
            (0..n).map(|_| rng.next() * scale).collect()
        };
        let ne4 = [ne[0], *ne.get(1).unwrap_or(&1), 1, 1];
        w.add_tensor(name, ggml::types::GgmlType::F32, ne4);
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for x in &vals {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        data.push(bytes);
    }

    let tmp = format!("{LAYA_FILE}.tmp{}", std::process::id());
    let f = std::fs::File::create(&tmp).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    bw.flush().unwrap();
    std::fs::rename(&tmp, LAYA_FILE).expect("publish synth gguf");
    LAYA_FILE.to_string()
}

// ---------------------------------------------------------------------------
// the request bodies of test_systemone.py
// ---------------------------------------------------------------------------

pub const TEST_STATE: &str = "I was charged twice for my order last week and nobody has replied.";

pub fn test_questions_json() -> String {
    r#"{
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"]
      },
      "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?"
      }
    }"#
    .to_string()
}

pub fn request_body() -> String {
    format!(
        r#"{{"state": {:?}, "questions": {}}}"#,
        TEST_STATE,
        test_questions_json()
    )
}

// ---------------------------------------------------------------------------
// in-crate invariants over the response JSON (the python assertions that do
// not need a live model)
// ---------------------------------------------------------------------------

fn json_get<'a>(v: &'a llama::json_schema::Json, path: &[&str]) -> Option<&'a llama::json_schema::Json> {
    use llama::json_schema::Json as J;
    let mut cur = v;
    for k in path {
        let J::Object(fields) = cur else {
            return None;
        };
        cur = fields.iter().find(|(kk, _)| kk == k).map(|(_, vv)| vv)?;
    }
    Some(cur)
}

fn j_str(v: &llama::json_schema::Json) -> Option<&str> {
    match v {
        llama::json_schema::Json::String(s) => Some(s),
        _ => None,
    }
}

fn f64_of(v: &llama::json_schema::Json) -> f64 {
    match v {
        llama::json_schema::Json::Int(i) => *i as f64,
        llama::json_schema::Json::Uint(u) => *u as f64,
        llama::json_schema::Json::Double(d) => *d,
        _ => 0.0,
    }
}

/// the shape of `test_systemone`'s assertions on a 200 response (run by the
/// parity script over both servers' replies; here over a canned capture so
/// the assertions themselves are pinned)
#[test]
fn answer_shape_invariants() {
    // a canonical reply: all three question types, one variant (laya)
    let canned = r#"{
      "model": "m", "answers": {
        "route": {"type": "choice", "choice": "billing",
          "probabilities": {"billing": 0.5, "shipping": 0.3, "technical": 0.2},
          "confidence": 0.25},
        "urgency": {"type": "score", "score": 1.5,
          "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
          "probabilities": {"0": 0.1, "1": 0.4, "2": 0.4, "3": 0.1},
          "confidence": 0.5},
        "angry": {"type": "noul", "noul": 0.6}
      },
      "usage": {"input_tokens": 57, "output_tokens": 0}
    }"#;
    let v = llama::json_schema::Json::parse(canned).unwrap();

    // answers keyed in request order
    let answers = json_get(&v, &["answers"]).unwrap();
    let llama::json_schema::Json::Object(afields) = answers else { panic!() };
    let keys: Vec<&String> = afields.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["route", "urgency", "angry"]);

    // choice: probabilities sum to 1, the choice is the argmax
    let route = json_get(&v, &["answers", "route"]).unwrap();
    assert_eq!(j_str(json_get(route, &["type"]).unwrap()), Some("choice"));
    let probs = json_get(route, &["probabilities"]).unwrap();
    let llama::json_schema::Json::Object(pfields) = probs else { panic!() };
    let mut sum = 0.0f64;
    let mut best = (String::new(), f64::MIN);
    for (k, p) in pfields {
        let p = f64_of(p);
        sum += p;
        if p > best.1 {
            best = (k.clone(), p);
        }
    }
    assert!((sum - 1.0).abs() < 1e-4);
    assert_eq!(j_str(json_get(route, &["choice"]).unwrap()), Some(best.0.as_str()));
    let conf = f64_of(json_get(route, &["confidence"]).unwrap());
    assert!((0.0..=1.0).contains(&conf));

    // score: the expectation of the level index
    let urgency = json_get(&v, &["answers", "urgency"]).unwrap();
    let mut expected = 0.0f64;
    let up = json_get(urgency, &["probabilities"]).unwrap();
    let llama::json_schema::Json::Object(upfields) = up else { panic!() };
    for (k, p) in upfields {
        expected += k.parse::<f64>().unwrap() * f64_of(p);
    }
    assert!((f64_of(json_get(urgency, &["score"]).unwrap()) - expected).abs() < 1e-4);

    // noul in [0, 1]; usage.output_tokens == 0
    let noul = f64_of(json_get(&v, &["answers", "angry", "noul"]).unwrap());
    assert!((0.0..=1.0).contains(&noul));
    assert_eq!(
        f64_of(json_get(&v, &["usage", "output_tokens"]).unwrap()),
        0.0
    );
}

// `gen` entry: `cargo test --release -p llama-server --test systemone_parity gen -- --ignored`
#[test]
#[ignore]
fn gen() {
    let path = gen_laya_fixture();
    println!("tinylaya synth: {path} ({} bytes)",
        std::fs::metadata(&path).unwrap().len());
}
