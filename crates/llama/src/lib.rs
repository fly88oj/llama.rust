//! llama — port of llama.cpp `src/llama-*.cpp` (model loading, architectures,
//! tokenizers, sampling, KV cache, contexts). See FILE_MAP.md at repo root.

pub mod adapter;
pub mod arch;
pub mod batch;
pub mod clef;
pub mod clip;
pub mod console;
pub mod context;
pub mod dflash;
pub mod display;
pub mod eagle;
pub mod grammar;
pub mod graph;
pub mod gemma4_assistant;
pub mod graph_arch;
pub mod hparams;
pub mod imatrix;
pub mod impl_log;
pub mod json_schema;
pub mod kv_cache;
pub mod meta;
pub mod mlock;
pub mod model;
pub mod mtmd;
pub mod mtmd_audio;
pub mod ngram_cache;
pub mod ngram_map;
pub mod ngram_mod;
pub mod quant;
pub mod regex_lite;
pub mod sampling;
pub mod saver;
pub mod speculative;
pub mod unicode;
pub mod unicode_data;
pub mod vocab;

pub fn version() -> &'static str {
    "0.1.0 (target: llama.cpp def4d406a)"
}
pub mod chat;
pub mod chat_parsers;
pub mod chat_tools;
pub mod peg;

/// `llama_time_us` (llama.cpp:153-155 -> `ggml_time_us`): microseconds since the
/// library's clock started. ggml starts its clock at `ggml_time_init` (first
/// call); the port's lazy static starts at first use — both are monotonic
/// microsecond counters used only for elapsed-time reporting.
pub fn time_us() -> i64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_micros() as i64
}
