# llama.rust

A from-scratch Rust implementation of [llama.cpp](https://github.com/ggml-org/llama.cpp)
targeting CPU inference, developed as a line-by-line port with a
bit-exactness discipline: wherever the C implementation defines an exact
numeric result, the port reproduces it bit-for-bit, verified against
reference dumps produced by instrumented builds of the original code.

## What is implemented

- **ggml crate** — tensor library: compute graph, CPU executor, the full set
  of quantization formats (K-quants, IQ family, MXFP4/NVFP4), AVX2/AVX512
  SIMD kernels (quantization, vec_dot, GEMM, flash attention, tinyBLAS,
  tiled matmul, repack), recurrent kernels (WKV/GLA/GDN), GGUF reader and
  writer, and a backend abstraction layer that can drive foreign
  (dlopen-able) ggml backends such as Vulkan through a graph
  re-emission bridge.
- **llama crate** — model loading and forward graphs for the llama.cpp model
  zoo (Llama/Qwen/Gemma/Phi/Mistral/DeepSeek MLA/DSA/MoE families, linear
  attention, mamba hybrids, and more), KV cache variants (including SWA and
  compressed DSV4), tokenizers (SPM/BPE/byte-level/UMM/MMLU-style),
  grammar-constrained (GBNF) and JSON-schema-constrained sampling, chat
  templates with a mini-jinja engine and tool-call protocols, LoRA adapters,
  importance-matrix quantization, speculative decoding (draft/eagle3/ngram/
  MTP), multimodal vision (CLIP) and audio (mtmd) input, and state
  save/restore.
- **tools** — `llama-cli`, `llama-server` (OpenAI-compatible endpoints,
  tool calls, MCP, slot state), `llama-quantize`, `llama-imatrix`,
  `llama-tokenize`, `llama-perplexity`, `llama-gguf`, `llama-gguf-split`,
  `llama-bench`, `llama-mtmd-cli`.
- **parity/** — the verification harness: reference dumps, C probe programs
  that instrument the original implementation, and comparison scripts used
  to validate the port.

## Build and test

```sh
cargo build --release --workspace
cargo test --workspace
```

Tests that require locally installed reference builds or local model files
skip themselves when those artifacts are absent; the rest of the suite is
self-contained.

## Local corpus paths

Tests and parity harnesses read models and the reference llama.cpp build from

    ${LLAMA_RUST_DATA_HOME}/<layout under the developer home>

which defaults to `$HOME`.  Point it wherever the corpus lives, e.g.
`export LLAMA_RUST_DATA_HOME=/data/corpus`.  Anything not found is skipped, not failed:
the committed fixtures under `parity/` carry the bit-exact expectations that need no
local models.

## Numerical parity notes

The parity methodology, the per-file upstream mapping and the feature coverage were
kept in working ledgers; those records now live in the commit history
(`git log --grep parity`). Where upstream C compiles
floating-point contractions that change results, or where the reference
binary diverges from its own sources, the discrepancies are recorded there
rather than papered over.

## License

MIT. The tokenizer test fixtures under
`crates/llama/tests/tokenizer_fixtures/vocab/` originate from the
[llama.cpp](https://github.com/ggml-org/llama.cpp) model test corpus (MIT).
