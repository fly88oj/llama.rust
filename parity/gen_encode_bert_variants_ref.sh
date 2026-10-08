#!/usr/bin/env bash
# gen_encode_bert_variants_ref.sh — regenerate the BERT-variant encoder
# ground-truth dumps (parity/encode_bert_variants_*.bin) via the reference
# llama_encode path (arch batch 16: jina-bert-v2 / jina-bert-v3 / nomic-bert /
# nomic-bert-moe / neo-bert / modern-bert).
#
# Every model file is the *synthetic* GGUF the port's test suite writes to
# /tmp/bert-variants (crates/llama/tests/bert_variants_e2e.rs — the batch-15
# encoder protocol: SPM vocab fixture KV + the arch's own KV + exactly the
# tensors its load_arch_tensors asks for, all F32). The dumps are consumed by
# crates/llama/tests/bert_variants_e2e.rs (bert_variants_reference_parity,
# #[ignore]d): embeddings must be bit-exact (max |Δ| == 0.0).
#
#   parity/encode_bert_variants_jina-bert-v2.bin        --pool none (the folded-in GEGLU gate)
#   parity/encode_bert_variants_jina-bert-v2-gated.bin  --pool none (the separate ffn_gate tensor)
#   parity/encode_bert_variants_jina-bert-v3.bin        --pool none (rope'd, GELU-SEQ FFN)
#   parity/encode_bert_variants_nomic-bert.bin          --pool none (rope'd, SwiGLU-PAR FFN)
#   parity/encode_bert_variants_nomic-bert-moe.bin      --pool none (rope'd, MoE every 2nd layer)
#   parity/encode_bert_variants_neo-bert.bin            --pool none (RMS + fused QKV + SWIGLU-SEQ)
#   parity/encode_bert_variants_modern-bert.bin         --pool none (symmetric SWA + GEGLU)
#   parity/encode_bert_variants_modern-bert-mean.bin    --pool mean (the SWA encoder's mean path)
#   parity/encode_bert_variants_modern-bert-rank.bin    --pool rank (the GTE reranker head)
#   parity/encode_bert_variants_modern-bert-silu.bin    --pool none (hidden_activation = silu)
#
# FA: the dumps anchor --fa off — the port's encoder baseline is the non-FA
# branch of build_attn_mha (the bert_e2e.rs precedent; no kq_b in this family
# disables FA on the reference side). --fa on exists to measure the gap on the
# reference side itself.
#
# Usage: bash parity/gen_encode_bert_variants_ref.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin

DUMP="$ROOT/parity/ref_encode_dump"

g++ -O2 -std=c++17 -o "$DUMP" "$ROOT/parity/ref_encode_dump.cpp" \
    -I"$PIN/include" -I"$PIN/ggml/include" \
    -L"$REF" -lllama -Wl,-rpath,"$REF"

# keep the log noise out of the sidecars
export LLAMA_LOG_LEVEL=error

# 1) write the synthetic files (the port's byte-exact GGUF writer)
cargo test -p llama --release --test bert_variants_e2e bert_variants_write_synth -- --ignored --nocapture

IDS="1,2,3,4,5,6,7,8,9,10,11,12"

cell() { # <gguf> <out> <pool>
    echo "== $2 (pool $3) =="
    "$DUMP" "/tmp/bert-variants/$1" "$ROOT/parity/encode_bert_variants_$2.bin" \
        --ids "$IDS" --pool "$3" --fa off --threads 8
}

cell jina-bert-v2-synth.gguf     jina-bert-v2        none
cell jina-bert-v2-synth-gated.gguf jina-bert-v2-gated none
cell jina-bert-v3-synth.gguf     jina-bert-v3        none
cell nomic-bert-synth.gguf       nomic-bert          none
cell nomic-bert-moe-synth.gguf   nomic-bert-moe      none
cell neo-bert-synth.gguf         neo-bert            none
cell modern-bert-synth.gguf      modern-bert         none
cell modern-bert-synth.gguf      modern-bert-mean    mean
cell modern-bert-synth-rank.gguf modern-bert-rank    rank
cell modern-bert-synth-silu.gguf modern-bert-silu    none

echo
echo "md5:"
md5sum "$ROOT"/parity/encode_bert_variants_*.bin
