#!/usr/bin/env bash
# gen_encode_ref.sh — regenerate the BERT / T5-encoder ground-truth dumps
# (parity/encode_*.bin{,.txt}) via the reference llama_encode path.
#
#   parity/encode_bert_pertoken.bin    bge-m3   --pool none    (T=14, last hidden state)
#   parity/encode_bert_pooled.bin      bge-m3   --pool default (T=14, file pooling_type = 2 = CLS)
#   parity/encode_bert_mean.bin        bge-m3   --pool mean    (T=14, build_inp_mean matmul path)
#   parity/encode_bert_single*.bin     bge-m3   --ids 0        (T=1: exactness anchor — at T=1
#                                                              no GEMM reaches llamafile
#                                                              tinyBLAS and every dot is a
#                                                              multiple of 64, so the port can
#                                                              match to 1-2 ulp)
#   parity/encode_bert_t2.bin          bge-m3   --ids 0 581    (T=2: seed/amplification curve)
#   parity/encode_bert_t64.bin         bge-m3   --ids 0..63    (T=64: all dots are 64-multiples,
#                                                              but the F32 attention GEMM now
#                                                              has m = 64 -> tinyBLAS)
#   parity/encode_t5enc_pertoken.bin   t5enc    --pool none    (T=17)
#   parity/encode_t5enc_mean.bin       t5enc    --pool mean
#   parity/encode_t5enc_single.bin     t5enc    --ids 0        (T=1 anchor)
#
# The .bin files are consumed by crates/llama/tests/bert_e2e.rs /
# t5_e2e.rs (the ids are inside the artifact, so tokenization is not part of
# the comparison). The .txt sidecars are informational.
#
# Why the T=1 anchors matter: see tests/bert_e2e.rs's header and PARITY.md.
# The reference routes F32 GEMMs to llamafile tinyBLAS whenever
# m % 4 == 0 && k % 16 == 0 and Q8_0 GEMMs (whose activation type is also
# Q8_0) unconditionally — `parity/ref_mulmat_q8_dump.c` pins both: Q8_0 is
# bit-identical to the port, tinyBLAS-F32 is not (nor is the port's f32
# vec_dot for n % 64 != 0). Those 1e-4-level seeds are amplified by the Q8_0
# activation quantization into a ~1% deviation of the final hidden state for
# T >= 2; at T=1 every GEMM takes the same kernel on both sides.
#
# The reference build has no embedding example binary at this revision
# (`ls /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin`) — libllama.so is the
# ground truth, so the dumper links against it directly.
#
# Usage: bash parity/gen_encode_ref.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin

BERT_MODEL=/home/jeffrey/localai/models/bge-m3-Q8_0.gguf
T5_MODEL=/home/jeffrey/localai/models/t5-v1_1-xxl-encoder-Q5_K_S.gguf

BERT_TEXT="The quick brown fox jumps over the lazy dog"
T5_TEXT="translate English to German: The quick brown fox jumps over the lazy dog"

DUMP="$ROOT/parity/ref_encode_dump"

g++ -O2 -std=c++17 -o "$DUMP" "$ROOT/parity/ref_encode_dump.cpp" \
    -I"$PIN/include" -I"$PIN/ggml/include" \
    -L"$REF" -lllama -Wl,-rpath,"$REF"

# keep the log noise out of the sidecars
export LLAMA_LOG_LEVEL=error

echo "== bert (bge-m3-Q8_0) =="
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_pertoken.bin" --pool none    --text "$BERT_TEXT" --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_pooled.bin"   --pool default --text "$BERT_TEXT" --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_mean.bin"     --pool mean    --text "$BERT_TEXT" --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_single.bin"      --pool none --ids 0 --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_single_cls.bin"  --pool cls  --ids 0 --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_single_mean.bin" --pool mean --ids 0 --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_t2.bin"   --pool none --ids "0 581" --threads 8
"$DUMP" "$BERT_MODEL" "$ROOT/parity/encode_bert_t64.bin"  --pool none --ids "$(seq -s' ' 0 63)" --threads 8

echo "== t5 encoder (t5-v1_1-xxl-encoder-Q5_K_S) =="
"$DUMP" "$T5_MODEL" "$ROOT/parity/encode_t5enc_pertoken.bin" --pool none --text "$T5_TEXT" --threads 8
"$DUMP" "$T5_MODEL" "$ROOT/parity/encode_t5enc_mean.bin"     --pool mean --text "$T5_TEXT" --threads 8
"$DUMP" "$T5_MODEL" "$ROOT/parity/encode_t5enc_single.bin"   --pool none --ids 0 --threads 8

# --- kernel-level ground truth for the encoder GEMMs (tests/encode_gemm_probe.rs,
#     see PARITY.md "根因"): Q8_0 GEMMs through the reference's production
#     mul_mat in the encoder shapes, F32 attention GEMMs in the tinyBLAS band.
echo "== encoder GEMM kernel probe =="
GEMM_DUMP="$ROOT/parity/ref_mulmat_q8_dump"
gcc -O2 -std=c11 -o "$GEMM_DUMP" "$ROOT/parity/ref_mulmat_q8_dump.c" \
    -I"$PIN/ggml/include" -L"$REF" -lggml -lggml-cpu -lggml-base -lm -Wl,-rpath,"$REF"
"$GEMM_DUMP" "$ROOT/parity/mulmat_q8_bert_ref.bin"

# --- t5 relative-position buckets, verbatim C copy (llama-graph.cpp:3890-3923)
echo "== t5 relative position buckets =="
BUCKET="$ROOT/parity/ref_t5_bucket"
gcc -O2 -o "$BUCKET" "$ROOT/parity/ref_t5_bucket.c" -lm
"$BUCKET" > "$ROOT/parity/t5_bucket_ref.txt"

echo
echo "md5:"
md5sum "$ROOT"/parity/encode_*.bin "$ROOT"/parity/mulmat_q8_bert_ref.bin "$ROOT"/parity/t5_bucket_ref.txt