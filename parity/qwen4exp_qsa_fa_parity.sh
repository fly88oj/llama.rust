#!/usr/bin/env bash
# qwen4exp_qsa_fa_parity.sh — batch 20: the QSA **flash-attention arm**
# regeneration + compare (the F16 kq_mask route llama-graph.cpp:38-39 /
# :2633-2669 builds when FA is on — batch 19 verified the non-FA path only).
#
# Produces the four stream fixtures the default acceptance tests consume:
#   parity/qwen4exp_qsa_nodes_ref.bin     (--fa off, batch 19's baseline)
#   parity/qwen4exp_qsa_nodes_fa_ref.bin  (--fa on,  batch 20)
#   parity/qwen4exp_qsa_nodes_port.bin    (port, fa off)
#   parity/qwen4exp_qsa_nodes_fa_port.bin (port, fa on)
#
# usage: bash parity/qwen4exp_qsa_fa_parity.sh [ref|port|cmp]
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"
REFBIN=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
PINNED=/home/jeffrey/llm/llama.cpp-pinned
WORK=/tmp/qfa-noembd
SYNTH=/tmp/hq-q4e/qwen4exp-qsa-ple.gguf

mkdir -p "$WORK"

# 1. the synthetic qwen4exp file (compress_ratios [0,0,4,0] + layer-1 PLE)
if [ ! -f "$SYNTH" ]; then
    ./parity/limited.sh -- cargo test --release -p llama --test qwen4exp_qsa_dump \
        qwen4exp_qsa_write_synth_file -- --ignored --nocapture
fi

# 2. the no-embeddings reference probe over the NEW reference (the
#    embeddings output extraction aborts on the QSA graph — PARITY 批次 19;
#    NOTE the probe needs the explicit --no-embd flag, without it
#    cparams.embeddings stays true and the same abort fires)
if [ "$MODE" = ref ] || [ "$MODE" = all ]; then
    gcc -O2 -I"$PINNED/include" -I"$PINNED/ggml/include" \
        parity/ref_decode_dump_noembd.c -o "$WORK/ref_decode_dump_noembd" \
        -L"$REFBIN" -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,"$REFBIN"
    "$WORK/ref_decode_dump_noembd" "$SYNTH" parity/qwen4exp_qsa_nodes_ref.bin \
        --text hi --no-embd --fa off --decode-tail 12
    "$WORK/ref_decode_dump_noembd" "$SYNTH" parity/qwen4exp_qsa_nodes_fa_ref.bin \
        --text hi --no-embd --fa on --decode-tail 12
fi

# 3. the port's two streams (Q4E_FA selects the FA arm)
if [ "$MODE" = port ] || [ "$MODE" = all ]; then
    Q4E_DUMP_OUT="$PWD/parity/qwen4exp_qsa_nodes_port.bin" \
        ./parity/limited.sh -- cargo test --release -p llama --test qwen4exp_qsa_dump \
        qwen4exp_qsa_prefill_node_dump -- --ignored --nocapture
    Q4E_FA=1 Q4E_DUMP_OUT="$PWD/parity/qwen4exp_qsa_nodes_fa_port.bin" \
        ./parity/limited.sh -- cargo test --release -p llama --test qwen4exp_qsa_dump \
        qwen4exp_qsa_prefill_node_dump -- --ignored --nocapture
fi

# 4. the named-node bit compare (both arms; the python matcher prints the
#    structural diff noise, the default Rust tests are the hard gate — the
#    head pipe closes the stream early, || true keeps the script going)
if [ "$MODE" = cmp ] || [ "$MODE" = all ]; then
    python3 parity/decode_dump_cmp.py \
        parity/qwen4exp_qsa_nodes_ref.bin parity/qwen4exp_qsa_nodes_port.bin \
        --max-report 3 2>/dev/null | head -4 || true
    python3 parity/decode_dump_cmp.py \
        parity/qwen4exp_qsa_nodes_fa_ref.bin parity/qwen4exp_qsa_nodes_fa_port.bin \
        --max-report 3 2>/dev/null | head -4 || true
fi
