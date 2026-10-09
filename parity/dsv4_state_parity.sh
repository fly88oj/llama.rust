#!/usr/bin/env bash
# dsv4_state_parity.sh — the DSV4 sequence-state byte-format cell (PARITY.md's
# dsv4-state section): the port's `DecodeContext::state_seq_get_data` blob
# must equal the reference's `llama_state_seq_get_data` byte-for-byte on the
# same synthetic deepseek4 model (the batch-7 generator, tests/
# dsv4_state_e2e.rs), after the same fixed 16-token prefill — and again with
# an 8-step tail that rolls the compressor rings past block boundaries.
#
# The context geometry is pinned on both sides: n_ctx 512, n_seq_max 1 (one
# compressed stream), n_ubatch 512, flash attention ON (the dsv4 raw pair is
# built with `v_trans = !flash_attn`, llama-model.cpp:2480 — only the
# v_trans = 0 layout matches the port's always-!v_trans caches).
#
# KNOWN BLOCKER (sync batch to c35b66744): the reference library SIGFPEs
# while creating the dsv4 raw pair on this model — 210791069's n_rot_k loop
# divides by 2*n_rot_k, the DeepSeek-indexer clause (llama-kv-cache.cpp:321-
# 332) sets attn_rot_k without re-checking n_embd_head_k_all > 0, and the
# all-SWA base half keeps n_embd_head_k_all = 0, so `while (0 % (2*n_rot_k)
# == 0)` doubles n_rot_k until 2*n_rot_k wraps the u32 to 0 (division by
# zero; gdb: llama_kv_cache::llama_kv_cache <- llama_kv_cache_iswa <-
# llama_kv_cache_dsv4). The probe dies before any state is written.
# The port writes the guarded values (base half n_rot 0/0 with its 0 layers,
# swa half 64/0) — structurally pinned by tests/dsv4_state_e2e.rs
# (dsv4_state_format_self_consistency) and by hand-decoding the blob:
#   base: cells=16 v_trans=0 n_layer=0 n_rot_k=0 n_rot_v=0
#   swa:  cells=16 v_trans=0 n_layer=4 n_rot_k=64 n_rot_v=0
# Byte-parity resumes once upstream guards the clause (or the tree moves).
#
# usage: bash parity/dsv4_state_parity.sh   (regenerates the model + blobs)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIN=/home/jeffrey/llm/llama.cpp-pinned
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
OUT=/tmp/arch-dsv4state
mkdir -p $OUT

fail=0

# 1. the port side: the synthetic model + the two blobs
cargo test --release -p llama --test dsv4_state_e2e dsv4_state_dump_blob -- --ignored \
    > $OUT/port-dump.log 2>&1 || { echo "PORT DUMP FAILED"; tail -5 $OUT/port-dump.log; exit 1; }

# 2. the reference probe (parity/ref_dsv4_state.c)
PROBE=$OUT/ref_dsv4_state
# NOTE (sync batch to c35b66744): the probe must compile against the NEW
# tree's headers — the pinned headers (a7b94df2c) ABI-mismatch the NEW
# library (llama_context_params gained moe_cache_size, llama.h @c35b66744),
# which crashes the probe mid-load
gcc -O2 "$ROOT/parity/ref_dsv4_state.c" -o "$PROBE" \
    -I /home/jeffrey/llm/llama.cpp-next/include -I /home/jeffrey/llm/llama.cpp-next/ggml/include \
    -L "$REF" -lllama -Wl,-rpath,"$REF" || { echo "PROBE BUILD FAILED"; exit 1; }

for tail in 0 8; do
    LLAMA_LOG_LEVEL=error "$PROBE" "$OUT/deepseek4-synth-state.gguf" "$OUT/ref-state$([ $tail -eq 0 ] && echo '' || echo -n -tail).bin" $tail \
        > $OUT/probe-$tail.log 2>&1 || { echo "PROBE RUN FAILED (tail $tail)"; tail -3 $OUT/probe-$tail.log; fail=1; continue; }
done

# 3. byte comparison
for pair in "port-state.bin ref-state.bin prefill-only" \
            "port-state-tail.bin ref-state-tail.bin with-8-step-tail"; do
    set -- $pair
    if cmp -s "$OUT/$1" "$OUT/$2"; then
        echo "  $3: port == reference, byte-identical ($(stat -c%s "$OUT/$1") bytes) PASS"
    else
        echo "  $3: BYTE MISMATCH (port $(stat -c%s "$OUT/$1") vs ref $(stat -c%s "$OUT/$2")) FAIL"
        cmp "$OUT/$1" "$OUT/$2" | head -2
        fail=1
    fi
done

if [ $fail -eq 0 ]; then
    echo "DSV4 STATE BYTE FORMAT PARITY PASS"
else
    echo "DSV4 STATE PARITY FAILURES"
fi
exit $fail
