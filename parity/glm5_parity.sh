#!/usr/bin/env bash
# glm5_parity.sh — the NEW glm5-next arch (def4d406a) loader-side acceptance
# vs the NEW reference build:
#   1. write the synthetic glm5-next GGUF with the port's writer
#      (tests/glm5_e2e.rs glm5_write_synth_file),
#   2. load it with the NEW reference library (default-params load via
#      parity/ref_model_saver.c against /home/jeffrey/llm/llama.cpp-next)
#      — exercising the reference's load_arch_hparams/load_arch_tensors on
#      the same file the port must accept,
#   3. byte-compare the print_info banner payload lines port vs reference.
#
# Batch 19 (graph side landed): steps 5-7 regenerate the node-stream
# acceptance — the reference probe (ref_decode_dump.c, --fa off
# --decode-tail 12) vs the port's in-process dump driver
# (tests/glm5_dump.rs), byte-compared through parity/decode_dump_cmp.py.
# The stored artifacts are parity/glm5/nodes_ref.bin + nodes_port.bin; the
# DEFAULT test tests/glm5_dump.rs::glm5_graph_nodes_bit_exact_vs_reference
# re-verifies the port side on every suite run.
#
# usage: bash parity/glm5_parity.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=/tmp/syncm-glm5
mkdir -p "$OUT"
SYNTH="$OUT/glm5-next-synth.gguf"
NEXT=/home/jeffrey/llm/llama.cpp-next
REF="$NEXT/build-rust-ref/bin"

echo "== 1) port writer"
bash parity/limited.sh -- cargo test --release -p llama --test glm5_e2e glm5_write_synth_file -- --ignored --nocapture || exit 1
[ -f "$SYNTH" ] || { echo "missing $SYNTH" >&2; exit 1; }

echo "== 2) NEW reference load (default-params, ref_model_saver probe)"
gcc -O2 parity/ref_model_saver.c -o "$OUT/ref_model_saver" \
    -I"$NEXT/include" -I"$NEXT/ggml/include" -L"$REF" -lllama -lggml -lggml-base || exit 1
if ! bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_model_saver" "$SYNTH" "$OUT/ref_saved.gguf" \
        2> "$OUT/ref_load.err"; then
    echo "REFERENCE LOAD FAILED:" >&2
    tail -5 "$OUT/ref_load.err" >&2
    exit 1
fi
grep -E "^print_info:" "$OUT/ref_load.err" | sed 's/^print_info: //' > "$OUT/banner_ref.txt"
echo "   reference banner: $(wc -l < "$OUT/banner_ref.txt") lines"
rm -f "$OUT/ref_saved.gguf"

echo "== 3) port load banner"
# the graph abort after the banner is expected (the builder is not wired);
# capture stderr through the abort
bash parity/limited.sh -- ./target/release/llama-cli -m "$SYNTH" -p hi -n 1 -t 4 -c 256 --temp 0 \
    > /dev/null 2> "$OUT/port_run.err" || true
grep -E "^print_info:" "$OUT/port_run.err" | sed 's/^print_info: //' > "$OUT/banner_port.txt"
echo "   port banner: $(wc -l < "$OUT/banner_port.txt") lines"

echo "== 4) diff"
diff -u "$OUT/banner_ref.txt" "$OUT/banner_port.txt" > "$OUT/banner_diff.txt" || true
n=$(grep -cE '^[+-][^+-]' "$OUT/banner_diff.txt" || true)
if [ "$n" -eq 0 ]; then
    echo "glm5-next loader banner: identical ($(wc -l < "$OUT/banner_ref.txt") lines)"
else
    echo "glm5-next loader banner: $n DIFFERING lines:" >&2
    cat "$OUT/banner_diff.txt" >&2
    exit 1
fi

# keep the synth + banners (durable record of the run); /tmp is volatile, the
# AUDIT row cites the counts
echo "ok — artifacts in $OUT"

# ---- 5) the graph-side node-stream acceptance (batch 19) ----
if [ "${GLM5_GRAPH:-1}" = "1" ]; then
    echo "== 5) node-stream acceptance (graph side)"
    gcc -O2 parity/ref_decode_dump.c -o "$OUT/ref_decode_dump" \
        -I"$NEXT/include" -I"$NEXT/ggml/include" \
        -L"$REF" -lllama -lggml -lggml-base -lggml-cpu -lm \
        -Wl,-rpath,"$REF" || exit 1
    if ! bash parity/limited.sh -- env LD_LIBRARY_PATH="$REF" "$OUT/ref_decode_dump" "$SYNTH" "$OUT/nodes_ref.bin" \
            --fa off --decode-tail 12 --text hi 2> "$OUT/graph_ref.err"; then
        echo "REFERENCE GRAPH RUN FAILED:" >&2; tail -3 "$OUT/graph_ref.err" >&2; exit 1
    fi
    bash parity/limited.sh -- env GLM5_DUMP_MODEL="$SYNTH" GLM5_DUMP_OUT="$OUT/nodes_port.bin" \
        cargo test --release -p llama --test glm5_dump -- --ignored --nocapture glm5_prefill_node_dump \
        > /dev/null 2>&1 || exit 1
    n=$(python3 parity/decode_dump_cmp.py "$OUT/nodes_ref.bin" "$OUT/nodes_port.bin" 2>/dev/null | grep -cE 'DIVERGENT' || true)
    if [ "$n" -eq 0 ]; then
        echo "   glm5-next graph: node streams bit-identical (0 divergent)"
    else
        echo "   glm5-next graph: $n DIVERGENT nodes" >&2
        exit 1
    fi
fi
