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
    # acceptance = the default glm5_dump test's named-node pairing (76 named
    # nodes x 13 graphs, byte-identical). The legacy positional comparator
    # (decode_dump_cmp.py) can no longer judge this pair: since 0bb496dbd the
    # reference keeps all three build_inp_embd branches in the graph via
    # ggml_build_forward_select, and the branches not selected for the batch
    # are never computed — their dump slots read as zeros, so a live port node
    # paired against a dead ref branch shows as a spurious DIVERGENT. The
    # port resolves the selection at step-graph build time (documented in
    # graph.rs build_inp_embd) and has no dead branches; named pairing
    # compares only live nodes on both sides.
    cp "$OUT/nodes_ref.bin"  parity/glm5/nodes_ref.bin
    cp "$OUT/nodes_port.bin" parity/glm5/nodes_port.bin
    if bash parity/limited.sh -- cargo test --release -p llama --test glm5_dump -- --nocapture glm5_graph_nodes_bit_exact_vs_reference \
        > /dev/null 2>&1; then
        echo "   glm5-next graph: named-node streams bit-identical (76 names x 13 graphs)"
    else
        echo "   glm5-next graph: named-node comparison FAILED" >&2
        exit 1
    fi
fi
