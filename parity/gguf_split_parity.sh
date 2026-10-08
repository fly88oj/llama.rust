#!/usr/bin/env bash
# gguf-split cross-tool parity: the port's llama-gguf-split vs the reference
# (bd4f514db) — byte-identical split parts and merges in every direction, plus
# the reader-side split.* handling.
#
# Usage: bash parity/gguf_split_parity.sh [model.gguf]     (default: qwen2.5-0.5b)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF_BIN="${REF_BIN:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-gguf-split}"
PORT_BIN="${PORT_BIN:-$ROOT/target/release/llama-gguf-split}"
MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
WORK="${WORK:-/tmp/gguf-split-parity}"

REF="$REF_BIN"
PORT="$PORT_BIN"
rm -rf "$WORK"; mkdir -p "$WORK"/{ref,port}

fail=0
check() { if [ "$1" = 0 ]; then echo "PASS: $2"; else echo "FAIL: $2"; fail=1; fi; }

for MODE in "--split-max-tensors 30" "--split-max-tensors 100 --no-tensor-first-split" "--split-max-size 200M"; do
    name=$(echo "$MODE" | tr -c 'a-zA-Z0-9\n' '_' )
    mkdir -p "$WORK/ref/$name" "$WORK/port/$name"
    $REF $MODE "$MODEL" "$WORK/ref/$name/m.gguf" >"$WORK/ref/$name.out" 2>"$WORK/ref/$name.err"
    ref_rc=$?
    $PORT $MODE "$MODEL" "$WORK/port/$name/m.gguf" >"$WORK/port/$name.out" 2>"$WORK/port/$name.err"
    port_rc=$?
    [ "$ref_rc" = "$port_rc" ]; check $? "[$MODE] exit codes match ($ref_rc)"
    # stdout/stderr: only the output-path prefix may differ
    sed -i "s|$WORK/ref/$name|OUT|g;s|$WORK/ref|OUT|g" "$WORK/ref/$name.out" "$WORK/ref/$name.err"
    sed -i "s|$WORK/port/$name|OUT|g;s|$WORK/port|OUT|g" "$WORK/port/$name.out" "$WORK/port/$name.err"
    diff -q "$WORK/ref/$name.out" "$WORK/port/$name.out" >/dev/null; check $? "[$MODE] stdout identical"
    diff -q "$WORK/ref/$name.err" "$WORK/port/$name.err" >/dev/null; check $? "[$MODE] stderr identical"
    n=0
    for f in "$WORK"/ref/$name/*.gguf; do
        cmp -s "$f" "$WORK/port/$name/$(basename "$f")" || { echo "FAIL: [$MODE] $(basename "$f") differs"; fail=1; }
        n=$((n+1))
    done
    check 0 "[$MODE] $n split parts byte-identical"
done

# merge: reference's parts merged by both tools
FIRST=$(ls "$WORK"/ref/__split_max_tensors_30/m.gguf-*.gguf | head -1)
N=$(basename "$FIRST" | sed 's/.*-of-0*//;s/.gguf//')
$REF --merge "$FIRST" "$WORK/ref/merged.gguf" >/dev/null 2>"$WORK/ref/merge.err"
$PORT --merge "$FIRST" "$WORK/port/merged.gguf" >/dev/null 2>"$WORK/port/merge.err"
cmp -s "$WORK/ref/merged.gguf" "$WORK/port/merged.gguf"; check $? "merge byte-identical (port merges reference parts)"
# the reference consumes the port's split output
P1=$(ls "$WORK"/port/__split_max_tensors_30/m.gguf-*.gguf | head -1)
$REF --merge "$P1" "$WORK/ref/merged-from-port.gguf" >/dev/null 2>&1
cmp -s "$WORK/ref/merged-from-port.gguf" "$WORK/ref/merged.gguf"; check $? "reference merge of port parts == reference merge of reference parts"

# ---- loader: the port loads the parts natively with identical outputs ----
if [ -x "$ROOT/target/release/llama-cli" ]; then
    ORIG="$MODEL"
    PART1=$(ls "$WORK"/ref/__split_max_tensors_30/m.gguf-*.gguf | head -1)
    PROMPT="The capital of France is"
    for v in orig part merged; do
        case $v in
            orig)   M=$ORIG ;;
            part)   M=$PART1 ;;
            merged) M="$WORK/ref/merged.gguf" ;;
        esac
        $ROOT/target/release/llama-cli -m "$M" -p "$PROMPT" -n 32 -t 8 -c 512 --temp 0 \
            2>/dev/null | grep "gen tokens" >"$WORK/gen-$v.txt"
    done
    cmp -s "$WORK/gen-orig.txt" "$WORK/gen-part.txt"; check $? "port: generation from part-1 (natively) == single-file"
    cmp -s "$WORK/gen-orig.txt" "$WORK/gen-merged.txt"; check $? "port: generation from merged == single-file"
fi

echo
if [ "$fail" = 0 ]; then echo "ALL gguf-split parity checks PASSED"; else echo "gguf-split parity FAILURES present"; exit 1; fi
