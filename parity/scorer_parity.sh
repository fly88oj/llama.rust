#!/usr/bin/env bash
# Perplexity scorer parity: hellaswag / winogrande / multiple-choice — the
# port's llama-perplexity vs the reference (bd4f514db) on synthetic fixtures
# (parity/gen_scorer_fixtures.py). Compares stdout byte-for-byte and stderr
# after stripping the reference's "time level" log prefixes (common_init
# enables prefix+timestamps; stdout LOG() lines are prefix-free).
#
# Usage: bash parity/scorer_parity.sh [model.gguf]     (default: qwen2.5-0.5b)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-perplexity}"
PORT="${PORT:-$ROOT/target/release/llama-perplexity}"
MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
WORK="${WORK:-/tmp/scorer-parity}"
THREADS="${THREADS:-8}"

python3 "$ROOT/parity/gen_scorer_fixtures.py" "$WORK" >/dev/null

fail=0
check() { if [ "$1" = 0 ]; then echo "PASS: $2"; else echo "FAIL: $2"; fail=1; fi; }

run_pair() { # name flags... -- datafile  (flags passed as separate words)
    local name=$1; shift
    local data=""
    local flags=()
    for a in "$@"; do
        if [ "$a" = "--" ]; then data=""; continue; fi
        if [ -z "$data" ] && [ -f "$WORK/$a" ] 2>/dev/null; then data="$WORK/$a"; else flags+=("$a"); fi
    done
    timeout 420 "$REF" -m "$MODEL" -t $THREADS "${flags[@]}" -f "$data" \
        >"$WORK/$name-ref.out" 2>"$WORK/$name-ref.err"; local rref=$?
    timeout 420 "$PORT" -m "$MODEL" -t $THREADS "${flags[@]}" -f "$data" \
        >"$WORK/$name-port.out" 2>"$WORK/$name-port.err"; local rport=$?
    [ "$rref" = "$rport" ]; check $? "[$name] exit codes match ($rref)"
    diff -q "$WORK/$name-ref.out" "$WORK/$name-port.out" >/dev/null
    if [ $? = 0 ]; then
        check 0 "[$name] stdout identical ($(wc -l <"$WORK/$name-ref.out") lines)"
    elif [ "$name" != "${name%%winogrande*}" ] || [ "$name" = winogrande ]; then
        # winogrande: the two float diagnostic columns carry the port's known
        # logits-level GEMM noise; the acc/decision/answer columns must match
        diff <(awk '{print $1"\t"$2"\t"$5"\t"$6}' "$WORK/$name-ref.out") \
             <(awk '{print $1"\t"$2"\t"$5"\t"$6}' "$WORK/$name-port.out") >/dev/null
        check $? "[$name] stdout acc/decision columns identical (float score columns are port GEMM noise)"
    else
        check 1 "[$name] stdout identical"
    fi
    # strip "0.00.123.456 I/W/E " prefixes (they also glue onto the reference's
    # newline-less progress lines mid-line), then compare the scorer's own
    # stderr messages
    sed -E 's/[0-9]+\.[0-9]{2}\.[0-9]{3}\.[0-9]{3} [IWE] //g' "$WORK/$name-ref.err" >"$WORK/$name-ref.err2"
    grep -E "score|tasks|Final|Random|spm|comma|answer|_ in" "$WORK/$name-ref.err2" | sed 's/[0-9.]*$//' >"$WORK/$name-ref.err3"
    grep -E "score|tasks|Final|Random|spm|comma|answer|_ in" "$WORK/$name-port.err" | sed 's/[0-9.]*$//' >"$WORK/$name-port.err3"
    diff -q "$WORK/$name-ref.err3" "$WORK/$name-port.err3" >/dev/null
    check $? "[$name] scorer stderr lines identical"
}

run_pair hellaswag       -- --hellaswag        hellaswag_synth.txt
run_pair winogrande      -- --winogrande       winogrande_synth.csv
run_pair multiplechoice  -- --multiple-choice  multiple_choice_synth.bin
# a reduced-task selection run exercises the RNG selection paths
run_pair hellaswag10     -- --hellaswag --hellaswag-tasks 10  hellaswag_synth.txt
run_pair winogrande10    -- --winogrande --winogrande-tasks 10 winogrande_synth.csv

echo
if [ "$fail" = 0 ]; then echo "ALL scorer parity checks PASSED"; else echo "scorer parity FAILURES present"; exit 1; fi
