#!/usr/bin/env bash
# anchor_newref.sh — the token anchors of the sync batch vs the NEW reference
# (a7b94df2c, /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin): a FRESH
# llama-server's FIRST /completion (temperature 0, cache_prompt false,
# return_tokens) per model, captured into the repo (parity/anchors/ — /tmp is
# volatile), then the port's greedy llama-cli run compared token-for-token
# (the mtp2_parity.sh convention: the ref's "tokens" JSON vs the port's
# "gen tokens: [...]" line).
#
# The anchors are the batch's ground truth: any arch the models+tables lane
# touched must still reproduce them (qwen2 exercises the plain tables, gpt-oss
# the MoE + sm-tensor path).
#
# usage:
#   bash parity/anchor_newref.sh ref      # capture the NEW-ref anchors only
#   bash parity/anchor_newref.sh cmp      # port side + compare (needs the build)
#   bash parity/anchor_newref.sh          # both
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-both}"
REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin
OUT=parity/anchors
PORT_CLI=./target/release/llama-cli
mkdir -p "$OUT"

QWEN25=/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
GPTOSS=/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf
PROMPT="The capital of France is"

run_ref() { # model port out_json
    local model=$1 port=$2 out=$3
    free -g | head -2
    bash parity/limited.sh -- "$REF/llama-server" -m "$model" -c 512 -t 8 \
        --host 127.0.0.1 --port "$port" --jinja >/dev/null 2>"$out.err" &
    local pid=$!
    local ok=""
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    if [ -z "$ok" ]; then
        kill $pid 2>/dev/null; pkill -P $pid 2>/dev/null; wait $pid 2>/dev/null
        return 1
    fi
    # the FIRST request against the fresh server is the anchor
    curl -s -X POST "http://127.0.0.1:$port/completion" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false,\"return_tokens\":true}" \
        -o "$out"
    kill $pid 2>/dev/null; pkill -P $pid 2>/dev/null; wait $pid 2>/dev/null
    return 0
}

port_tokens_of() { sed -n 's/^gen tokens: \[\(.*\)\]$/\1/p' "$1" | head -1 | tr -d ' ' | tr ',' '\n' | head -16 | tr '\n' ' ' | sed 's/ $//'; }

fail=0
for pair in "qwen25:$QWEN25:8851" "gptoss:$GPTOSS:8852"; do
    name=${pair%%:*}; rest=${pair#*:}; model=${rest%%:*}; port=${rest##*:}
    echo "=== $name ==="
    if [ "$MODE" = ref ] || [ "$MODE" = both ]; then
        if run_ref "$model" "$port" "$OUT/$name.ref.json"; then
            python3 -c "import json;d=json.load(open('$OUT/$name.ref.json'));print(' '.join(map(str,(d.get('tokens') or [])[:16])))" > "$OUT/$name.ref.tokens"
            echo "  ref anchor: $(cat "$OUT/$name.ref.tokens")"
        else
            echo "  REF SERVER FAILED (see $OUT/$name.ref.json.err)"; fail=1; continue
        fi
    fi
    if [ "$MODE" = cmp ] || [ "$MODE" = both ]; then
        [ -f "$OUT/$name.ref.tokens" ] || { echo "  missing anchor (run 'ref' first)"; fail=1; continue; }
        free -g | head -2
        # gpt-oss: the reference server runs FA ON by default, so the port
        # side must too (-fa on; llama-cli defaults to off — without it the
        # gptoss anchor flips at step 1 on a -1.83 vs -1.86 tie, a false
        # divergence; verified 16/16 with FA on, sync batch D)
        FA=off; [ "$name" = gptoss ] && FA=on
        bash parity/limited.sh -- "$PORT_CLI" -m "$model" -p "$PROMPT" -n 16 -t 8 -c 512 -fa $FA \
            --temp 0.0 > "$OUT/$name.port.txt" 2>"$OUT/$name.port.err" \
            || { echo "  PORT RUN FAILED (see $OUT/$name.port.err)"; fail=1; continue; }
        pt=$(port_tokens_of "$OUT/$name.port.txt")
        rt=$(cat "$OUT/$name.ref.tokens")
        if [ "$pt" = "$rt" ]; then
            echo "  token anchor IDENTICAL ($(wc -w <<< "$rt") tokens)"
        else
            echo "  token anchor DIFFERS: ref=[$rt] port=[$pt]"; fail=1
        fi
    fi
done
exit $fail
