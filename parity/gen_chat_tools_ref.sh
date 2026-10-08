#!/usr/bin/env bash
# gen_chat_tools_ref.sh — regenerate the chat tool-calling parity artifacts.
#
#   parity/chat_tools_cases.json  inputs (3 templates x 15 request shapes)
#   parity/chat_tools_ref.json    reference outputs (prompts, generation
#                                 prompts, parser dumps, grammars, parses)
#
# Consumed by crates/llama/tests/chat_tools_parity.rs.
#
# Requires the pinned reference build:
#   /home/jeffrey/llm/llama.cpp-next               (source, a7b94df2c)
#   /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin (libllama-common.so)
#
# NOTE: run with TZ=UTC (the reference formats datetime/date_string with
# std::localtime; the test normalizes rendered dates away either way).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# a7b94df2c (sync batch C2): the pinned worktree stays at the diff base
# (def4d406a); the probe builds against the NEW tree's headers + library
PIN="${PIN:-/home/jeffrey/llm/llama.cpp-next}"
REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"

DUMP="$ROOT/parity/ref_chat_tools_dump"

g++ -O2 -std=c++17 -o "$DUMP" "$ROOT/parity/ref_chat_tools_dump.cpp" \
    -I"$PIN/src" -I"$PIN/include" -I"$PIN/common" -I"$PIN/ggml/include" \
    -L"$REF" -lllama-common -Wl,-rpath,"$REF"

export TZ=UTC
export LLAMA_LOG_LEVEL=error

"$DUMP" \
    "$ROOT/parity/chat_tools_cases.json" \
    "$ROOT/parity/chat_tools_ref.json" 2>/dev/null

python3 - "$ROOT/parity/chat_tools_ref.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ok = sum(1 for r in d if r.get("ok"))
print(f"chat_tools_ref.json: {ok}/{len(d)} reference cases ok")
EOF
