#!/usr/bin/env bash
# cli_interactive_parity.sh — scripted interactive-session parity: the port's
# llama-cli -i vs the pinned reference's llama-completion
# (tools/completion/completion.cpp), stdout transcripts byte-for-byte.
#
# Protocol (PARITY.md): the reference's LOG()-level output is stdout
# (log.cpp:113) — the generation echo, "EOF by user", the conversation
# "\n> " marker, the prefix/suffix echoes. LOG_INF/WRN/ERR go to stderr and
# are not compared (the port keeps its own diagnostics there).
#
# Sampling is pinned (--temp 0 --seed 42): the reference's default seed is
# random (LLAMA_DEFAULT_SEED), and prompts outside the established greedy
# anchors hit the documented pos19-class tie divergence (PARITY.md) — the
# anchor prompt "The capital of France is" is 16/16 clean in both FA modes.
#
# Usage: parity/cli_interactive_parity.sh [model.gguf]
set -u

MODEL="${1:-/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
REF_BIN_DIR="${REF_BIN_DIR:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT_BIN="${PORT_BIN:-$(dirname "$0")/../target/release/llama-cli}"
REF="$REF_BIN_DIR/llama-completion"

if [ ! -x "$REF" ]; then
    echo "reference llama-completion not built: make -C $REF_BIN_DIR llama-completion" >&2
    exit 2
fi
if [ ! -x "$PORT_BIN" ]; then
    echo "port binary not built: cargo build --release -p llama-cli" >&2
    exit 2
fi

export LD_LIBRARY_PATH="$REF_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

COMMON=(-m "$MODEL" -p 'The capital of France is' --temp 0 --seed 42 -c 256 -t 8)
PASS=0
FAIL=0

run_case() {
    local name="$1"; shift
    local stdin_data="$1"; shift
    timeout 300 "$REF" "${COMMON[@]}" "$@" <<<"$stdin_data" >/tmp/clip-ref.out 2>/dev/null
    timeout 300 "$PORT_BIN" "${COMMON[@]}" "$@" <<<"$stdin_data" >/tmp/clip-port.out 2>/dev/null
    if cmp -s /tmp/clip-ref.out /tmp/clip-port.out; then
        echo "MATCH  $name"
        PASS=$((PASS+1))
    else
        echo "DIFF   $name"
        diff /tmp/clip-ref.out /tmp/clip-port.out | head -6
        FAIL=$((FAIL+1))
    fi
}

# 1-2: the basic turn-taking loop, both console reader paths (the advanced
# reader echoes the piped line to stdout because /dev/tty is unavailable in
# a scripted run — console.cpp:141-144 falls back to stdout)
run_case "simple-io turn loop"     $'The capital of Germany is\n' -i -no-cnv -n 8 --simple-io
run_case "advanced reader turn loop" $'The capital of Germany is\n' -i -no-cnv -n 8

# 3: EOF on the first read of an interactive session
run_case "EOF on first read"       '' -i -no-cnv -n 4 --simple-io

# 4: multiple user turns before EOF
run_case "two user turns"          $'The capital of Germany is\nThe capital of Spain is\n' -i -no-cnv -n 6 --simple-io

# 5: '\' continuation merges lines (multiline OFF: one continuation)
run_case "backslash continuation"  $'The capital\\\nof Germany is\n' -i -no-cnv -n 6 --simple-io

# 6: -mli flips the default: plain lines continue, '\' submits
run_case "-mli multiline input"    $'line one\nline two\n' -i -no-cnv -n 4 -mli --simple-io

# 7: reverse prompt — generation halts at "Paris", control returns
run_case "reverse prompt -r"       $'again\n' -i -no-cnv -n 8 -r 'Paris' --simple-io

# 8: single-token reverse prompt (antiprompt_token path, completion.cpp:757-770)
run_case "reverse prompt token"    $'again\n' -i -no-cnv -n 8 -r '.' --simple-io

# 9: --no-display-prompt hides the prompt echo (display stays on for the reply)
run_case "no-display-prompt"       $'hello world\n' -i -no-cnv -n 4 --no-display-prompt --simple-io

# 10: --in-prefix/--in-suffix echo before/after the read (completion.cpp:817-820, :853-857)
run_case "in-prefix/suffix"        $'The sky\n' -i -no-cnv -n 6 --in-prefix ' [' --in-suffix ']' --simple-io

# 11: --in-prefix-bos prefixes BOS to the user input tokens
run_case "in-prefix-bos"           $'The sky\n' -i -no-cnv -n 4 --in-prefix-bos --simple-io

# 12: -if waits for input before generating anything
run_case "interactive-first"       $'The capital of Germany is\n' -if -no-cnv -n 6 --simple-io

# 13: empty line passes control back without adding tokens. Kept short: a
# session that generates past ~19 tokens crosses the reference's documented
# greedy tie (PARITY.md pos19, 1083 vs 7407, margin 0.038) — a pre-existing
# decode-path divergence, not an interactive-surface one
run_case "empty line pass-back"    $'\nok\n' -i -no-cnv -n 3 --simple-io

# 14: -sp special tokens are printed as text (same pos19 note as case 13)
run_case "special token output"    $'ok\n' -i -no-cnv -n 3 -sp --simple-io

# 15: '/' at end of line submits without newline (readline_simple marker)
run_case "slash submit marker"     $'The capital of Germany is/' -i -no-cnv -n 4 --simple-io

# 16: escape processing of the user line (-e is the default)
run_case "escape sequences"        $'line\\ttwo\n' -i -no-cnv -n 4 --simple-io

# 17: conversation mode — the chat-template turns, the "\n> " markers and the
# invisible special tokens (rendered empty) all ride the same transcript
run_case "conversation mode -cnv"  $'What is France?\n' -cnv -n 16 --simple-io

echo
echo "cli interactive parity: $PASS MATCH, $FAIL DIFF (of $((PASS+FAIL)))"
[ "$FAIL" -eq 0 ]
