#!/usr/bin/env bash
# Differential test: our llama-tokenize vs the pinned reference binary.
# Usage: bash parity/tok_diff.sh <model.gguf> [more models...]
set -u
REF=${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/llama-tokenize}
RUST=${RUST:-./target/debug/llama-tokenize}
BAD=${BAD:-/tmp/bad_utf8.bin}
printf 'a\xffb' > "$BAD"

fail=0; n=0
for M in "$@"; do
  for P in "" "hello world" "  spaces  " "a<|im_start|>b" "中文测试" $'tab\ttab' "mixed é ü 日本"; do
    for EXTRA in "--no-bos" "--no-bos --parse-special"; do
      n=$((n+1))
      a=$($REF  -m "$M" -p "$P" --ids $EXTRA 2>/dev/null | tail -1)
      b=$($RUST -m "$M" -p "$P" --ids $EXTRA 2>/dev/null | tail -1)
      if [ "$a" != "$b" ]; then
        fail=$((fail+1))
        echo "DIFF $(basename "$M") P=[$P] EXTRA=[$EXTRA]"
        echo "  ref : $a"
        echo "  rust: $b"
      fi
    done
  done
  for E in "" "--escape"; do
    n=$((n+1))
    a=$($REF  -m "$M" -f "$BAD" --ids --no-bos $E 2>/dev/null | tail -1)
    b=$($RUST -m "$M" -f "$BAD" --ids --no-bos $E 2>/dev/null | tail -1)
    if [ "$a" != "$b" ]; then
      fail=$((fail+1))
      echo "DIFF raw-bytes $(basename "$M") E=[$E]"
      echo "  ref : $a"
      echo "  rust: $b"
    fi
  done
  # detokenize: the no-ids output prints "  <id> -> '<piece>'" per token
  for P in "hello <world> 42" "  double  space" "中文😀"; do
    n=$((n+1))
    a=$($REF  -m "$M" -p "$P" --no-bos 2>/dev/null)
    b=$($RUST -m "$M" -p "$P" --no-bos 2>/dev/null)
    if [ "$a" != "$b" ]; then
      fail=$((fail+1)); echo "DIFF detok $(basename "$M") P=[$P]"; echo "  ref : $a"; echo "  rust: $b"
    fi
  done
done
echo "checked $n pairs, $fail diffs"