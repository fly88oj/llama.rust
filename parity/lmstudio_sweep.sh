#!/usr/bin/env bash
# LM Studio 本地模型加载/生成实测：对每个本机 GGUF 跑 llama-cli 载入 + 8 token 贪心生成。
# 记录：退出码、加载时长、生成 token、首 60 字符。
set -u
CLI=/home/jeffrey/works/personal/github/fly88oj/llama.rust/target/release/llama-cli
OUT=/tmp/lmstudio_sweep.tsv
P="The capital of France is"
: > "$OUT"
MODELS=(
  "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  "/home/jeffrey/localai/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
  "/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
  "/home/jeffrey/localai/models/bge-m3-Q8_0.gguf"
  "/home/jeffrey/localai/models/t5-v1_1-xxl-encoder-Q5_K_S.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-small-GGUF/granite-4.0-h-small-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/ERNIE-4.5-21B-A3B-PT-GGUF/ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/unsloth/Seed-OSS-36B-Instruct-GGUF/Seed-OSS-36B-Instruct-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-GGUF/Olmo-3-32B-Think-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Laguna-S-2.1-GGUF/Laguna-S-2.1-Q4_K_M-00001-of-00002.gguf"
  "/home/jeffrey/.lmstudio/models/unsloth/Phi-4-reasoning-GGUF/phi-4-reasoning-UD-Q6_K_XL.gguf"
  "/home/jeffrey/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf"
  "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M.gguf"
)
for M in "${MODELS[@]}"; do
  [ -f "$M" ] || { printf "%s\tMISSING\t-\t-\t-\n" "$(basename "$M")" >> "$OUT"; continue; }
  T0=$(date +%s)
  R=$(timeout 900 "$CLI" -m "$M" -p "$P" -n 8 -t 8 -c 512 --temp 0 -fa off 2>&1)
  RC=$?
  T1=$(date +%s)
  TOK=$(echo "$R" | grep -oE "gen tokens: \[[0-9, ]+\]" | head -1 | tr -d '[]' | sed 's/gen tokens: //' | cut -c1-40)
  ERR=$(echo "$R" | grep -iE "^error|panic" | head -1 | cut -c1-80)
  printf "%s\t%d\t%ds\t%s\t%s\n" "$(basename "$M")" "$RC" "$((T1-T0))" "${TOK:-$ERR}" "$ERR" >> "$OUT"
  echo "done: $(basename "$M") rc=$RC" >> /tmp/lmstudio_sweep.log
done
echo SWEEP_COMPLETE >> /tmp/lmstudio_sweep.log