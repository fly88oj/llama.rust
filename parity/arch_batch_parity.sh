#!/usr/bin/env bash
# Synthetic-GGUF parity for the arch batches.
#
#   parity/arch_batch_parity.sh [arch ...]           # batch 1 (default)
#   ARCH_BATCH2=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH3=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH4=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH4_CLI=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH5=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH5_CLI=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH6=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH6B=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH7=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH8=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH9=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH10=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH11B=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH11A=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH12=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH14=1 parity/arch_batch_parity.sh [arch ...]
#   ARCH_BATCH15=1 parity/arch_batch_parity.sh [arch ...]
#
# Batch 1: gpt2 / phi2 / starcoder2 / command-r / gptneox / olmo2
#   (crates/llama/tests/arch_batch_e2e.rs, files in /tmp/arch-batch/)
# Batch 2: codeshell / orion / olmo / xverse / internlm2 / exaone / gemma /
#          falcon
#   (crates/llama/tests/arch_batch2_e2e.rs, files in /tmp/arch-batch2/)
# Batch 3: baichuan13 / baichuan7 / bloom / mpt / mptfull / starcoder /
#          refact / plamo / stablelm / granite / minicpm — the ALiBi family +
#          the cheap no-rope archs
#   (crates/llama/tests/arch_batch3_e2e.rs, files in /tmp/arch-batch3/)
# Batch 4: qwen2moe / qwen3moe / granite-moe / phimoe / arctic / olmoe /
#          ernie4-5-moe / ernie-nosh / smollm3 / seed-oss / openelm — the MoE
#          family
#   (crates/llama/tests/arch_batch4_e2e.rs, files in /tmp/arch-batch4/)
# Batch 5: mamba / mamba2 / jamba / jamba-moe / nemotron-h / nemotron-h-moe —
#          the mamba family (recurrent-state archs); N = 48 tokens so the
#          recurrence runs well past the prompt, and an appended "-long" runs
#          the >64-token prompt cell that exercises the conv-state slide
#   (crates/llama/tests/arch_batch5_e2e.rs, files in /tmp/arch-batch5/)
# Batch 6: deepseek2 / deepseek2-lite / deepseek2-legacy / deepseek2-ocr /
#          deepseek — the DeepSeek MLA family (absorbed-MQA MLA with the
#          compressed K-only cache + wv_b decompression, the lite wq variant,
#          the legacy unsplit wkv_b decompressed-MHA variant incl. rope
#          offset, the OCR MHA branch and the non-MLA v2 base); "-long"
#          appends the >64-token prompt cell. Batch 6 drives llama-cli itself
#          (batch-1 protocol) — this batch's ForwardWeights/CLI arms landed
#          with the graph
#   (crates/llama/tests/arch_batch6_e2e.rs, files in /tmp/arch-batch6/)
# Batch 6b: nemotron / grok / grok-dense / chameleon / chameleon-swin /
#           deci / deci-mixed / jais / falcon-h1 / plamo2 — the batch-7
#           queue's dense post-norm + ALiBi + hybrid-mamba entries. "-long"
#           appends the >64-token prompt cell (jais's ALiBi distances, the
#           rope of grok/deci, the hybrids' recurrence past the ubatch
#           boundary). Like batch 6 this batch drives llama-cli itself —
#           its ForwardWeights/CLI arms landed with the graph
#   (crates/llama/tests/arch_batch6b_e2e.rs, files in /tmp/arch-batch6b/)
# Batch 7: deepseek4 — hyper-connections (4 streams, the fused ggml_dsv4_hc_*
#   ops) + the compressed DSV4 KV cache (iswa raw pair + csa/hca/lid
#   compressed caches + the three compressor states). The synthetic file's 4
#   layers pin all three compression ratios (0/4/128) plus the hash layer;
#   "-long" runs the >64-token prompt cell that pushes positions past the
#   64-token SWA window. Drives llama-cli itself (batch-1 protocol)
#   (crates/llama/tests/arch_batch7_e2e.rs, files in /tmp/arch-batch7/)
# Batch 8: hunyuan-moe / dots1 / bailingmoe / bailingmoe2 / glm4moe /
#   minimax-m2 / cohere2moe / exaone-moe — the MoE long-tail family
#   (`build_moe_ffn` compositions). cohere2moe / exaone-moe carry n_swa 64, so
#   their "-long" cells (>64-token prompt) push positions past the window
#   (the real iswa purge/mask path on both sides — llama-cli now creates the
#   iswa pair like llama-server). Like batch 6/6b/7 this batch drives
#   llama-cli itself — its ForwardWeights/CLI arms landed with the graph
#   (crates/llama/tests/arch_batch8_e2e.rs, files in /tmp/arch-batch8/)
# Batch 9: plamo3 / qwen3next / kimi-linear / bailingmoe3 — the
#   linear-attention family (the KDA/GDN hybrid archs run the fused
#   GGML_OP_GATED_DELTA_NET with K = 1; plamo3 is the SWA + post-norm + swiglu
#   dense relative). The KDA archs' MLA layers cache the compressed
#   [kv_lora|rope] rows (the deepseek2 trick); "-long" appends the >64-token
#   prompt cell (the conv-state slide + the SWA window edge). Drives
#   llama-cli itself (batch-1 protocol)
#   (crates/llama/tests/arch_batch9_e2e.rs, files in /tmp/arch-batch9/)
# Batch 10: smallthinker / llada-moe / minimax-01 / graniteswitch — the
#   small-arch + EXP-op batch. smallthinker carries the probs_in + ReGLU MoE
#   (its "-swa" variant is verified in-port, arch_batch10_variant_swa — the
#   iswa graph template); llada-moe is the non-causal diffusion attention (the
#   reference server generates through build_attn_inp_no_cache — a decode step
#   attends only its own ubatch; the port mirrors that default); minimax-01 is
#   lightning attention (the la decay inputs + GGML_UNARY_OP_EXP + the
#   n_embd_head_la recurrent state — 48 tokens so the recurrence runs well
#   past the prompt); graniteswitch routes its switch-LoRA adapters through an
#   in-graph single-head router layer. Drives llama-cli itself (batch-1
#   protocol) (crates/llama/tests/arch_batch10_e2e.rs, files in
#   /tmp/arch-batch10/)
# Batch 12: hrm-text / laguna / laguna-full / maple — the final long-tail
#   queue (the last three src/models/*.cpp files without a ported graph).
#   hrm-text threads its learned [n_embd] low-cycle state through the
#   alternating low/high stacks over aliased cache slots; laguna is the
#   sigmoid-routed MoE + softplus attention gate arch (the default file is
#   the XS.2-like hybrid with per-layer-type RoPE — YaRN on the full layer,
#   plain rope on the SWA layers — "-full" is the M.1-like all-full file);
#   maple ropes ONLY its SWA layers over the iswa pair. "-long" appends the
#   >64-token prompt cell (positions past the n_swa 64 windows). Drives
#   llama-cli itself (batch-1 protocol)
#   (crates/llama/tests/arch_batch12_e2e.rs, files in /tmp/arch-batch12/)
#
#   cargo test -p llama --test arch_batch_e2e  -- --ignored arch_batch_write_synth
#   cargo test -p llama --test arch_batch2_e2e -- --ignored arch_batch2_write_synth
#   cargo test -p llama --test arch_batch3_e2e -- --ignored arch_batch3_write_synth
#   cargo test -p llama --test arch_batch4_e2e -- --ignored arch_batch4_write_synth
#   cargo test -p llama --test arch_batch5_e2e -- --ignored arch_batch5_write_synth
#   cargo test -p llama --test arch_batch6_e2e -- --ignored arch_batch6_write_synth
#   cargo test -p llama --test arch_batch7_e2e -- --ignored arch_batch7_write_synth
#   cargo test -p llama --test arch_batch8_e2e -- --ignored arch_batch8_write_synth
#   cargo test -p llama --test arch_batch9_e2e -- --ignored arch_batch9_write_synth
#   cargo test -p llama --test arch_batch10_e2e -- --ignored arch_batch10_write_synth
#   cargo test -p llama --test arch_batch11b_e2e -- --ignored arch_batch11b_write_synth
#
# Each (arch, -fa on|off) pair is one *fresh* reference llama-server and the
# FIRST /completion request on it (PARITY.md protocol: temperature 0,
# logprobs 20, cache_prompt false), compared against ./target/release/llama-cli
# through parity/run_cli_arch_parity.sh. The token match and the worst
# |dlogprob| over the shared top-k come from parity/arch_batch_cmp.py.
#
# Batches 1 (and batch 4 with ARCH_BATCH4_CLI=1, after the ForwardWeights
# wiring landed in context.rs) drive ./target/release/llama-cli itself via
# parity/run_cli_arch_parity.sh. The plain batch-3/4/5 cells still drive the
# env-driven greedy drivers `arch3_cli_driver` / `arch4_cli_driver` /
# `arch5_cli_driver` in arch_batch{3,4,5}_e2e.rs (they print llama-cli's exact
# debug format), keeping the two port sides independently checkable. Append
# "-long" to a batch-3 arch name to run the *long* (>64 token) prompt cell
# that exercises the ALiBi mask's distance dependence.
#
# Artifacts: /tmp/parity-<arch>-<fa>-{server.log,ref.json,cli.out,cli.err}
set -uo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin}"
PORT="${PORT:-8790}"

if [ -n "${ARCH_BATCH15:-}" ]; then
  DIR=/tmp/arch-batch15
  # batch 15 — the P1+P2 queue of AUDIT_models.md: qwen(v1) / maincoder /
  # pangu-embed / plm / cogvlm / spark2-5 / muse-glimmer / hunyuan-vl
  # (+mrope) / hunyuan-dense / granite-swa (+moe) / afmoe / mellum /
  # paddleocr / hy-v3 / mimo2 / step35 / hy-v4 (+dsa: the lightning indexer
  # over the shared-indexer layers) + mistral4 (the deepseek2-MLA recipe
  # under the mistral4 arch name). The variants: hunyuan-vl-mrope (sections +
  # the XDRoPE alpha), granite-swa-moe (the softmax MoE + the SWIGLU fused
  # shexp), afmoe-swa / mellum-swa / mimo2-vscale (the in-port variants —
  # afmoe-swa/mellum-swa re-run the same graphs under iswa so they stay
  # runnable explicitly). llada is NOT in the default set (the diffusion
  # family creates no reference memory — verified in-port,
  # arch_batch15_inport_only_cells); gemma-embedding / llama-embed are the
  # encoder pair (the ref_encode_dump protocol, the ignored
  # arch_batch15_encoder_reference_parity). "-long" runs the >64-token prompt
  # cell (the n_swa 32 windows, step35's halved rope, hy-v4's lid rows).
  # Drives llama-cli itself (batch-1 protocol)
  # (crates/llama/tests/arch_batch15_e2e.rs)
  # cogvlm is NOT in the default set:
  #   * cogvlm — the pinned reference SERVER diverges from the reference
  #     LIBRARY's own decode (ref_decode_dump: the port matches the library
  #     bit-exactly — all 224000 logits + 33/33 comparable nodes; the server
  #     flips the step-0 top-1 at a 0.11-logprob near-tie) — the minimax-m3
  #     precedent (PARITY.md 批次 11a).
  # mimo2 / step35 (+ their -long cells) ARE in the default set: the former
  #   ~3e-3 iswa-layer kq divergence is fixed — the router selection bias
  #   (ffn_exp_probs_b) was folded into the logits BEFORE the gating function
  #   (sigmoid(l+b) instead of the reference's sigmoid(l)+b on the top-k
  #   scores only, llama-graph.cpp:2063-2067), which perturbed every layer's
  #   MoE weights from layer 0's l_out on; the iswa kq nodes were pure
  #   downstream drift (layer 0's kq sits upstream of the FFN, hence was
  #   bit-exact). The second-order fix: the MoE ffn_moe_swiglu rows (n_ff_exp
  #   24, not a multiple of 16) ran the v512-polynomial silu on the scalar
  #   tail lanes where the reference's ggml_vec_swiglu_f32 runs libm expf
  #   (1 ulp). Both archs now dump bit-exact against the reference library
  #   (arch_batch15_prefill_node_dump + ref_decode_dump: 0/224000 final
  #   logits, both FA modes).
  DEFAULT=(qwen maincoder pangu-embed plm spark2-5 muse-glimmer            hunyuan-vl hunyuan-vl-mrope hunyuan-dense granite-swa granite-swa-moe            afmoe mellum paddleocr hy-v3 hy-v4 hy-v4-dsa mistral4            mimo2 step35            spark2-5-long granite-swa-long hy-v4-dsa-long mimo2-long step35-long)
elif [ -n "${ARCH_BATCH14:-}" ]; then
  DIR=/tmp/arch-batch14
  # batch 14 — the P0 new-mechanism queue: the RWKV family (rwkv6 /
  # rwkv6qwen2 / rwkv7 / arwkv7 — pure-recurrent, the token-shift + WKV state
  # cells; the WKV scans are composed from the existing ops until the fused
  # ggml kernels land as integrator items) and gemma3n (per-layer embeddings +
  # altup/laurel + the KV-reuse layers). RWKV is attention-free, so its -fa
  # cells degenerate (the flag only flips the mask dtype; both cells run).
  # "-long" runs the >64-token prompt cell: the RWKV recurrence + token shift
  # across ~100 tokens, gemma3n's positions past the n_swa 512 windows are
  # not reachable at this ctx, so its -long cell re-exercises the reuse
  # layers' attention over a longer history. Drives llama-cli itself
  # (batch-1 protocol) (crates/llama/tests/arch_batch14_e2e.rs)
  # rwkv7-long is IN the default set since the fused WKV7 kernel landed
  # (crates/ggml/src/wkv.rs, bit-exact vs the reference — parity/wkv_ref.bin):
  # the composed stand-in's mul+add rounding (vs the kernel's FMA) used to
  # flip near-tied tokens at ~72 tokens; with the fused op the cell matches
  # 48/48 both FA modes.
  # arwkv7 stays out: a real, unexplained divergence — at -t 1 (both sides
  # deterministic, ref stable across fa modes and fresh servers) the port's
  # step-0 distribution is ~0.18 nats off the ref's top-token logprob and
  # token 1 diverges outright; every piece it shares with rwkv7 (the whole
  # time mix incl. the rk bonus, the WKV7 scan, the state write-back) is
  # verified by rwkv7/rwkv7-long matching 48/48, so the bug hides in the
  # arwkv7-specific arms (tsc-1 token shift / SwiGLU FFN / RMS norms /
  # loader). At -t 8 the ref itself additionally sits on a thread-noise
  # cliff (its top token flips across runs) — the -t 1 protocol above is
  # the honest comparison. Both stay runnable explicitly and verified
  # in-port (arch_batch14_pin_and_smoke).
  DEFAULT=(rwkv6 rwkv6qwen2 rwkv7 rwkv7-long gemma3n rwkv6-long)
elif [ -n "${ARCH_BATCH13:-}" ]; then
  DIR=/tmp/arch-batch13
  # batch 13 — the P0 standard-attention queue: llama4 / qwen3vl /
  # qwen3vlmoe / qwen2vl / glm4 / glm-dsa / chatglm / mistral3 / cohere2 /
  # minicpm3 / exaone4 / bitnet / dbrx / ernie4-5 (dense) + nemotron-h-moe
  # (the loader arm; its graph is nemotron-h's, batch 5). The variants:
  # mistral3-temp (attention.temperature_scale 0.5 + the yarn floor),
  # glm4-mrope (rope.dimension_sections → ggml_rope_multi + 4 pos/token),
  # exaone4-swa (the 64-layer 32B shape, window 32), llama4-noswa
  # (sliding_window 0 → CHUNKED off, always rope) and glm-dsa-shared
  # (attention.indexer_types [1,0,0,0] — the shared-indexer layers). The
  # qwen3vl/qwen2vl cells are the TEXT-side graph (mrope sections with the
  # same pos repeated in all 4 blocks + qwen3vl's zero-padded deepstack —
  # what the reference does for text-only batches). "-long" runs the
  # >64-token prompt cell (the iswa windows of cohere2 / exaone4-swa at
  # n_swa 32, llama4's chunked pattern + no-rope layers, glm-dsa's lid
  # rows). Drives llama-cli itself (batch-1 protocol)
  # (crates/llama/tests/arch_batch13_e2e.rs)
  # llama4-noswa is NOT in the default set: the PINNED REFERENCE itself
  # aborts on the file — build_inp_attn_scale is called unconditionally
  # (llama4.cpp:119-121) and llm_graph_input_attn_temp::set_input asserts
  # f_attn_temp_scale != 0 (llama-graph.cpp:161), but the noswa variant
  # carries no attention.temperature_scale (sliding_window 0 = the always-rope
  # branch, llama4.cpp:8-11). No reference output exists to compare against;
  # the port's graph (which skips the temp input exactly when the scale is 0)
  # is verified in-port (arch_batch13_variants_differ /
  # arch_batch13_llama4_temp_scale_below_first_chunk) and stays runnable
  # explicitly.
  DEFAULT=(cohere2 chatglm bitnet dbrx ernie4-5 mistral3 mistral3-temp \
           minicpm3 glm4 glm4-mrope exaone4 exaone4-swa llama4 \
           qwen2vl qwen3vl qwen3vlmoe glm-dsa glm-dsa-shared nemotron-h-moe \
           cohere2-long exaone4-swa-long llama4-long glm-dsa-long)
elif [ -n "${ARCH_BATCH12:-}" ]; then
  DIR=/tmp/arch-batch12
  # batch 12 — the final long-tail queue: every arch creates a memory in the
  # pinned reference, so all four cells run against the reference server.
  # "-long" runs the >64-token prompt cell: laguna / maple positions past the
  # n_swa 64 windows (both sides of the iswa pair), hrm-text's zH/zL state
  # across ~100 tokens. maple-allswa (the rope-every-layer variant) is NOT in
  # the default set — it is verified in-port
  # (arch_batch12_maple_swa_rope_changes_logits) and stays runnable explicitly.
  DEFAULT=(hrm-text laguna laguna-full maple maple-long)
elif [ -n "${ARCH_BATCH11B:-}" ]; then
  DIR=/tmp/arch-batch11b
  # dream / rnd1 are NOT in the default set: the pinned reference creates NO
  # memory for the diffusion archs (llama-model.cpp:2289-2295 — the same
  # LLADA-family list llada-moe is on) and both llama-server ("the current
  # context does not logits computation", server-context.cpp:3177-3180) and
  # llama-cli refuse to generate on a memory-less context — there is no
  # reference output to compare against. The port's no-cache graphs are
  # verified in-port (arch_batch11b_inport_only_pins /
  # arch_batch11b_long_prompt_cells). eurobert is the batch's encoder (bert/
  # t5 `llama_encode` path — no generation either): verified against the
  # reference dumper by the ignored arch_batch11b_eurobert_reference_parity.
  DEFAULT=(arcee jais2 talkie nanbeige nanbeige-loops2)
elif [ -n "${ARCH_BATCH11A:-}" ]; then
  DIR=/tmp/arch-batch11a
  # batch 11a — the long-tail queue, first half. The "-long" cells are part
  # of the default set: dots3note-long crosses the >64-token SWA window edge
  # with the lid cache rows live, minimax-m3-long drives the MSA block
  # selection at ~100 tokens (fa on) and the dense fallback (fa off), and
  # the GDN/KDA hybrids run their conv-state slide + recurrent state well
  # past the prompt. qwen4exp runs the legal dense/no-PLE configuration
  # (no attention.compress_ratios / PLE keys — see PARITY.md 批次 11a).
  # minimax-m3's fa=on cells are SKIPPED below (reference-side execution-
  # mode divergence — see the note in the cell loop).
  DEFAULT=(apertus grovemoe qwen35moe kimi-k3 dots3note dots3note-long \
           minimax-m3 minimax-m3-long qwen4exp qwen4exp-long)
elif [ -n "${ARCH_BATCH10:-}" ]; then
  DIR=/tmp/arch-batch10
  # smallthinker-swa is NOT in the default set: the iswa graph template
  # (the window key re-pins n_swa 4096 / the *_swa rope pair) is verified
  # in-port (arch_batch10_variant_swa) and stays runnable explicitly.
  # llada-moe is NOT in the default set either: the pinned reference creates
  # NO memory for the diffusion archs (llama-model.cpp:2289-2295 —
  # LLADA/LLADA_MOE/DREAM/RND1 → `res = nullptr`) and every generation
  # driver refuses a memory-less context ("the current context does not
  # logits computation", server-context.cpp:3177-3180; llama-cli errors the
  # same) — there is no reference output to compare against. The port's
  # graph (the non-causal no-cache attention) is verified in-port
  # (arch_batch10_pin_and_smoke / arch_batch10_long_prompt_cells).
  DEFAULT=(smallthinker minimax-01 graniteswitch)
elif [ -n "${ARCH_BATCH9:-}" ]; then
  DIR=/tmp/arch-batch9
  # qwen3next-legacy / kimi-linear-legacy are NOT in the default set: the
  # legacy GGUF layouts (the fused ssm_in QKVZ tensor / the unsplit wkv_b MHA
  # cache) are verified in-port (arch_batch9_variant_legacy) and stay
  # runnable explicitly
  DEFAULT=(plamo3 qwen3next kimi-linear bailingmoe3)
elif [ -n "${ARCH_BATCH8:-}" ]; then
  DIR=/tmp/arch-batch8
  # cohere2moe-sep-ln is NOT in the default set: the fused-gate_up + RMS file
  # is the reference cell; the separate-tensor + LLM_NORM variant is verified
  # in-port (synth_cohere2moe_sep_ln_variant) and stays runnable explicitly
  DEFAULT=(hunyuan-moe dots1 bailingmoe bailingmoe2 glm4moe minimax-m2 cohere2moe exaone-moe)
elif [ -n "${ARCH_BATCH7:-}" ]; then
  DIR=/tmp/arch-batch7
  DEFAULT=(deepseek4)
elif [ -n "${ARCH_BATCH6B:-}" ]; then
  DIR=/tmp/arch-batch6b
  # chameleon-swin is NOT in the default set: the reference server computes
  # different numbers than chameleon.cpp's own swin_norm wiring for it (the
  # port and an independent numpy model of the C source agree with each
  # other — see PARITY.md's batch-6b section); it stays runnable explicitly
  DEFAULT=(nemotron grok grok-dense chameleon deci deci-mixed jais falcon-h1 plamo2)
elif [ -n "${ARCH_BATCH6:-}" ]; then
  DIR=/tmp/arch-batch6
  DEFAULT=(deepseek2 deepseek2-lite deepseek2-legacy deepseek2-ocr deepseek)
elif [ -n "${ARCH_BATCH5:-}" ]; then
  DIR=/tmp/arch-batch5
  DEFAULT=(mamba mamba2 jamba jamba-moe nemotron-h nemotron-h-moe)
elif [ -n "${ARCH_BATCH4:-}" ]; then
  DIR=/tmp/arch-batch4
  DEFAULT=(qwen2moe qwen3moe granite-moe phimoe arctic olmoe ernie4-5-moe smollm3 seed-oss openelm)
elif [ -n "${ARCH_BATCH3:-}" ]; then
  DIR=/tmp/arch-batch3
  DEFAULT=(baichuan13 baichuan7 bloom mpt starcoder refact plamo stablelm granite minicpm)
elif [ -n "${ARCH_BATCH2:-}" ]; then
  DIR=/tmp/arch-batch2
  DEFAULT=(codeshell orion olmo xverse internlm2 exaone gemma falcon)
else
  DIR=/tmp/arch-batch
  DEFAULT=(gpt2 phi2 starcoder2 command-r gptneox olmo2)
fi

ARCHS=("$@")
if [ ${#ARCHS[@]} -eq 0 ]; then
  ARCHS=("${DEFAULT[@]}")
fi

PROMPT="The capital of France is"
# the batch-3 alibi long-prompt cell: ~100 SPM tokens so attention distances
# beyond 64 exist (where -|p0-p1| is no longer ~constant) but still fit the
# synthetic files' 256-token context with 16 generated tokens
LONG_PROMPT="The history of Rome spans more than two and a half thousand years of political, religious and cultural development, during which the city grew from a small Italian village into the center of a vast empire that dominated the Mediterranean world, and its legacy still shapes the languages, laws and institutions of modern Europe in ways both obvious and subtle."
if [ -n "${ARCH_BATCH14:-}" ]; then
  # batch 14: 48 — the RWKV recurrence (token shift + WKV state) must run
  # well past the prompt (the files' context is 512)
  N=48
elif [ -n "${ARCH_BATCH15:-}" ]; then
  # batch 15: 16 greedy tokens; the "-long" cells push rope/positions past
  # the n_swa 32 windows and step35's halved full-attention rope
  N=16
elif [ -n "${ARCH_BATCH5:-}" ]; then
  # batch 5: 48 generated tokens (the recurrence must be exercised well past
  # the prompt; the batch-5 files' context is 512)
  N=48
elif [ -n "${ARCH_BATCH13:-}" ]; then
  # batch 13: 16 greedy tokens; the "-long" cells push rope/positions past
  # the n_swa 32 windows of cohere2 / exaone4-swa and the glm-dsa lid rows
  N=16
elif [ -n "${ARCH_BATCH12:-}" ]; then
  # batch 12: 16 greedy tokens (plain attention archs; the "-long" cell
  # pushes rope/positions past the n_swa 64 windows)
  N=16
elif [ -n "${ARCH_BATCH11B:-}" ]; then
  # batch 11b: 16 greedy tokens (plain attention archs; the "-long" cell
  # pushes rope/positions past 64)
  N=16
elif [ -n "${ARCH_BATCH11A:-}" ]; then
  # batch 11a: 48 — the KDA/GDN recurrence (kimi-k3 / qwen35moe / qwen4exp)
  # must run well past the prompt, and the MSA block selection of minimax-m3
  # evolves every decode step
  N=48
elif [ -n "${ARCH_BATCH6B:-}" ]; then
  # batch 6b: 48 too — the falcon-h1 / plamo2 hybrids recur at every step
  N=48
elif [ -n "${ARCH_BATCH9:-}" ]; then
  # batch 9: 48 — the KDA/GDN recurrence must run well past the prompt (the
  # conv-state slide and the recurrent state carry across 48 decode steps)
  N=48
elif [ -n "${ARCH_BATCH10:-}" ]; then
  # batch 10: 48 — minimax-01's lightning-attention recurrence must run well
  # past the prompt (the la state + the block decay evolve every step); the
  # others are plain attention archs and just generate 48 tokens
  N=48
elif [ -n "${ARCH_BATCH7:-}" ]; then
  # batch 7: 16 greedy tokens (the comp plans + both attention branches are
  # exercised every step); the -long cell crosses the window + block edges
  N=16
else
  N=16
fi
FAIL=0

# batch-3 cell name -> synth file
b3_model() {
  case "$1" in
    baichuan13) echo "$DIR/baichuan-synth.gguf" ;;
    baichuan7)  echo "$DIR/baichuan-synth-7b.gguf" ;;
    mptfull)    echo "$DIR/mpt-synth-full.gguf" ;;
    stablelm3b) echo "$DIR/stablelm-synth-3b.gguf" ;;
    *)          echo "$DIR/$1-synth.gguf" ;;
  esac
}

# batch-4 cell name -> synth file (the pretty cell names map onto the C arch
# names that prefix the GGUF KV: granitemoe / seed_oss / ernie4_5-moe)
b4_model() {
  case "$1" in
    ernie-nosh)   echo "$DIR/ernie4_5-moe-synth-nosh.gguf" ;;
    granite-moe) echo "$DIR/granitemoe-synth.gguf" ;;
    seed-oss)    echo "$DIR/seed_oss-synth.gguf" ;;
    ernie4-5-moe) echo "$DIR/ernie4_5-moe-synth.gguf" ;;
    *)           echo "$DIR/$1-synth.gguf" ;;
  esac
}

# batch-5 cell name -> synth file (jamba/nemotron_h are the GGUF arch names;
# the -moe / -long suffixes compose)
b5_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    jamba-moe)       echo "$DIR/jamba-synth-moe.gguf" ;;
    nemotron-h)      echo "$DIR/nemotron_h-synth.gguf" ;;
    nemotron-h-moe)  echo "$DIR/nemotron_h-synth-moe.gguf" ;;
    *)               echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-7 cell name -> synth file
b7_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    *) echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-8 cell name -> synth file
b8_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    cohere2moe-sep-ln) echo "$DIR/cohere2moe-synth-sep-ln.gguf" ;;
    *)                  echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-6 cell name -> synth file (suffixes sit after -synth)
b6_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    deepseek2-lite)   echo "$DIR/deepseek2-synth-lite.gguf" ;;
    deepseek2-legacy) echo "$DIR/deepseek2-synth-legacy.gguf" ;;
    deepseek2-ocr)    echo "$DIR/deepseek2-ocr-synth.gguf" ;;
    *)                echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-6b cell name -> synth file (the variant suffix sits after -synth;
# falcon-h1's own dash is accounted for)
b6b_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    grok-dense)         echo "$DIR/grok-synth-dense.gguf" ;;
    chameleon-swin)     echo "$DIR/chameleon-synth-swin.gguf" ;;
    deci-mixed)         echo "$DIR/deci-synth-mixed.gguf" ;;
    *)                  echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-9 cell name -> synth file (the GGUF arch names keep their dashes;
# the variant suffixes sit after -synth)
b9_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    qwen3next-legacy)     echo "$DIR/qwen3next-synth-legacy.gguf" ;;
    kimi-linear-legacy)   echo "$DIR/kimi-linear-synth-legacy.gguf" ;;
    *)                    echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-10 cell name -> synth file (smallthinker's iswa variant sits after
# -synth; the parity prompt tokens never hit graniteswitch's activate ids —
# the router readback still runs on every token through slot 0)
b10_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    smallthinker-swa) echo "$DIR/smallthinker-synth-swa.gguf" ;;
    *)                echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-11b cell name -> synth file (nanbeige's loops2 variant sits after
# -synth)
b11b_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    nanbeige-loops2) echo "$DIR/nanbeige-synth-loops2.gguf" ;;
    *)               echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-11a cell name -> synth file (the GGUF arch names keep their dashes;
# the files are flat <arch>-synth.gguf)
b11a_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    *) echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-14 cell name -> synth file (flat <arch>-synth.gguf; the variant
# suffixes sit after -synth)
b14_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    *) echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-15 cell name -> synth file (flat <arch>-synth.gguf; the variant
# suffixes sit after -synth; pangu-embed maps onto the pangu-embedded GGUF
# arch name)
b15_model() {
  local cell="$1" base="${1%-long}"
  local stem="${base%%-*}" rest="${base#*-}"
  # candidates in resolution order: the literal name; the whole-dash
  # underscore form; the variant suffix after -synth (stem or underscored
  # stem); pangu-embed is pangu-embedded
  local c=(
    "$DIR/$base-synth.gguf"
    "$DIR/${base//-/_}-synth.gguf"
    "$DIR/$stem-synth-$rest.gguf"
    "$DIR/${stem//-/_}-synth-$rest.gguf"
    "$DIR/${stem}_${rest%%-*}-synth-${rest##*-}.gguf"
  )
  [ "$base" = "pangu-embed" ] && c=("$DIR/pangu-embedded-synth.gguf")
  for f in "${c[@]}"; do
    [ -f "$f" ] && { echo "$f"; return; }
  done
  echo "${c[0]}"
}

# batch-13 cell name -> synth file (the arch names keep their dashes /
# underscores; the variant suffixes sit after -synth)
b13_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    ernie4-5)       echo "$DIR/ernie4_5-synth.gguf" ;;
    glm-dsa-shared) echo "$DIR/glm-dsa-synth-shared.gguf" ;;
    mistral3-temp)  echo "$DIR/mistral3-synth-temp.gguf" ;;
    glm4-mrope)     echo "$DIR/glm4-synth-mrope.gguf" ;;
    exaone4-swa)    echo "$DIR/exaone4-synth-swa.gguf" ;;
    llama4-noswa)   echo "$DIR/llama4-synth-noswa.gguf" ;;
    nemotron-h-moe) echo "$DIR/nemotron_h_moe-synth.gguf" ;;
    *)              echo "$DIR/$base-synth.gguf" ;;
  esac
}

# batch-12 cell name -> synth file (the hrm-text cell maps onto the hrm_text
# GGUF arch name; laguna's -full variant sits after -synth; maple-allswa is
# the in-port rope-every-layer variant)
b12_model() {
  local cell="$1" base="${1%-long}"
  case "$base" in
    hrm-text)     echo "$DIR/hrm_text-synth.gguf" ;;
    laguna-full)  echo "$DIR/laguna-synth-full.gguf" ;;
    maple-allswa) echo "$DIR/maple-synth-allswa.gguf" ;;
    *)            echo "$DIR/$base-synth.gguf" ;;
  esac
}

# One parity cell for the env-driver batches (3/4): fresh reference server +
# first request, then the port's env-driven greedy driver printing llama-cli's
# debug format, then the tokenizer-agreement check and arch_batch_cmp.py.
# Args: cell fa model prompt tag ENV_PREFIX TEST_FILE TEST_NAME
run_env_driver_cell() {
  local cell="$1" fa="$2" model="$3" prompt="$4" tag="$5" eprefix="$6" testfile="$7" testname="$8"
  echo "=== $cell -fa $fa"
  # 1) fresh reference server + first request (run_cli_arch_parity.sh's
  #    protocol, minus the CLI half)
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 2
  "$REF/llama-server" -m "$model" -c "${CTX:-512}" -t "${THREADS:-8}" -fa "$fa" \
    --port "$PORT" --host 127.0.0.1 >"/tmp/parity-$tag-server.log" 2>&1 &
  local srv=$!
  trap 'kill "$srv" 2>/dev/null || true' EXIT
  for _ in $(seq 1 300); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 1
  done
  python3 parity/cli_parity_cmp.py body tokens "$prompt" "" "$N" \
    >"/tmp/parity-$tag-body.json"
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d @"/tmp/parity-$tag-body.json" -o "/tmp/parity-$tag-ref.json"
  # sanity: the reference's own tokenization of the prompt
  curl -s "http://127.0.0.1:$PORT/tokenize" -H 'Content-Type: application/json' \
    -d "{\"content\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$prompt"),\"add_special\":true}" \
    -o "/tmp/parity-$tag-tokens.json"
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  trap - EXIT

  # 2) the port side: the env-driven greedy driver (same debug format the
  #    comparator already parses for llama-cli)
  env "${eprefix}_MODEL=$model" "${eprefix}_FA=$fa" "${eprefix}_PROMPT=$prompt" "${eprefix}_N=$N" \
    cargo test -q --release -p llama --test "$testfile" -- --ignored --nocapture "$testname" \
    >"/tmp/parity-$tag-cli.out" 2>"/tmp/parity-$tag-cli.err" || true

  # 3) tokenization agreement (both sides must see the same prompt ids)
  python3 - "$tag" <<'PYEOF'
import json, re, sys
tag = sys.argv[1]
ref = json.load(open(f"/tmp/parity-{tag}-tokens.json"))["tokens"]
out = open(f"/tmp/parity-{tag}-cli.out").read()
m = re.search(r"prompt tokens \((\d+)\) \[(.*?)\]", out)
port = [int(x) for x in m.group(2).split(",")] if m else []
if port != ref:
    print(f"{tag:>18}: TOKENIZER MISMATCH ref={ref[:12]}... port={port[:12]}...")
    sys.exit(1)
PYEOF
  [ $? -ne 0 ] && FAIL=1

  python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
}

if [ -n "${ARCH_BATCH15:-}" ]; then
  # batch 15 drives llama-cli itself (batch-1 protocol) — like batch 6-14,
  # the batch's ForwardWeights/CLI/server arms landed with the graph.
  # "-long" runs the >64-token prompt cell (the iswa windows at n_swa 32,
  # step35's halved rope, hy-v4's lid rows + shared-indexer top-k).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b15_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" 2>&1
      FAIL=1
      continue
    fi
    for fa in off on; do
      # plm fa=on: the pinned reference ABORTS at load — plm.cpp's 2D Vcur
      # ([n_embd_head_v*n_head, T], plm.cpp:131-133) hits ggml_set_rows'
      # ne[0] assert inside build_attn's FA-only V path (ggml.c:4005); the
      # port re-3D-shapes the contiguous rows (value-identical — the -fa off
      # cell is bit-exact 16/16)
      if [ "$base" = "plm" ] && [ "$fa" = "on" ]; then
        echo "=== $a -fa on (llama-cli) SKIPPED — reference set_rows assert on the 2D Vcur (see PARITY.md 批次 15)"
        continue
      fi
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH14:-}" ]; then
  # batch 14 drives llama-cli itself (batch-1 protocol) — like batch 6-13,
  # the batch's ForwardWeights/CLI/server arms landed with the graph. "-long"
  # runs the >64-token prompt cell (the RWKV token-shift + WKV recurrence
  # across ~100 tokens in one ubatch).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b14_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" 2>&1
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH13:-}" ]; then
  # batch 13 drives llama-cli itself (batch-1 protocol) — the batch's
  # ForwardWeights/CLI arms landed with the graph. "-long" runs the
  # >64-token prompt cell (the iswa windows at n_swa 32, llama4's chunked
  # pattern, glm-dsa's lid rows).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b13_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" 2>&1
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH12:-}" ]; then
  # batch 12 drives llama-cli itself (batch-1 protocol) — like batch 6-11,
  # the batch's ForwardWeights/CLI arms landed with the graph. "-long" runs
  # the >64-token prompt cell (positions past the n_swa 64 windows of
  # laguna / maple, hrm-text's state across ~100 tokens).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b12_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" 2>&1
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH11B:-}" ]; then
  # batch 11b drives llama-cli itself (batch-1 protocol) — like batch 6-10,
  # the batch's ForwardWeights/CLI arms landed with the graph. "-long" runs
  # the >64-token prompt cell (rope past 64, nanbeige's loop boundary norm
  # mid-stack on the aliased slots).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b11b_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH11A:-}" ]; then
  # batch 11a drives llama-cli itself (batch-1 protocol) — like batch 6-10,
  # the batch's ForwardWeights/CLI arms landed with the graph. "-long" runs
  # the >64-token prompt cell: dots3note's SWA window edge + lid rows,
  # minimax-m3's MSA block selection (fa on) / dense fallback (fa off), and
  # the GDN/KDA hybrids' recurrence past the prompt.
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b11a_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      # minimax-m3 fa=on: the port is BIT-EXACT against the reference
      # library's own decode (parity/ref_decode_dump.c prefill + a 40-step
      # decode-tail replaying the exact greedy ids — every computing node
      # identical, PARITY.md 批次 11a), but the reference SERVER (4-slot
      # unified cache, "graphs reused = 47", with -fit on AND off alike)
      # computes a different trajectory from step ~38 of the 6-token prompt
      # cell — the reference disagrees with itself between the two drivers,
      # so there is no server output to compare against. fa=off is the
      # reference's documented DENSE fallback (minimax-m3.cpp:239-244) and
      # IS compared.
      if [ "$base" = "minimax-m3" ] && [ "$fa" = "on" ]; then
        echo "=== $a -fa on (llama-cli) SKIPPED — reference server/lib divergence (see PARITY.md 批次 11a)"
        continue
      fi
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH10:-}" ]; then
  # batch 10 drives llama-cli itself (batch-1 protocol) — like batch 6-9, the
  # batch's ForwardWeights/CLI arms landed with the graph. "-long" runs the
  # >64-token prompt cell: llada-moe's [T, T] non-causal mask over ~100
  # tokens, minimax-01's diag_decay matrix + the decays at pos_rel ~100.
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b10_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH9:-}" ]; then
  # batch 9 drives llama-cli itself (batch-1 protocol) — like batch 6-8, the
  # batch's ForwardWeights/CLI arms landed with the graph. "-long" runs the
  # >64-token prompt cell: the KDA conv-state slide + recurrent state across
  # the ubatch boundary, plamo3's positions past the n_swa 64 window.
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b9_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH8:-}" ]; then
  # batch 8 drives llama-cli itself (batch-1 protocol) — like batch 6/6b/7,
  # the batch's ForwardWeights/CLI arms landed with the graph. "-long" runs
  # the >64-token prompt cell: positions past the n_swa 64 window of
  # cohere2moe / exaone-moe (and rope >64 for the rest).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b8_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH7:-}" ]; then
  # batch 7 drives llama-cli itself (batch-1 protocol) — like batch 6, the
  # batch's ForwardWeights/CLI arms landed with the graph. "-long" runs the
  # >64-token prompt cell (positions past the 64-token SWA window, CSA block
  # boundaries at every 4th token, the HCA one at 128).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b7_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH6B:-}" ]; then
  # batch 6b drives llama-cli itself (batch-1 protocol) — like batch 6, the
  # batch's ForwardWeights/CLI arms landed with the graph. "-long" runs the
  # >64-token prompt cell (jais's ALiBi distances / the rope of grok+deci /
  # the hybrids' recurrence past the ubatch boundary).
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b6b_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH6:-}" ]; then
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b6_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa (llama-cli)"
      ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
elif [ -n "${ARCH_BATCH5:-}" ]; then
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b5_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      if [ -n "${ARCH_BATCH5_CLI:-}" ]; then
        # the integrator's wiring: batch 5 reached `ForwardWeights`/llama-cli,
        # so drive the release CLI itself (batch-1's protocol)
        tag="$a-$fa"
        echo "=== $a -fa $fa (llama-cli)"
        ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$prompt" "$N" \
          | tail -3
        python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
      else
        run_env_driver_cell "$a" "$fa" "$model" "$prompt" "$a-$fa" ARCH5 arch_batch5_e2e arch5_cli_driver
      fi
    done
  done
elif [ -n "${ARCH_BATCH4:-}" ]; then
  for a in "${ARCHS[@]}"; do
    model="$(b4_model "$a")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      if [ -n "${ARCH_BATCH4_CLI:-}" ]; then
        # the integrator's wiring: batch 4 reached `ForwardWeights`/llama-cli,
        # so drive the release CLI itself (batch-1's protocol via
        # run_cli_arch_parity.sh: fresh reference server + first request vs
        # LLAMA_RUST_DEBUG=1 ./target/release/llama-cli)
        tag="$a-$fa"
        echo "=== $a -fa $fa (llama-cli)"
        ./parity/run_cli_arch_parity.sh tokens "$model" "$fa" "$PORT" "$tag" "$PROMPT" "$N" \
          | tail -3
        python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
      else
        run_env_driver_cell "$a" "$fa" "$model" "$PROMPT" "$a-$fa" ARCH4 arch_batch4_e2e arch4_cli_driver
      fi
    done
  done
elif [ -n "${ARCH_BATCH3:-}" ]; then
  for a in "${ARCHS[@]}"; do
    base="${a%-long}"
    prompt="$PROMPT"
    [ "$base" != "$a" ] && prompt="$LONG_PROMPT"
    model="$(b3_model "$base")"
    if [ ! -f "$model" ]; then
      echo "$a: $model missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      run_env_driver_cell "$a" "$fa" "$model" "$prompt" "$a-$fa" ARCH3 arch_batch3_e2e arch3_cli_driver
    done
  done
else
  for a in "${ARCHS[@]}"; do
    MODEL="$DIR/$a-synth.gguf"
    if [ ! -f "$MODEL" ]; then
      echo "$a: $MODEL missing — run the generator test first" >&2
      FAIL=1
      continue
    fi
    for fa in off on; do
      tag="$a-$fa"
      echo "=== $a -fa $fa"
      ./parity/run_cli_arch_parity.sh tokens "$MODEL" "$fa" "$PORT" "$tag" "$PROMPT" "$N" \
        | tail -3
      python3 parity/arch_batch_cmp.py "$tag" || FAIL=1
    done
  done
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "arch batch parity: all runs matched"
else
  echo "arch batch parity: at least one run diverged (see above / PARITY.md)"
fi
exit "$FAIL"
