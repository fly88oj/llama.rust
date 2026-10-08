# AUDIT: src/models/ 逐文件逐符号迁移审计（pinned bd4f514db1）

审计日期: 2026-09-24（对照当日端口树）。审计人: 代理AUDIT-MODELS。
范围: 参考 `src/models/`（同步批次 A 起 = NEW 基线 `/home/jeffrey/llm/llama.cpp-next` @ def4d406a）全部 **156 个 .cpp 文件**（旧基线 bd4f514db1 为 155; def4d406a 新增 glm5-next.cpp）+ `src/models/models.h`（148 个 `llama_model_*` 类 + 4 个 `llm_build_*_base` 类）+ `src/llama-arch.cpp` 三张表（LLM_ARCH_NAMES / LLM_TENSOR_NAMES / LLM_KV_NAMES）。
端口落点: `crates/llama/src/model.rs`（load_arch_tensors, 95 arch 臂 + `arch_tensors_support`）、`graph_arch.rs`（118 个 `build_*_forward` + MTP/草稿图 + helper）、`eagle.rs` / `dflash.rs`（草稿头自有装载+图）、`meta.rs`（hparams 臂）、`crates/tools/llama-cli` / `llama-server`（forward_weights 臂）。

## 判定图例

| 码 | 含义 |
|---|---|
| ✅ | 全符号 1:1 移植（loader + graph + helper + CLI + server；括注=已开档的子项缺口，非静默） |
| ✅+ | 装载-only 文件（C 里 `using graph = 父类::graph`），其图宿主已移植 |
| 🟡 | 部分移植：trunk/一侧已移植，明确列出的子符号未移植（有出处） |
| ⛔ | documented-skip：PARITY.md 有明文理由，整档未移 |
| ❌ | **MISSED**：无端口符号、无 PARITY.md 记录（或记录仅涉及周边而非本文件） |

**汇总: 157 文件 = ✅130 + ✅+7 + 🟡0**（同步批次 A2 (2026-10-05, a7b94df2c): **+clef.cpp ✅**（决策模型 arch 全量, 合成文件全模型分数 vs NEW 参考位精确）+ qwen4exp QSA→kpool 重写（节点流位同 NEW 参考 ×2FA）+ modern-bert 决策头 + qwen35 cls_out 投影 + glm5-next 散射掩码改型（glm5 位比对不变）; 批次 19 (2026-10-02) glm5-next.cpp 🟡→✅; MTP 批次 17 后 🟡=0, ❌=0, ⛔=0; 下方"已知状态"行为批次 15 时的历史快照）。（批次 15 后 — 本批移走 19 个 ❌（其中 hy-v3/mimo2/step35 以 🟡 入列：trunk + MTP 张量加载、graph_mtp 开档），gemma4-assistant 以 ⛔ 入列；llada 参考无生成路径按家族先例 ✅。**批次 15 §integrator 后续**: mimo2/step35 修复晋升默认集（16/16 ×2FA 含 -long），gemma4-assistant ⛔→🟡 已移（gemma4_assistant.rs 全模块 + mem-shared 草稿驱动, 真机对位精确））
对照"已知状态"声明（"全部 ~156 文件都有 builder"）: **不成立** — 43 个文件没有任何 builder/loader/CLI/server 臂，其中大多数连 PARITY.md 记录都没有。

## MISSED 清单（43 文件, 按 triage）

### P0 — 有真实公开 GGUF、用户会直接撞上的拒载
| 文件 | 缺失符号（file:line） | 备注 |
|---|---|---|
| llama4.cpp | load_arch_hparams:3 / load_arch_tensors:44 / graph<iswa>:105-269 | **已移 (arch batch 13)**: LLAMA4 臂 + build_llama4_forward (iswa/CHUNKED + 温度缩放 + no-rope 层 + sigmoid MoE + 共享专家), CLI/server 臂, 合成文件 16/16 双 FA (parity/arch_batch13)。收尾轮: MoE 专家权重改**前乘**（llama-graph.cpp:2018 的 LLM_ARCH_LLAMA4 特例, `weight_before_ffn` — 修复前 -long 格 0.018-nat 近平局翻牌）; llama4-noswa 变体移出默认集（pinned 参考自身 abort, llama-graph.cpp:161 断言） |
| qwen3vl.cpp | ldh:3 / ldt:16 / graph:60-196 | **已移 (arch batch 13)**: QWEN3VL 臂 + build_qwen3vl_forward (IMRoPE + q/k norm + deepstack 零填充), 合成文件 16/16 双 FA |
| qwen3vlmoe.cpp | ldh:3 / ldt:16 / graph:64-189 | **已移 (arch batch 13)**: QWEN3VLMOE 臂 + 同一 build_qwen3vl_forward 的 MoE 分支 (softmax), 合成文件 16/16 双 FA |
| qwen2vl.cpp | ldh:3 / ldt:8 / graph:42-142 | **已移 (arch batch 13)**: QWEN2VL 臂 + build_qwen2vl_forward (MRoPE 必需 sections, n_embd 宽 Q), 合成文件 16/16 双 FA |
| ~~rwkv6.cpp / rwkv6qwen2.cpp / rwkv6-base.cpp~~ | ~~缺失~~ | ✅ 批次 14 已移; 2026-09-28 起图直调融合 WKV6/GLA 内核（wkv.rs, 位一致）, rwkv6/rwkv6-long 48/48 ×2FA 0.0000 |
| ~~rwkv7.cpp / arwkv7.cpp / rwkv7-base.cpp~~ | ~~缺失~~ | ✅ 批次 14 已移; 2026-09-28 起图直调融合 WKV7 内核, rwkv7 + rwkv7-long（晋升默认集）48/48 ×2FA 0.0000; **arwkv7 存在真实步 0 分歧（~0.18 nat, -t 1 双侧确定）, 仍移出默认集（PARITY.md 批次 14 节新证据）** |
| ~~gemma3n.cpp~~ | ~~缺失~~ | ✅ 批次 14 已移（KV-reuse 为 builder 侧复用） |
| glm4.cpp | ldt:15 / graph:68-184（hparams 臂已在 meta.rs） | **已移 (arch batch 13)**: GLM4 臂 + build_glm4_forward (post-norm + SWIGLU-SEQ + 可选 M-RoPE), 合成文件 16/16 双 FA (含 -mrope 变体) |
| glm-dsa.cpp | ldh:29 / ldt:74 / graph:196 / graph_mtp:539（769 行全缺） | **已移 (arch batch 13)**: GLM_DSA 臂 + build_glm_dsa_forward (DSA 闪电索引器 + 共享索引器层 + sigmoid MoE 宽共享专家) + build_glm_dsa_mtp_forward + MtpForward::GlmDsa, 合成文件 16/16 双 FA (含 -shared 变体) |
| chatglm.cpp | ldh:3 / ldt:25 / graph:58 | **已移 (arch batch 13)**: CHATGLM 臂 + build_chatglm_forward (SWIGLU-SEQ ffn_up {n_embd,2*n_ff}), 合成文件 16/16 双 FA |
| mistral3.cpp | ldh:3 / ldt:29 / graph:93 | **已移 (arch batch 13)**: MISTRAL3 臂 + build_mistral3_forward (可选温度缩放/长rope因子/MoE 尾), 合成文件 16/16 双 FA (含 -temp 变体) |
| cohere2.cpp | ldh:3 / ldt:21 / graph:50 | **已移 (arch batch 13)**: COHERE2 臂 + build_cohere2_forward (LLM_NORM + 仅 SWA 层 rope + 双残差 + logit_scale, iswa), 合成文件 16/16 双 FA |
| minicpm3.cpp | ldh:3 / ldt:14 / graph:63 | **已移 (arch batch 13)**: MINICPM3 臂 + build_minicpm3_forward (半-MLA: k_pe 跨头重复 + rope_set_offset + scale_embd/scale_depth), 合成文件 16/16 双 FA |
| exaone4.cpp | ldh:3 / ldt:24 / graph<iswa>:81-190 | **已移 (arch batch 13)**: EXAONE4 臂 + build_exaone4_forward (QK-norm 先于 rope, 无 attn_norm, iswa), 合成文件 16/16 双 FA (含 64 层 -swa 变体) |
| bitnet.cpp | ldh:3 / ldt:12 / graph:51 | **已移 (arch batch 13)**: BITNET 臂 + build_bitnet_forward (子范数 + .scale 权重 + tok_embd 绑定头), 合成文件 16/16 双 FA |
| dbrx.cpp | ldh:3 / ldt:13 / graph:47 | **已移 (arch batch 13)**: DBRX 臂 + build_dbrx_forward (钳位融合 QKV + L2 norm + softmax MoE), 合成文件 16/16 双 FA |
| ernie4-5.cpp | ldh:3 / ldt:23 / graph:76 | **已移 (arch batch 13)**: ERNIE4_5 臂 + 复用 build_ernie45_moe_forward 全稠密分支 (n_layer_dense_lead = n_layer, 与 C 的 arch 守卫等价), 合成文件 16/16 双 FA |

### P1 — 有公开模型但用户面较窄 / 继承 typedef 缺臂
| 文件 | 缺失符号 | 备注 |
|---|---|---|
| step35.cpp | ldh:3 / ldt:37 / graph:191 / graph_mtp:368 | **已移 (arch batch 15; §integrator 修复后晋升默认集)**: STEP35 臂 + build_step35_forward (iswa + 半维 rope + 可选 QK-norm + 逐头 attn 门 + sigmoid MoE + shexp), MTP 张量随装载臂加载 (graph_mtp ✅ 已移, MTP 批次 17 逐位相同), 16/16 双 FA 含 -long (路由偏置根因已修, 节点 dump 位同) |
| mimo2.cpp | ldh:3 / ldt:25 / graph:91 / graph_mtp:261 | **已移 (arch batch 15; §integrator 修复后晋升默认集)**: MIMO2 臂 + build_mimo2_forward (iswa + sinks + value_scale + sigmoid MoE), MTP 张量加载 (graph_mtp ✅ 已移, MTP 批次 17); 16/16 双 FA 含 -long — 复跑 iswa 层 kq ~3e-3 实差的根因（路由偏置进 gating 前 + swiglu 尾道 ulp）已修, 节点 dump 位同 (PARITY 批次 15) |
| hunyuan-vl.cpp + hunyuan-dense.cpp | hunyuan-vl ldh:3/ldt:23/graph:60; dense = 纯 typedef | **已移 (arch batch 15)**: HUNYUAN_VL(+HUNYUAN_DENSE 共臂) + build_hunyuan_vl_forward (可选 M-RoPE sections + XDRoPE alpha 重基 + rope 后 QK-norm), 两 arch 名合成文件 16/16 双 FA (-mrope 变体在档) |
| hy-v3.cpp | ldh:3 / ldt:22 / graph:106 / graph_mtp:239 | **已移 (arch batch 15)**: HY_V3 臂 + build_hy_v3_forward (sigmoid MoE + 裸名 exp_probs_b + shexp), MTP 张量加载 (graph_mtp ✅ 已移, MTP 批次 17), 合成文件 16/16 双 FA |
| hy-v4.cpp | ldt:70 / graph:497 + 6 helper（hparams 臂已在 meta.rs HY_V4） | **已移 (arch batch 15)**: HY_V4 装载臂 + hparams 臂 + build_hy_v4_forward (iHC 2*hc 系数 + 门控 MLA + 可选 DSA 索引器 + sigmoid MoE 宽 shexp; hy_v4_hc_pre/post/head/reduce + hy_v4_attn_mla 本地 helper), 合成文件 16/16 双 FA (-dsa 变体在档, 含 -long) |
| spark2-5.cpp | ldh:3 / ldt:20 / graph:56 | **已移 (arch batch 15)**: SPARK2_5 臂 + build_spark25_forward (iswa + 逐头 sigmoid attn 门 + GELU FFN), 合成文件 16/16 双 FA (含 -long) |
| muse-glimmer.cpp | ldh:3 / ldt:21 / graph:57 | **已移 (arch batch 15)**: MUSE_GLIMMER 臂 + build_muse_glimmer_forward (反转 rope 规则 + 宽 attn 门 + 双 1e-8 后置 norm + logit_scale/tanh softcap), 合成文件 16/16 双 FA (-softcap 变体在档) |
| llada.cpp | ldh:3 / ldt:19 / graph:66 | **已移 (arch batch 15)**: LLADA 臂 + build_llada_forward (无 cache 非因果扩散注意力; 参考无生成路径, 端口内验证 — llada-moe/dream/rnd1 先例) |
| cogvlm.cpp | ldh:3 / ldt:12 / graph:53 | **已移 (arch batch 15)**: COGVLM 臂 + build_cogvlm_forward (文本路径 + visexp 双权重装载; 朴素 ggml_rope) — 图与参考**库**位精确 (224k logits), 参考服务器与自身库分歧 (minimax-m3 先例, PARITY 批次 15) |
| granite-swa.cpp | ldh:5 / ldt:56 / graph:121 + build_attention_layer:200/build_layer_ffn:241 | **已移 (arch batch 15)**: GRANITE_SWA 臂 + build_granite_swa_forward (iswa + sinks + granite 三 scale + softmax MoE + SWIGLU 融合 shexp + deepstack 注入), 合成文件 16/16 双 FA (含 -moe/-long) |
| plm.cpp | ldh:3 / ldt:13 / graph:48 | **已移 (arch batch 15)**: PLM 臂 + build_plm_forward (共享 k_pe 跨头重复 + rope_set_offset + relu² FFN + 绑定头), 合成文件 16/16 (fa off; 参考 fa on 在装载时 set_rows 断言 — 见 PARITY 批次 15) |
| pangu-embed.cpp | ldh:3 / ldt:13 / graph:58 | **已移 (arch batch 15)**: PANGU_EMBED 臂 + build_pangu_embed_forward (必需 wo 偏置), 合成文件 16/16 双 FA |
| maincoder.cpp | ldh:3 / ldt:12 / graph:47 | **已移 (arch batch 15)**: MAINCODER 臂 + build_maincoder_forward (rope 后 QK-norm), 合成文件 16/16 双 FA |
| minicpm3.cpp（见 P0） | | |
| arwkv7.cpp（见 P0） | | |

### P2 — 小众/研究型/嵌入型
| 文件 | 缺失符号 | 备注 |
|---|---|---|
| afmoe.cpp | ldh:3 / ldt:38 / graph:107 | **已移 (arch batch 15)**: AFMOE 臂 + build_afmoe_forward (MuP sqrt 缩放 + 双 norm + 宽 attn 门 + (il+1)%4 rope 规则 + sigmoid MoE + 宽 shexp), 合成文件 16/16 双 FA |
| qwen.cpp | ldh:3 / ldt:13 / graph:43 | **已移 (arch batch 15)**: QWEN 臂 + build_qwen1_forward (融合 wqkv+bias + 半宽 SwiGLU), 合成文件 16/16 双 FA |
| exaone4.cpp（见 P0） | | |
| gemma-embedding.cpp | ldh:3 / ldt:30 / graph:73 | **已移 (arch batch 15)**: GEMMA_EMBEDDING 臂 + build_gemma_embedding_forward (对称 SWA 无 cache 编码器 + 双后置 norm + Q 预缩放; EncoderContext/kq_mask_swa) — 参考自身 llama_encode 崩溃 (GGML_ASSERT(buffer), ggml-backend.cpp:205), 端口内验证 |
| gemma4-assistant.cpp | ldh:3 / ldt:21 / graph:84 | **🟡 已移 (arch batch 15 §integrator)**: crates/llama/src/gemma4_assistant.rs — ldh/ldt 逐行 + graph + attach/decode_gemma4_assistant（ctx_other 的端口内倒置: 头张量物化进主干 Context, 草稿图视图 share 映射层的主干 iswa KV, llama-model.cpp:2698-2703）+ CommonSpeculativeImplDraftMtp 的 is_mem_shared 模式 + CLI/server 接线; 真机对（26B-A4B 主干 + 重钉名头副本）5 步草稿位精确 ×2FA（parity/ref_gemma4_assistant_dump.c, 参考侧三陷阱见 PARITY.md 批次 15） |
| mellum.cpp | ldh:3 / ldt:27 / graph<iswa>:74-219 | **已移 (arch batch 15)**: MELLUM 臂 + build_mellum_forward (全 MoE softmax norm_w + SWA 层朴素 rope; 注: 是因果解码器不是编码器), 合成文件 16/16 双 FA (-swa 变体在档) |
| mistral4.cpp | build_arch_graph（typedef of deepseek2） | **已移 (arch batch 15)**: MISTRAL4 与 DEEPSEEK2 共 hparams/装载/图臂 (models.h:1393-1395), 合成文件 (MLA 配方) 16/16 双 FA |
| llama-embed.cpp | build_arch_graph（graph<embed=true> typedef of llama） | **已移 (arch batch 15)**: LLAMA_EMBED 与 LLAMA 共 hparams/装载臂 + build_llama_embed_encoder (无 cache 因果图<true>; llama_encode 强制非因果 mask, llama-context.cpp:1526-1529) — 参考 llama_encode dump 位精确 |
| paddleocr.cpp | graph:7-107（hparams 臂已在 meta.rs；张量装载继承 ernie4_5 亦缺） | **已移 (arch batch 15)**: PADDLEOCR 与 ERNIE4_5 共 hparams/装载臂 (models.h:1984-1986) + build_paddleocr_forward (M-RoPE sections), 合成文件 16/16 双 FA |
| t5.cpp 的 decoder 半边 | graph<false>:110-262 + dec_* 张量装载（graph<true> 已移） | 归入 🟡 见下 |

注：43 个 ❌ 文件 = P0 21（真实公开 GGUF 会直接撞上拒载）+ P1 14（小众/typedef 缺臂）+ P2 8（研究型/嵌入型/typedef）。t5.cpp 的 decoder 半边归入 🟡（trunk=encoder 已移）。

## 🟡 部分移植（trunk 已移, 子符号未移）—— **清零（MTP 批次 17, 2026-09-29）**

全部 8 个 🟡 行已在下方逐文件表晋升 ✅（9 个 graph_mtp 构建器 + t5 解码器 + lfm2 稠密装载臂 + bailingmoe3 的 MTP 装载循环 + qwen3next 的 nextn 三元组; 接受准则 = 合成 nextn/enc-dec 文件上 vs 参考**自身** MTP/编码-解码上下文的 t_logits/t_h_nextn 逐位相同, parity/{ref_mtp2_dump.c,ref_t5_dec_dump.c} + tests/{mtp2,t5_dec}_e2e.rs, PARITY.md MTP 批次 17 节）。先前 12 个 🟡 计数中其余 3 行（hy-v3/mimo2/step35 在 P1 段内注记）与 1 行记号问题（gemma4-assistant: 该文件本无 graph_mtp, 全档即 MTP 头, 真机对位早已位同）同批处理。

（deepseek2 / deepseek32 / deepseek4 的 graph_mtp **已移植**：build_deepseek{2,32,4}_mtp_forward + MtpForward 分发 — "MTP 三连"。）

## ⛔ documented-skip（10 文件 → **全部清零**, 2026-09-28）
| 文件 | PARITY 依据 |
|---|---|
| ~~jina-bert-v2.cpp / jina-bert-v3.cpp~~ | ✅ 批次 16 已移（装载臂 + bert::graph 键控分支; 合成文件位对齐 768/768 — PARITY.md 批次 16 节; 原 :381/:638 记录作废） |
| ~~nomic-bert.cpp / nomic-bert-moe.cpp~~ | ✅ 批次 16 已移（nomic 的 gate/moe_every_n_layers; MoE 格位对齐 768/768） |
| ~~neo-bert.cpp~~ | ✅ 批次 16 已移（enc.output_norm + 融合 QKV + SWIGLU; 768/768） |
| ~~modern-bert.cpp~~ | ✅ 批次 16 已移（对称 SWA + GEGLU/hidden_activation=silu + GTE rank 头 cls/cls.norm/cls.output; none/mean/rank/silu 四格位对齐） |
| ~~qwen3tts.cpp~~ | **已移 (音频批次 5, 2026-09-28)**: 原 :4577 记录作废 — QWEN3TTS 与 QWEN3VL 共装载/图臂（models.h:625-627 纯 typedef）+ tts 头窄化 3072（qwen3vl.cpp:19-23）; 合成文件参考 llama_decode 位精确 ×fa。生成器半边（qwen3tts_gen 图/驱动）同期全量入 clip.rs/mtmd.rs — PARITY.md 音频批次 5 节 |
| ~~pockettts.cpp~~ | **已移 (音频批次 5, 2026-09-28)**: 原 :4577 记录作废 — POCKETTTS 装载臂 + build_pockettts_forward; 合成文件 32000 logits ×fa/nofa 0.0000（tts_parity.sh A 节） |
| ~~wavtokenizer-dec.cpp~~ | **已移 (音频批次 5, 2026-09-28)**: 原 :4577 记录作废 — WAVTOKENIZER_DEC 装载臂 + build_wavtokenizer_dec_forward + group_norm_lm 组合内核（初版 flat 序按 numpy 行主序排组 — 修复后 t_embd 波形 64/64 位同 vs 参考, float64 numpy 裁判定位, PARITY.md 音频批次 5 节） |
| gemma4-assistant.cpp | **已移 (批次 15 §integrator)**: gemma4_assistant.rs 全模块（ldh/ldt/graph + mem-shared 草稿驱动）; 参考侧三陷阱（spec 驱动装载错路径 common/speculative.cpp:2562 / 测试套 FIXME / 本机文件新版命名）与真机对位精确证据见 PARITY.md 批次 15 |
| nemotron-h-moe.cpp（⛔ → ✅+⛔） | graph_mtp ⛔ 已档（批次 5 §5 "MTP/NextN 图未移植"）；~~未档缺口~~ **已补 (arch batch 13)**: NEMOTRON_H_MOE 进 arch_tensors_support/load_arch_tensors/hparams（与 NEMOTRON_H 共臂，models.h:1539 继承）+ CLI/server 路由 NemotronH 图；真实 arch=nemotron_h_moe 合成文件 16/16 双 FA |

## Wiring 缺口（已移架构的接线洞, 非文件缺失）
1. ~~**llama-cli 无 BERT / EUROBERT 臂**~~ **已补 (arch batch 13)**：llama-cli `-fe/--embedding` + `--pooling` + `--embd-ids` 编码器模式（EncoderContext / llama_encode 路径，与 llama-server 的 -fe 分支同一构造；pinned 参考版自己的 cli 拒收该旗标 arg.cpp:824，故按参考 llama_encode dump + 参考 server /embedding 验证）。
2. ~~**llama-cli 与 llama-server 都无 T5ENCODER 臂**~~ **已补 (arch batch 13)**：两端都接 EncoderWeights::T5Encoder。
3. ~~**NEMOTRON_H_MOE** 未进 arch_tensors_support/load_arch_tensors/ForwardWeights~~ **已补 (arch batch 13)**（见上）。
4. ~~**LFM2（arch=lfm2 稠密）** 有意 Unsupported~~ **已补（MTP 批次 17）**：lfm2 与 lfm2moe 共装载臂（lfm.cpp 的 load_arch_tensors 与 lfm2moe 同体, n_layer_dense_lead==n_layer 全稠密）, arch_tensors_support(LFM2) → Partial, 钉死测试改写 + 合成稠密文件全张量集测试。
5. port `LlmTensor` 有两个无 template() 的死变体 POST_ATTN_NORM/POST_MLP_NORM（arch.rs:1065-1066，无引用；非 parity 破坏）。

## models.h 审计
- 148 个 `llama_model_*` 类与 156 个 .cpp 一一对应（152 类文件 + 4 个 base 文件：mamba-base / rwkv6-base / rwkv7-base / delta-net-base，NEW 基线 +glm5-next.cpp/llama_model_glm5_next）；无孤儿类（`llama_model_params` 是参数类非模型类）。
- 继承/别名关系全部核对：loader-only 文件（jina×2, nomic×2, mamba2, granite-moe, t5encoder, deepseek2ocr, minicpm, lfm2moe, phimoe, mistral4, qwen3tts, hunyuan-dense, llama-embed, nemotron-h-moe, paddleocr, ernie4-5-moe）在端口按"共臂/复用 builder"落地，与 C 的 `using graph =` 一致。
- base 类 4 个：mamba-base ✅、delta-net-base ✅（fused 路径; 非 fused 已档）、rwkv6-base ✅（批次 14: build_rwkv6_time_mix / build_rwkv6_channel_mix + 组合 WKV6/GLA 扫描; 融合算子 integrator 项）、rwkv7-base ✅（批次 14: build_rwkv7_time_mix / build_rwkv7_channel_mix + 组合 WKV7 扫描; 同上）。

## llama-arch.cpp 三表审计（全 parity）
| 表 | 参考条目 | 端口状态 |
|---|---|---|
| LLM_ARCH_NAMES | 153 | **153/153 名字逐字相同**（arch.rs 的 name() 臂），无缺无错 |
| LLM_TENSOR_NAMES | 273 | **273/273 模板串逐字相同**（LlmTensor::template()）；装载器引用的每个张量名都存在 → 加载真实文件不会因表缺名而断 |
| LLM_KV_NAMES | 243 | **243/243 模板串在端口中逐字存在** |

## 逐文件表（156 行, 按判定分组）
（符号列缩写: ldh=load_arch_hparams, ldt=load_arch_tensors；graph xN 含模板双实例化）

| 文件 (src/models/) | 行 | 符号 | 判定 | 未移符号 | 端口落点 / 依据 |
|---|---|---|---|---|---|
| apertus.cpp | 171 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | model.rs APERTUS 臂 + graph_arch build_apertus_forward + CLI/server 臂 |
| arcee.cpp | 158 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | ARCEE 臂 + build_arcee_forward（批次 11b） |
| arctic.cpp | 181 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | ARCTIC 臂 + build_arctic_forward（批次 4） |
| baichuan.cpp | 156 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BAICHUAN 臂 + build_baichuan_forward（批次 3） |
| bailingmoe.cpp | 181 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BAILINGMOE 臂 + build_bailingmoe_forward（批次 8） |
| bailingmoe2.cpp | 212 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BAILINGMOE2 臂 + build_bailingmoe2_forward（批次 8） |
| bert.cpp | 222 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BERT 臂 + build_bert_forward（PARITY BERT 节; llama-server ✓; llama-cli -fe 编码器模式已补, arch batch 13） |
| bloom.cpp | 152 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BLOOM 臂 + build_bloom_forward（批次 3） |
| chameleon.cpp | 205 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | CHAMELEON 臂 + build_chameleon_forward（批次 6b） |
| codeshell.cpp | 154 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | CODESHELL 臂 + build_codeshell_forward（批次 2） |
| command-r.cpp | 145 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | COMMAND_R 臂 + build_command_r_forward（批次 6） |
| deci.cpp | 192 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DECI 臂 + build_deci_forward（批次 6b） |
| deepseek.cpp | 195 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DEEPSEEK 臂 + build_deepseek_forward（批次 6） |
| deepseek2.cpp | 714 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | DEEPSEEK2 臂 + build_deepseek2_forward + build_deepseek2_mtp_forward（MTP 三连） |
| deepseek32.cpp | 727 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | DEEPSEEK32 臂 + build_deepseek32_forward + build_deepseek32_mtp_forward |
| deepseek4.cpp | 1503 | ldh;ldt;graph x1;graph_mtp x1;helpers(13):build_attention,build_attention_impl,build_cs… | ✅ | 0（Sinkhorn 非 fused 回退 :317-352 已档） | DEEPSEEK4 臂 + build_deepseek4_forward + _mtp + build_hc_* / build_lid_top_k 等全部 helper（graph_arch.rs:11512-12690） |
| delta-net-base.cpp | 607 |  | ✅ | 0（llama 层面 chunked 半边不可达: 参考 CPU 构建恒开 fused, llama-context.cpp:233-234, 无用户旗标 — 见下行批次 18 的链级验收） | fused 路径全量（gated_delta_net 算子 + graph_arch.rs:2825-2940 conv/state/recurrent 块）; **MTP 批次 17**: 非 fused 自回归半边 :289-374 已移（build_delta_net_autoregressive, vs 参考 ggml 同链 parity/ref_dnet_ar_dump.c **逐位相同**, tests/mtp2_e2e.rs::mtp2_dnet_ar_bitcompare）; **批次 18**: chunked 半边 :17-287 已移（build_delta_net_chunking, 含 GGML_OP_SOLVE_TRI/DIAG/SET + UNARY-NEG 入 crates/ggml — ops.cpp:10824/5434/4769/ unary-ops.cpp:11 逐线转录, vs 参考 ggml 同链 parity/ref_dnet_ch_dump.c **逐位相同**（GDA CS=64 双 chunk+padding+双 seq、KDA CS=16 三 chunk）, tests/mtp2_e2e.rs::mtp2_dnet_ch_bitcompare + mtp2_tri_ops_bitcompare; solve_tri 单算子在 k≥4 形状与参考 .so 差 ≤6.5e-5 相对误差 — 参考 .so 的 -ffp-contract=fast 按行程数混排 vmulps/vaddss 向量体与 vfmadd231ss 标量尾, 无单一 Rust 形式可复刻, 小 k 形状逐位相同; fused 与 chunked 两路径自身差 ≤2.6e-5 相对误差（实测 o 2.52e-5 / s_new 1.11e-5） = 参考设计内两条 fp 路线之差, 各自均已对位 .so） |
| dflash.cpp | 1044 | ldh;ldt;build_inp_embd_enc + graph<true>+graph<false> + graph_dsv4 | ✅ | 0（M-RoPE 草稿拒载已档） | dflash.rs 全量（loader + graph<true>/<false> + graph_dsv4 + dflash2 conv/selector, PARITY DFLASH2 节） |
| dots1.cpp | 194 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DOTS1 臂 + build_dots1_forward（批次 8） |
| dots3note.cpp | 477 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DOTS3NOTE 臂 + build_dots3note_forward（批次 11a; 该 arch 无 graph_mtp） |
| dream.cpp | 139 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DREAM 臂 + build_dream_forward（批次 11b; 参考无生成路径, PARITY:3450） |
| eagle3.cpp | 339 | ldh;ldt;build_inp_embd_enc + graph<true>+graph<false> | ✅ | 0 | eagle.rs 头加载器 + graph_arch build_eagle3_{encoder,decoder}_forward + speculative.rs 驱动（PARITY EAGLE3 节） |
| ernie4-5-moe.cpp | 134 | graph x1;other:build_arch_graph | ✅ | 0 | ERNIE4_5_MOE 臂 + build_ernie45_moe_forward（批次 4; hparams 继承自 ernie4_5 由 moe 臂一体处理） |
| eurobert.cpp | 125 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（pooling 仅 NONE 验证已档） | EUROBERT 臂 + build_eurobert_forward（批次 11b; llama-cli -fe 编码器模式已补, arch batch 13） |
| exaone-moe.cpp | 240 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | EXAONE_MOE 臂 + build_exaone_moe_forward（批次 8） |
| exaone.cpp | 137 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | EXAONE 臂 + build_exaone_forward（批次 2） |
| falcon-h1.cpp | 210 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | FALCON_H1 臂 + build_falcon_h1_forward（批次 6b, 复用 build_mamba2_mixer） |
| falcon.cpp | 162 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | FALCON 臂 + build_falcon_forward（批次 2） |
| gemma.cpp | 140 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GEMMA 臂 + build_gemma_forward/build_gemma1_forward（批次 2） |
| gemma2.cpp | 176 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GEMMA2 臂 + build_gemma2_forward |
| gemma3.cpp | 224 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | GEMMA3 臂 + build_gemma3_forward（iswa 双实例化合一, ForwardWeights::Gemma3 带 iswa） |
| gemma4.cpp | 499 | ldh;ldt;graph x1;helpers(2):build_inp_per_layer,project_per_layer_inputs;other:build_ar… | ✅ | 0 | GEMMA4 臂 + build_gemma4_forward + build_gemma4_per_level 等（graph_arch.rs:2213-2660） |
| gpt2.cpp | 149 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GPT2 臂 + build_gpt2_forward（批次 6） |
| gptneox.cpp | 220 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GPTNEOX 臂 + build_gptneox_forward（批次 6） |
| granite-hybrid.cpp | 303 | ldh;ldt;graph x1;helpers(2):build_attention_layer,build_layer_ffn;other:build_arch_graph | ✅ | 0 | GRANITE_HYBRID 臂 + build_granite_forward（hybrid 参数; build_mamba2_layer） |
| granite-switch.cpp | 428 | ldh;ldt;graph x1;inputs:switch;helpers(4):build_attention_layer,build_layer_ffn,build_s… | ✅ | 0 | GRANITE_SWITCH 臂 + build_graniteswitch_forward + build_switched_lora_delta/_mm + llm_graph_input_switch 等价物（批次 10） |
| granite.cpp | 321 | ldh;ldt;graph x1;helpers(2):build_attention_layer,build_layer_ffn;other:build_arch_graph | ✅ | 0 | GRANITE 臂 + build_granite_forward + build_granite_attention/build_granite_ffn（批次 3） |
| grok.cpp | 224 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GROK 臂 + build_grok_forward（批次 6b） |
| grovemoe.cpp | 194 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GROVEMOE 臂 + build_grovemoe_forward（批次 11a） |
| hrm-text.cpp | 214 | ldh;ldt;graph x1;helpers(1):build_stack;other:build_arch_graph | ✅ | 0 | HRM_TEXT 臂 + build_hrm_text_forward（build_stack 内联, 批次 12） |
| hunyuan-moe.cpp | 188 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | HUNYUAN_MOE 臂 + build_hunyuan_moe_forward（批次 8） |
| internlm2.cpp | 140 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | INTERNLM2 臂 + build_internlm2_forward（批次 2） |
| jais.cpp | 133 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | JAIS 臂 + build_jais_forward（批次 6b） |
| jais2.cpp | 156 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | JAIS2 臂 + build_jais2_forward（批次 11b） |
| jamba.cpp | 199 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | JAMBA 臂 + build_jamba_forward（批次 5） |
| kimi-k3.cpp | 619 | ldh;ldt;graph x1;helpers(4):build_kda_layer,build_latent_moe,build_mla_layer,res_mix;ot… | ✅ | 0 | KIMI_K3 臂 + build_kimi_k3_forward + res_push/res_mix/build_kda_layer/build_mla_layer/build_latent_moe 等价物（批次 11a; fused-qkv 显式拒载 model.rs:3323-3326） |
| kimi-linear.cpp | 563 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | KIMI_LINEAR 臂 + build_kimi_linear_forward（批次 9） |
| laguna.cpp | 332 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | LAGUNA 臂 + build_laguna_forward（批次 12） |
| llada-moe.cpp | 164 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | LLADA_MOE 臂 + build_llada_moe_forward（批次 10; 参考无生成路径已档 PARITY:3314） |
| llama.cpp | 251 | ldh;ldt;graph<false>+graph<true>(embed) | ✅ | 0（rope_factors/LONGROPE 未移已档 graph_arch.rs:104） | LLAMA 臂 + build_llama_forward; graph<true>(embed) 由 embd_layer_inp 抽头等价（graph_arch.rs:126-137） |
| mamba-base.cpp | 305 |  | ✅ | 0 | build_mamba1_mixer/build_mamba2_mixer/build_mamba2_layer（graph_arch.rs:7396-1558 区; PARITY 批次 5 §2 逐行） |
| mamba.cpp | 138 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MAMBA 臂 + build_mamba_forward + build_mamba1_mixer（批次 5） |
| maple.cpp | 151 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MAPLE 臂 + build_maple_forward（批次 12） |
| minimax-01.cpp | 485 | ldh;ldt;graph x1;inputs:la;other:build_arch_graph | ✅ | 0（多序列 la decay 未移已档 graph_arch.rs:16534） | MINIMAX_01 臂 + build_minimax01_forward + llm_graph_input_la 等价物（批次 10） |
| minimax-m2.cpp | 169 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MINIMAX_M2 臂 + build_minimax_m2_forward（批次 8） |
| minimax-m3.cpp | 609 | ldh;ldt;graph x1;inputs:msa;helpers(1):build_attn_msa_fa;other:build_arch_graph | ✅ | 0（QSA 压缩注意力未移已档 graph_arch.rs:19897） | MINIMAX_M3 臂 + build_minimax_m3_forward + MinimaxMsaCache/Step（批次 11a） |
| mpt.cpp | 171 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（Q/K 全宽 norm 分支参考不可达已档 PARITY:1800） | MPT 臂 + build_mpt_forward（批次 3） |
| nanbeige.cpp | 186 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | NANBEIGE 臂 + build_nanbeige_forward（批次 11b） |
| nemotron-h.cpp | 341 | ldh;ldt;graph x1;helpers(2):build_attention_layer,build_ffn_layer;other:build_arch_graph | ✅ | 0（MTP/NextN 图未移已档 PARITY 批次 5 §5） | NEMOTRON_H 臂 + build_nemotron_h_forward + build_ffn_relu_sqr/build_moe_ffn_relu_sqr（MTP 头张量装载已移） |
| nemotron.cpp | 151 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | NEMOTRON 臂 + build_nemotron_forward（批次 6b） |
| olmo.cpp | 143 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | OLMO 臂 + build_olmo_forward（批次 2） |
| olmo2.cpp | 209 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | OLMO2 臂 + build_olmo2_forward（iswa 双实例化合一, 批次 6） |
| olmoe.cpp | 174 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | OLMOE 臂 + build_olmoe_forward（批次 4） |
| openai-moe.cpp | 176 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | OPENAI_MOE 臂 + build_gpt_oss_forward + build_moe_ffn_gpt_oss（gpt-oss） |
| openelm.cpp | 172 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | OPENELM 臂 + build_openelm_forward（批次 4） |
| orion.cpp | 142 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | ORION 臂 + build_orion_forward（批次 2） |
| phi2.cpp | 143 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | PHI2 臂 + build_phi2_forward（批次 6） |
| phi3.cpp | 197 | ldh;ldt;graph<iswa>×2 | ✅ | 0（rope_factors 未移已档 graph_arch.rs:510） | PHI3 臂 + build_phi3_forward（iswa, 批次 1） |
| plamo.cpp | 137 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | PLAMO 臂 + build_plamo_forward（批次 3） |
| plamo2.cpp | 427 | ldh;ldt;graph x1;helpers(2):build_plamo2_attn_layer,build_plamo2_mamba_layer;other:buil… | ✅ | 0 | PLAMO2 臂 + build_plamo2_forward + build_plamo2_{attn,mamba}_layer（批次 6b） |
| plamo3.cpp | 196 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | PLAMO3 臂 + build_plamo3_forward（iswa, 批次 9） |
| qwen2.cpp | 155 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN2 臂 + build_llama/qwen2 前向（锚点架构, Full） |
| qwen2moe.cpp | 195 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN2MOE 臂 + build_qwen2moe_forward（批次 4） |
| qwen3.cpp | 160 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN3 臂 + build_qwen3_forward |
| qwen3moe.cpp | 180 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN3MOE 臂 + build_qwen3moe_forward（批次 4） |
| clef.cpp | 616 | ldh(qwen35 继承);ldt(qwen35 继承+头);graph x1;inputs:decision;helpers(5):build_layer_attn,build_layer_attn_linear,build_head,build_head_attn/norm/ffn | **✅ (同步批次 A2, 99b95488c)** | 0 | crates/llama/src/clef.rs 全模块: CLEF arch（arch.rs 154→155, 决策张量/KV 全表）+ hparams（meta.rs QWEN35 共臂 + decision.routing/joint/head_count + f_norm_eps + n_embd_out=1; 路由/头数入 hparams 的 clef_* 字段 — router_layer 先例）+ 装载（model.rs QWEN35 共臂 + load_clef_head: dec.blk.* 头块[routing 仅 cross+norm_kv, joint 另有 self]、decision.* 投影/范数/评分器、token_types[3]、无后缀的 decision.scales）+ `clef_weights/clef_params` + 无记忆驱动 `ClefState::decode`（因果无缓存掩码 + GDN 零态不回写 — create_memory=nullptr 语义, gemma4_assistant 模块先例）+ `input_decision` 输入（clef_get_spans 跨度 + 池化矩阵/掩码/NaN 状态行, batch.rs 的 decision_order 消费）+ `build_head`（选项×问题跨注意力头 + l2_norm 先验/联合余弦 + 4·n_embd_h 评分器, **新增 ggml l2_norm 算子** — Norm+params[1]==2 编码, C++ 链接位同, f64 累加 CPU 内核）; 验收: 合成文件（2 层 trunk: GDN+门控全注意力; 1 routing + 1 joint 头块）[1, n_tokens] 决策分数 vs NEW 参考 (`llama_process` + `llama_batch_ext_set_decision_order` — /v1/systemone 公共路径; 参考把无记忆 arch 改道 encode, clef.cpp:393 预期) **位精确**（parity/clef_scores_ref.bin + ref_clef_dump.c[staging 符号需 C++ 链接] + gen_clef_ref.sh + tests/clef_e2e.rs; 另有 spans 单测） | ClefModelWeights/ClefParams + LlamaModel::clef_weights/clef_params |
| qwen4exp.cpp | 1295→1567 | ldh;ldt;graph x1;graph_mtp x1;inputs:kpool+ple;helpers;other:build_arch_graph | ✅ | graph_mtp 驱动（图级开档） | **同步批次 A2 (a7b94df2c, 5 commits)**: hparams 补 QSA 单一 compress_ratio 校验 + kpool_row=2/by_order=true（meta.rs）; 装载臂 load_block(il,flags) 重构 + mtp_only + NEXTN_HC_* 张量（eh_proj/enorm/hnorm[带 hc 流]/hc_head_* 尾块, model.rs）; **QSA 重写为 kpool**（上游 66e0c17ee: `llm_graph_input_qsa` 删除, build_qsa_sel/build_inp_kpool 共享 kpool 输入 — raw|pooled 双列 F16 行, 池化键旋转到首成员位置, fused lightning indexer 打分（889edf43d 省一半评分内存）, cumsum dump 行散射掩码（4e2713c16）; graph_arch.rs Qwen4KpoolStep + context.rs 步构建）; 验收: 13 图节点流 vs NEW 参考**位同 0 分歧 ×2FA**（qwen4exp_qsa_nodes{,_fa}_ref.bin 重生成）— 关键坑已修: 池化列视图偏移用存储元素宽度（F16）非 F32; new_pool_pos 的 y/x 行 = 时间位置（参考 is_pos_2d()=n_pos>=3 对 M-RoPE 批恒真, ext.y/x 读广播 1D pos 行, kv_cache.rs）; 批次 19 的 QSA+PLE 其余部分不变。**graph_mtp（c061df198）图级未移** — nextn 张量已装载, 驱动接线（embeddings_nextn 抽头 + MTP 上下文过滤）为开档缺口 | QWEN4EXP 臂 + build_qwen4exp_forward + hc_mix/hc_combine/qkvz/norm_gated/layer_attn(_linear)/layer_ffn |
| refact.cpp | 161 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（MoE 分支参考不可达已档 graph_arch.rs:5743） | REFACT 臂 + build_refact_forward（批次 3） |
| rnd1.cpp | 178 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | RND1 臂 + build_rnd1_forward（批次 11b; 参考无生成路径 PARITY:3450） |
| seed-oss.cpp | 152 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | SEED_OSS 臂 + build_seed_oss_forward（批次 4） |
| smallthinker.cpp | 189 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | SMALLTHINKER 臂 + build_smallthinker_forward（iswa, 批次 10） |
| smollm3.cpp | 153 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | SMOLLM3 臂 + build_smollm3_forward（批次 4） |
| stablelm.cpp | 173 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | STABLELM 臂 + build_stablelm_forward（批次 3） |
| starcoder.cpp | 146 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | STARCODER 臂 + build_starcoder_forward（批次 3） |
| starcoder2.cpp | 159 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | STARCODER2 臂 + build_starcoder2_forward（批次 6） |
| t5encoder.cpp | 45 | ldh;ldt;other:build_arch_graph | ✅ | 0 | T5ENCODER 臂 + build_t5_encoder_forward + load_arch_hparams_t5（PARITY BERT/T5 节; CLI/server 编码器臂已补, arch batch 13） |
| talkie.cpp | 150 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（ffn_down_s 死参数已档 PARITY:3461） | TALKIE 臂 + build_talkie_forward（批次 11b） |
| xverse.cpp | 137 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | XVERSE 臂 + build_xverse_forward（批次 2） |
| clip.cpp | 19 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | C 桩语义 1:1（meta.rs:386 CLIP 早退; 运行时在 clip.rs=多模态代理域; quantize 不经 load_model） |
| deepseek2ocr.cpp | 81 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only 文件; DEEPSEEK2OCR 臂 + Deepseek2Params::is_ocr（deepseek2.cpp:473-494 分支） |
| granite-moe.cpp | 85 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only; GRANITE_MOE 臂（批次 4）+ 复用 build_granite_forward（models.h: using graph = llama_model_granite::graph） |
| lfm2moe.cpp | 86 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only; LFM2MOE 臂 + 复用 build_lfm2_forward（models.h: using graph = llama_model_lfm2::graph<iswa>） |
| mamba2.cpp | 89 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only; MAMBA2 臂 + 复用 build_mamba_forward（models.h: using graph = llama_model_mamba::graph; build_mamba2_mixer） |
| minicpm.cpp | 90 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only; GRANITE|MINICPM 共臂（model.rs:3301）+ 复用 build_granite_forward（批次 3） |
| phimoe.cpp | 56 | ldh;ldt;other:build_arch_graph | ✅+ | 0 | 装载-only; PHIMOE 臂 + build_phimoe_forward（models.h: using graph = llama_model_phi3::graph<iswa>） |
| bailingmoe3.cpp | 541 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_bailingmoe3_forward；**MTP 批次 17**: build_bailingmoe3_mtp_forward（门控 MLA + K-only cache + LAYER_OUT_NORM 头范数, bailingmoe3.cpp:126-159 的 MTP 装载循环补齐）+ 合成 nextn 文件 12 步草稿链 vs 参考自身 ctx_type=MTP 上下文 t_logits/t_h_nextn **逐位相同**（parity/ref_mtp2_dump.c, tests/mtp2_e2e.rs） |
| cohere2moe.cpp | 440 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_cohere2moe_forward；**MTP 批次 17**: build_cohere2moe_mtp_forward（RMS-vs-LN 范数开关 + 范数输入 FFN + 0.5 共享专家 + 双残差 + logit_scale）—— 逐位相同（同上探针/驱动） |
| glm4-moe.cpp | 445 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_glm4_moe_forward；**MTP 批次 17**: build_glm4_moe_mtp_forward —— 逐位相同（同上） |
| lfm2.cpp | 300 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | builder ✅ build_lfm2_forward（iswa）+ ForwardWeights::Lfm2; **MTP 批次 17**: 装载臂补齐 — lfm2 与 lfm2moe 共臂（lfm2.cpp 的 load_arch_tensors 与 lfm2moe 同体, n_layer_dense_lead == n_layer 使每层走稠密 FFN, meta.rs 的 LFM2 hparams 臂）, arch_tensors_support(LFM2) → Partial（原 Unsupported 钉死测试同步改写, 合成稠密文件全张量集测试 tests/mtp2_e2e.rs::mtp2_lfm2_dense_loads） |
| qwen35.cpp | 645→659 | ldh;ldt;graph x1;graph_mtp x1;helpers(5) | ✅ | 0 | trunk ✅ build_qwen35_forward；MTP 批次 17: build_qwen35_mtp_forward 逐位相同。**同步批次 A2 (a4cb4c61f)**: 可选 embeddings 投影 cls_out/cls_out_b（`cls.output.*`, n_embd_out 宽; t_embd 换投影, LM 头仍读 normed 行; clef 的 lexical 头复用 output） |
| qwen35moe.cpp | 742 | ldh;ldt;graph x1;graph_mtp x1;helpers(5):build_layer_attn,build_layer_attn_linear,build… | ✅ | 0 | trunk ✅ build_qwen35moe_forward；**MTP 批次 17**: build_qwen35moe_mtp_forward（MoE + sigmoid 门控共享专家, 复用 build_qwen35moe_ffn）—— 逐位相同（同上） |
| qwen3next.cpp | 823 | ldh;ldt;graph x1;graph_mtp x1;helpers(5):build_layer_attn,build_layer_attn_linear,build… | ✅ | 0（chunked delta-net 见 delta-net-base.cpp 行 — 批次 18 已移并链级验收） | trunk ✅ build_qwen3next_forward；**MTP 批次 17**: build_qwen3next_mtp_forward（朴素 rope + 图中 inp_out_ids 收集 + MoE/门控共享专家; nextn 三元组装载补齐）—— 逐位相同（同上）。非 fused delta-net 两半均已移: 自回归半边（批次 17）+ chunked 半边（**批次 18**, build_delta_net_chunking, vs 参考 ggml 同链逐位相同, 见 delta-net-base.cpp 行） |
| t5.cpp | 371 | ldh;ldt;graph<false>(dec)+graph<true>(enc) | ✅ | 0 | **MTP 批次 17**: graph<false> 解码器 ✅ build_t5_decoder_forward（自注意力带相对位置偏置的 KV-cache 路径 attn_kv_cached_bias + 跨注意力（enc 状态 + cross mask + build_attn_inp_cross 等价物）+ RELU-SEQ/GELU-PAR FFN）+ dec.blk.* 张量装载臂（t5.cpp:60-107, dec_n_layer > n_layer 的 layers.resize 亦移）+ enc.output_norm 双范数槽; encoder graph<true> ✅ build_t5_encoder_forward。**relu 分支升级**: ggml 已有 GGML_UNARY_OP_RELU, 编码器/解码器的 RELU-SEQ FFN 分支从已档缺口变为已移。合成 enc+dec 文件 12 步解码链 vs 参考 llama_encode/llama_decode 对 **逐位相同**（parity/ref_t5_dec_dump.c, tests/t5_dec_e2e.rs） |
| jina-bert-v2.cpp | 67 | ldh;ldt;other:build_arch_graph | ⛔ | load_arch_hparams/load_arch_tensors | PARITY:381 "jina-bert-v2/v3…与它们的 load_arch_tensors" 未移（graph=bert::graph 已移） |
| jina-bert-v3.cpp | 50 | ldh;ldt;other:build_arch_graph | ⛔ | load_arch_hparams/load_arch_tensors | 同上（PARITY:381） |
| modern-bert.cpp | 173→305 | ldh;ldt;graph x1;decision_head;other:build_arch_graph | ✅（批次 16 移主体; **A2 补 decision head**） | graph_mtp 无 | 批次 16 移（装载臂 + graph, none/mean/rank/silu 四格位对齐）; **同步批次 A2 (a7b94df2c)**: `decision.block_count` hparams（is_swa 头层重置 + n_embd_out=3）+ 头块张量（attn_norm/wqkv/wo/ffn_norm/ffn_up/ffn_down 全带偏置, 尾部 n_layer_decision 块）+ token_types/cls_b/cls_norm_b + `build_decision_head`（3 个题型各跑一遍头, 逐题型 type 行种子 + 无位置编码注意力 + LLM_FFN_RELU + cls→gelu_erf→cls_out 评分器, [3, n_out] 拼接取代 t_embd） |
| neo-bert.cpp | 135 | ldh;ldt;graph x1;other:build_arch_graph | ⛔ | load_arch_hparams/load_arch_tensors/graph | PARITY:381 未移 |
| nomic-bert-moe.cpp | 57 | ldh;ldt;other:build_arch_graph | ⛔ | load_arch_hparams/load_arch_tensors | PARITY:381 未移 |
| nomic-bert.cpp | 52 | ldh;ldt;other:build_arch_graph | ⛔ | load_arch_hparams/load_arch_tensors | PARITY:381 未移（graph=bert::graph 已移） |
| pockettts.cpp | 147 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | **音频批次 5 (2026-09-28)**: POCKETTTS 装载臂（LLM_NORM 双 bias + tok_embd 复用为头 :23-24）+ build_pockettts_forward; 合成文件 llama_decode 32000 logits ×fa/nofa 0.0000（parity/tts_parity.sh A 节 + ref_lm_tts_dump） |
| qwen3tts.cpp | 4 |  | ✅ | 0 | **音频批次 5 (2026-09-28)**: QWEN3TTS 与 QWEN3VL 共 hparams/装载/图臂（models.h:625-627 纯 typedef）+ n_vocab_out 3072 头特例（qwen3vl.cpp:19-23）; 合成文件 ×fa 位精确 |
| wavtokenizer-dec.cpp | 265 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | **音频批次 5 (2026-09-28)**: WAVTOKENIZER_DEC 装载臂（ldh 读 group-norm 三键 :3-7; conv1d/posnet 6 块/tok_norm/convnext/输出头 :9-112）+ build_wavtokenizer_dec_forward（:118-264 逐行: sigmoid 门 resnet/单头双向 attn/convnext/波形头 = res->t_embd）+ group_norm_lm 组合内核; 合成文件 t_embd 64/64 位同（tts_parity.sh, 参考侧 logits 为零哑 buffer — 比较面是 t_embd） |
| nemotron-h-moe.cpp | 165 | graph_mtp x1;other:build_arch_graph | ✅+⛔ | graph_mtp(:11-16)（档: PARITY 批次 5 §5） | graph_mtp ⛔ 已档; **NEMOTRON_H_MOE 装载臂已移 (arch batch 13)** — 与 NEMOTRON_H 共臂, CLI/server 路由 NemotronH 图 |
| afmoe.cpp | 284 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | AFMOE 臂 + build_afmoe_forward（arch batch 15; MuP 缩放 + (il+1)%4 rope 规则 n_no_rope_layer_step 默认 4 + 宽 shexp, 合成文件 16/16 双 FA） |
| arwkv7.cpp | 203 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | ARWKV7 臂 + build_arwkv7_forward（5/6-plane lerp_fused 回退 + 可选门控, 批次 14） |
| bitnet.cpp | 171 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | BITNET 臂 + build_bitnet_forward（arch batch 13） |
| chatglm.cpp | 162 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | CHATGLM 臂 + build_chatglm_forward（arch batch 13） |
| cogvlm.cpp | 159 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | COGVLM 臂 + build_cogvlm_forward（arch batch 15; 文本路径 + visexp 双装载; 图与参考库位精确, 参考服务器自身分歧 — minimax-m3 先例, PARITY 批次 15） |
| cohere2.cpp | 160 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | COHERE2 臂 + build_cohere2_forward（arch batch 13） |
| dbrx.cpp | 155 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | DBRX 臂 + build_dbrx_forward（arch batch 13） |
| ernie4-5.cpp | 165 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | ERNIE4_5 臂 + 复用 build_ernie45_moe_forward 全稠密分支（arch batch 13） |
| exaone4.cpp | 191 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | EXAONE4 臂 + build_exaone4_forward（arch batch 13; 64 层 -swa 变体在档） |
| gemma-embedding.cpp | 177 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（参考 llama_encode 自身崩溃 GGML_ASSERT(buffer) — 端口内验证） | GEMMA_EMBEDDING 臂 + build_gemma_embedding_forward（arch batch 15; 对称 SWA 无 cache 编码器, EncoderContext::kq_mask_swa） |
| gemma3n.cpp | 465 | ldh;ldt;graph x1;helpers(8):altup_compute_router_modalities,altup_correct,altup_predict… | ✅ | 0（多模态 per-level 分支文字路径验证; build_inp_per_level 的 embd 半边为文字-only） | GEMMA3N 臂 + build_gemma3n_forward + 全部 helper（批次 14; KV-reuse 层为 builder 侧指向复用层缓存行）。收尾轮: `gemma3n_attn` 补漏 wo 投影（节点级 DECDMP1 定位, 修复后 47/47 ×2FA 0.0000） |
| gemma4-assistant.cpp | 201 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（真机对位: 26B-A4B 主干 + 重钉名头, 5 步草稿 t_logits/t_h_nextn/主干 h 抽头全位同 ×2FA; 该文件没有 graph_mtp 类 — 整档就是 MTP 头, 先前的 🟡 为记号问题, MTP 批次 17 改记） | gemma4_assistant.rs（批次 15 §integrator）: 装载臂 + graph + attach/decode_gemma4_assistant（ctx_other 倒置, share 映射层视图主干 iswa KV）+ CommonSpeculativeImplDraftMtp::is_mem_shared + CLI/server 接线 |
| glm-dsa.cpp | 770 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | GLM_DSA 臂 + build_glm_dsa_forward + build_glm_dsa_mtp_forward + MtpForward::GlmDsa（arch batch 13; -shared 索引器变体在档） |
| glm4.cpp | 187 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | GLM4 臂 + build_glm4_forward（arch batch 13; -mrope 变体在档） |
| glm5-next.cpp | 1013 | ldh;ldt;graph x1;inputs:kpool;helpers(6):build_kda_layer,build_kpool_select,build_dsa_layer,build_hc_*(3);other:build_arch_graph | **✅ (批次 19)** | graph 全量 + `llm_graph_input_kpool` | **同步批次 A（def4d406a 新 arch）**: 表/枚举/rope_type/sm_tensor ✅（arch.rs 再生成, 154 arch）+ hparams 臂 ✅（meta.rs, indexer_kpool/kpool_select_tail/mHC 断言/is_indexer_full 填 1）+ 装载臂 ✅（model.rs, conv 4D→3D 回退/共享索引器层 NOT_REQUIRED/NextN 块）+ LLM_TYPE_320B_A18B ✅（display.rs）+ 合成文件装载验收 tests/glm5_e2e.rs + parity/glm5_parity.sh（对 NEW 参考装载横幅逐行一致）; 图侧 ✅ (批次 19): `llama_memory_hybrid_idx` 记忆模块全量（kv_cache.rs HybridIdxCache: kpool_layout/kpool_state/set_input_kpool/set_input_qsa + seq_*/state 挂钩; context.rs 记忆选择 + `Glm5KpoolStep`） + graph 族（graph_arch.rs: glm5_hc_pre/post/sinkhorn/mean + build_glm5_kda_layer/build_glm5_dsa_layer/build_kpool_select + gather/scatter 双路）; 验收: 合成文件 13 图（2-token 预填 + 12 解码尾步, --fa off, 尾 token 100）节点流 vs NEW 参考 (`parity/glm5/nodes_ref.bin`, ref_decode_dump.c) **逐节点位同 0 分歧**（decode_dump_cmp.py; 76 具名节点 ×13 出现位同）; 默认测试 tests/glm5_dump.rs::glm5_graph_nodes_bit_exact_vs_reference。**同步批次 A2 (4e2713c16)**: build_kpool_select 的 -inf/0 散射改自有类型化张量（弃 sel_idx 视图种子, ggml_build_forward_expand 保活; kq_mask 类型即 FA 下 zeros 变 F16 — 0/-inf 转换精确无数值变化）; glm5_parity.sh 复跑: 装载横幅 56 行一致 + 节点流位同 0 分歧 |
| granite-swa.cpp | 320 | ldh;ldt;graph x1;helpers(2):build_attention_layer,build_layer_ffn;other:build_arch_graph | ✅ | 0 | GRANITE_SWA 臂 + build_granite_swa_forward（arch batch 15; 两 helper 内联, iswa + sinks + SWIGLU 融合 shexp + deepstack 注入, 合成文件 16/16 双 FA 含 -moe/-long） |
| hunyuan-dense.cpp | 7 | other:build_arch_graph | ✅ | 0 | HUNYUAN_DENSE 与 HUNYUAN_VL 共臂（arch batch 15, models.h:2101-2103） |
| hunyuan-vl.cpp | 192 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | HUNYUAN_VL 臂 + build_hunyuan_vl_forward（arch batch 15; 可选 M-RoPE + XDRoPE alpha + rope 后 QK-norm, 合成文件 16/16 双 FA 含 -mrope） |
| hy-v3.cpp | 391 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_hy_v3_forward + MTP 张量随装载臂加载（arch batch 15）；**MTP 批次 17**: build_hy_v3_mtp_forward（rope 前 QK-norm + 稠密|sigmoid-MoE FFN + vLLM final_layernorm 语义）—— 逐位相同（parity/ref_mtp2_dump.c） |
| hy-v4.cpp | 602 | ldh;ldt;graph x1;helpers(6):build_attention,build_attention_dsa,build_hc_head,build_hc_… | ✅ | 0 | HY_V4 装载+hparams 臂 + build_hy_v4_forward + hy_v4_hc_pre/post/head/reduce + hy_v4_attn_mla + build_indexer_top_k 内联（arch batch 15; iHC + 门控 MLA + DSA 索引器, 合成文件 16/16 双 FA 含 -dsa/-long） |
| llada.cpp | 154 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | LLADA 臂 + build_llada_forward（arch batch 15; 参考无生成路径, 端口内验证 — PARITY 批次 15） |
| llama-embed.cpp | 7 | other:build_arch_graph | ✅ | 0 | LLAMA_EMBED 与 LLAMA 共装载臂 + build_llama_embed_encoder（arch batch 15; EncoderContext 路径, 参考 llama_encode dump 位精确） |
| llama4.cpp | 273 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | LLAMA4 臂 + build_llama4_forward（arch batch 13; CHUNKED iswa + 温度缩放 + no-rope 层 + sigmoid MoE + 共享专家, -noswa 变体在档） |
| maincoder.cpp | 152 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MAINCODER 臂 + build_maincoder_forward（arch batch 15; rope 后 QK-norm, 合成文件 16/16 双 FA） |
| mellum.cpp | 220 | ldh;ldt;graph<iswa>×2 | ✅ | 0 | MELLUM 臂 + build_mellum_forward（arch batch 15; iswa 双实例化合一 — swa_type NONE/STANDARD 由参数区分, SWA 层朴素 rope; 因果解码器非编码器, 合成文件 16/16 双 FA） |
| mimo2.cpp | 397 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_mimo2_forward + MTP 张量加载（arch batch 15; §integrator 修复 iswa kq 实差后晋升默认集, 16/16 ×2FA 含 -long, 节点 dump 位同）；**MTP 批次 17**: build_mimo2_mtp_forward（iswa + sinks + value_scale + 融合 qkv 拆分 + 收集后的 h_nextn + layer_out_norm 头范数回退链）—— 逐位相同（同上） |
| minicpm3.cpp | 253 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MINICPM3 臂 + build_minicpm3_forward（arch batch 13） |
| mistral3.cpp | 236 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MISTRAL3 臂 + build_mistral3_forward（arch batch 13; -temp 变体在档） |
| mistral4.cpp | 7 | other:build_arch_graph | ✅ | 0 | MISTRAL4 与 DEEPSEEK2 共 hparams/装载/图臂（arch batch 15, models.h:1393-1395; ForwardWeights::Deepseek2 路由, NEMOTRON_H_MOE 先例; MLA 合成文件 16/16 双 FA） |
| muse-glimmer.cpp | 204 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | MUSE_GLIMMER 臂 + build_muse_glimmer_forward（arch batch 15; 反转 rope 规则 + 双 1e-8 后置 norm + softcap 头, 合成文件 16/16 双 FA 含 -softcap） |
| paddleocr.cpp | 108 | graph x1;other:build_arch_graph | ✅ | 0 | PADDLEOCR 与 ERNIE4_5 共装载臂（arch batch 15, models.h:1984-1986）+ build_paddleocr_forward（M-RoPE sections, 合成文件 16/16 双 FA） |
| pangu-embed.cpp | 163 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | PANGU_EMBED 臂 + build_pangu_embed_forward（arch batch 15; 必需 wo 偏置, 合成文件 16/16 双 FA） |
| plm.cpp | 207 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0（参考 -fa on 在其 2D Vcur 上 set_rows 断言 — PARITY 批次 15） | PLM 臂 + build_plm_forward（arch batch 15; 共享 k_pe 跨头重复 + relu² FFN; fa off 16/16） |
| qwen.cpp | 141 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN 臂 + build_qwen1_forward（arch batch 15; 融合 wqkv+bias + 半宽 SwiGLU, 合成文件 16/16 双 FA） |
| qwen2vl.cpp | 144 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN2VL 臂 + build_qwen2vl_forward（arch batch 13; MRoPE sections 必需） |
| qwen3vl.cpp | 198 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN3VL 臂 + build_qwen3vl_forward（arch batch 13; IMRoPE + deepstack 零填充文本路径） |
| qwen3vlmoe.cpp | 191 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | QWEN3VLMOE 臂 + build_qwen3vl_forward 的 MoE 分支（arch batch 13） |
| rwkv6-base.cpp | 165 | helpers(2):build_rwkv6_channel_mix,build_rwkv6_time_mix | ✅ | 0（WKV 扫描为组合实现，融合算子为在途项） | build_rwkv6_time_mix/channel_mix + 组合 wkv6/GLA 扫描（批次 14; 融合 GGML_OP_RWKV_WKV6/GATED_LINEAR_ATTN 为 integrator 项） |
| rwkv6.cpp | 186 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | RWKV6 臂 + build_rwkv6_forward（批次 14, 组合 WKV 扫描; -legacy 变体在档） |
| rwkv6qwen2.cpp | 168 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | RWKV6QWEN2 臂 + build_rwkv6qwen2_forward（is_qrwkv GLA 路径 + GQA repeat, 批次 14） |
| rwkv7-base.cpp | 138 | helpers(2):build_rwkv7_channel_mix,build_rwkv7_time_mix | ✅ | 0（WKV 扫描为组合实现，融合算子为在途项） | build_rwkv7_time_mix/channel_mix + rwkv_l2_norm 组合 + 组合 wkv7 扫描（批次 14; 融合 GGML_OP_RWKV_WKV7 为 integrator 项） |
| rwkv7.cpp | 212 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | RWKV7 臂 + build_rwkv7_forward（v_first 跨层残差, 批次 14） |
| spark2-5.cpp | 147 | ldh;ldt;graph x1;other:build_arch_graph | ✅ | 0 | SPARK2_5 臂 + build_spark25_forward（arch batch 15; 逐头 sigmoid attn 门 + GELU FFN, 合成文件 16/16 双 FA 含 -long） |
| step35.cpp | 561 | ldh;ldt;graph x1;graph_mtp x1;other:build_arch_graph | ✅ | 0 | trunk ✅ build_step35_forward + MTP 张量加载（arch batch 15; 半维 rope + 逐头门 + sigmoid MoE; §integrator 修复后 16/16 ×2FA 含 -long, 节点 dump 位同）；**MTP 批次 17**: build_step35_mtp_forward（iswa + 可选逐头门 + rope_freqs 仅全注意力层 + 收集后的 h_nextn）—— 逐位相同（同上） |

总计: ✅=128, ❌=0, ⛔=0, 🟡=0, ✅+=7, ⛔+❌=0 （批次 15 移走 19 个 ❌: …, gemma4-assistant ⛔→🟡（§integrator）; 音频批次 5 清零最后三个 ⛔; **MTP 批次 17 (2026-09-29): 最后 12 个 🟡 清零** — qwen35/qwen35moe/qwen3next/glm4-moe/cohere2moe/bailingmoe3/hy-v3/mimo2/step35 的 graph_mtp 构建器（9 个, 逐位相同 vs 参考自身 ctx_type=MTP 上下文, 12 步草稿链 ×9 arch）、t5 的 graph&lt;false&gt; 解码器 + dec_* 张量（逐位相同 vs 参考 llama_encode/llama_decode 对）、lfm2 稠密装载臂、qwen3next nextn 三元组、bailingmoe3 MTP 装载循环、gemma4-assistant 记号修正（本无 graph_mtp）; 附带: 非 fused delta-net 的自回归半边（逐位相同）+ t5 的 RELU-SEQ FFN 分支（原已档缺口, ggml 已有 relu）。**src/models/ 155 文件: ✅128 + ✅+7, 无 ❌ 无 ⛔ 无 🟡**; **同步批次 A (2026-10-01, def4d406a): +glm5-next.cpp 🟡（装载/表/枚举 ✅, 图待 kpool）→ 156 文件 = ✅128 + ✅+7 + 🟡1**）

## 审计方法与可复核性
1. 符号清单: 对每个 .cpp 正则抽取 `load_arch_hparams/load_arch_tensors`、graph/graph_mtp 构造、`llm_graph_input_*` 类、全部 `graph::helper` 成员（含 std::pair 返回型）。
2. 覆盖判定: model.rs 的 `arch_tensors_support`(L594-776) + `load_arch_tensors` 95 个 arch 臂（L1108-10280）+ `load_arch_hparams_batch` 46 臂 + meta.rs 39 个显式 hparams 臂 + graph_arch.rs 118 个 `build_*_forward` + 3 个 MTP builder + eagle.rs/dflash.rs + llama-cli/llama-server forward_weights 臂，全部用词边界匹配（避免 COHERE2⊂COHERE2MOE 类前缀假阳）。
3. 抽样复核建议: 任意取 ❌ 行 grep `LlmArch::<变体>` 于 crates/llama/src/{model,graph_arch,context,meta}.rs — 命中数 ≤3 且均在枚举/rope 分组/hparams 臂处（GLM4/PADDLEOCR/HUNYUAN_VL/HY_V4/QWEN3TTS 有 meta.rs hparams 臂，其余仅 rope 分组）。
