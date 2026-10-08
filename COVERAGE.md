# COVERAGE.md — 参考面完整覆盖清单（自动生成 + 人工状态）

基线: /home/jeffrey/llm/llama.cpp-pinned @ bd4f514db1 | CPU 相关文件 583 个 / 334893 行

说明: ⛔ 排除的 GPU/非 CPU 后端不计入; ➖ 头文件/胶水/可选性能路径以等价形式并入; 🔨 进行中; ⏳ 未开始。

## 文件级清单（大→小, 前 60 + 分类汇总）

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 12206 | ggml/src/ggml-cpu/ops.cpp | crates/ggml/src/{compute,ssm,flash_attn}.rs | 🔨 推理所需算子✅(含 FA/SSM/MoE; 批次6 增 fill/lightning_indexer/top_k —— top_k 为 libstdc++ partial_sort 堆选择的逐算子复刻, oracle 测试钉死并列集合; ggml.c 侧 rope_set_offset 补齐); 其余 op 按需 |
| 8167 | ggml/src/ggml.c | crates/ggml/src/{types,tensor,graph}.rs | ✅ 核心算子/类型表; 部分 op 变体随架构推进补 |
| 7034 | src/unicode-data.cpp | crates/llama/src/unicode*.rs | ✅ |
| 6596 | ggml/src/ggml-cpu/arch/riscv/quants.c | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 6407 | ggml/src/ggml-cpu/arch/x86/repack.cpp | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 6087 | tools/mtmd/clip.cpp | — | ⏳/⛔ 按需 |
| 5768 | ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp | — | ⏳ 未映射 |
| 5638 | ggml/src/ggml-quants.c | crates/ggml/src/{quants,quants_k}.rs | ✅ 位精确(含 IQ/MXFP4/NVFP4) |
| 5558 | tools/server/server-context.cpp | crates/tools/llama-server/src/{engine.rs,main.rs,api.rs,http.rs,chat.rs} | 🔨 槽位生命周期(idle→Started→ProcessingPrompt→DonePrompt→Generating)、`get_available_slot`(id/相似度/LRU)、`pre_decode`/`post_decode`、`cache_prompt` 的 LCP 前缀复用+`[TAG_PROMPT_LOGITS]` 规则、多序列 batched decode、`process_token` 停止条件(eos/stop 词/limit/UTF-8 半字符, partial 发送点=停止条件之前)、`n_probs`、SSE、`handle_completions_impl` 三型 res_type(NONE/OAI_CMPL/OAI_CHAT, :4257-4560)、`handle_embeddings_impl`+`send_embedding`(:5390-5491/:2152-2192)、`get_models`/`get_slots`、rerank 501; 未移植(见 PARITY.md): tools 解析链/jinja、mtmd/检查点/sleep/purge/alora/n_cmpl 子槽/`POST /slots`; **context shift 已补**(`pre_decode` 的搬移块 :2909-2972 → `Engine::context_shift`, `--context-shift`/`--keep` 默认同参考=关) |
| 5465 | ggml/src/ggml-cpu/arch/arm/repack.cpp | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 5253 | ggml/src/ggml-cpu/repack.cpp | crates/ggml/src/repack.rs | ✅ x86 有实例的 3 族全位精确: MXFP4 8x8 + Q4_K 8x8 + Q4_0 8x8(repack_ref/q4k_repack/q4_0_repack_ref.bin); q2_K/iq4_nl_8x8 无 x86 消费者(PARITY 已记录) |
| 4763 | common/arg.cpp | crates/common(最小)+llama crate 内 | 🔨 CLI 子集(含 -j/--json-schema + -jf, 集成者); grammar/json/speculative/ngram/jinja 其余待定 |
| 4526 | src/llama-vocab.cpp | crates/llama/src/vocab.rs | ✅ 12 fixture 逐 id |
| 4412 | src/llama-context.cpp | crates/llama/src/context.rs | 🔨 单序列 decode✅ + 17 架构 ForwardWeights 分发✅（批次5 增 Mamba/Jamba/NemotronH; 批次6 增 Deepseek2/Deepseek/Deepseek32 —— 32 的 dsa 双 cache 在 `step_inputs` 造 lid row_idx/F16 mask/k_rot 输入）+ **多序列 `decode_batch`✅**(llama-batch.cpp allocr→多序列 ubatch、逐 token `seq_id` KQ mask、`n_outputs`/`output_ids`/`output_reorder`) + `seq_rm`/`seq_pos_min|max` + **`llama_state_seq_*`✅**(代理DSV4STATE: :3142-3213 的 io_magic 帧 + `llama_set_causal_attn` :1245-1247 → `DecodeContext::state_seq_{get,set}_data/get_size`/`set_causal_attn`; blob 逐字节同参考, 见 kv_cache 行); speculative/base-sampling 待做 |
| 4385 | src/llama-sampler.cpp | crates/llama/src/sampling.rs | ✅ 核心链位精确(grammar/dry 待做) |
| 4319 | ggml/src/ggml-cpu/arch/arm/quants.c | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 4164 | ggml/src/ggml-cpu/llamafile/sgemm.cpp | crates/ggml/src/tinyblas.rs | ✅ 6 组合路由表+tile_tail 逐位(tinyblas_ref.bin 6360 例) |
| 4108 | ggml/src/ggml-cpu/arch/x86/quants.c | vec_dot.rs/repack.rs/simd_x86.rs lane 复刻 | ✅ 全部 x86 派发量化 lane(8 族 + 2026-09-27 审计轮 Q1_0/Q2_0/NVFP4/IQ×9, vecdot3_ref.bin) |
| 3944 | ggml/src/ggml-cpu/ggml-cpu.c | crates/ggml/src/compute.rs | 🔨 主路径✅(mul_mat/mul_mat_id/repack 分支; vec_dot_type 全量化类型已接); 部分 op |
| 3915 | src/llama-graph.cpp | crates/llama/src/{graph,graph_arch}.rs | 🔨 45 架构 builder(批次4 起 MoE 家族全接, 批次6b 起 grok softcap/GELU-MoE 变体 + plamo2 mixer: build_moe_ffn_silu 的 norm_w 两种/门控 shexp/路由偏置; 批次5 起 mamba 家族; 批次6 起 MLA: attn_k_cached_mla 的 K-only cache + wv_b 解压、build_qkv、DSA top-k mask 合成 + lightning indexer); 其余待移植 |
| 3359 | src/llama-model.cpp | crates/llama/src/model.rs | 🔨 60 架构加载(批次6 +4 MLA 家族; 批次6b +7: nemotron/grok/chameleon/deci/jais/falcon-h1/plamo2); 其余待移植 |
| 3178 | ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp | — | ⏳ 未映射 |
| 3022 | ggml/include/ggml.h | Rust 类型系统 | ➖ 头文件→类型 |
| 2997 | common/speculative.cpp | crates/llama/src/speculative.rs | 🔨 代理SPEC: **draft-simple 全量状态机**(类型表/`are_compatible`/impl 统计/synth 接受率/`n_max`/init/begin·process·draft·accept/print_stats)+ `common/sampling.cpp:678-715` 验证规则 + `examples/speculative-simple` 驱动; 7B(0.5B draft) 36 token == 参考 plain/spec, 接受率 100%, 目标前向 -73%, 但 0.50x(慢, Q4_K 无 gemm); **+ ngram 五员家族✅**(代理DSV4RS/NGRAM: ngram-simple/map-k/map-k4v/mod/cache 五 impl = :1769-2181, 参数 common.h:352-384, `--spec-ngram-*` 值旗标 arg.cpp:4254-4378 两端 CLI 已接; 0.5B 自投机五类型提交流==plain greedy, ngram-cache 接受 14/前向 29→18, 参考服务器全新首请求五类型全 MATCH —— parity/run_server_ngram_parity.sh); **+ draft-eagle3✅**(代理EAGLE 2026-09-24: `common_speculative_impl_draft_eagle3` = :426-907 逐行 —— 延迟边界 (token[P+1], g_embd[P])/跨 ubatch 桥/verify 快照重锚/分块编码器; 头双图 graph_arch.rs = src/models/eagle3.cpp:103-326, 加载器+ctx_other 零拷贝 eagle.rs, eagle 上下文/encode/目标侧 layer_inp 抽头 context.rs; `--spec-type draft-eagle3 -md` 两端已接; 合成对 parity 4 格全 PASS(trunk-unchanged + draft parity 16/16 ×2FA), 服务器侧草稿计数 39/39 同参考 —— parity/eagle_parity.sh, 见 PARITY.md §EAGLE3); **+ draft-dflash/draft-dspark✅**(代理DSV4STATE 2026-10: :910-1328 同一 impl 双类型; dflash 双模解码器+加载器在**新文件 dflash.rs**(dflash.cpp:572-853 含 DSpark markov/conf 头 :295-406), 非因果 mask/`DecodeContext::new_dflash`/rope=NEOX 在 context.rs; parity 8 格全 PASS + 草稿链 153/153 逐行同参考(ref_dflash_chain.cpp 同驱动探测) —— parity/dflash_parity.sh, 见 PARITY.md §DFlash; 未移: DFlash2 selector 图/DSV4 骨干/M-RoPE/backend-sampling); 未移植: backend-sampling + 检查点回退 (PARITY.md §speculative/§ngram/§EAGLE3/§DFlash) |
| 537 | common/ngram-map.cpp | crates/llama/src/ngram_map.rs | ✅ 代理DSV4RS/NGRAM 全量(simple 反向搜索 + map begin/draft/accept, reasoning-chat 收缩清理; 262144 槽哈希表按 C 可选项置空) |
| 286 | common/ngram-cache.cpp | crates/llama/src/ngram_cache.rs | ✅ 代理DSV4RS/NGRAM 全量(update/draft/save/load/merge, save/load 与参考二进制字节兼容) |
| 63 | common/ngram-mod.cpp | crates/llama/src/ngram_mod.rs | ✅ 代理DSV4RS/NGRAM 全量(LCG 开放表) |
| 2817 | src/llama-kv-cache.cpp | crates/llama/src/kv_cache.rs | 🔨 统一非SWA✅ + iswa 双 cache/mask✅(363 行 llama-kv-cache-iswa.cpp 并入, 长上下文 16/16) + **DSA 双 cache✅**(批次6: 262 行 llama-kv-cache-dsa.cpp 并入 —— MLA K-only + lid 索引键 cache `KvLidCache`/`new_dsa` + gen_hadamard + k_rot/lid-mask step 输入; deepseek32 16/16 ×2FA+long); **DSV4✅**(批次7: 2253 行 llama-kv-cache-dsv4.cpp 并入 —— iswa raw + csa/hca/lid 压缩 K cache + 三压缩器状态平面 + `Dsv4Plan` 单序列切片; deepseek4 16/16 ×2FA+long; **多序列+seq_cp+回滚平面✅**(代理DSV4RS/NGRAM: `n_stream=n_seq_max` 行块分流 :1287, seq_cp :1512-1527, n_rs_seq>0 平面 :653-716 索引逐字+宿主行复制+`DecodeContext::new_with_dsv4`; `seq_rm` head 回退 :403-419 补齐; 双序列 decode_batch 与两次单序列运行逐位相同, 回滚逐位复现); **状态序列化✅**(代理DSV4STATE 2026-10: `llama_kv_cache::state_write/read` :2055-2628 → `raw_state_write/read`+`KvCache::state_seq_{write,read,size}`+`StateWriter/StateReader`, io_magic 帧 → `DecodeContext::state_seq_{get,set}_data/get_size` llama-context.cpp:3142-3213; dsv4 blob 与参考逐字节相同×2 场景 + 往返逐位, parity/dsv4_state_parity.sh)); **位置搬移族+K-shift✅**(seq_add/seq_keep/seq_div/seq_cp :451-657 → `kv_cache.rs` 含 per-cell shift 累计器; `build_graph_shift`+`update`/`init_update` :2003-2053/:817-896 → `context.rs build_graph_shift`/`memory_update`, 每个解码入口先跑 llama-context.cpp:1805 的 memory_update; **defrag 上游已删** —— commit 9ebebef62 "remove KV cache defragmentation logic (#15473)", pinned 修订版无此函数, `-dt/--defrag-thold` 仅剩弃用警告 arg.cpp:2530-2536); MSA cache✅（批次 11a: MinimaxMsaCache 挂 context.rs）; **状态序列化全类✅**(代理DFLASH2 2026-09-27: dsa lid 半 llama-kv-cache-dsa.cpp:164-172 → `lid_state_{write,read}` + state_seq_{write,read} 的 dsa 分支（dsa_iswa :236-242 顺序）; MSA idx 半 llama-kv-cache-msa.cpp:160-168 → context.rs `msa_state_seq_{write,read}`（idx K 行 + 零 V 行 + lockstep 元数据）; 整上下文头 llama-context.cpp:3341-3381 → `state_{get,set}_data/get_size`; minimax-m3 seq/full×2 状态 4 blob 与参考逐字节同, deepseek32 结构+dense-lead 行逐位同, 往返逐位 —— parity/state_kinds_parity.sh + tests/state_kinds_e2e.rs); recurrent 待做 |
| 2733 | tools/mtmd/mtmd.cpp | — | ⏳/⛔ 按需 |
| 2662 | src/models/models.h | crates/llama/src/{graph_arch,model}.rs | 🔨 **136/156 架构**（批次 15 后的 `arch_tensors_support` 口径）。批次 15（PARITY.md「批次 15」节）: +21 P1+P2 arch — qwen(v1 融合 QKV 半宽 FFN)/maincoder(rope 后 QK-norm)/pangu-embed(必需 wo 偏置)/plm(共享 k_pe MLA-lite+relu²)/cogvlm(visexp 双权重, 图与参考库位精确、服务器自分歧)/spark2-5(逐头 sigmoid attn 门)/muse-glimmer(反转 rope 规则+双 1e-8 后置 norm+softcap 头)/llada(扩散 no-cache, in-port)/hunyuan-vl(+dense 共臂, 可选 M-RoPE+XDRoPE alpha)/granite-swa(iswa+sinks+SWIGLU 融合 shexp+deepstack)/afmoe(MuP+(il+1)%4 rope 规则)/mellum(全 MoE, 因果解码器)/paddleocr(ernie4_5 共臂 M-RoPE)/hy-v3(裸名 exp_probs_b)/mimo2(sinks+value_scale)/step35(半维 rope+共享 rope_freqs; 三者 graph_mtp 🟡 开档)/hy-v4(iHC 2*hc+门控 MLA+可选 DSA 索引器)/mistral4(deepseek2 共臂)/llama-embed(graph<true>, llama_encode 非因果)/gemma-embedding(对称 SWA 编码器, 参考自身崩溃); gemma4-assistant 🟡 已移（批次 15 §integrator: gemma4_assistant.rs + mem-shared 草稿驱动, 真机对位精确 ×2FA; 参考侧三陷阱见 PARITY.md 批次 15）。原批次 13/14 记录:批次 13/14（前代理中断后收尾, PARITY.md「架构批次 13/14」节）: +20 P0 标准注意力与新机制 arch（llama4(CHUNKED iswa+no-rope 温度层+MoE 权重前乘)/qwen3vl(+moe)/qwen2vl(mrope)/glm4(+mrope)/glm-dsa(MLA+DSA 索引器)/chatglm/mistral3(温度缩放)/cohere2/minicpm3(半-MLA)/exaone4/bitnet/dbrx/ernie4-5 + nemotron-h-moe 装载臂 + RWKV 四员 rwkv6(+qwen2)/rwkv7(+arwkv7) 的 rwkv6/7-base mix + 组合 WKV 扫描（融合算子 integrator 项）+ gemma3n 每层嵌入+altup/laurel+KV-reuse 层）; llama-server 的批次 13/14 臂 + parity/run_server_arch_parity.sh 扩到批次 9-14 同批验证。此前: 批次12 +3: 终局长尾 hrm-text(交替低/高栈, 缓存槽别名物理块, zH+zL 状态穿线)/laguna(sigmoid MoE+选中偏置+softplus 双宽输出门+逐层头数+逐层类型 RoPE)/maple(iswa 对上的 softmax MoE, 仅 SWA 层 rope) —— **架构队列清空**, llama-server 的 forward_weights 同批补齐批次 9-12 全部 22+1 arch(weights.rs=CLI 逐字副本, eurobert 走 EncoderContext; /completion 的 prompt 字段改 tokens.detokenize(special) 对齐 server-context.cpp:2111); 批次11b +7: 长尾队列后半 arcee/jais2/talkie/nanbeige(num_loops 展开)/dream/rnd1(llada 家族, 参考 memory=nullptr, in-port)/eurobert(EncoderContext 与参考 llama_encode 位同) —— 零新算子; 批次11a +7: kimi-k3/qwen4exp/qwen35moe/dots3note/minimax-m3/grovemoe/apertus; 批次10 +4: small-arch + EXP-op 批 smallthinker/llada-moe/minimax-01/graniteswitch —— `ggml_exp`(UNARY 13, ref_exp_dump 位同)+pad+round+reglu 件; minimax-01 lightning attention(la 衰减输入+n_embd_head_la 状态), graniteswitch in-graph adapter 路由+switched LoRA, llada-moe 非因果 no-cache(参考 memory=nullptr 无生成, in-port); 批次9 +4: 线性注意力家族 plamo3/qwen3next/kimi-linear/bailingmoe3 —— KDA/GDN 层走融合 GATED_DELTA_NET(K=1)+RecurrentState, MLA 层压缩 [kv_lora\|rope] 行; 批次8 +8: MoE 长尾家族 |
| 2631 | tools/server/server-models.cpp | — | ⛔ 未移植(路由器模式/模型下载/切换; 单模型 `GET /models` 已在 server-context.cpp 侧移) |
| 2517 | ggml/src/ggml-backend-meta.cpp | — | ⏳ 未映射 |
| 2513 | ggml/src/ggml-backend.cpp | 并入 compute.rs/context | ➖ CPU-only 简化 |
| 2511 | ggml/src/ggml-cpu/amx/mmq.cpp | tinyBLAS 公式已推导(vec_dot.rs); 其余不移植 | ➖ 性能可选 |
| 2507 | tools/llama-bench/llama-bench.cpp | crates/tools/llama-bench | ✅ 代理BENCH: 旗标/默认值/26 维测试矩阵/计时协议/avg-stdev 公式/5 种输出格式(表头逐字节相同, JSON 43 键中非时间字段全同); 缺口: GPU/后端旗标、量化 KV、embeddings、线程池旋钮按 WARNING 忽略(见 PARITY.md 末节); 性能 0.19-0.55x |
| 2361 | common/common.cpp | crates/common(最小)+llama crate 内 | 🔨 CLI 子集; grammar/json/speculative/ngram/jinja 其余待定 |
| 2309 | ggml/src/ggml-cpu/arch/loongarch/quants.c | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 2304 | ggml/src/ggml-cpu/arch/powerpc/quants.c | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 2253 | src/llama-kv-cache-dsv4.cpp | crates/llama/src/kv_cache.rs | 🔨 批次7 并入(压缩 K cache×3 + 压缩器状态×3 + Dsv4Plan); 多序列/seq_cp/回滚平面/逐序列清零已补(代理DSV4RS/NGRAM, 双序列逐位==单序列、回滚逐位复现); **状态序列化已补**(代理DSV4STATE: `state_write/read` :1594-1673 + `dsv4_state_write/read_k_cache` :332-412 + `llama_dsv4_comp_state::state_write/read` :1080-1158 + 流 helpers :206-330/`dsv4_make_k_only` :1253 → `kv_cache.rs`; blob 与参考逐字节相同×2 场景, 往返逐位 — parity/dsv4_state_parity.sh) |
| 2172 | tools/server/server-tools.cpp | — | ⛔ 未移植(MCP/工具运行时) |
| 2124 | common/peg-parser.cpp | crates/llama/src/peg.rs | ✅ PEG 组合子/执行器(lenient NEED_MORE_INPUT 流式)/AST/序列化/GBNF 生成(until/ac Aho-Corasick 补集), trie.cpp 一并; 解析器结构 dump 与参考逐字节一致 (代理CT, 见 PARITY.md chat tools 节) |
| 2098 | tools/perplexity/perplexity.cpp | crates/tools/perplexity | ✅ 已建(代理Y) |
| 1988 | tools/server/server-common.cpp | crates/tools/llama-server/src/{api.rs,chat.rs,main.rs} | 🔨 `format_error_response`/`get_token_probabilities`/`validate_utf8`/`string_find_partial_stop`/`token_to_piece`/`random_string`/`server_slot_stats::to_json`/`common_embd_normalize`/`format_embeddings_response_oaicompat`/`format_oai_sse`(=http.rs terminal_done)/`oaicompat_chat_params_parse`(:1151-1408) 已移; metrics/log 格式化/lora 工具/媒体下载未移 |
| 1919 | ggml/src/ggml-cpu/kleidiai/kleidiai.cpp | tinyBLAS 公式已推导(vec_dot.rs); 其余不移植 | ➖ 性能可选 |
| 1911 | ggml/src/ggml-common.h | crates/ggml/src/blocks.rs | ✅ 位级布局 |
| 1901 | tools/server/server-task.cpp | crates/tools/llama-server/src/{api.rs,engine.rs} | 🔨 `task_params::to_json`(both 变体)/非 OAI 响应 JSON(`to_json_non_oaicompat`, partial+final)/OAI 响应族(`to_json_oaicompat` :374-411、`to_json_oaicompat_chat{,_stream}` :414-527、partial :1073-1164、`usage_json_oaicompat`、embd :1466-1486)/`stop_type`/`probs_vector_to_json`/SSE 帧已移; Responses/Anthropic/ASR 解析、tool-call diff 累积(:157-241)未移 |
| 1811 | src/llama-model-loader.cpp | crates/llama/src/model.rs | ✅ |
| 1795 | ggml/src/ggml-cpu/spacemit/repack.cpp | — | ⏳ 未映射 |
| 1253 | ggml/src/ggml-cpu/iqp.cpp | — | ➖ IQ panel gemm 性能 lane（ggml-cpu.c:1377 接线; batch≥8 且 ne[1]%16==0 时替代行式 vec_dot, 求和序不同; 端口走 vec_dot = 参考解码路径, AUDIT_ggml.md §5-B.6 归档） |
| 1094 | ggml/src/ggml-opt.cpp | — | ⛔ 训练 API; llama 层零调用（CROSS_ENTROPY_LOSS/OPT_STEP_* 只进 opt; AUDIT_ggml.md §5-B.5 归档, 补 >1500 行截断遗漏） |
| 55 | ggml/src/ggml-cpu/hbm.cpp | — | ➖ Fujitsu A64FX HBM buffer（GGML_USE_CPU_HBM off; AUDIT_ggml.md §5-B.6 归档） |
| 1743 | tools/mtmd/mtmd-image.cpp | — | ⏳/⛔ 按需 |
| 1742 | ggml/src/ggml-cpu/spacemit/ime.cpp | — | ⏳ 未映射 |
| 1714 | ggml/src/gguf.cpp | crates/ggml/src/{gguf,gguf_write}.rs | ✅ 读+写(写与参考字节级一致, parity/gguf_write_ref.bin) |
| 1703 | ggml/src/ggml-cpu/arch/riscv/repack.cpp | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 1646 | include/llama.h | Rust 类型系统 | ➖ 头文件→类型 |
| 1635 | common/chat-diff-analyzer.cpp | crates/llama/src/chat_tools.rs | ✅ 差分模板分析(reasoning/content/tools/call-id 标记检测, 11 个 workaround 表)与 helpers 的 calculate_diff_split/segmentize 等全同参考(C++ oracle 逐 case 验证) (代理CT) |
| 1586 | common/jinja/value.cpp | crates/llama/src/chat.rs (mini_jinja 模块) | ✅ 值模型(none/tuple/对象内可变/函数)/global 内建(namespace/range/raise_exception/strftime_now/tojson)/逐类型 builtin 表(string 方法、数组 join/sort/min/max/selectattr/map、对象 get/items/dictsort、none/undefined 空表)/py 切片/tojson+ensure_ascii/value_compare——按 vendor 模板所需面 1:1 (代理JINJA, 2026-10; 131/131 真模板夹具逐字节) |
| 1570 | ggml/src/ggml-cpu/vec.h | crates/ggml/src/{vec_dot,ops}.rs | ✅ f32/f16/bf16/tanh/gelu/silu/softmax 语义 |
| 1557 | tools/mtmd/mtmd-audio.cpp | crates/llama/src/mtmd_audio.rs | ✅ 全 11 预处理器 + FFT + ISTFT 逐位 (代理AUDIO, 2026-10; whisper-enc 图侧已由代理AUDIO2 补齐, 见 clip.rs 行; 12 个预处理器全接进 mtmd.rs init_audio 家族, 代理AUDIO4) |
| 1534 | ggml/src/ggml-cpu/arch/s390/quants.c | vec_dot.rs/repack.rs lane 复刻 | ➖ 仅 x86 lane 针对性复刻(参考构建) |
| 1533 | common/chat.cpp | crates/llama/src/{chat,chat_tools,chat_parsers}.rs | ✅ 含特化模板分发(chat.cpp:1090-1223)与 continue_final_message(:1252-1271); 未移: ASR/reasoning-budget 预设 (代理CT + 代理CHATPARSER) |
| 1526 | src/llama-grammar.cpp | crates/llama/src/grammar.rs | ✅ 解析器/匹配器/piece 表位同参考(代理Z); lazy 未移植 (json-schema 转换器由代理A 补齐, 见下) |
| 1028 | common/json-schema-to-grammar.cpp | crates/llama/src/json_schema.rs | ✅ JSON schema → GBNF **201/201 逐字节同参考** + 248 条 accept/reject 全同 (代理A; 同文件另含 json-schema.cpp 514 行与 common_json 子集 434 行); **llama-cli `-j`/`-jf` 已接线**(集成者, 首次请求 11/11) |
| 1502 | src/models/deepseek4.cpp | crates/llama/src/{graph_arch,model}.rs | ✅ 批次7（hyper-connection 流 + 三比压缩注意力 + hash 层 + SQRT_SOFTPLUS/swiglu_clamp; MTP 图/embedding 抽头/vl 偏置未移, PARITY.md 批次7 §4） |
| 1486 | src/llama-quant.cpp | crates/llama/src/quant.rs + tools/quantize | ✅ 25/25 case 与参考 llama-quantize 逐字节一致(含 Q1_0/Q2_0) |
| 1408 | src/unicode.cpp | crates/llama/src/unicode*.rs | ✅ |
| 1339 | ggml/src/ggml-cpu/quants.c | crates/ggml/src/vec_dot.rs | ✅ 量化 vec_dot 位精确 |
| 1324 | src/llama-memory-recurrent.cpp | crates/llama/src/{kv_cache,context}.rs | ✅ recurrent 单序列 + state_write/read（代理RSSTATE: :766-1224 → `recurrent_state_seq_{write,read}`, mamba2 blob 与参考逐字节同, /slots 解禁; PLE 行/ON_DEVICE 未移, PARITY.md RSSTATE） |
| 1294 | src/models/qwen4exp.cpp | crates/llama/src/{graph_arch,model}.rs | 🔨 35/156 架构(见支持矩阵) |
| 1249 | ggml/src/ggml-alloc.c | compute.rs 内 arena 分配 | ➖ 等价简化 |
| 1232 | common/chat-peg-parser.cpp | crates/llama/src/chat_tools.rs | ✅ ChatPegBuilder 全部 tag 助手 + standard_json_tools 三模式 + constructed/python 式 + ChatPegMapper 状态机 + tagged_peg_parser; gemma4/minimax-m3 专用 mapper 未移 (代理CT) |
| 1213 | common/common.h | crates/common(最小)+llama crate 内 | 🔨 CLI 子集; grammar/json/speculative/ngram/jinja 其余待定 |
| 1193 | tools/imatrix/imatrix.cpp | crates/tools/imatrix + crates/llama/src/imatrix.rs | ✅ 代理IMATRIX: CLI/采集/归一化/GGUF+dat 写入/载入合并/show-statistics 全移植（+ `common/imatrix-loader.cpp` 173 行并入同一模块）; GGUF 元数据+张量表与参考相同, in-file 重写与 dat 输出**逐字节相同**; 另补 `ggml/src/quants_impl.rs`（`quant_weights != NULL` 侧量化器）⇒ `llama-quantize --imatrix` 19/19 ftype 逐字节（含 Q2_K_S/IQ4_NL/IQ4_XS）; 缺口: IQ1/IQ2/IQ3 量化器（需 `iq2xs/iq3xs_init_impl` 的 kmap/neighbours 运行时表）、MoE 路径未实测 |
| 1166 | src/llama-arch.cpp | crates/llama/src/arch.rs | ✅ 表+测试 |
| 1166 | common/console.cpp | crates/common(最小)+llama crate 内 | 🔨 CLI 子集; grammar/json/speculative/ngram/jinja 其余待定 |
| 1113 | common/fit.cpp | crates/common(最小)+llama crate 内 | 🔨 CLI 子集; grammar/json/speculative/ngram/jinja 其余待定 |

（完整 583 文件清单见 /tmp/ref_files.txt 与 FILE_MAP.md 大表；本表覆盖全部 >1500 行文件与分类规则）

## CLI 覆盖补齐：11 个 builder 全部可达 + json-schema 旗标（集成者, 2026-09-24）

`llama-cli` 的架构分发此前只到 qwen2/llama/gemma2/gemma3/phi3；本轮把 `crates/llama/src/context.rs::ForwardWeights` 扩到与 `crates/llama/src/graph_arch.rs` 的 builder 一一对应（新增 `Qwen3`/`GptOss`），并在 CLI 里补齐 `QWEN3`/`OPENAI_MOE`/`GEMMA4`/`GRANITE_HYBRID`/`LFM2MOE`/`QWEN35` 六条臂；无 builder 的 arch 走 `other` 臂，打印 `arch '<name>' (<LlmArch>) has a loader but no forward builder …` 并 exit 1（参考版在 graph 构建中 abort）。

同时接线 `-j/--json-schema` 与 `-jf/--json-schema-file`（common/arg.cpp:2279-2300）：四个语法旗标写同一个 `params.sampling.grammar`，**命令行后者覆盖前者**；失败文案与参考 `diff` 逐字节相同（`error while handling argument "-j": JSON schema conversion failed:\n…` + 该选项的 usage 块）。

实测（每次全新 reference server + 首请求, `temperature=0`）：qwen3 (fa on/off) 16/16、gpt-oss MXFP4 16/16、gemma4 16/16、granitehybrid 16/16、lfm2moe 16/16、qwen35 16/16、phi3 16/16、qwen2 16/16；`-j` 小 schema 与服务端 `json_schema` 字段 **11/11**，输出 `{"name":"John","age":30}` 且经 `jsonschema` 校验合法。脚本 `parity/run_cli_arch_parity.sh` + `parity/cli_parity_cmp.py`；细节见 `PARITY.md`/`FILE_MAP.md` 同名小节。

## 编码器架构 BERT / T5-encoder（代理BERT, 2026-09-25）

| 参考文件 | 行数 | Rust 位置 | 状态 |
|---|---|---|---|
| src/models/bert.cpp | 221 | `crates/llama/src/graph_arch.rs::build_bert_forward` + `crates/llama/src/model.rs`（`LlmArch::BERT` 装载臂） | ✅ BERT 分支（token/type/pos 嵌入 + post-LN + GELU-SEQ FFN + CLS/MEAN/NONE/LAST pooling）；真实文件 bge-m3-Q8_0 **T=1 99.02% 位精确**；jina/nomic/modern/euro/neo-bert 变体未移植（⊘，见 PARITY.md） |
| src/models/t5.cpp | 370 | `graph_arch::build_t5_encoder_forward`（`graph<true>`，:264-358）+ `model.rs`（`LlmArch::T5ENCODER` 装载臂）+ `model.rs::load_arch_hparams_t5` | ✅ 编码器路径（相对位置 bucket + kq_scale=1.0 + RMSNorm + gated GELU + enc_norm）；真实文件 t5-v1_1-xxl-encoder-Q5_K_S **T=1 100% 位精确**；解码器 `graph<false>`（:109-262）与 cross-attention ⊘ 未移植 |
| src/models/t5encoder.cpp | 44 | 同上（与 t5 共用 `graph<true>`） | ✅ |
| src/models/eagle3.cpp | 339 | `graph_arch::{build_eagle3_encoder_forward, build_eagle3_decoder_forward, Eagle3Weights, Eagle3Params}` + `crates/llama/src/eagle.rs`（头加载器 + ctx_other 零拷贝外置存储; model.rs 的通用加载器无 EAGLE3 臂, 见 PARITY.md §EAGLE3）+ `context.rs::{new_eagle3, encode_eagle3, eagle 分支}`（含 llama.cpp:127 的 t_layer_inp 记录） | ✅ 编码器 `graph<true>`（:103-146, t_h_nextn 输出）+ 解码器 `graph<false>`（:151-326, d2t scatter 含）; `--spec-type draft-eagle3` 全链 parity 4 格 PASS ×2FA（parity/eagle_parity.sh）; 驱动 `common_speculative_impl_draft_eagle3` 见 speculative.cpp 行 |
| src/models/dflash.cpp | 1043 | `crates/llama/src/dflash.rs`（加载器 + KV 注入臂 :608-677 + 噪声块臂 :679-853 含 DSpark markov/置信度头 :295-406 + d2t scatter :825-839 + DFlash2 conv/selector :408-567 + DSV4 DSpark 骨干 :52-93/:855-1028; ctx_other 零拷贝同 eagle.rs）+ `context.rs::{new_dflash(含 dsv4 iswa 环分支), dflash 分支, set_causal_attn}` + speculative.rs 的 `CommonSpeculativeImplDraftDflash` | ✅ 代理DSV4STATE 2026-10 + 代理DFLASH2 2026-09-27: 纯 DFlash1/DSpark✅（rope=NEOX, 链 153/153）; DFlash2✅（lattice 候选 id 逐位同 + dump ≤7.9e-5 ulp + chain 17/17 轮 + 服务器格 ×2FA）; DSV4 DSpark✅（rope=NORM per llama-model.cpp:3053-3059, 注入态逐字节同 5864B, 链 153/153, ×2FA）; 拒载: M-RoPE（:598-606）; 未移: markov d2t 散射（:351-359） |
| src/llama-graph.cpp:2740-2763 / :2451-2567 / :2480-2513 / :2587-2599 / :3890-3923 | ~120 | `graph_arch::{attn_no_cache, build_pos_bias, relative_position_bucket}` + `EncodeInputs`/`EncoderParams` | ✅ no-cache attention 非 FA 分支 + t5 位置偏置（bucket 表 433/433 同参考） |
| src/llama-graph.cpp:3677-3772 | 96 | `graph_arch::build_pooling`（NONE/MEAN/CLS/LAST） | ✅；RANK 头（`cls`/`cls_out`+tanh，:3722-3766）⊘ 张量已装载、pooling 未实现 |
| src/llama-context.cpp:1466-1659 / :964-1001 | ~250 | `crates/llama/src/context.rs::EncoderContext`（`build`/`run`/`encode`）+ `EncodeEmbeddings` + `resolve_pooling` | ✅ 端口新增的 `llama_encode` 入口（解码 `DecodeContext` 零改动）；FA 分支⊘（缺 F32→F16 cast 算子） |
| ggml/src/ggml-cpu/llamafile/sgemm.cpp（tinyBLAS F32 / Q8_0 内核） | — | — | Q8_0：经 `encode_gemm_probe` 证明与端口 `vec_dot` **逐位相同**；F32 tinyBLAS ⊘ 未建模（1e-4 级差异种子，见 PARITY.md 根因节） |
| parity/ref_encode_dump.cpp / ref_mulmat_q8_dump.c / ref_t5_bucket.c + gen_encode_ref.sh | — | `parity/encode_*.bin`（11 档）、`parity/mulmat_q8_bert_ref.bin`、`parity/t5_bucket_ref.txt` | ✅ 真值工具（参考版本 revision 无 embedding 示例程序，直连 libllama.so） |

## LoRA 适配器（代理LORA, 2026-09-25）

`src/llama-adapter.cpp/.h`（523+92 行）此前未移植 ⇒ `--lora` 静默不可用；本轮补齐。

| 参考文件 / 行 | 行数 | Rust 位置 | 状态 |
|---|---|---|---|
| src/llama-adapter.cpp:151-423 `llama_adapter_lora_init_impl` | 273 | `crates/llama/src/adapter.rs::adapter_lora_init_impl` | ✅ 元数据 dump/契约（`general.type`/`general.architecture`/`adapter.type`/`adapter.lora.alpha`/`adapter.alora.invocation_tokens`）、`.lora_a`/`.lora_b` 配对、形状校验、原始 F16/F32 载荷落地；错误文案与 C 逐字相同 |
| src/llama-adapter.cpp:425-510（init/free/meta_*） | 86 | `adapter::{load_adapter_lora, AdapterLora::meta_*}` | ✅ 路径入口 + 4 个 meta 访问器；`free` 由 `Rc`/`Context` 生命周期取代；FILE* 入口 ⊘ |
| src/llama-adapter.h:48-88（weight/adapter 结构） | 41 | `adapter::{LoraWeight, AdapterLora}` + `get_scale`/`get_weight`/`get_n_nodes` | ✅ |
| src/llama-adapter.cpp:16-136 `llama_adapter_cvec` + llama-graph.cpp:1508-1512 + llama-context.cpp:1376-1389 | ~160 | — | ⏳/⊘ 控制向量未移植（`--control-vector*` 亦未接，common/common.cpp:2002-2100） |
| src/llama-graph.cpp:1514-1543 `build_lora_mm` / :1545-1581 `build_lora_mm_id` / :2389-2405 `build_inp_embd` lora | 109 | `adapter::{lora_mm, lora_mm_s, lora_mm_id, lora_embd}` | ✅ `lora_mm + add` 模式原样；已接入 11 个解码 builder（qwen2/llama/qwen3/phi3/gemma2/3/gemma4/gpt-oss/granite/lfm2/qwen35）的**全部**权重投影 + 词嵌入 + lm head，外加 encoder-only 的 bert/t5 投影（`attn_no_cache` 的 wo + 各自 q/k/v 与 FFN）；`w_s`（NVFP4 per-tensor scale）参数保留但端口无该机制 |
| src/llama-context.cpp:1332-1374 `set_adapters_lora`/`are_same` + llama.h:723 `llama_set_adapters_lora` | 43 | `adapter::{set_adapters_lora, adapters_lora_are_same, clear_adapter_lora}` | ✅ 零 scale 丢弃、同指针保留首个 scale；活动集落在 `adapter.rs` 线程局部（`context.rs` 不在本轮所有权内，见 PARITY.md 的接线说明） |
| common/arg.cpp:2950-2974 `--lora`/`--lora-scaled` + :1347-1389 `parse_csv_row` | 65 | `crates/tools/llama-cli/src/main.rs`（`lora_adapters` + `parse_csv_row`/`string_split`/`stof`） | ✅ 可重复 + 逗号分隔 + `FNAME:SCALE`；格式错误文案与参考逐字节相同 |
| common/common.cpp:1342-1358 / :1507-1508 / :1667-1676（装载→施加） | ~40 | CLI 主流程 | ✅ adapter 在 `DecodeContext` 之前装载（张量必须低于图 watermark），随后设 scale |

验证：合成 rank-8 adapter（端口 GGUF writer 生成，sha256 钉住）在 qwen2.5-0.5b 上
**`-fa on`/`-fa off` 各 16/16 token 同参考**，top-5 logprob 差 ≤0.382；基座模型不变（16/16）；
`--lora-scaled :0` 与基座 prefill logits **逐位相同**；`:0.5` 亦 16/16（含 0.035 logprob 的平局步骤）。
细节与命令见 `PARITY.md` 同名小节，测试在 `crates/llama/src/adapter.rs`（默认 6 个）与
`crates/llama/tests/lora_e2e.rs`（`#[ignore]`，含参考捕获常量）。

## 架构批次 6 个: gpt2 / phi2 / starcoder2 / command-r / gptneox / olmo2（代理BATCH, 2026-09-24）

架构计数 **12/156 → 18/156**（`model.rs::arch_tensors_support` 标 Partial 的集合 + qwen2/llama 两个 Full）。

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 148 | src/models/gpt2.cpp | graph_arch.rs `build_gpt2_forward` + model.rs GPT2 臂 | ✅ 合成 GGUF 与参考 16/16（-fa off/on），Δlogprob 1e-4 |
| 142 | src/models/phi2.cpp | `build_phi2_forward` + PHI2 臂 | ✅ 16/16，1e-4 |
| 158 | src/models/starcoder2.cpp | `build_starcoder2_forward` + STARCODER2 臂 | ✅ 16/16，1e-4 |
| 144 | src/models/command-r.cpp | `build_command_r_forward` + COMMAND_R 臂 | ✅ 16/16，6e-4/1.3e-3（`logit_scale`+per-head q/k LN+并行残差） |
| 219 | src/models/gptneox.cpp | `build_gptneox_forward` + GPTNEOX 臂 | ✅ 16/16（`use_par_res` 两支变体各一份文件），1e-4 |
| 208 | src/models/olmo2.cpp | `build_olmo2_forward` + OLMO2 臂 | ✅ 16/16，1e-4；SWA 变体仅元数据校验（见下） |
| — | crates/llama/tests/arch_batch_e2e.rs（新） | 合成 GGUF 生成器 + 每 arch loader 钉（9 个测试） | ✅ 默认 8 跑 / 1 `#[ignore]` 造文件 |
| — | parity/arch_batch_parity.sh + arch_batch_cmp.py（新） | 12 次对照驱动 + token/Δlogprob 汇总 | ✅ 12/12 全 16/16 |

未覆盖：**olmo2 `graph<true>`（SWA）**只做了元数据/hparams 校验（`KvCache::layer_is_swa` 与 rope 覆盖已在 builder 内，但 CLI 走统一 cache；需要 `new_with_swa` 才能跑 SWA 前向）；`LLM_TYPE_*` 未建模；未移植的 `build_ffn` 变体（`act_scales`/`swiglu_clamp_shexp`）本批不涉及。下一批队列（含每个 arch 的 C file:line）见 `FILE_MAP.md` 同名小节。

## 架构批次 2 个: codeshell / orion / olmo / xverse / internlm2 / exaone / gemma(v1) / falcon（代理BATCH2, 2026-09-25）

架构计数 **18/156 → 26/156**（同口径）。协议/合成文件契约沿用第一批（见 PARITY.md 同名小节），
合成文件在 `/tmp/arch-batch2/`、测试 `crates/llama/tests/arch_batch2_e2e.rs`（默认 10 跑 + 2 `#[ignore]`）、
对照驱动 `ARCH_BATCH2=1 ./parity/arch_batch_parity.sh`。

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 153 | src/models/codeshell.cpp | graph_arch.rs `build_codeshell_forward` + model.rs CODESHELL 臂 | ✅ 合成 GGUF 与参考 16/16（-fa off/on），Δlogprob 1e-4 |
| 141 | src/models/orion.cpp | `build_orion_forward` + ORION 臂 | ✅ 16/16，1e-4（LN+bias、SILU-PAR、NEOX） |
| 142 | src/models/olmo.cpp | `build_olmo_forward` + OLMO 臂 | ✅ 16/16，0（**图里三处 `ggml_norm` 无权重**；`clamp_kqv` 可选） |
| 136 | src/models/xverse.cpp | `build_xverse_forward` + XVERSE 臂 | ✅ 16/16，1e-4（RMS+SILU-PAR，rope NORM） |
| 139 | src/models/internlm2.cpp | `build_internlm2_forward` + INTERNLM2 臂 | ✅ 16/16，1e-4（与 xverse 同构，仅 rope mode 不同；output 必填） |
| 136 | src/models/exaone.cpp | `build_exaone_forward` + EXAONE 臂 | ✅ 16/16，1e-4（`rope_freqs` 走 rope src[2]） |
| 139 | src/models/gemma.cpp | `build_gemma1_forward` + GEMMA 臂 | ✅ 16/16，0（无 post-norm/无 logit softcap 的 v1 图；GELU-PAR） |
| 161 | src/models/falcon.cpp | `build_falcon_forward` + FALCON 臂 | ✅ 15/15 对齐，1e-4（参考少报 1 个 byte token；`attn_norm_2` 40B 形状 = loader 钉） |
| — | crates/llama/tests/arch_batch2_e2e.rs（新） | 合成 GGUF 生成器 + 每 arch loader 钉 + falcon context probe | ✅ 默认 10 跑 / 2 `#[ignore]` |
| — | parity/arch_batch_parity.sh（`ARCH_BATCH2=1`）+ arch_batch_cmp.py（子序列对齐） | 16 次对照驱动 + token/Δlogprob 汇总 | ✅ 16/16 格全对齐 |

参考侧记录性质（**不是**端口 bug，任何 arch 生成到 SPM byte token 都会遇到）：pinned server 的
`completion_probabilities` 会**漏掉** piece 非 UTF-8 的 token 条目（`tokens_predicted` 计数、字节并进下一条；
server-task.cpp:285-290 `validate_utf8`），falcon 两格因此从 11/15 变成 15/15 对齐 + 1 条"参考少报"。归因链与复现工具见 PARITY.md §4。

未覆盖：无真实文件（8 个 arch 全为合成 GGUF，与第一批同）；falcon 40B 形状只做 loader/结构钉；
exaone 的 `rope_freqs` 只在端口侧钉了"接上/不接会产生不同 logits"；alibi 系（baichuan/bloom/mpt/starcoder/refact）
需要改 `kv_cache.rs`/`graph.rs` 的 mask + softmax `max_bias`，不在本批文件所有权内，已并入下一批队列（FILE_MAP.md）。

## 多模态视觉路径：clip.cpp + mtmd-image.cpp + mtmd.cpp（代理MM, 2026-09-25）

参考面：`tools/mtmd/clip.cpp` 6087 行 + `clip-impl.h`/`clip-model.h`/`clip-graph.h` + `models/qwen3vl.cpp`
+ `mtmd-image.cpp` 1743 行 + `mtmd.cpp` 视觉半边 + `mtmd-helper.cpp` 解码粘合。
只接 `qwen3vl_merger`（本机 3 个 mmproj 全为该类型）。

| 参考文件 / 行 | 行数 | Rust 位置 | 状态 |
|---|---|---|---|
| tools/mtmd/clip.cpp（视觉路径：hparams 1244-2070 / 张量表 2109-2600 / ViT 图 312-819 / encode 4427-5870）| ~2000 | `crates/llama/src/clip.rs` | ✅ 端到端跑通，embedding 与参考 L2 rel 1.2e-4~2.7e-4、cos ≥0.99999997（PARITY.md） |
| tools/mtmd/models/qwen3vl.cpp + qwen2vl.cpp `build_inp_with_temporal_merge` | 391 | `clip::build_graph` | ✅ 含 2×2 空间合并、位置嵌入双线性插值（ALIGN_CORNERS）、M-RoPE、GELU FFN、4× merger；deepstack 分支未接（本机文件 `is_deepstack_layers` 全 false，且无 `v.deepstack.*` 张量） |
| tools/mtmd/mtmd-image.cpp `img_tool` + dyn_size/fixed_size 预处理 | ~850 | `clip::img_tool::*`、`ClipContext::image_preprocess` | ✅ Pillow 兼容 22-bit 定点重采样（bilinear/bicubic/lanczos）、smart_resize、PAD_CEIL/NEAREST/NONE；llava-uhd 系切割器 ⏳ |
| tools/mtmd/mtmd.cpp 视觉半边（marker/tokenizer/chunk/encode）| ~600 | `crates/llama/src/mtmd.rs` | ✅ 含 BOS/EOS 语义、`<|vision_start|>`/`<|vision_end|>` 包装、M-RoPE `n_pos=max(nx,ny)`、`decoder_pos` |
| tools/mtmd/mtmd-helper.cpp eval 粘合 + :305-362 音频解码 | ~250 | `mtmd::MtmdEval` + `mtmd::{is_wav_file,is_mp3_file,is_flac_file,audio_from_wav_bytes,audio_from_flac_bytes}` | ✅ 分批发 eval + M-RoPE 位置（端口 `LlamaBatch` 无 `embd` 字段，等价走"输入矩阵替换 tok_embd"，见 PARITY.md）; 音频解码 WAV + **FLAC（代理AUDIO4: spec 子集解码器, dr_flac 数值管线逐式照抄, 6 个 ffmpeg 产文件对参考嵌入逐位同, `parity/audio_flac_parity.sh` + `tests/mtmd_flac`）**; mp3 保持响亮报错（PARITY.md 音频第四轮 §C） |
| mtmd-helper.cpp:360-420 图像解码（stb_image）| — | `mtmd::{bitmap_from_bytes,bitmap_from_file}` | ➖ 自写 PNG（inflate 全类型块 + 5 种 filter）/BMP 24-32bit/PPM-PGM；JPEG/WebP/GIF/HDR ⛔ |
| tools/mtmd/mtmd-cli.cpp | 587 | `crates/tools/llama-mtmd-cli` | ✅ 单轮图像问答 + parity 工具旗标；交互 chat/音频/视频/`--jinja` ⛔ |
| common/parsers/*.cpp (16 个特化处理器, 2571 行) | 2571 | crates/llama/src/chat_parsers.rs | ✅ 全量 + gemma4/minimax mapper; 43 synth 用例逐字节、真模板 73 用例逐字节(mini-jinja 特性补齐后, 代理JINJA) (代理CHATPARSER) |
| tools/mtmd/mtmd-audio.cpp | 1557 | crates/llama/src/mtmd_audio.rs | ✅ 逐位 (见上行 mtmd-audio) |
| tools/mtmd/models/whisper-enc.cpp（音频图, 137 行）| 137 | `crates/llama/src/clip.rs`（`build_whisper_graph` + `build_vit_audio` 共享 build_vit 路径） | ✅ 代理AUDIO2 (2026-09): conv1d stem、分离 qkv ViT、AvgPool1d（AUDIO3 起为字面 GGML_OP_POOL_1D）、StackAudioFrames、6 arch 投影器尾（qwen2a/ultravox/voxtral/meralion/glma/musicflamingo）; 合成协议对参考逐位同（FA on/off 双路, `parity/audio_mtmd_parity.sh`）; clip.cpp 的 `has_audio_encoder` 加载臂/`n_output_tokens`/`n_mmproj_embd` 音频路径同步补齐 |
| tools/mtmd/models/qwen3a.cpp + gemma4ua.cpp + conformer.cpp | 323 | `crates/llama/src/clip.rs`（`build_qwen3a_graph`/`build_gemma4ua_graph`/`build_conformer_graph`） | ✅ 代理AUDIO3 (2026-09): qwen3a/gemma4ua 逐位（`parity/audio_mtmd_parity2.sh`, FA 双路）; lfm2a 参考接受、≤5.7e-6（卷积模块 ULP 残差开档）; 伴随 ggml 新算子 roll/conv_2d_dw_direct/conv_2d_direct/relu/pool_1d 全 C-probe 逐位（parity/conformops_ref.bin + parity/pool1d_ref.bin）; granite-speech/gemma4a/parakeet/mimo/qwen3tts/pockettts 未移（PARITY.md 音频第三轮 §5 逐条 C file:line） |
| tools/mtmd/models/{granite-speech,gemma4a,parakeet,mimo-audio,qwen3tts-spkenc,pockettts-spkenc(+seanet)}.cpp | ~1670 | `crates/llama/src/clip.rs`（六个 `build_*_graph` + hparams/加载/set_inputs/n_tokens/n_embd 全臂） | ✅ 代理AUDIO4 (2026-09): **gemma4a/mimo_audio/qwen3tts_spkenc/pockettts_spkenc 逐位**（`parity/audio_mtmd_parity3.sh`, FA 双路）; granite_speech/parakeet 参考接受、≤4.2e-4/1.4e-6——根因=参考 ssm_conv AVX-512 向量化（PARITY.md §D 三重证明）; 伴随 sub/sin/cos/sqr/mean/pad_reflect_1d/elu/pad_ext 新算子 C-probe 逐位（parity/audioops_ref.bin）; qwen3tts_gen/pockettts_gen（输出生成器）仍未移 |
| 其它 ~50 个 projector（models/*.cpp）| ~8000 | — | ⛔ 按需（`ProjectorType::from_str` 能识别并显式报错） |
| ggml 新算子 | — | `ops.rs`/`compute.rs`: `Im2col`(GGML_OP_IM2COL)、`Upscale`(GGML_OP_UPSCALE/ggml_interpolate)、`cont_{2,3,4}d`、`cast` | ✅ 全部附加；核级真值 `parity/interp_ref.bin`（`parity/ref_interp_dump.c`） |

顺带修复：`mrope_cache_init` 缺 `indep_sects`（ops.cpp:5994）导致 VISION 模式 rope 从第 2 个 section 起全错 —— 修后视觉误差降 100 倍（PARITY.md）。

验证：`parity/mtmd_parity.sh`（fixture 生成 + 参考/端口两侧 dump + 对比报告）、
`parity/gen_mtmd_fixture.py`（PNG/BMP/PPM 生成器，仅用标准库）。
默认测试：`clip::tests::synth_mmproj_load_preprocess_encode`（合成 mmproj，无需大文件）、
`clip::tests::real_mmproj_metadata_and_geometry`、`mtmd::tests::image_decoders_agree`、
`compute::tests::interpolate_align_corners_vs_reference`。

## 架构批次 3 个: baichuan / bloom / mpt / starcoder / refact / plamo / stablelm / granite(dense) / minicpm（代理BATCH3, 2026-09-27）

架构计数 **26/156 → 35/156**（同口径：`model.rs::arch_tensors_support` 标 Partial 的集合 + qwen2/llama 两个 Full）。
协议/合成文件契约沿用前两批；合成文件在 `/tmp/arch-batch3/`、测试 `crates/llama/tests/arch_batch3_e2e.rs`
（默认 11 跑 + 2 `#[ignore]`：造文件、env 驱动的 parity driver）、对照驱动
`ARCH_BATCH3=1 ./parity/arch_batch_parity.sh`（**26/26 格全对齐**，含 baichuan-13B 的 72-token 长提示格；
端口侧由 `arch3_cli_driver` 打印 llama-cli 同格式 debug 行，`arch_batch_cmp.py` 零改动复用）。

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 121 | src/models/baichuan.cpp | graph_arch.rs `build_baichuan_forward` + model.rs BAICHUAN 臂 | ✅ 16/16（13B alibi + 7B rope 两变体 + 长提示格），Δlogprob ≤2.1e-3 |
| 151 | src/models/bloom.cpp | `build_bloom_forward` + BLOOM 臂 | ✅ 16/16，0（alibi 8 + fused qkv+bias + token_embd_norm） |
| 171 | src/models/mpt.cpp | `build_mpt_forward` + MPT 臂 | ✅ 16/16（最小 + `-full`），≤1.9e-3（alibi 走 GGUF KV、clamp、act-scales；Q/K 全宽 norm 分支在参考里不可达——mpt.cpp:43 FIXME） |
| 154 | src/models/starcoder.cpp | `build_starcoder_forward` + STARCODER 臂 | ✅ 16/16，0（**本 revision 无 alibi**，pos_embd 必需） |
| 160 | src/models/refact.cpp | `build_refact_forward` + REFACT 臂 | ✅ 16/16，0（alibi 8、rope NONE；MoE 文件显式拒） |
| 136 | src/models/plamo.cpp | `build_plamo_forward` + PLAMO 臂 | ✅ 16/16，≤1e-4 |
| 172 | src/models/stablelm.cpp | `build_stablelm_forward` + STABLELM 臂 | ✅ 16/16（12B parallel-residual + 3B partial-rope 两变体），≤1e-4 |
| 320 | src/models/granite.cpp（dense） | 复用 `build_granite_forward` + `GraniteParams::dense` + GRANITE 臂 | ✅ 16/16，0（4 个 scale + MoE 装载臂；deepstack 未移植） |
| 96 | src/models/minicpm.cpp（graph = granite） | 同 granite + MINICPM 臂 | ✅ 16/16，0（默认 scale 三连） |
| — | **ALiBi 机制**（llama-graph.cpp:2643/:2705 + llama-kv-cache.cpp:1692-1697 + llama-model.cpp:1419） | graph_arch.rs `attn_kv_cached` 的 `max_bias` 参数 + graph.rs `fill_kq_mask_alibi{,_f16}` + model.rs `use_alibi` 重判 | ✅ 斜率内核既有（compute.rs/flash_attn.rs），本批接通到图并钉死（mask 手算值 + slope [1/4,1/16,1/64,1/256]） |
| — | crates/llama/tests/arch_batch3_e2e.rs（新） | 合成 GGUF 生成器 + 每 arch loader 钉 + alibi softmax vs 朴素 | ✅ 默认 11 跑 / 2 `#[ignore]` |

未接线: DecodeContext/CLI 的 9 个 ForwardWeights 臂与 alibi mask 填充（context.rs 归并行代理，
集成者接线；批次 3 的 arch 只能从测试文件驱动）。回归: 批次 1/2 parity 全对齐、qwen2.5 锚点
16/16 双 FA、`cargo test --workspace` 492/0。批次 4 队列（mamba 家族 / MLA / MoE 变体等，含 C file:line）
见 FILE_MAP.md 同名小节。

## 投机解码 CLI/server 接入 + 批次 3 驱动接线（集成者INTEG, 2026-09-28）

* llama-cli / llama-server 的 `--spec-*`/`-md` 参数面（arg.cpp:4135-4253 已接; `--spec-ngram-*` 值旗标 arg.cpp:4254-4378 已接
  —— 代理DSV4RS/NGRAM 2026-09-26, 设备类旗标未接, 见 PARITY.md §1 表）; CLI 走 speculative-simple 驱动, server 走按槽状态机（单请求对照 36/36 token 与参考
  相同, 接受率双侧 100%, port 1.41x / ref ~2.1x）; 多槽投机已接未对照。
* `ForwardWeights` 批次 3 的 7 个新变体 + granite-dense/minicpm 复用; alibi mask 在 `DecodeContext` 的
  单序列/多序列两条填充路径均接（`use_alibi` 语义）; llama-cli 9 arch 臂经 llama-cli 本体对照
  13 格 × fa off/on = **26/26** 与参考一致（此前只能从 arch_batch3_e2e.rs 驱动）。
* 回归: 工作区 492/0; 批次 1/2 parity 全对齐; qwen2.5 双 FA 锚点 16/16; server parity MATCH。

## 架构批次 4 个: MoE 家族 — qwen2moe / qwen3moe / granite-moe / phimoe / arctic / olmoe / ernie4-5-moe + smollm3 / seed-oss / openelm（代理BATCH4, 2026-09-28）

架构计数 **35/156 → 45/156**（同口径：`model.rs::arch_tensors_support` 标 Partial 的集合 +
qwen2/llama 两个 Full）。协议/合成文件契约沿用前三批；合成文件在 `/tmp/arch-batch4/`、测试
`crates/llama/tests/arch_batch4_e2e.rs`（默认 11 跑 + 2 `#[ignore]`）、对照驱动
`ARCH_BATCH4=1 ./parity/arch_batch_parity.sh`（**22/22 格全对齐**, 含 ernie 的无 shexp 变体格;
每格全新参考 server + 首请求, fa off/on 各 16 token）。

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 194 | src/models/qwen2moe.cpp | graph_arch.rs `build_qwen2moe_forward` + model.rs QWEN2MOE 臂 | ✅ 16/16（off/on），Δlogprob 1e-4（**norm_w=false** + sigmoid 门控共享专家 :147-165 + 1-D `ffn_gate_inp_shexp`） |
| 179 | src/models/qwen3moe.cpp | `build_qwen3moe_forward` + QWEN3MOE 臂 | ✅ 16/16，0（per-head q/k norm + norm_w=true, tie 头） |
| 84 | src/models/granite-moe.cpp（graph = granite 的, models.h:1680） | 复用 `build_granite_forward` + GRANITE_MOE 臂 | ✅ 16/16，0（专家=稠密 n_ff; 不读 rope_finetuned） |
| 55 | src/models/phimoe.cpp（graph = phi3 的, models.h:661） | `build_phimoe_forward` + PHIMOE 臂（+`build_norm_rms_b`） | ✅ 16/16，2e-4（唯一带偏置 RMS 的 MoE; rope_factors 长短选择未对照——合成两侧都不带） |
| 180 | src/models/arctic.cpp | `build_arctic_forward` + ARCTIC 臂 | ✅ 16/16，1e-4（双 FFN: 稠密方阵 + **inpSA 上的 MoE** :136-139） |
| 173 | src/models/olmoe.cpp | `build_olmoe_forward` + OLMOE 臂 | ✅ 16/16，3e-4（全宽 q/k norm, reshape=false; [n_embd] k-norm 仅 MHA 成立——GQA 在参考也 abort） |
| 133 | src/models/ernie4-5-moe.cpp（hparams = ernie4-5.cpp 的 arch 分支, meta.rs 已 Full） | `build_ernie45_moe_forward` + ERNIE4_5_MOE 臂 | ✅ 16/16 + `-nosh` 变体 16/16，3e-4（dense-lead/step 切层, `exp_probs_b` 路由偏置, 可选无门控 shexp） |
| 152 | src/models/smollm3.cpp | `build_smollm3_forward` + SMOLLM3 臂 | ✅ 16/16，4e-4（nope: `n_no_rope_layer_step=4` 每 4 层第 4 层不 rope） |
| 151 | src/models/seed-oss.cpp | `build_seed_oss_forward` + SEED_OSS 臂 | ✅ 16/16，0（attn_post_norm 即 FFN norm, 无 ffn_norm） |
| 171 | src/models/openelm.cpp | `build_openelm_forward` + OPENELM 臂 | ✅ 16/16，1.3e-3（fa on）（**逐层头数/FFN 宽**走 GGUF 数组 + 融合 qkv 头段切分 + per-head q/k norm; per-layer KV 宽度 `KvCache::new_with_dims` 先例） |
| — | crates/llama/tests/arch_batch4_e2e.rs（新） | 合成 GGUF 生成器 + 每 arch loader 钉 + 3×n_moe_layer 个 MUL_MAT_ID 结构钉 + env 驱动 parity driver | ✅ 默认 11 跑 / 2 `#[ignore]` |
| — | parity/arch_batch_parity.sh（`ARCH_BATCH4=1`; 批次 3/4 公共格逻辑合并为 `run_env_driver_cell`） | 22 次对照驱动 + token/Δlogprob 汇总 | ✅ 22/22 格全对齐 |

未接线: DecodeContext/CLI 的 10 个 ForwardWeights 臂（context.rs 归并行代理, 集成者接线;
注意 openelm 的 per-layer KV 宽度与 olmoe 的 MHA 约束）。nemotron 因 RELU_SQR 需 ggml crate
新增 RELU/SQR 算子（不在本批所有权）移入批次 5 队列。回归: 批次 1/2/3 parity 全对齐、
qwen2.5 双 FA 锚点 16/16、`cargo test --workspace` **503/0**。

## llama-server OpenAI 兼容面（代理SRV, 2026-09-28）

`crates/tools/llama-server` 新增 `chat.rs`（`oaicompat_chat_params_parse` + 模板渲染 + 语法包裹 +
特殊 token 重复修复版 tokenize）、`engine.rs` 的 OAI 响应族/嵌入任务/`GET /slots` 快照、
`main.rs` 的 `-fe/--embeddings`/`--pooling` 与全部新路由。实测 `parity/run_server_parity_chat.sh`
**24/24 检查全对齐**（qwen2.5: chat 非流式/流式/usage/logprobs/json_schema/json_object/stop/
alias、`/v1/completions` 双式、`/models`、`/slots`、rerank 501; bge-m3: embeddings 四式
**逐位相同**）; 既有 `run_server_parity.sh` 仍 MATCH, 工作区 **506/0**。
发现的所有权外问题（已报给集成者, 见 PARITY.md §3）: `vocab.rs::tokenizer_st_partition`
重复尾片 bug（端口 `chat.rs::tokenize_prompt` 临时绕开 + 回归测试）;
`context.rs::decode_batch` 需 `output_all` hook 才能支持解码器 `-fe` embeddings;
`chat.rs` mini-jinja 需 tools 绑定 + `tojson` 才能支持 chat `tools`。

## 批次 4 CLI/DecodeContext 接线（集成者INTEG, 2026-09-28）

* 批次 4 的 10 arch（§上节）全部 CLI 可达: `ForwardWeights` 9 新变体 + llama-cli 9 arch 臂
  （granite-moe 复用 `Granite` + `GraniteParams::dense`, 同一臂扩为 `GRANITE|MINICPM|GRANITE_MOE`）。
* 特例落地: openelm 逐层 KV 宽度（`kv_dims` Openelm 臂 → `KvCache::new_with_dims`）; olmoe 的
  MHA 前置守卫（GQA 文件显式报错, 参考亦 abort）; ernie 的 dense-lead/MoE-step 切层透传
  `n_moe_layer_step/n_layer_dense_lead/n_ff_shexp`; qwen2moe 的 `ffn_gate_inp_shexp` sigmoid
  门控路由随权重组装进入 CLI。
* 验证: `ARCH_BATCH4_CLI=1 ./parity/arch_batch_parity.sh`（llama-cli 本体, 22/22 逐 token 16/16,
  worst Δlogprob ≤1.3e-3）+ env 驱动 22/22 复验; 批次 1/2/3、qwen2.5 双 FA 锚点不回归;
  `cargo test --workspace` **507/0**（批次 4 代理记录的 503 之后另有并行代理 +4）。

## 架构批次 5 个: mamba 家族 — mamba / mamba2 / jamba / nemotron-h（代理BATCH5, 2026-09-24）

架构计数 **+4**（`arch_tensors_support` 口径 46 → 50；任务账面 55 → 59/156）。recurrent-state
架构首次成批落地：`RecurrentState` 驱动泛化（granite-hybrid 先例），mamba1/mamba2 两个 mixer
（mamba-base.cpp 9-151/153-302）+ 三张 arch 图。**N=48** 生成（递归越过 prompt）+ `-long`
长提示格（conv-state 滑窗 T>1）。合成文件 `/tmp/arch-batch5/`、测试
`crates/llama/tests/arch_batch5_e2e.rs`（默认 8 跑 + 2 `#[ignore]`）、驱动
`ARCH_BATCH5=1 ./parity/arch_batch_parity.sh` —— **20/20 格全对齐**（12 基础 + 8 长提示；
`ARCH_BATCH5_CLI=1` 走 llama-cli 本体）。

| 行数 | 参考文件 | Rust 对应 | 状态 |
|---|---|---|---|
| 151 | src/models/mamba-base.cpp（`llm_build_mamba_base` 的两个 mixer） | graph_arch.rs `build_mamba1_mixer` / `build_mamba2_mixer`（granite 的 `build_mamba2_layer` 改为委托后者, 逐字保留原函数体） | ✅ 4 arch 复用, 20/20 格 |
| 112 | src/models/mamba.cpp（mamba1; mamba2 经 models.h:942 复用此图） | `build_mamba_forward` + model.rs MAMBA/MAMBA2 臂 | ✅ 47-48/48 ×2 + long ×2 each, Δlogprob ≤1e-4 |
| 198 | src/models/jamba.cpp（hybrid mamba1+无 rope attention; 逐层 dense\|MoE; dt/B/C RMS 三元组） | `build_jamba_forward` + JAMBA 臂 | ✅ jamba 48/48 ×2（≤1e-4）+ jamba-moe 48/48 ×2（≤8e-4） |
| 340 | src/models/nemotron-h.cpp（三态层; relu² dense\|MoE; latent 投影; MTP 头加载 1:1） | `build_nemotron_h_forward` + `build_ffn_relu_sqr`（clamp(0,inf)+mul 组合, 无需 ggml RELU/SQR 算子）+ `build_moe_ffn_relu_sqr` + NEMOTRON_H 臂 | ✅ dense 48/48 ×2 + long ×2（on 1.1e-3）; MoE 48/48 ×2（≤2e-4） |
| — | crates/llama/tests/arch_batch5_e2e.rs（新） | 合成 GGUF 生成器 + 每 arch loader/hparams/is_recr 钉 + **batch-vs-step 状态一致性**（6 spec）+ env 驱动 parity driver | ✅ 默认 8 跑 / 2 `#[ignore]` |
| — | parity/arch_batch_parity.sh（`ARCH_BATCH5=1`, b5_model/-long/N=48） | 20 次对照驱动 + token/Δlogprob 汇总 | ✅ 20/20 格全对齐 |

接线同步落地（不再等集成者）: `ForwardWeights::{Mamba,Jamba,NemotronH}`（mamba/mamba2 共用
Mamba）+ `recurrent_dims()` 三新臂 + llama-cli 4 arch 臂（`ARCH_BATCH5_CLI=1` 抽查 4/4 格对齐）。
注意: granite 的 `granite_state_vs_full_recompute`（batch-vs-step 位同断言）在本批之前的树上已
失败（原始函数体逐位复现同值 0.94211185, PARITY.md 既有"granite 4/16 大残差"开档, 与本批无关）;
`granite_4_0_h_tiny_reference_parity` 参考对照锚点**通过**（文档带内）。回归: 批次 1/2/3/4 parity
全对齐（12+16+20+20 格）、qwen2.5 双 FA 锚点 16/16、`cargo test --workspace` **522/0**（基线 514 + 8）。

## 增量（2026-09-30）: llama-server chat tools 面

* `POST /v1/chat/completions`(+`/chat/completions`) 的 `tools`/`tool_choice`/
  `tool_calls`/流式增量/`finish_reason:"tool_calls"`: **已接线**（库级
  `crates/llama/src/chat_tools.rs` jinja+autoparser 路径）——`parity/run_server_parity_chat.sh`
  由 24 检查（tools 为预期差异）扩为 **29 检查全 MATCH**。`/props` 的
  `chat_template_caps` 换为真实探测（9 项与参考逐项同）。
* 仍不在覆盖面（与 PARITY.md 同节一致）: 端口 ggml 的 GGML_OP_POOL_1D 与单张量（src1==NULL）GLU 构造器（whisper-enc 图用 pool_2d/视图+swiglu_split 的位同等价构造过渡, 见 PARITY.md 接线项）、真模板 jinja 特性（macro/namespace, 见 PARITY.md 接线项; `common/parsers/*` 已移入 chat_parsers.rs）、
  `continue_final_message` 前缀拼接、reasoning-budget 键、触发器 PATTERN 形态、
  Responses/Anthropic 等其余未移植端点维持原状。

## 架构批次 6b（2026-09-24）: nemotron(dense) / grok / chameleon / deci / jais / falcon-h1 / plamo2

计数 **54 → 61/156**（任务账面 63 → 70/156）。28/28 合成 parity 格全 48/48 对齐
（9 arch × fa off/on + 5 `-long` 长提示格, `ARCH_BATCH6B=1 ./parity/arch_batch_parity.sh`）;
`cargo test --workspace` 564/0。详见 PARITY.md 批次 6b 节（含 4 个排障坑与
chameleon-swin 变体的未对齐记录）。

## 架构批次 7: deepseek4 + 压缩 DSV4 KV cache（代理BATCH7, 2026-09-26）

落地: src/models/deepseek4.cpp（1502）+ src/llama-kv-cache-dsv4.cpp（2253, 第四种 cache 形态）
+ 新 ggml 算子 `ggml_dsv4_hc_{comb,pre,post}`/`ggml_swiglu_clamp`/`ggml_sqrt`/`ggml_rope_ext_back`
（逐位 oracle: parity/ref_dsv4_dump.c → parity/dsv4_ops_ref.bin）+ 两个隐形数值契约修复
（HINT_SRC0_IS_HADAMARD → 快速 Walsh-Hadamard 变换内核; mul_mat_id F32 的 vec_dot_f32）。
合成 parity 16/16 ×2FA + `-long` 16/16 ×2（PARITY.md 批次7）。

架构计数 **61 → 62/156**（任务账面 70 → 71/156）。

## 架构批次 8: MoE 长尾家族 — hunyuan-moe / dots1 / bailingmoe / bailingmoe2 / glm4-moe / minimax-m2 / cohere2moe / exaone-moe（代理BATCH8, 2026-09-30）

`build_moe_ffn`-composition 条目一次落地 8 个 arch（hparams + 加载器 + 图 +
ForwardWeights/CLI 臂同批）。**32/32 合成 parity 格全 16/16**（8 arch × fa off/on ×
短/`-long` 长提示; cohere2moe/exaone-moe 带 n_swa=64, `-long` 格真实越窗 ——
llama-cli 的 DecodeContext 构造本批补了 llama-model.cpp:2687-2690 的 iswa 分裂判定）。
新增共享 helper `build_moe_ffn_silu_gu`（融合 `ffn_gate_up_exps` 分支,
llama-graph.cpp:2167-2190）。回归: 批次 1-7 parity 串行全对齐 + qwen2.5 双 FA 锚点
16/16 + `cargo test --workspace` 580/0（基线 569 + 11）。
未移（集成者条目）: minimax-01（lightning attention, 需 ggml crate 的 `ggml_exp`）、
bailingmoe3（KDA delta-net + MLA）、MTP 图类。详见 PARITY.md 批次 8 节。

架构计数 **62 → 70/156**（任务账面 71 → 79/156）。

## 架构批次 11a: 长尾队列前半 — apertus / grovemoe / qwen35moe / kimi-k3 / dots3note / minimax-m3 / qwen4exp（代理BATCH11A, 2026-10, append-only）

新增 ggml 件: `ggml_xielu`（unary, apertus FFN）、dup I32→F32（grovemoe 块专家选中 id
的 F32 缩放, ops.cpp:570-575）、repeat 的 2 字节形态（F16/BF16/I16, minimax-m3 块掩码）、
`pool_2d`（MSA 块分数 max 池化）。修复: pool_2d 的平面写偏移（≥1 平面全零）、xielu 的
FMA 收缩序（对参考逐位）、三处 fused wqkv 视图的 K/V 字节偏移、MSA cache 视图 stride、
MSA 两分支的 wo 漏乘与 kfa/vfa F16 回转、dots3note iswa 的 live 行视图、
`context.rs::n_pos_per_embd` 补 IMROPE（qwen35moe/qwen4exp）两臂（llama-model.cpp:3061-3070）。

机制: grovemoe 双 MoE（同 router logits, sigmoid 选择 + 块 id 除法映射, llama-graph.cpp:
2076-2118）; kimi-k3 的跨层残差注意库（res_push/res_mix + `dsv4_hc_pre` 复用, kimi-k3.cpp:
186-234）与潜空间 SITU MoE; dots3note 的 DSA lightning indexer 叠两侧不同 lora 几何的 iswa
对 + lid cache; minimax-m3 的 MSA（idx cache + 块 top-k + batch 掩码/decode gather 两路 +
`build_attn_msa_fa` 组折叠 FA, 单序列）; qwen4exp 的 hc 残差流（dsv4_hc_pre_gated/post 融合）。

验证: 合成 GGUF 18 对照格全对齐（48 贪心 ×2FA + `-long` ×2; minimax-m3 fa=on 两格为参考
服务器/库自分歧格 —— 端口经 ref_decode_dump 探针（预填 + 40 步贪心重放）对参考库全节点
位精确, 参考服务器与自己的库在第 ~38 步分岔, 见 PARITY.md 批次 11a §2）; qwen2.5 双 FA
锚点与批次 1-10 回归见 PARITY.md。新工具: ref_decode_dump 的 --decode-tail/--decode-ids +
`arch_batch11a_dump.rs`。

架构计数 **85 → 92/156**（`arch_tensors_support` 口径, 含本批 +7; 任务账面 87 → 94/156）。

## 架构批次 12: 终局长尾 hrm-text / laguna / maple + llama-server 批次 9-12 接线（代理BATCH12, 2026-10, append-only）

最后三个 src/models/*.cpp 图文件（hrm-text / laguna / maple）—— 零新 ggml 算子、零新 cache
形态，全为既有 helper 组合（qwen3next 的 sigmoid 注意力门、attn_kv_cached 的 iswa 选择、
build_moe_ffn_silu 的 SIGMOID/偏置门控、逐层 AttnParams 覆写（openelm 先例））。逐 arch
机制与 C 行号见 PARITY.md「架构批次 12」节（hrm 的缓存槽别名 + zH/zL 状态、laguna 的双宽
softplus 门 + 逐层类型 RoPE、maple 的 SWA-only rope）。合成 GGUF 对照全格对齐（16/16 ×2FA
×{hrm-text, laguna, laguna-full, maple, maple-long}, dlogprob 0.0000-0.0009）。

**llama-server 闭环**（批次 9/10/11a/11b/12 的全部 22 decoder arch + eurobert 编码器）:
`forward_weights` 分发臂 + `src/weights.rs`（llama-cli 捆绑函数逐字副本）+ eurobert 的
`EncoderContext` 分支；/completion 的 `prompt` 字段改 `tokens.detokenize(ctx, true)`
（server-context.cpp:2111）。证据：run_server_parity.sh 0-diff（maple/smallthinker ×2、
bailingmoe3 首请求；其 n_probs 第二请求为既有 KDA 槽位复用缺口）、eurobert 服务器嵌入
8×128 逐位同、端口 server==CLI 25 格 ×2FA=50/50、qwen2.5 锚点双 FA + chat 29/29 MATCH。

架构计数 **92 → 95/156 →（批次 13/14）115/156 →（批次 15）136/156**（`arch_tensors_support`
口径）。批次 15 的 21 个新 arch 里 17 个进了 ARCH_BATCH15=1 默认集（17 格 ×2FA + 3 -long
格 all matched），llada（参考无 memory）与 gemma-embedding（参考 llama_encode 自身崩溃）
端口内验证，llama-embed 与参考 dump 位精确；cogvlm 与参考**库**位精确（服务器自分歧,
minimax-m3 先例）、plm fa-on 参考装载断言、mimo2/step35 的 iswa 层 kq ~3e-3 实差
（DECDMP1 定位, 开档在-port项）—— 四者默认集外、显式可跑。
