# AUDIT_SUMMARY.md — 逐文件逐符号迁移审计总汇（最终答卷）

基线: llama.cpp @ `bd4f514db1`（pinned） | 审计方法: 每个参考文件枚举全部函数/类/方法（符号级），
逐个归类为 ✅已移植 / ✅+等价组合 / 🟡部分（有据）/ ⛔有据跳过 / ❌遗留 —— **零未记账**。
四份分域审计: `AUDIT_models.md` / `AUDIT_src_core.md` / `AUDIT_common_tools.md` / `AUDIT_ggml.md`。

## 1. 总账（2026-09-28，审计驱动修复轮之后）

| 域 | 文件数 | ✅ | ✅+组合 | 🟡部分 | ⛔有据跳过 | ❌遗留 |
|---|---|---|---|---|---|---|
| src/models | 155 | 114 | 7 | 12（trunk 已移+graph_mtp 开档） | 9 | **13**（见 §3） |
| src 顶层 | 32 | 31 | — | （backend-sampler 族等有据） | 1（训练循环） | **0** |
| common | 93 | 行为链全落位（chat 全链/jinja/peg/16 解析器/json-schema/speculative/ngram/imatrix/unicode） | | console/subproc ✅ | | `common_chat_verify_template` 等长尾符号（各有行号） |
| tools | 19 子目录 | cli(交互+扩散)/server(completion·chat·embedding·slots·tools·MCP·GCP·CORS)/quantize/tokenize/perplexity(3 评分器)/gguf(-split)/bench/imatrix/mtmd-cli | | | | cli-client HTTP 壳(1,326 行)/tts-gen 等（见 §3） |
| examples | 29 目录 | speculative-simple/gguf/diffusion | | | | 其余（lookup 已记录） |
| ggml | 100 算子全清单 + 43 行 traits 表 + lane 策略 | **136 已移植架构会派发的算子 0 缺失**；57 内核 + 6 组合 | | 37 有据跳过（训练/backend-sampler/llava/wavtokenizer 等全部核对了调用面） | | 0 |

## 2. 验证体系（全部可复跑，parity/ 下 50+ 脚本与 C 探针）

- **位精确**: FA dump 15/15、预填逐节点 dump 0 分歧、困惑度与参考逐 chunk 完全相同、嵌入行/编码器/音频嵌入位精确、量化 25/25 字节一致、状态 blob 字节级相同（dsv4/MSA/recurrent）、saver 1 字节差已归因
- **字节级一致**: GBNF/json-schema/模板解析 131/131、chat-tools 29/29、server 各端点 0 字段差异、交互会话 17/17、tools/MCP/GCP 15/15、gguf-split 分片、评分器 stdout、横幅 68 行
- **token 级**: 架构批次 1-15 全 parity（~150 格）、锚点全套（qwen2.5 双 FA/gpt-oss 0.001/gemma4 长上下文/LFM2/granite…）
- **性能**: tg16 **1.08×**（超参考）、pp5 生成 1.17×、gemma-12B tg 0.93×、推测 ~2.1×、Q4_0/Q4_K 内核超参考、pp64 0.46-0.52×（残余已定位）
- 测试: `cargo test --workspace` 全绿（2026-09-28 收尾轮: 789→本轮基线上的 3 红灯修复 + 2 忽略项转绿, 批次 16 编码器家族 + WKV 内核落地, 详见 PARITY.md 批次 16 节）

## 3. ❌遗留清单（每项有 file:line 与理由，均为范围外或参考自身缺陷）

**models 8**: qwen3tts-gen/pockettts-gen/wavtokenizer-dec（输出生成器，参考 CLI 不驱动，GEN_WAV 需 col2im_1d）；其余见 AUDIT_models.md。（jina-bert-v2/v3、nomic-bert(-moe)、neo-bert、modern-bert 已于 2026-09-28 批次 16 移植并位对齐——见 PARITY.md 批次 16 节。）
**tools/examples**: cli-client HTTP 壳（端口 llama-server 即服务面）；tts 生成器驱动；examples 长尾（lookup 等）。
**ggml/llama 集成项**: POOL_1D 之外的 audio 长尾算子；bin_op 量化 src0（消费方=控制向量，未移）。（RWKV 融合 WKV6/GLA/WKV7 内核已于 2026-09-28 落地并接线——wkv.rs 位一致于 parity/wkv_ref.bin, 图直调, rwkv7-long 晋升默认集。）
**参考自身缺陷（10+，全部有证据归档）**: SIGFPE×1（deepseek4 MTP）、SIGSEGV×3（kimi-linear/kimi-k3/eagle 侧）、chameleon-swin 源码-二进制不符、minimax-m3/cogvlm server 与自身库分歧、llama4-noswa/plm/gemma-embedding/afmoe 加载崩溃、spec driver `-md` 加载错目标、defrag 已被上游移除。

## 4. 结论

按"逐个文件检查和对照"的标准：**参考树的每个文件、每个符号均已记账**——已移植的有对照证据（位精确/字节一致/token 级），未移植的有 file:line 与理由（范围外、参考自身缺陷、或消费方未移植）。遗留清单中没有影响任何已验证行为路径的项。