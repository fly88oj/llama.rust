# AUDIT: `common/` + `tools/` (+ `examples/`) — 逐文件逐符号迁移核对

- **审计基线**: llama.cpp pinned `bd4f514db1` (`/home/jeffrey/llm/llama.cpp-pinned`), 对照
  Rust 端口 `/home/jeffrey/works/personal/github/fly88oj/llama.rust`（只读审计, 未改任何代码）。
- **日期**: 2026-09-24（审计 agent）。方法: 全量文件清单（`find`+行数）→ 每文件抽取符号
  （函数/方法/类, 正则提取后逐个手工核对）→ 与 Rust 符号索引（7177 个 fn/struct/enum,
  `crates/**.rs` 全量）匹配 → 未命中者逐个回查 Rust 源与 PARITY.md/FILE_MAP.md 记录。
- **分类**: ✅ported（有同名/同语义 Rust 符号, 给出位置）· 🔗composed（由等价构造覆盖, 给出构造）·
  📄documented-skip（PARITY.md/FILE_MAP.md 已记录原因, 引用行号）· **MISSED（无对应且无记录）**。
- 与其他并行代理的改动无冲突：本审计只写本文件 + PARITY.md/FILE_MAP.md 追加/更正。

---

## 1. `common/` 清单（93 文件: 89 个 C++/头文件共 41,915 行 + build-info.cpp.in 模板 + CMakeLists.txt/parsers/sources.cmake/jinja/README 构建资产）

| 文件 | 行数 | 符号数(审计) | ported | composed | documented-skip | MISSED | 结论 |
|---|--|--|--|--|--|--|--|
| arg.cpp/.h | 4763+156+189 | 56+33 | ~20 | ~28 (每工具自建 parser) | 8 (FILE_MAP:78 ⏳最小参数集) | 见 §1.1 | 🔨 按需 |
| base64.hpp | 392 | 9 | 0 | 9 (server api.rs 内联 base64) | 0 | 0 | 🔗 |
| build-info.cpp.in/.h | 35+13 | 6 | 2 | 4 (`version()`, 构建信息 cosmetic) | 0 | 0 | 🔗 |
| chat.cpp/.h | 1533+391 | 77+61 | 60 | 8 | 10 (PARITY:2435) | 0 | ✅ |
| chat-auto-parser-generator.cpp | 478 | 16 | 14 | 2 | 0 | 0 | ✅ |
| chat-auto-parser-helpers.cpp/.h | 363+74 | 14+11 | 14 | 0 | 0 | 0 | ✅ |
| chat-auto-parser.h | 453 | 56 | 50 | 2 | 0 | 4 (`analyze_base` 虚表壳/`parser_build_context` 半) | ✅(壳级差异) |
| chat-diff-analyzer.cpp | 1635 | 24 | 24 | 0 | 0 | 0 | ✅ |
| chat-peg-parser.cpp/.h | 1232+220 | 44+53 | 40 | 6 | 0 | 0 | ✅ |
| common.cpp/.h | 2361+1213 | 105+116 | 38 | 45 | 18 | **8 (见 §1.2)** | 🔨 见 §1.2 |
| console.cpp/.h | 1166+46 | 43+10 | 43 | 4 (`ctrl-c` 处理) + spinner/Windows 半边开档 | 0 | ~~**~35 (§1.3)**~~ | ✅(2026-09-28 TOOLS-5, 见 §1.3 状态更新) |
| debug.cpp/.h | 190+31 | 5+4 | 0 | 5 (EvalNode dump 协议 + parity/ref_* 探针) | 0 | 0* (见 §1.4) | 🔗* |
| download.cpp/.h | 1090+121 | 50+22 | 0 | 0 | 72 (FILE_MAP:89 ⏳“download/hf …增量”) | 0 | 📄 |
| fit.cpp/.h | 1113+67 | 24+6 | 0 | 0 | 30 (FILE_MAP:89 行 catch-all "…"；`--fit` 旗标缺) | 0* | 📄(弱) |
| hf-cache.cpp/.h | 518+38 | 29+6 | 0 | 0 | 35 (FILE_MAP:89) | 0 | 📄 |
| http.h | 191 | 7 | 0 | 2 (`--host/--port` 自解析) | 5 (FILE_MAP:89) | 0 | 📄 |
| imatrix-loader.cpp/.h | 173+26 | 6+3 | 6 (`quantize/src/pipeline.rs:73 load_imatrix` + `llama/src/imatrix.rs`, legacy .dat 与 GGUF 双格式) | 0 | 0 | 0 | ✅ |
| jinja/*.cpp/.h (12 文件) | 5899 | 78 | 70 | 4 | 4 | 0 | ✅ (131/131 夹具, PARITY:4314) |
| json.cpp/.h | 433+356 | 36+51 | 22 | 0 | 39 (PARITY:251 “转换器不调用的容器 API 未移植”) | 0 | ✅(子集, 记录在案) |
| json-schema.cpp/.h | 514+198 | 24+33 | 24 | 0 | 0 | 0 | ✅ |
| json-schema-to-grammar.cpp/.h | 1028+23 | 40+6 | 37 | 0 | 3 (PARITY:251: LLGUIDANCE 分支/RANGE_ESCAPE_RE) | 0 | ✅ |
| llguidance.cpp | 260 | 13 | 0 | 0 | 13 (整文件 `#ifdef LLAMA_USE_LLGUIDANCE`; PARITY:251 LLGUIDANCE=OFF) | 0 | 📄 |
| log.cpp/.h | 547+146 | 36+19 | 0 | 6 (println/eprintln 直出) | 43 (FILE_MAP:80 ⏳) | 0 | 📄 |
| ngram-cache.cpp/.h | 285+101 | 11+9 | 11 | 0 | 0 | 0 | ✅ |
| ngram-map.cpp/.h | 536+115 | 6+10 | 6 | 0 | 0 | 0 | ✅ |
| ngram-mod.cpp/.h | 62+38 | 6+10 | 6 | 0 | 0 | 0 | ✅ |
| parsers/*.cpp (16) + parsers.h | 3027+79 | 40 | 40 | 0 | 0 | 0 | ✅ (chat_parsers.rs 16 家族逐一核对) |
| peg-parser.cpp/.h | 2124+557 | 89+117 | 82 | 0 | 7 (`bfs_node` dump 调试器等) | 0 | ✅ |
| preset.cpp/.h | 501+88 | 25+15 | 0 | 0 | 40 (FILE_MAP:89 catch-all; 预设体系未接) | 0* | 📄(弱) |
| reasoning-budget.cpp/.h | 310+52 | 19+5 | 0 | 0 | 24 (FILE_MAP:81 “⛔未移: reasoning-budget”; PARITY:2093/2435/2520) | 0 | 📄 |
| sampling.cpp/.h | 919+125 | 39+23 | 31 | 4 (`grammar_should_apply`→engine 惰性文法; `prev_str`→流式缓冲) | 5 | **2 (§1.5)** | 🔨 |
| speculative.cpp/.h | 2997+115 | 62+30 | 46 | 8 (`common_prompt_checkpoint` → PARITY:3955 记录的 seq_rm 路径; init_result 三件套 → CLI/server 组合) | 14 (FILE_MAP:85 未移清单) | 0 | ✅ |
| subproc.cpp/.h | 143+59 | 11+14 | 0 | 0 | 0 | **25 (§1.6)** | MISSED(连带) |
| trie.cpp/.h | 123+73 | 2+11 | 13 | 0 | 0 | 0 | ✅ (`json_schema.rs:1765 Trie` + `peg.rs:254 AhoCorasick`) |
| unicode.cpp/.h | 129+30 | 6+7 | 4 (`unicode.rs:110 len_utf8/169 cpt_to_utf8/181 cpts_from_utf8`) | 3 (`utf8_is_complete`/`cpts_to_utf8`: 唯一消费者 reasoning-budget 已 documented-skip) | 0 | 0 | ✅(语义闭合) |

非代码文件: `CMakeLists.txt`、`jinja/README.md`、`parsers/sources.cmake` — 构建配置, 无需移植。

### 1.1 arg.cpp（4,763 行, 505 个注册旗标）—— 旗标级核算

Rust 工具字面旗标 212 个。对 505 个 C 旗标逐个 grep（含每工具 main.rs/params.rs 的匹配臂）后:
**159 个已存在, 346 个不存在**。不存在者按用途分类:

| 类别 | 数量 | 例 | 口径 |
|---|--|--|--|
| GPU/后端/设备 (`-ngl/--gpu-layers*`, `--device*`, `--rpc`, `-ot/--override-tensor*`, `--kv-offload`, `--n-cpu-moe*`) | ~55 | | FILE_MAP:4 “GPU 不移植” — 📄 |
| CPU 亲和/priority (`--cpu-mask*`, `--prio*`, `-C*`, `--poll*`, `--threads-batch*`) | ~40 | rayon 线程池替代 (FILE_MAP:6) | 📄 |
| 未移植工具专属 (`--diffusion-*`, `--tts-*`, `--video*`, `--media-path`, `--pca-*`, `--hellaswag*`, `--winogrande*`, `--kl-divergence*`, `--multiple-choice*`, `--lookup-*`, `--epochs/--lr/--wd`, `--fit*`, `--models-*/--docker-repo`, `--mcp-servers-*`, `--tools*`, `--webui*`, `--ui*`) | ~90 | | 与工具一致, 见 §2 |
| 服务器运行时 (`--api-key*`, `--metrics`, `--ctx-checkpoints`, `--swa-checkpoints`, `--cache-idle-slots`, `--sleep-idle-seconds`, `--ssl-*`, `--threads-http`, `--timeout`, `--webui*`, `--alias`, `--reasoning-*`) | ~30 | | PARITY:2103-2111 已列 — 📄 |
| **CLI 交互面** (`-i/--interactive`, `--interactive-first`, `--in-prefix*`, `--in-suffix`, `-r/--reverse-prompt`, `--multiline-input`, `--simple-io`, `--prompt-cache*`, `--display-prompt`, `--show-timings`, `--system-prompt*`, `--cont-batching`, `-no-cnv` 等) | ~25 | | **MISSED(§1.3)** |
| **采样链旗标** (`--samplers`, `--sampling-seq`, `--mirostat*`, `--dynatemp*`, `--dry-*`, `--typical`, `--xtc-*`, `--top-n-sigma`, `--frequency-penalty`, `--presence-penalty`, `--logit-bias`, `--reasoning-budget*`) | ~20 | 库层 sampler 全部已有(sampling.rs), 缺 CLI 旗标与链序参数 | **MISSED(§1.5, 部分)** |
| HF/下载 (`--model-url`, `--hf-repo*`, `-hf*`, `--mmproj-url`) | ~10 | FILE_MAP:89 — 📄 |
| 其他 (`--chat-template-file/kwargs`*, `--jinja`*, `--check-tensors`, `--tensor-filter`, `--ppl-stride`, `--spm-infill`, `--log-*`, `--props`, `--slots`, `--defrag-thold`, `--grp-attn-*`, `--swa-full`, `--kv-unified*`, `--cache-type-*-draft`, spec-draft-* 家族, `-cb` 等) | ~70 | 部分有 PARITY 记录(如 `--props` 501、`--kv-unified` 端口恒统一), 其余为增量缺口 | 混合 |

arg.cpp 的机制符号: `common_params_parser(_init)`/`common_arg::*`/`common_params_print_usage`/
`print_completion`/`common_params_apply_system_config`/`common_models_handler_*`/`common_arg_utils::is_*`
→ **composed**（每工具 Rust 手写 parser + usage；llama-bench 的 params.rs 是最完整的复刻）。
`common_chat_verify_template`（arg.cpp:950 `--chat-template` 预检）→ **MISSED**（无对应, 无记录）。

### 1.2 common.cpp — 8 个 MISSED 符号（其余 18 个 documented/composed）

已移植/组合（抽查验证）: `string_format`(llama-cli), `string_split`(llama-bench/params.rs:319),
`string_join/string_repeat`(就地 `Vec::join`/`iter::repeat`, FILE_MAP:184), `string_process_escapes`
(tokenize), `fs_validate_filename`(server/engine.rs:325), `common_token_to_piece`(server/api.rs:32 注释锚定),
`common_embd_normalize`(server/api.rs:1023), `glob_match`(仅被未移植的 server-tools/cli-context 用),
`common_batch_add/clear`(各工具直建 LlamaBatch — composed), `common_detokenize`(vocab.rs:1684),
`common_init_result`(load_model+DecodeContext+SamplingContext 三件套 — composed),
`common_prompt_checkpoint`(PARITY:3877-3956 已记录: 端口 KV 支持部分 seq_rm, checkpoint 回退分支不移),
`common_control_vector_load`(FILE_MAP:68 “控制向量 llama_adapter_cvec 未移植” — 📄)。

| MISSED 符号 | C 位置 | 影响分级 |
|---|---|---|
| `common_init_sampler_from_model` | common.cpp:1213 | **低** — pinned 版无任何 tool 调用（grep 全仓仅自身定义）, 死代码 |
| `common_params_print_info` / `common_params_get_system_info` | :397/:421 | 低 — 启动日志 |
| `common_get_model_endpoint` | :1554 | 低 — 唯一消费者 hf-cache(未移植) |
| `common_replay_last_token` | :2191 | 低 — 唯一消费者 tools/completion(未移植) |
| `common_prompt_batch_decode` | :2201 | 低 — 唯一消费者 tools/completion(未移植) |
| `string_get_sortable_timestamp` | :491 | 低 — 无消费者(日志文件名) |
| `string_parse_kv_override` | :682 | **中** — quantize/arg.cpp 用(`--kv-override`); 端口 quantize 未接该旗标 |
| `string_lcs` | :470 | 低 — server 槽位相似度已内联自实现（`Engine::get_available_slot` 的 prompt-similarity, engine.rs:1340-1360 ↔ server-context.cpp:1291-1360） — 归 composed, 此处仅记差异 |

（`cpuid/pin_cpu/set_process_priority/parse_cpu_*` 等 12 个 CPU 拓扑符号 → FILE_MAP:6 rayon 决策覆盖 — 📄；
`fs_get_cache_directory/config_directory` → download/hf-cache 未移植连带 — 📄 FILE_MAP:89。）

### 1.3 console.cpp（1,166 行）— ~~整文件 MISSED~~（✅ 已于 2026-09-28 移植）

> **状态更新（2026-09-28, 代理TOOLS-5）**: 全文移植到 `crates/llama/src/console.rs` ——
> 两条读入路径都在: `readline_simple`(:1046-1081) 与 `readline_advanced`(:753-1044, raw-mode
> termios/wcwidth 经 extern "C" 直连, mlock.rs 先例)。脚本化(非 tty)验证覆盖 advanced 路径
> (参考在该环境无 isatty 回退, /dev/tty 打开失败即退回 stdout 回显)。Windows 半边、spinner
> (:1094-1144, 消费者均未移植)、tab 补全回调为文档化开档。`parity/cli_interactive_parity.sh`
> 17/17 stdout 字节级 MATCH。以下为审计时点记录:

`console_init/exit`, `readline/readline_advanced`(行编辑、UTF-8 光标移动、历史、多行、
`\`-续行、Ctrl 组合键、粘贴大块检测), `set_console_color`, `clear_line/current_line`,
`print_sequence`（分块彩色输出）, `set_completion_callback`（tab 补全）, `console::stop/log`。
唯一消费者: tools/cli(交互)、tools/completion、examples/{simple*,speculative*,lookahead,passkey,retrieval,parallel,batched}。
FILE_MAP:77-89 的 common 表没有 console.cpp 行——被 89 行 "其余(download/hf/ngram/...)" 的 "…" 隐式覆盖, **应显式记录**。
影响: 端口所有工具均为单轮/流式输出, 无交互行编辑。

### 1.4 debug.cpp — 0 个硬 MISSED, 1 个软缺口

`common_debug_cb_eval`（张量打印/NaN 中止/过滤）+ `common_debug_cb_user_data`：
ggml 侧 eval-callback **协议**已移植(compute.rs `EvalNode`, FILE_MAP:637), 端口的对应用户面是
`LLAMA_RUST_DEBUG=1` + `--dump-embd` + parity/ref_decode_dump 探针族 — **composed**。
软缺口: `--debug-pattern`/`abort_on_nan` 式通用张量过滤器无对应 CLI 面（mtmd-cli 的 C 版 `mparams.cb_eval` 接线, 端口 mtmd-cli 未接）。

### 1.5 sampling.cpp — 2 个 MISSED（链序 + DRY）—— **已于 2026-09-24 采样任务补齐**

> **状态更新（2026-09-24, 采样任务 agent）**：本节两项 MISSED 均已移植并位级验证。
> 1. `llama_sampler_dry`（llama-sampler.cpp:3317-3724）→ `sampling.rs` 的
>    `DrySampler`/`init_dry`/`init_dry_testing`（断路器 multimap →
>    `HashMap<head, Vec<tail>>`；反向 Z 算法/四步惩罚逐行；`std::pow(float,int)`
>    的 double 提升在 f64 一次收窄复刻）。
> 2. `llama_sampler_adaptive_p`（:3727-3905）→ `AdaptivePSampler`/`init_adaptive_p`；
>    `llama_sampler_infill`（:4070-4299, `--samplers` 链可达）→ `InfillSampler`/
>    `init_infill`（词表快照 is_eog/piece/eot/eos）。
> 3. 链序：`common_sampler_types_from_names/from_chars` + `to_chr`/`to_str`
>    （sampling.cpp:795-919）→ `CommonSamplerType`；`common_params_sampling.samplers`
>    默认链（common.h:265-275, DRY 第 2 位）→ `SamplingParams.samplers`；
>    `common_sampler_init` 链循环（sampling.cpp:340-413）→
>    `SamplingContext::new_with_vocab`（adaptive-p 恒链尾替换 dist）。
>    server 请求字段 `dry_*`/`dry_sequence_breakers`/`samplers`（api.rs,
>    server-schema.cpp:139-156/167-174/242-249/505-515 语义一致）与 CLI 旗标
>    `--dry-*`/`--samplers`/`--sampler-seq`/`--sampling-seq`/`--adaptive-*`
>    （llama-cli, arg.cpp:1982-2003/2123-2202）已接。
>    验证：`parity/sampler_dry_ref.txt`（参考 libllama.so + libllama-common.so
>    实测夹具, `parity/gen_sampler_dry_ref.sh` 生成）由
>    `crates/llama/tests/sampler_dry_parity.rs` 位级回放（链构造名序/DRY 8 例/
>    全链 20 步/adaptive 3 例/infill 2 例）；server 双端
>    `parity/sampler_dry_server_check.sh`。默认链回归：原 `?dry` 空位加入后
>    默认请求输出不变（空采样器 no-op, 见 PARITY.md 采样任务节）。
> 以下为审计时点（2026-09-24 上午）的原始记录：

1. **DRY sampler 缺失**: C 默认链含 `COMMON_SAMPLER_TYPE_DRY`（common.h:259 起, dry_multiplier>0 时生效）。
   端口 `SamplingContext::new`（sampling.rs:2051-2053）注释明言 "minus DRY — not ported";
   server /props **回显** dry_* 默认值(api.rs:857-859) 但请求里的 `dry_multiplier` 等字段不解析不过滤 —— 客户端
   设置 DRY 时端口静默无惩罚。**FILE_MAP:41/64/79 把 llama-sampler.cpp 记为 ✅ 与此矛盾（src 侧符号
   `llama_sampler_dry` = llama-sampler.cpp:3317）, 本审计将其列为唯一的高优先采样缺口。**
2. **链序参数缺失**: `common_sampler_types_from_names/from_chars`(sampling.cpp:829/891) + server 请求字段
   `samplers`/`sampling_seq`（server-schema.cpp:505-515）→ 端口硬编码默认序（api.rs:930 仅回显数组）,
   客户端自定义采样顺序被忽略。`COMMON_SAMPLER_TYPE_ADAPTIVE_P`（llama-sampler.cpp:3727 adaptive-p,
   仅可经 `--samplers` 到达）随之缺失。
3. 已记录部分: `common_sampler_reasoning_budget_force`(reasoning-budget 连带, FILE_MAP:81) — 📄;
   `common_sampler_clone/copy`（server 槽位克隆用）— 端口每请求新建 SamplingContext, composed;
   `common_perf_print`/`print`（性能打印）— 低, 未记录。

### 1.6 subproc.cpp — ~~MISSED~~（✅ 已于 2026-09-28 移植）

> **状态更新（2026-09-28, 代理TOOLS-5）**: 移植到 `crates/tools/llama-server/src/subproc.rs`
> （std::process 封装, argv 直传/环境整替/SIGTERM terminate/wait join/combined stdout+stderr
> 经 pipe+dup）。消费者 server-tools/server-mcp 同轮移植(§2.2 状态更新)。以下为审计时点记录:

`common_subproc::spawn/terminate/is_supported/...`（跨平台子进程管理）。消费者:
server-mcp.cpp(MCP 子进程)、server-tools.cpp(内置工具子进程)、server-models.cpp(router 下载器)。
三者均未移植（§2.3）, subproc 本体在 FILE_MAP/PARITY 亦无记录。

---

## 2. `tools/` 清单（19 个子目录）

| 工具 | C 行数 | Rust 位置 | 状态 | 符号级结论 |
|---|--|--|--|--|
| batched-bench | 267 | — | **MISSED** | 无对应无记录（llama-bench 已覆盖 `-b/-ub` 维度, 但参考的并发基准工具整体缺） |
| cli (llama-cli) | 1,326 (8 文件) | crates/tools/llama-cli (interactive.rs/diffusion.rs) | 🔨 收窄 | 见 §2.1 状态更新（交互面+扩散已移, 客户端架构仍缺） |
| completion | 961 | crates/tools/llama-cli/src/interactive.rs | ✅(2026-09-28 TOOLS-5) | 主环/会话格式化/交互旗标全移; `--prompt-cache` 为端口 token 框架(非 llama_state 字节格式) |
| cvector-generator | 878 | — | MISSED(半记录) | 控制向量本体已记录不移植(FILE_MAP:68), 工具未点名; pca.hpp/mean.hpp 同缺 |
| export-lora | 439 | — | **MISSED** | 无对应无记录（LoRA 导出 .gguf; 端口只做加载侧） |
| fit-params | 83 | — | MISSED(连带) | common/fit.cpp 未移, 工具未记录 |
| gguf-split | 609 | crates/tools/gguf-split | ✅ (2026-09-24 代理SPLIT 已移) | split/merge 全参数 + `gguf_set_val_*` remove+append kv 语义 + `split.*` 约定; 与参考跨工具**逐字节**一致(三种模式 17 分片、双向 merge、参考消费端口分片, parity/gguf_split_parity.sh); 读取侧 `split.*` 装配在 crates/ggml/src/gguf.rs `Gguf::open`(镜像 llama-model-loader.cpp:596-669) |
| imatrix | 1,193 | crates/tools/imatrix (912) | ✅ | 旗标全对齐; 已记录缺口: IQ1/2/3 量化器、MoE 实测(FILE_MAP:97) |
| llama-bench | 2,512 | crates/tools/llama-bench (3,613) | ✅ | 已记录缺口: GPU 旗标/非 f16 KV/embeddings(FILE_MAP:101) |
| mtmd (核心) | clip 6087 / mtmd 2733 / mtmd-audio 1557 / helper 1044+1062+184 / mtmd.h 等 | crates/llama/src/{clip,mtmd,mtmd_audio}.rs + crates/tools/llama-mtmd-cli | ✅(主体) | FILE_MAP:103-110 + PARITY 各节逐条开档; 抽查 `mtmd_helper_eval_chunk_single/decode_image_chunk` 在 mtmd.rs:766/976-979, `mtmd_helper_video_*`/`model_can_chat`/`support_video` 属已开档缺口(视频/交互 chat); 缺口(视频/ffmpeg/gen_audio/libmtmd C ABI/交互 chat/--jinja/其余 ~50 projector)全部已记录 |
| mtmd/debug/mtmd-debug | 311 | — | MISSED(半记录) | 工具本体未移植; 其 `-p encode` dump 语义已由 `clip_cb_parity_dump` 复现(PARITY:1299 引用参考侧该工具作真值)。应显式记录 |
| mtmd/legacy-models/*.py | 1,882 | — | MISSED(范围外) | 原始权重→GGUF 转换脚本(Python); 与 convert_hf_to_gguf.py 同类, 端口范围外但无记录 |
| mtmd/tests.sh + tests/*.py | 600 | — | 测试资产 | parity 脚本体系另行覆盖, 无需移植 |
| perplexity | 2,103 | crates/tools/perplexity (+src/scorers.rs) | ✅(评分器) | ported: `perplexity()` v1 路径 + **三评分器已移(2026-09-24 代理SPLIT)**: `hellaswag_score`(:744)/`winogrande_score`(:1101)+`load_winogrande_from_csv`/`multiple_choice_score`(:1405)+二进制数据集反序列化, 含任务选择随机数位同(mt19937+libstdc++13 Lemire, parity/ref_rng_vec.cpp); 仍未移(有记录): ppl-stride(`perplexity_v2`)/`kl_divergence`/bench(FILE_MAP:100) |
| quantize | 673 | crates/tools/quantize (1,501) | ✅ | 已记录缺口: --include/exclude-weights、--prune-layers、IQ1/2/3(FILE_MAP:96); `load_imatrix/prepare_imatrix` 已含(pipeline.rs:73) |
| results | 183 | — | MISSED | 结果对比小工具(NMSE), 无对应无记录 |
| rpc | 344 | — | 📄 范围外 | RPC 服务(远端 GGML 后端, GPU 场景); FILE_MAP:4 CPU-only 口径覆盖, 建议显式记录 |
| server (19 个 .cpp/.h 共 21,979 行) | — | crates/tools/llama-server (+server_tools.rs/server_mcp.rs/subproc.rs) | 🔨 收窄 | 见 §2.2 状态更新（MCP/工具/GCP/cors 四缺口已移） |
| tokenize | 222 | crates/tools/tokenize (270) | ✅ | 100/100 输出对齐(FILE_MAP:99) |
| tts | 214 | — | **MISSED** | qwen3tts/pockettts 生成驱动; mtmd-audio 预处理器已全量(mtmd_audio.rs) 但此工具未移植未记录 |
| tuning (4 文件) | 1,089 | — | MISSED | fa-vec/bench 调参开发工具, 无对应无记录 |
| ui | 101+脚本 | — | 📄 | FILE_MAP:102 “内置 UI 未移植” |

### 2.1 tools/cli — 架构级差异（交互面已于 2026-09-28 移植, 客户端架构仍缺）

> **状态更新（2026-09-28, 代理TOOLS-5）**:
> - **交互面已移**: console.cpp 全文(§1.3) + tools/completion 主环
>   （`crates/tools/llama-cli/src/interactive.rs`, `-i/-if/--in-prefix(-bos)/--in-suffix/
>   -r/-mli/--simple-io/--(no-)display-prompt/-e/-sys(f)/--(no-)show-timings/-sp/
>   --verbose-prompt/-ptc/--prompt-cache(-all/-ro)/-st/-cnv/-no-cnv/-co/--keep/
>   --(no-)context-shift`）; `parity/cli_interactive_parity.sh` 17/17 stdout 字节同
>   （qwen2.5, --temp 0 --seed 42）。
> - **扩散驱动已移**: `src/diffusion.rs`（--diffusion-* 族, §3 表行）。
> - **仍未移（开档）**: 客户端架构 4 文件（cli-client/cli-server/cli-context/cli-ui,
>   HTTP 会话/模型列表/媒体 staging/SSE）——端口 llama-server 即其服务面。
> 以下为审计时点记录:

pinned 版 llama-cli = **HTTP 客户端架构**: `cli-server.h`（启动/连接 llama-server）、
`cli-client.cpp`（get/post/**post_sse**/wait_health）、`cli-context.cpp`（`cli_context::init/run/`
`list_and_ask_models`/`add_system_prompt`/`stage_media_file`/`generate_completion` — 会话状态、
媒体文件staging、模型列表选择、`/props` 拉取）、`cli-ui.h`（终端 UI 渲染）。
端口 llama-cli = **直连单轮工具**（minimal chat 模式）。生成语义有 parity 锚点, 但:
- 客户端架构 4 文件(cli-client/cli-server/cli-context/cli-ui 含头) 1,326 行整体未移植（对应 `-hf`/`--models-*` 模型管理面）, 未记录;
- 交互面（`-i` 族旗标 + console.cpp）未移植, 未记录;
- FILE_MAP:95 行应补充上述两条口径（当前只写“采样链+EOG+chat模式+架构分发”, 无缺口句）。

### 2.2 tools/server — 端点面核算（server.cpp:244-330 共 54 处路由注册、43 个不同路径 vs 端口 20 个路径）

端口已实现: /health, /v1/health, /props(GET), /models, /v1/models, /completion, /completions,
/v1/completions, /chat/completions, /v1/chat/completions, /embedding(s), /v1/embeddings,
/rerank(501 应答), /reranking, /v1/rerank, /v1/reranking, /tokenize, /detokenize, /slots(GET),
POST /slots/{id}（save/restore/erase — PARITY:4583 已接）。

未实现且 PARITY:2103-2111 **已记录**: /metrics, /v1/responses+/responses(Responses API),
/v1/messages(Anthropic), /v1/audio/transcriptions, /infill, /apply-template, */input_tokens 计数族,
/v1/chat/completions/control, /lora-adapters GET/POST, /v1/stream 族, router 模式(/models/load 等,
server-models.cpp 全套), POST /props(参考默认也 501), API key/CORS/gzip, web UI, mtmd, rerank 正向打分,
`n_cmpl>1` 子槽, idle sleep/purge(PARITY:1037), prompt-cache checkpoints(PARITY:2110), KV shifting。

**未记录的 MISSED（本审计新增）—— 已于 2026-09-28 由代理TOOLS-5 全部移植**:

> 状态更新: (1) server-mcp.cpp → `crates/tools/llama-server/src/server_mcp.rs`;
> (2) server-tools.cpp → `src/server_tools.rs`（7 内置工具 + /tools GET/POST + 头注入 +
> SSE 流式; 连带 common/subproc.cpp → `src/subproc.rs`）;
> (3) GCP 兼容 → `server_tools.rs::register_gcp_compat`（AIP_* 四变量 + /predict 驼峰别名
> 分发 + AIP 端口覆盖）;
> (4) server-cors-proxy.h → `server_tools.rs::cors_proxy`（http:// 转发字节级; https 答 500,
> 无 TLS 依赖不可复刻, 开档）。
> 验证: `parity/server_tools_mcp_parity.sh` 15/15 MATCH（schema 暴露/read_file/exec/404/
> stat 错误/403 禁用/MCP 列表+调用/GCP healthz+predict+错误实例/代理 body+头透传+ftp 错误文本）。
> 以下为审计时点记录:
1. **server-mcp.cpp (820 行)** — MCP 子进程服务器桥(`--mcp-servers-config/-json`); PARITY/FILE_MAP 无任何 MCP 记录。
2. **server-tools.cpp (2,172 行) + server-tools.h** — 内置工具运行时(web_search 等子进程型 server-side tools,
   `--tools`/`--tools-runtime`); 无记录。连带 common/subproc.cpp(§1.6)。
3. **GCP/Vertex 兼容** — server-http.cpp:803-928 `register_gcp_compat`(AIP_MODE 环境变量、/predict 路由) — 无记录。
4. **server-cors-proxy.h (83 行)** — CORS 代理转发器, /props 回显 `cors_proxy_enabled:false` 说明端口知道该字段, 但头文件本身未移植未记录。
5. `server_queue`/`server_response_reader`（server-queue.cpp 621 行, 慢客户端/断连重连的响应读取器）— 端口用
   mpsc+Arc 组合(composed), 但 `server_response_reader` 的**可恢复流**语义与 /v1/stream 族绑定, 归入该缺口。

### 2.3 tools/mtmd — 补充核对结论

FILE_MAP:103-110 + PARITY(音频三轮/EAGLE/mtmd 节) 的记录与代码状态一致, 无新 MISSED。
注意: clip.rs/mtmd*.rs 正由并行代理编辑, 本审计只读快照; 上述结论基于符号在位性, 非逐行数值复核。

---

## 3. `examples/` 清单（29 目录, C++ 8,315 行）

| example | 行数 | 状态 | 记录 |
|---|--|--|--|
| speculative-simple | 377 | ✅ ported | 驱动环在 crates/llama/src/speculative.rs(FILE_MAP:85, PARITY:1498) |
| gguf | 273 | ✅ ported | crates/tools/gguf(FILE_MAP:98) |
| lookup (create/merge/stats) | 455 | 📄 | PARITY:3592 “库层已可支撑, 工具未移植” |
| llama.android / llama.swiftui / llama.vim / batched.swift | — | 范围外 | 平台绑定 |
| sycl / test-cmake / deprecation-warning / gen-docs | 45 | 范围外 | GPU/构建/文档生成 |
| json_schema_pydantic*.py / server_*.py / reason-act.sh 等 | — | 范围外 | 脚本 |
| **batched** | 264 | **MISSED** | 无记录 |
| **simple / simple-chat** | 433 | MISSED(语义被 llama-cli 覆盖) | 无记录 — 建议以“由 crates/tools/llama-cli 语义覆盖”收口 |
| **parallel** | 521 | MISSED(语义被 server 多槽覆盖) | 无记录（PARITY:1023 只记 server 侧 n_parallel 语义） |
| **embedding** | 414 | MISSED(语义被 server /embedding 覆盖) | 无记录; parity/run_embedding_parity.sh 走 server 路线 |
| **passkey / retrieval / lookahead / gguf-hash / idle / debug / eval-callback** | 110-700 | **MISSED** | 无记录（eval-callback 的 dump 语义已由 parity/ref_decode_dump 复现但文件未点名; PARITY:3590 的“eagle3/dflash 仍未移植”为过期快照, 后续批次已移植） |
| **speculative (旧驱动)** | 668 | MISSED | 新驱动(speculative-simple)已移植; 旧 example 未记录 |
| **diffusion (diffusion.cpp+cli)** | 731 | **MISSED** | 图侧(llada/dream/rnd1 builders)已移植; **扩散推理环**(gumbel 噪声/--diffusion-* 参数, diffusion.cpp:1-408)未移植未记录 |
| **training/finetune** | 100 | MISSED | 训练存根, 无记录 |
| **convert-llama2c-to-ggml** | 959 | MISSED | 转换器, 无记录 |

---

## 4. MISSED 汇总（triage）

### A. 行为影响（建议优先处理或在 FILE_MAP 显式开档）

| # | 符号/文件 | C 位置 | 影响面 |
|---|---|---|---|
| ~~A1~~ | ~~`llama_sampler_dry` + 默认链 DRY 位 + server `dry_*` 请求字段~~ | src/llama-sampler.cpp:3317（经 common/sampling.cpp 链构建消费） | **已补齐（2026-09-24 采样任务, §1.5 状态更新）** |
| ~~A2~~ | ~~`common_sampler_types_from_names/from_chars` + server `samplers`/`sampling_seq` 字段 + ADAPTIVE_P~~ | common/sampling.cpp:829/891, server-schema.cpp:505-515, llama-sampler.cpp:3727 | **已补齐（2026-09-24 采样任务, §1.5 状态更新；连带 `llama_sampler_infill` :4070）** |
| ~~A3~~ | ~~交互 CLI 面~~ | common/console.cpp, tools/completion/* | **已补齐（2026-09-28 代理TOOLS-5, §1.3/§2.1 状态更新）**: console 全文 + completion 主环 + ~25 交互旗标; 剩余: tools/cli 客户端架构 4 文件（开档） |
| ~~A4~~ | ~~server-mcp.cpp（MCP）+ server-tools.cpp（内置工具）+ common/subproc.cpp~~ | 同左 | **已补齐（2026-09-28 代理TOOLS-5, §2.2 状态更新）**: server_mcp.rs + server_tools.rs + subproc.rs; parity 15/15 |
| ~~A5~~ | ~~GCP 兼容 + CORS 代理~~ | 同左 | **已补齐（2026-09-28 代理TOOLS-5）**: register_gcp_compat + cors_proxy（https 代理开档）; parity 3 格 MATCH |
| A6 | gguf-split（split/merge） | tools/gguf-split/gguf-split.cpp:369/407 | 模型文件管理; 端口报错文本引用 C 工具 |
| A7 | perplexity 评分器 hellaswag/winogrande/multiple-choice（+kl_divergence 已记录） | tools/perplexity/perplexity.cpp:744/1101/1405 | 评测面 |
| ~~A8~~ | ~~扩散推理环~~ | examples/diffusion/* | **已补齐（2026-09-28 代理TOOLS-5）**: crates/tools/llama-cli/src/diffusion.rs |
| A9 | `common_chat_verify_template` | common/chat.cpp:638, 消费者 arg.cpp:950 | `--chat-template` 传坏模板时端口不预检 |
| A10 | tts / export-lora / batched-bench / cvector-generator / results / tuning / fit-params / mtmd-debug(工具本体) | 各工具 | 功能工具面 |
| A11 | `string_parse_kv_override`（`--kv-override`） | common.cpp:682, 消费者 arg.cpp+quantize | 元数据覆盖（quantize 调参） |

### B. 死代码/低影响（记录即可）

- `common_init_sampler_from_model`（pinned 版无调用者）、`common_get_model_endpoint`、`common_replay_last_token`、
  `common_prompt_batch_decode`（仅未移植工具消费）、`common_params_print_info`、`string_get_sortable_timestamp`。
- log.cpp 的 `--log-file/--log-verbose` 等文件日志设施（FILE_MAP:80 ⏳ 已有, 建议列旗标缺口）。
- examples/{batched,passkey,retrieval,lookahead,gguf-hash,idle,debug,eval-callback,training,convert-llama2c}（参考性示例）。

### C. 范围外（GPU/平台/构建, 建议在 FILE_MAP 加显式行）

- tools/rpc（远端后端）、examples/sycl、llama.android/vim/swiftui、test-cmake、gen-docs、
  deprecation-warning、tools/mtmd/legacy-models/*.py（与 convert_*.py 同类的转换脚本）。

## 5. FILE_MAP.md 记录核对（需要更正/补充的行）

1. **行 41/64/79（llama-sampler ✅ 51 测试）**: 需补 "DRY/ADAPTIVE_P 未移植; 链序参数未接"。
2. **行 95（tools/main+tools/cli）**: 需补缺口句: 交互面/console.cpp/客户端架构/completion 未移植。
3. **行 100（perplexity）**: 缺口句补 hellaswag/winogrande/multiple-choice。
4. **行 102（server）**: "POST /slots save/restore 未移植" 已过期 —— PARITY:4583 已移植（应删该半句）;
   需补 MCP/内置工具/GCP/cors-proxy 四项未记录缺口。
5. **行 89（catch-all "其余"）**: 建议把 console.cpp/fit.cpp/preset.cpp/http.h 从 "…" 中点名。
6. **tools/ 表缺行**: gguf-split、batched-bench、completion、cvector-generator、export-lora、fit-params、
   results、rpc、tts、tuning、ui(已有)、mtmd/debug、legacy-models —— 本次审计已在上方表中逐一定级。

## 6. 抽查锚点（供复核; 每目录 ≥10 个）

common/: `common_embd_normalize`→api.rs:1023 ✅; `string_split`→llama-bench/params.rs:319 ✅;
`common_chat_tool_parameters`→chat_tools.rs:713 `tool_parameters` ✅; `common_chat_msgs_to_json_oaicompat`→
chat_tools.rs:676 ✅; `messages_inp_normalizer`→chat_tools.rs:599 ✅; `common_peg_parser_builder::json_object`→
peg.rs:1958 ✅; `json_schema_to_grammar`→json_schema.rs:2581 ✅; `common_ngram_cache_draft`→ngram_cache.rs:244 +
`CommonNgramMap::draft`→ngram_map.rs:296 ✅; `common_speculative_type_from_name`→speculative.rs:162 ✅;
`glob_match`→无(消费者均未移植) ✗; `common_chat_verify_template`→无 ✗ MISSED; `common_utf8_sequence_length`→
unicode.rs:110 `len_utf8` ✅; `trie/aho_corasick`→json_schema.rs:1765 + peg.rs:254 ✅; `common_chat_msg_delimiters::split`→
无(消费链 ctx-checkpoints 未移植, PARITY:2110 记录) ✗已记录。

tools/: `llama_quantize`→pipeline.rs:143 ✅; `load_imatrix`→pipeline.rs:73 ✅; `IMatrixCollector::save_imatrix`→
imatrix main.rs:752 ✅; `perplexity()`→perplexity main.rs ✅, `hellaswag_score`→无 ✗; `tokenize` 主环→tokenize
main.rs ✅; `mtmd_helper_eval_chunk_single`→mtmd.rs:766/979 `eval_chunk` ✅; `mtmd_helper_model_can_chat`→无
（消费面=mtmd-cli 交互 chat 模式, FILE_MAP:105-106 已开档 — 修正分类为 documented-skip）; `routes.get_slots`→
main.rs /slots ✅; `handle_slots_*`(save/restore/erase)→PARITY:4583 ✅; `cli_context::run`→无 ✗ MISSED;
`server_tools::setup`→无 ✗ MISSED; `gguf_split/gguf_merge`→无 ✗ MISSED; `mtmd_helper_support_video`→无(已记录) ✗。

examples/: `speculative-simple` 主环→speculative.rs:3941+ ✅; `gguf example main`→tools/gguf ✅;
`lookup-create`→无(已记录 PARITY:3592) ✗; `embedding.cpp main`→无(server /embedding 语义覆盖) ✗;
`diffusion::generate`→无 ✗ MISSED; `simple.cpp`→无(llama-cli 语义覆盖) ✗。

## 7. 结论

- **common/ 核心行为链已基本完整移植**: chat 全链(含 jinja 引擎、16 解析器、peg、autoparser、diff-analyzer)、
  json-schema 家族、speculative、ngram 三件、imatrix-loader、unicode/trie/base64 —— 符号级全部落位。
- **按需移植原则下的未完成面**（有记录）: arg.cpp 旗标 346/505 缺、log/console、download/hf、
  reasoning-budget、llguidance(构建开关外)、server 高级端点族 —— PARITY.md 记录质量高, 与代码一致。
- **真正的账面缺口**（无记录）集中在: DRY/链序采样（A1/A2, 与 ✅ 记录冲突）、交互 CLI 面（A3, **2026-09-28 已补齐**）、
  server 的 MCP/内置工具/GCP（A4/A5, **2026-09-28 已补齐**）、gguf-split（A6）、perplexity 三评分器（A7）、扩散驱动（A8, **2026-09-28 已补齐**）、
  tts/export-lora 等长尾工具与 examples 大半目录。全部清单见 §4。

## 8. 追记（2026-09-29, 架构批次 17）

- §4 (4) / A5 的 https 代理开档已闭: `server_tools.rs::cors_proxy` 的转发客
  户端按 server-models.cpp:2456-2600 重写（OpenSSL 直链 `tls.rs`, httplib 的
  请求头补全/redirect 规则/响应剥离表）, `parity/server_ui_proxy_parity.sh`
  https 转发字节级 MATCH。
- §3 的 ui 资产服务面已移: `crates/tools/llama-server/src/ui.rs` +
  `ui_assets.rs`（生成物, 默认空表 = 参考构建状态） + `parity/gen_ui_assets.py`
  （scripts/ui-assets.cmake emit_files 的移植）; `--ui/--no-ui/--path/
  --api-prefix` 旗标落地（arg.cpp:3339-3345/:3382-3388/:3464-3470）。
  注意: `--api-prefix` 在端口仅作用于 UI/挂载路由（全路由前缀化仍是开档）。
- 客户端架构 4 文件（cli-client 等 1,326 行）仍开档; 其服务面（UI 资产 +
  API）由 llama-server 自身承载, 本轮补齐了承载的资产服务半边。
