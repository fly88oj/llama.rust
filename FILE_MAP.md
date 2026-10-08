# FILE_MAP.md — llama.cpp → llama.rust 逐文件对照清单

- **基线**: llama.cpp master `bd4f514db1` (2026-09-24)，固定于 worktree `/home/jeffrey/llm/llama.cpp-pinned`（原目录会被外部自动更新，勿作对照）；参考构建 `/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/`（commit 一致）
- **对标范围**: llama.cpp 的 **CPU 构建**（GPU 后端 CUDA/Vulkan/Metal/SYCL/HIP/... 不移植；这在 C 原版也是合法构建配置）。
- **Rust 优化原则**（在保持数值等价的前提下）:
  - rayon 线程池替代 ggml 手写 threadpool；
  - `#[repr(C)]` + `bytemuck` 零拷贝块布局；
  - `memmap2` 权重映射；
  - 类型安全: `GgmlType`/`Value` 枚举替代裸 int + union；
  - `Result` 错误处理替代 GGML_ABORT（用户数据路径），内部不变量仍用断言。
- **状态**: ✅ 完成(含校验) · 🔨 进行中 · ⏳ 未开始 · ➖ 部分语义 · ⛔ CPU 范围外

## ggml 核心

| 参考文件 | 行数 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|---|
| ggml/src/ggml-common.h | 1911 | crates/ggml/src/blocks.rs | ✅ | const asserts == C static_asserts |
| ggml/src/ggml.c (类型表/命名) | 8167 | crates/ggml/src/types.rs | ✅ | 测试 vs ggml.c 表 |
| ggml/src/ggml.c (tensor/context/op 构建器) | 8167 | crates/ggml/src/tensor.rs | 🔨 | Context/TensorId/Storage ✅; 算子构建方法进行中 (rope op_params 布局已核对: [1]=n_dims,[2]=mode,[4]=n_ctx_orig,[5..10]=f32位存储,[11..14]=sections,[15]=n_offs) |
| ggml/src/ggml.c (graph) | 8167 | crates/ggml/src/graph.rs | ✅ | build_forward 拓扑排序, 单测通过 |
| ggml/src/gguf.cpp | 1714 | crates/ggml/src/{gguf,gguf_write}.rs | ✅ | 读取: 实测 qwen2.5-0.5b 交叉验证; **写入: 与参考写入器逐字节一致**(parity/gguf_write_ref.bin, ref_gguf_write.c); **split.* 多分片**: `Gguf::open` 复刻 llama-model-loader.cpp:596-669 的分片装配(part 0 kv + 各分片张量并入、`split.no/count/tensors.count` 校验、`llama_split_path/prefix` 命名), `TensorInfo::part`+`tensor_storage` 供 model.rs 每张量指向所在分片的 mmap(7B 两分片原生加载实测); `Gguf::open_single`=原单文件读取(gguf-split 工具用) |
| ggml/src/ggml-quants.c | 5638 | crates/ggml/src/quants.rs + quants_k.rs | ✅ | dequant **全类型**位精确: Q4_0..Q8_1/Q2K..Q6K/IQ2/3/4,IQ1/**MXFP4/NVFP4**(代理O: 5024/5024 位同 + 真实 gpt-oss 张量 17280 f32 位同); 对照 parity/quants_ref.bin(+_iq.bin+_fp4.bin) |
| ggml/src/ggml-alloc.c | 1249 | crates/ggml/src/tensor.rs (`Context` arena) | 📄 | CPU-only 简化: 单 arena + `arena_resize_tensor`/`reset_graph_to`（COVERAGE.md "arena 分配 等价简化"）；gallocr 复用规划数值不可观测 |
| ggml/src/ggml-backend.cpp | 2513 | crates/ggml/src/compute.rs + crates/llama/src/context.rs | 📄 | CPU-only；行为面（eval_callback → compute.rs set_eval_callback、buffer 零初始化、cpu device description）已核（PARITY.md） |

## ggml-cpu 后端

| 参考文件 | 行数 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|---|
| ggml/src/ggml-cpu/ggml-cpu.c | 3944 | crates/ggml/src/compute.rs | ✅ | 主循环/mul_mat 全路由/mul_mat_id/type_traits_cpu 对应(见 AUDIT_ggml.md §2)；**vec_dot_type 全量化类型已接**（2026-09-27 审计轮，见 PARITY.md §AUDIT） |
| ggml/src/ggml-cpu/ops.cpp | 12206 | crates/ggml/src/ops.rs + compute.rs | ✅ | 代理A: 29测试, 量化matmul位精确, rms_norm位精确, rope/softmax/手算对照; 集成者补 set_rows(ops.cpp:5221)+30th测试 |
| ggml/src/ggml-cpu/vec.cpp/.h | 613 | crates/ggml/src/cpu/vec.rs → 并入 ops.rs(ggml_expf) + vec_dot.rs | ✅ | exp多项式标量复刻, f64累加; 代理M: tanh/gelu 无 SIMD — ggml_vec_tanh_f32(vec.h:909) 是纯 tanhf 标量循环(已证实 op 输出与 raw glibc tanhf 位同), gelu F32 走 f16 查表(vec.h:987, GGML_GELU_FP16 无条件定义); 位精确: parity/tanh_ref.bin |
| ggml/src/ggml-cpu/quants.c | 1339 | crates/ggml/src/vec_dot.rs | ✅ | 8种量化vec_dot整型位精确 |
| src/llama-vocab.cpp + unicode* | 12568 | crates/llama/src/vocab.rs + unicode*.rs | ✅ | 代理B: 16测试, 12个 ggml-vocab fixture 与参考版 llama-tokenize 逐id一致(qwen2/qwen3.5/llama-bpe/spm/gpt2/falcon/phi3/bert/command-r/deepseek/gemma4) |
| src/llama-arch.cpp/.h | ~2500 | crates/llama/src/arch.rs | ✅ | 代理C: 表尺寸/名称往返/块名模板测试 |
| src/llama-hparams.cpp/.h | ~300 | crates/llama/src/hparams.rs | ✅ | 代理C: qwen2.5 calculator 等测试 |
| src/llama-kv-cache.cpp + cells | ~3400 | crates/llama/src/kv_cache.rs | ✅ | 集成者: 非SWA统一缓存+cells+slot管理, set_rows 写入; **iswa 双 cache/mask 已补**(+363 行 src/llama-kv-cache-iswa.cpp: 第二个 cache + 每层选择, `find_slot` 的窗口复用/`apply_ubatch` 的 purge + `set_input_kq_mask` 的 `is_masked_swa` 分支); gemma4 1164-token 提示 16/16 (PARITY.md); **位置搬移族+K-shift 已补**(seq_add/seq_keep/seq_div/seq_cp :451-657 含 shift 累计器 + `build_graph_shift`/`memory_update` :2003-2053/llama-context.cpp:845-905→`context.rs`, `-c` 256 填充对齐 :290; **defrag 上游已删** — commit 9ebebef62, 修订版无此函数); DSA/MSA/recurrent 变体后续; **dsv4 多序列+回滚已补**(代理DSV4RS/NGRAM 2026-09-26: compressed 三 cache/三状态按 `n_stream=n_seq_max` 行块分流(:1287), `seq_cp` 全量复制(:1512-1527 含逐序列平面清零), `n_rs_seq>0` 回滚平面(:653-716 索引逐字+宿主行复制), `seq_rm` 的 head 回退规则补齐(llama-kv-cache.cpp:403-419); **状态序列化已补**(代理DSV4STATE 2026-10: `llama_kv_cache::state_write/read` 的 n_stream=1 切片 :2055-2628 → `raw_state_write/read`（base+swa 两半）, `llama_state_seq_*` 的宿主 IO 与 io_magic 帧 llama-context.cpp:3142-3213 → `StateWriter/StateReader` + `DecodeContext::state_seq_{get,set}_data`; dsv4 blob 与参考**逐字节相同**（预填+尾步两种, `parity/dsv4_state_parity.sh`）, 往返逐位恢复; dsa lid 半/MSA idx 半/整上下文头已补(代理DFLASH2 2026-09-27: llama-kv-cache-dsa.cpp:164-172 → kv_cache.rs `lid_state_{write,read}` + state_seq 分支（dsa_iswa :236-242 顺序）; llama-kv-cache-msa.cpp:160-168 → context.rs `msa_state_seq_{write,read}`; llama-context.cpp:3341-3381 → `state_{get,set}_data`; minimax-m3 4 blob 逐字节同参考, deepseek32 结构+dense-lead 行逐位, 往返逐位 —— parity/state_kinds_parity.sh)。未移: seq_add/seq_div 之外的压缩 cells 行滚动、ON_DEVICE blob; **attn 半层列表补 has_kv 过滤**(代理RSSTATE 2026-09-24: `raw_state_write/read`/`raw_view_base` 只序列化 k_row>0 的层——C 的 `layers[]` 不含 has_kv=false/被滤层 :165-175; `ForwardWeights::kv_dims` 为 jamba/nemotron-h/granite/lfm2/plamo2 补 0 宽逐层臂)。 |
| src/llama-graph.cpp (helpers) | 3915 | crates/llama/src/graph.rs | 🔨 | qwen2 非FA前向完成(toy smoke); FA路径/其他架构后续 |
| src/llama-context.cpp (decode 路径) | 4412 | crates/llama/src/context.rs | 🔨 | decode驱动+每步图重建(reset_graph_to); 对接 model.rs 后端到端 |
| src/llama-model-loader.cpp (tensors) | ~1000 | crates/llama/src/model.rs | ✅ | 代理E: 真实模型 mmap 零拷贝+Q5_0 解码与 python 32/32 位一致; 6 架构(qwen2/llama full + qwen3/gemma2/3/phi3 partial) |
| src/llama-sampling.cpp + llama-sampler.cpp | ~5800 | crates/llama/src/sampling.rs | ✅ | 代理F: 51 测试; mt19937/分布/libstdc++排序平局位精确; 注: 上游已删旧的 llama-sampling.cpp, 按 common/sampling.cpp+llama-sampler.cpp; **代理Z 追加 grammar sampler (llama-sampler.cpp:2658-2853, name="grammar")**; **审计(2026-09-24)曾记 DRY/ADAPTIVE_P/链序缺口(A1/A2) —— 采样任务(2026-09-24)已补齐: `DrySampler`+`init_dry`/`init_dry_testing`(:3317-3724), `AdaptivePSampler`+`init_adaptive_p`(:3727-3905), `InfillSampler`+`init_infill`(:4070-4299, `--samplers` 链可达), `CommonSamplerType`+`to_chr/to_str/from_names/from_chars`(common/sampling.cpp:795-919), `SamplingParams.samplers` 链序字段 + `new_with_vocab` 按 C 循环建链(sampling.cpp:340-413); 夹具 parity/sampler_dry_ref.txt(参考 libllama.so+libllama-common.so 实测) + tests/sampler_dry_parity.rs 位级回放; server 请求字段 `dry_*`/`dry_sequence_breakers`/`samplers`(api.rs) 与 CLI 旗标 `--dry-*`/`--samplers`/`--sampler-seq`/`--adaptive-*`(llama-cli) 已接** |
| src/llama-grammar.cpp/.h | 1526+194 | crates/llama/src/grammar.rs | ✅ | 代理Z: 8 个 `grammars/*.gbnf` 解析逐元素一致; 63 步 stacks/mask 位同参考; piece 表 151936 全同; `--grammar-file` e2e 4/5 语法 token 全同 (list.gbnf 在 0.013 logit 平局翻转, 数值尾巴) |
| ggml/src/ggml-cpu/unary-ops.cpp, binary-ops.cpp | 491 | crates/ggml/src/compute.rs (`forward_{tanh,…,xielu}` / `forward_bin_op`) | ✅ | 11/22 unary（其余 documented-skip, AUDIT §3）；bin_op 量化 src0 变体 = documented-skip（唯一消费者 cvec, 见 compute.rs::forward_bin_op 处注释） |
| ggml/src/ggml-cpu/ggml-cpu.cpp (接口/参数探测) | 716 | crates/llama/src/util.rs (cpu_info/线程参数) | ✅ | device description/props（PARITY.md） |
| ggml/src/ggml-cpu/repack.cpp (+arch/x86) | 11660 | crates/ggml/src/repack.rs (MXFP4 8x8 + **Q4_K 8x8**) + compute.rs 接线(2D mul_mat gemm/gemv 尾 + 3D mul_mat_id gemv) | ✅ | 代理S/代理REPACK: MXFP4 + Q4_K 布局/量化/gemv/gemm **位同参考** (parity/ref_repack_dump.cpp → repack_ref.bin; parity/ref_repack_kdump.cpp → q4k_repack_ref.bin: repack 字节全同 + AVX gemv/gemm 逐位 + 激活量化字节/反量化值全同); C 粒度 chunk(nth*4, NB_COLS 对齐)+wdata 4 行组跨线程; 懒缓存(默认开, LLAMA_RUST_REPACK=0 关); 路由: x86 K-quant 仅 Q4_K 有实例(repack.cpp:5006), Q5_K/Q6_K 无(repack.cpp:5050-5071, 两侧行式 vec_dot); 未做: q4_0/q2_k/iq4_nl 8x8(本机模型不可达/非 K-quant, Q4_0 端口走 tinyBLAS=位级路由分歧, 见 PARITY.md §Q4_K repack) |
| ggml/src/ggml-cpu/tiled/{tiled,tiled-kernel}.{cpp,h}（sync 批次 D 新增; 删除 iqp.{cpp,h}） | ~2400 | crates/ggml/src/tiled.rs + compute.rs 接线 | ✅ | VNNI 微内核+转置 84/84 位同（parity/tiled_kernel_ref.bin, dlsym 导出内核）; 整 op 52 MUL_MAT + 13 MUL_MAT_ID 段位同 + 线程不变（parity/tiled_ref.bin, NEW .so 图计算）; 详见 PARITY.md 批次 D |
| ggml/src/ggml-cpu/llamafile/sgemm.cpp | 4164 | crates/ggml/src/tinyblas.rs | ✅ | 6 组合(F32/F16/BF16/Q8_0/Q4_0/Q5_0) 路由表+尾块与参考 switch 逐位（parity/tinyblas_ref.bin 6360 例；旧"暂不移植"行已过时——AUDIT_ggml.md §5-B.7） |
| ggml/src/ggml-cpu/arch/x86 (SIMD 变体) | ~21k | crates/ggml/src/{simd_x86,vec_dot,repack}.rs | ✅ | 位精确验证: parity/vec_ref.bin + mulmat_ref.bin(+_c1) + repack/q4k/q4_0 系列 + **vecdot3_ref.bin**（2026-09-27 审计轮: Q1_0/Q2_0/NVFP4/IQ×9 vec_dot lane 位同, parity/ref_vecdot_dump3.c） | |

## src/ (llama 库)

| 参考文件 | 行数 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|---|
| src/llama-arch.cpp/.h | ~2500 | crates/llama/src/arch.rs | ✅ | 代理C: 表尺寸/名称往返/块名模板测试 |
| src/llama-hparams.cpp/.h | ~300 | crates/llama/src/hparams.rs | ✅ | 代理C: qwen2.5 calculator 等 |
| src/llama-vocab.cpp/.h | 4526 | crates/llama/src/vocab.rs | ✅ | 代理B: 12 个 ggml-vocab fixture 与参考 llama-tokenize 逐 id 一致 |
| src/unicode.cpp + unicode-data.cpp | 8442 | crates/llama/src/unicode.rs + unicode_data.rs | ✅ | 代理B: cpt roundtrip/nfd/utf8 测试 |
| src/llama-model-loader.cpp/.h | ~1000 | crates/llama/src/model.rs(ModelLoader) | ✅ | 见上 |
| src/llama-model.cpp/.h + src/models/* | 3359+2662 | crates/llama/src/model.rs + graph.rs + graph_arch.rs | 🔨 | 代理L 实测: qwen2 16/16 token 一致, llama-builder 与 qwen2-builder 位同, Phi-4-mini 前7 token 一致(idx7 1.79 logits OPEN, 侦查中N), gemma3 合成 PPL 1.3%; **P1 已修**(gemma norm 双 +1); gemma2 已通(代理M: ggml_tanh + softcap 接线; 真实 cap 50/30 下参考不再 NaN, PPL 2.4%, 8/8 token + top5 一致); gemma4/gpt-oss/lfm2moe/granitehybrid/qwen35 已移植并逐个与参考对照(见 PARITY.md); 新增 qwen3/bert/t5encoder |
| src/llama-graph.cpp/.h | 3915 | crates/llama/src/graph.rs + graph_arch.rs | 🔨 | qwen2 非FA+FA 完成; llama/gemma2/3/phi3 5 架构 builder + FA 分支(代理Q: FA-vs-refFA 16/16 首请求); **gemma/gpt-oss/lfm2 SWA 已接**(llama-graph.cpp:3131-3133 每层 cache/mask 选择 + iswa 双 mask); MoE/SSM/GDN 待补 |
| src/llama-kv-cache.cpp + cells | ~3400 | crates/llama/src/kv_cache.rs | ✅ | 非SWA统一缓存+cells+slot管理+set_rows 写入; **iswa 双 cache/mask 已补**(集成者 2026-09-24, 长上下文 16/16); DSA/MSA/recurrent/dsv4 变体已移(见上方行 37 与 PARITY.md) |
| src/llama-kv-cells.h, llama-memory*.cpp | ~2000 | crates/llama/src/context.rs + kv_cache.rs（无独立 memory.rs） | 🔨 | 代理RSSTATE 2026-09-24: **`llama_memory_recurrent::state_write/state_read` 已移**（llama-memory-recurrent.cpp:766-1224 → context.rs `recurrent_state_seq_{write,read}`、`recurrent_seq` 单活 cell、`is_pure_recurrent`/`state_write_attn_half` 的 create_memory 分发（mamba 纯递归 = recurrent 半独占; hybrid = attn 半+recurrent 半, llama-memory-hybrid.cpp:190-195）; mamba2 blob 与参考**逐字节相同**, /slots 全流程与参考服务器互证 —— PARITY.md RSSTATE 节）; rs_zero 的 `seq_rm` 清零修复随附; PLE 行/ON_DEVICE/多 cell 恢复未移; RWKV 记忆已移(批次 14: build_rwkv6/7_time_mix + token_shift 帮手 graph_arch.rs:32799/32811) |
| src/llama.cpp | 620 | crates/llama/src/lib.rs + model.rs + chat.rs + saver.rs 等 | 🟡 | `llama_version`→lib.rs `version()`; split_path/prefix→gguf.rs; chat_apply_template→chat.rs; 装载五入口→`Gguf::open`+`load_model` composed; supports_*→CPU-only 常量语义; backend_init/numa→no-op; `llama_time_us`→lib.rs `time_us()`(2026-09-28); `llama_print_system_info`/`llama_flash_attn_type_name`→display.rs(2026-09-28); `llama_model_save_to_file`→saver.rs(2026-09-28)。load_mode 文案函数随恒-mmap 设计失效(PARITY:748) |
| src/llama-context.cpp/.h | 4412 | crates/llama/src/context.rs | 🔨 | decode/decode_all+每步图重建+toy 全链路 0 误差; **多序列 `decode_batch`✅**(batch allocr→多序列 ubatch、逐 token `seq_id` 的 KQ mask、`out_ids`/`output_ids`/selection-sort reorder、`seq_rm`/`seq_pos_min|max`), 单序列路径逐字节未改; **dsv4 多序列/回滚驱动已补**(代理DSV4RS/NGRAM: `new_with_dsv4(n_seq_max, n_rs_seq)` 构造、`with_rs_rollback` 旋钮、逐序列/逐 token ubatch 切分、`rs_snapshot_step` 快照环、`seq_rm` 的 dsv4/回滚分支、`seq_cp` 压缩半边; n_rs_seq=0 时全部旁路); speculative 后续 |
| src/llama-batch.cpp/.h | ~700 | crates/llama/src/batch.rs | ✅ | `llama_batch_allocr` 全量: `init`(token/seq_id 校验、自动 pos(每序列 `seq_pos_max+1`)/n_seq_id/seq_id/logits、耦合序列、序列集合+`seq_idx`、一致性检查)+`split_simple`/`split_equal`/`split_seq`/`ubatch_reserve`/`split_reset`/`out_ids`/`seq_pos_min|max`; 10 个默认测试 |
| src/llama-sampler.cpp/.h | 4385 | crates/llama/src/sampling.rs | ✅ | 见上; 接线 CLI 进行中(代理J); 审计(2026-09-24)曾记 DRY(:3317)/ADAPTIVE_P(:3727) 缺口 —— **已补齐(2026-09-24 采样任务, 见行 41 补记)** |
| (旧 llama-sampling.cpp 已删除) | - | 并入 sampling.rs | ✅ | |
| src/llama-grammar.cpp/.h | ~2000 | crates/llama/src/grammar.rs | ✅ | 解析器/匹配器/piece 表位同参考(代理Z); lazy+json-schema 见 json_schema.rs | |
| src/llama-chat.cpp/.h | ~700 | crates/llama/src/chat.rs | ✅ | 代理H 交付 + 代理J 补 30 测试 + 集成者修复 jinja lstrip 索引移植偏差与模板尾换行 + 代理JINJA 重写 mini-jinja 为 minja 全语义引擎(common/jinja 四文件, 2026-10, 见 PARITY.md); 43/43 绿 |
| src/llama-adapter.cpp/.h (lora) | ~700 | crates/llama/src/adapter.rs | ✅ | 代理LORA: 元数据契约/`.lora_a`/`.lora_b` 配对/形状校验/`get_scale`(llama-adapter.h:53-57)/`meta_*` 全部移植; 图侧 `build_lora_mm`(:1514-1543)+`build_lora_mm_id`(:1545-1581)+`build_inp_embd` lora 分支(:2389-2405) 接入 11 个解码 builder 全部权重投影(外加 bert/t5 投影); `llama_set_adapters_lora` 语义(:1332-1374) 落在 adapter.rs 线程局部(见 PARITY.md); 控制向量 `llama_adapter_cvec` 未移植 |
| src/llama-cparams.cpp / llama-impl.cpp / llama-io.cpp | ~350 | batch.rs(const)/impl_log.rs/log routing + kv_cache.rs(io) | ✅ | cparams→`LLAMA_MAX_SEQ` const + DecodeContext 字段(composed); io→StateWriter/StateReader; impl→impl_log.rs(log_get/set/internal/callback_default + LLAMA_LOG_* 宏, llama-impl.cpp:28-71, quant.rs log_warn/error 已改路由) |
| src/llama-mmap.cpp | ~830 | memmap2 + crates/llama/src/mlock.rs | ✅ | mmap→memmap2 组合(PARITY:748 恒 mmap); `llama_mlock`→mlock.rs(extern mlock/munlock, 同款失败告警, `-lm mlock` 已接 llama-cli) |
| src/llama-model-saver.cpp | 505 | crates/llama/src/saver.rs | ✅ | 全档 1:1(add_kv_from_model KV 序 + add_tensors_from_model C 成员序 + per_layer 折叠 + rope FIXME dedup + vector<char> charsmap 的 C 同款 abort); byte 对照 492,021,632 字节仅 1 字节差(EOT id, vocab.rs:863 已档的 unordered_map 扫描序分歧); 参考装载往返生成 16/16 同; 见 PARITY.md §model-saver |

## common/ (按 CLI 所需子集)

| 参考文件 | 行数 | Rust 位置 | 状态 |
|---|---|---|---|
| common/common.cpp/.h | 2361+1213 | 分布式: 助手在各工具(llama-bench/params.rs 的 string_split、llama-server/api.rs 的 common_embd_normalize/common_token_to_piece、engine.rs 的 fs_validate_filename/槽位相似度) + crates/llama/src/vocab.rs(common_detokenize) | ⏳ 审计(2026-09-24): 符号级 38 ported/45 composed/18 documented/8 missed(死代码为主, 见 parity/AUDIT_common_tools.md §1.2; 原行所指 crates/common/src/common.rs 不存在) |
| common/arg.cpp/.h | 4763 | 每工具手写 parser(llama-bench/params.rs 最全) | ⏳ 审计(2026-09-24): 旗标级核算 505 个注册旗标中 159 在端口字面存在/346 缺(GPU/CPU 亲和 ~95 属范围外或 rayon, 未移植工具专属 ~90, server 运行时 ~30 已有 PARITY 记录); `common_chat_verify_template`(arg.cpp:950) 未移(原行所指 crates/common/src/arg.rs 不存在) |
| common/sampling.cpp/.h | 919 | crates/llama/src/sampling.rs（`SamplingContext` = common_sampler, 链构建/accept/reset/sample/惰性文法接线） | ✅ 审计(2026-09-24)曾记 DRY/链序缺口 —— **已补齐(2026-09-24 采样任务)**: `common_sampler_types_from_names/from_chars`/`to_chr`/`to_str`(:795-919) + `common_sampler_init` 链循环(:340-413) 移植(`SamplingContext::new_with_vocab`), 默认链含 DRY(common.h:265-275); reasoning-budget 连带 ⛔(原行所指 crates/common/src/sampling.rs 不存在, 状态 ⏳ 过期) |
| common/log.cpp/.h | 547 | 未移植(工具用 println/eprintln 直出) | ⏳ 审计(2026-09-24): `--log-file/--log-verbose/--log-colors` 等文件日志旗标面缺(原行所指 crates/common/src/log.rs 不存在) |
| common/chat*.cpp + peg-parser.cpp + common/parsers/* + common/jinja/* (chat tools 全链) | ~16700 | crates/llama/src/{peg,chat_tools,chat_parsers}.rs + chat.rs(mini-jinja 引擎+caps) | ✅ 全链已移 (代理CT 主体 2026-09-29 + 代理CHATPARSER 2026-10 + 代理JINJA 2026-10): peg-parser/chat-peg-parser/chat-auto-parser-generator/chat-diff-analyzer/chat.cpp jinja 路径 1:1; common/parsers/*.cpp 全部 **17** 个特化处理器(同步批次 C 增 llm-jp-harmony, chat.cpp:1102-1106 派发针 `chat_format=llm-jp-harmony-v1` 在 gpt-oss 之前) + gemma4/minimax 专用 mapper + continue_final_message(chat.cpp:1252-1271) → chat_parsers.rs; common/jinja/{lexer,parser,runtime,value,caps}.cpp → chat.rs mini_jinja(按 vendor 模板所需面: `{% macro %}`/块 set/namespace()/切片/kwargs/selectattr/字符串方法/**非调用式带参 test/`sameas`/`dict()`/强转数组属性**——批次 C 的六个上游提交, def4d406a 的 loop-scope 优化端口本就只按需克隆), **134/134 夹具用例(批次 C: +3 个 synth-llm-jp-harmony case 含并行工具调用; 夹具经 NEW 参考 def4d406a 重生成, 131 旧 case 输出零变化——六个 jinja 上游提交只解锁报错路径)** + 3 个真实 GGUF(gpt-oss-20b/gemma-4-12B/Qwen3.5-9B)模板与参考服务器 `/apply-template` 端到端逐字节; ⛔ 未移: ASR/reasoning-budget 预设; server 触发器序列化只写 WORD(接线项, 见 PARITY.md) |
| common/json-schema-to-grammar.cpp (+ .h) | 1028+23 | crates/llama/src/json_schema.rs | ✅ 代理A: **201/201 schema 的 GBNF 与参考逐字节同**, 33 个 integration grammar 的 248 条 accept/reject 判定全同 (`parity/json_schema_ref.txt`, 重生 `parity/gen_json_schema_ref.sh`) |
| common/json-schema.cpp (+ .h) | 514+609 | crates/llama/src/json_schema.rs | ✅ 代理A: schema 树 / `$ref` 解析 / `type` 数组展开 / 失败消息逐字同; `value_types()`/`may_be_string()` 一并移植 |
| common/json*.cpp (其余) | ~2600 | crates/llama/src/json_schema.rs (Json 子集) | ➖ 代理A: 只移植转换器所需子集 — `Json::parse`+`dump` 与 nlohmann `ordered_json` 逐字节同 (193 fixture round-trip, 含 Grisu2 float); 容器构造/变更 API 未移植; **不引入 serde_json** |
| common/speculative.cpp/.h | 2997+115 | crates/llama/src/speculative.rs | 🔨 代理SPEC: **draft-simple 全量**（类型表 / `are_compatible` / impl 计数器+计时 / synth 接受率 / n_max / `get_output_limits` / `common_speculative_init` / `begin·process·draft·accept` / `print_stats`）+ **draft-mtp**（deepseek2/32/4 的 graph_mtp 驱动, speculative.cpp:1330-1767 的单头模式; 图构建器在 graph_arch.rs、MTP 上下文与 `embeddings_nextn` 抽头在 context.rs; 合成 parity 11/12 格 16/16, 见 PARITY.md 的 MTP 节）+ `common/sampling.cpp:678-715` 验证规则 + `examples/speculative-simple.cpp:126-342` 驱动；0.5B 同模型 14 token 与 plain greedy 逐 token 同（temp 0；temp 0.8 见 PARITY.md §Q4_K repack 的前提失效说明）；0.5B draft + 7B target 36 token 与参考 plain/spec 完全一致、接受率 100%（mean len 4.00）、目标前向 37→10（省 73%）；Q4_K 8x8 repack 落地后 4 行前向 3.00x→**2.15x**、投机速度 0.50x→**0.81x**（剩余瓶颈在 draft 前向/驱动侧，非 K-quant 域）。**+ ngram 五员家族**（代理DSV4RS/NGRAM 2026-09-26: `ngram-simple`/`ngram-map-k`/`ngram-map-k4v`/`ngram-mod`/`ngram-cache` 的 impl = speculative.cpp:1769-2181, 参数结构 common.h:352-384, `n_max` 改读真实子参数 :2335-2369, 构造分支 :2676-2718; 库层在 `ngram_cache.rs`/`ngram_map.rs`/`ngram_mod.rs` 三新文件, save/load 与参考二进制格式字节兼容; 0.5B 自投机五类型提交流 == plain greedy, ngram-cache 接受 14 token/前向 29→18, 参考服务器五类型全新首请求全部 MATCH — `parity/run_server_ngram_parity.sh`, 见 PARITY.md）。**+ draft-eagle3**（代理EAGLE 2026-09-24: `common_speculative_impl_draft_eagle3` = speculative.cpp:426-907 逐行 —— 延迟边界 (token[P+1], g_embd[P]) 约定/跨 ubatch 桥/verify 快照重锚; 头双图在 graph_arch.rs `build_eagle3_{encoder,decoder}_forward` = src/models/eagle3.cpp:103-326, 头加载器+ctx_other 零拷贝在 eagle.rs, eagle 上下文/encode/目标侧 layer_inp 抽头在 context.rs（llama.cpp:127 的 t_layer_inp 记录加进 build_llama_forward）; 合成对（llama 目标+两头变体）parity **4 格全 PASS**（trunk-unchanged + draft parity 16/16 ×2FA）, 服务器侧 16 token 与草稿计数 39/39 逐字同参考 — `parity/eagle_parity.sh`, 见 PARITY.md 的 EAGLE3 节）。**+ draft-dflash / draft-dspark**（代理DSV4STATE 2026-10: `common_speculative_impl_draft_dflash` = speculative.cpp:910-1328 同一 impl 双类型（:2667-2675）; dflash 草稿模型双模解码器（KV 注入臂 dflash.cpp:608-677 + 噪声块臂 :679-853 含 DSpark markov/置信度头 :295-406）与加载器在**新文件 dflash.rs**, 非因果 mask（`llama_set_causal_attn` :1052 → context.rs `set_causal_attn`+fill_mask_seq 参数）、dflash 草稿上下文 `DecodeContext::new_dflash`; rope 为 **NEOX**（llama-model.cpp default 臂, 调试记录见 PARITY.md §DFlash）; parity **8 格全 PASS**（trunk-unchanged + draft parity × dflash/dspark × 2FA）+ **草稿链 153/153 逐行同参考**（`parity/ref_dflash_chain.cpp` = 参考 speculative-simple 驱动环探测, SPC_DBG 候选迹; 服务器与示例驱动注入前缀不同, 链对比必须同驱动）— `parity/dflash_parity.sh`, 见 PARITY.md 的 DFlash/DSpark 节）。未移植: `--draft-backend-sampling`、draft-mtp 的 chain_heads/is_mem_shared（step35/gemma4 模式）、eagle3 的 backend_sampling/get_state 边界 stash、DFlash2 conv/selector 图与 DSV4 DSpark 骨干（dflash.rs 拒载并报错, dflash.cpp:408-567/:855-1028）、M-RoPE 草稿（speculative.cpp:1014-1018）、检查点式部分接受回退 |
| common/ngram-cache.cpp/.h | 286+102 | crates/llama/src/ngram_cache.rs | ✅ 代理DSV4RS/NGRAM: `CommonNgram`/哈希/`update`/`draft`（lax+strict 阈值 :60-63）/`save`/`load`/`merge` 全量; save/load 与参考二进制格式字节兼容（条目顺序随 HashMap）; tie 的 argmax 依赖哈希迭代顺序（C 同） |
| common/ngram-map.cpp/.h | 537+116 | crates/llama/src/ngram_map.rs | ✅ 代理DSV4RS/NGRAM: `common_ngram_simple_draft`（:49-112）+ `CommonNgramMap::begin/draft/accept`（:121-536, key_only=ngram-map-k 否则 k4v; reasoning-chat 收缩清理 :128-135/:171-215）; 262144 槽哈希加速表按 C 的可选项置空（线性搜索恒在）; 参考无 ngram-proximity 类型 |
| common/ngram-mod.cpp/.h | 63+39 | crates/llama/src/ngram_mod.rs | ✅ 代理DSV4RS/NGRAM: `CommonNgramMod` LCG 开放表全量（idx/add/get/reset/size, 4M 槽由 speculative.rs 构造） |
| 其余 (download/hf/ngram/...) | — | 按工具需求增量移植 | ⏳ | **console.cpp 已移(2026-09-28 代理TOOLS-5)**: `crates/llama/src/console.rs` 全文双读入器(simple :1046-1081 + advanced :753-1044, termios/wcwidth 经 extern "C"), Windows 半边/spinner 不移(PARITY.md TOOLS-5 节); **subproc.cpp 已移**: `crates/tools/llama-server/src/subproc.rs`(std::process 封装, combined stdout+stderr 经 pipe+dup); 仍未移: fit.cpp(`--fit` 自动适配)、preset.cpp(预设体系)、http.h、download/hf-cache；llguidance.cpp 整文件 `#ifdef LLAMA_USE_LLGUIDANCE`(默认构建 OFF, PARITY:251)；reasoning-budget ⛔(行 81)；log.cpp 旗标面(`--log-file/--log-verbose`) ⏳ |

## tools/

| 参考文件 | Rust 位置 | 状态 |
|---|---|---|
| tools/main/main.cpp + tools/cli/* | crates/tools/llama-cli | 🔨 | 采样链+EOG+chat模式+**架构分发全覆盖(集成者: `graph_arch.rs` 的 11 个 builder 全部有 CLI 臂 + `-j/--json-schema` / `-jf/--json-schema-file`; 见文末"CLI 架构分发"节)**+--flash-attn; **采样链旗标已接(2026-09-27 采样任务): `--samplers`/`--sampler-seq`/`--sampling-seq`/`--dry-multiplier`/`--dry-base`/`--dry-allowed-length`/`--dry-penalty-last-n`/`--dry-sequence-breaker`(arg.cpp:1982-2003/2123-2182, 首用清默认+"none" 清空)/`--adaptive-target`/`--adaptive-decay`(arg.cpp:2184-2202), DRY 输出与参考 server 逐 token 一致(PARITY.md 采样任务节)**; qwen2/phi3 无回归; **qwen3/gpt-oss/gemma4/granite/lfm2/qwen35 经 CLI 与全新参考 server 首请求前 16 token 全同**; **交互面已移(2026-09-28 代理TOOLS-5)**: console.cpp 全文 + tools/completion 的交互 REPL(`src/interactive.rs`, `-i/-if/--in-prefix(-bos)/--in-suffix/-r/-mli/--simple-io/--(no-)display-prompt/-e/-sys(f)/--(no-)show-timings/-sp/--verbose-prompt/-ptc/--prompt-cache(-all/-ro)/-st/-cnv/-no-cnv/-co/--keep/--(no-)context-shift` 全接; `parity/cli_interactive_parity.sh` 17/17 stdout 字节同) + 扩散驱动(`src/diffusion.rs`, `--diffusion-*`); 仍未移: HTTP 客户端架构(cli-client/cli-server/cli-context/cli-ui: 会话状态/媒体 staging/模型列表/SSE)与 `common_chat_verify_template`(arg.cpp:950) |
| tools/quantize | crates/tools/quantize + llama/src/quant.rs | ✅ 代理X 25/25 逐字节 + **代理IMATRIX: `--imatrix` 接线后 19/19 ftype 逐字节**(Q2_K..Q6_K / Q4_0..Q5_1 / Q8_0 / F16 / **IQ4_NL / IQ4_XS** / COPY, 含此前造不出的 **Q2_K_S**); 缺口: `--include/exclude-weights`、`--prune-layers`、IQ1/IQ2/IQ3 量化器(需 kmap/neighbours 运行时表, 见 PARITY) |
| tools/imatrix | crates/tools/imatrix + llama/src/imatrix.rs (+ ggml/src/{compute.rs,quants_impl.rs}) | ✅ 代理IMATRIX: 写出的 GGUF 与参考**元数据/张量表完全相同**、`--in-file` 重写与 `--output-format dat` **逐字节相同**、`--show-statistics` 输出相同、被参考 llama-quantize 接受(跨兼容); 采集值受端口前向"数值一致但非逐位"限制(层0 三个张量逐位同, 深层 max rel 4e-4→4.5); 缺口: IQ2/IQ3/IQ1 量化器、MoE(`MUL_MAT_ID`)未实测 |
| tools/gguf (实为 examples/gguf, 自测读写器) | crates/tools/gguf | ✅ 代理Y: `r` 模式与参考**逐字节相同**(qwen2.5 645 行 + 19 词表 + gpt-oss + Bonsai-Q1_0 + clip-BF16); 另加 -m/--kv/--tensors dump 模式; `w` 未移植(写路径由 gguf_write 位对照覆盖) |
| tools/tokenize | crates/tools/tokenize | ✅ 代理Y: 20 模型 × 5 模式 **100/100 输出逐字节相同**; 缺口: 非 UTF-8 输入需 llama crate 加 tokenize_bytes 入口 |
| tools/perplexity | crates/tools/perplexity | ✅ 代理Y: chunk 语义逐位置验证(512/512 token, 255/255 评分点); 全文 PPL 差 −0.0074%(残余=图级噪声, 与参考自身 fa on/off 波动同分布); **代理SPLIT (2026-09-24): 三评分器已移**——`hellaswag_score`(:744)/`winogrande_score`(:1101)+`load_winogrande_from_csv`/`multiple_choice_score`(:1405)+二进制数据集反序列化, 数据解析/评分公式/输出格式/Wilson 区间逐行对照, 任务选择随机数与 g++ 13 libstdc++ 位同(mt19937+Lemire `_S_nd<u64>`+`int(scale*rng()*n)` f32 惯用法, parity/ref_rng_vec.cpp), 合成夹具 stdout 与参考逐字节相同(parity/scorer_parity.sh); 缺口: ppl-stride(`perplexity_v2`)/KL/bench 套件未移植; 性能 1/17 |
| tools/llama-bench | crates/tools/llama-bench | ✅ 代理BENCH: 全部旗标解析 + 26 维测试矩阵(批次 C: +`--repack <0|1>` 第 27 维→`ggml::repack::set_repack_override`, 965f89794; null-log 回调放行 ERROR, 7dad6db85; BUILD pin→def4d406a/11325)(顺序与 C 一致) + 计时协议(warmup/每 rep 清 KV/`-d` 深度) + `avg`/`stdev` 逐式 + **5 种输出格式表头逐字节相同**(md/csv/sql/jsonl/json; JSON 43 键非时间字段全同); token 数逐 rep 精确(pp64→[64,64], tg16→[16,16]); 性能 0.19-0.55x, 与 bench.sh 的 0.20x/0.40x 同带; 缺口: GPU/后端旗标、非 f16 KV、embeddings、线程池旋钮 WARNING 忽略 |
| tools/server (main/server/server-context/server-http/server-schema/server-task/server-common/server-chat) | crates/tools/llama-server | 🔨 | std TCP HTTP/1.1 + 槽位引擎（idle→processing→done、`cache_prompt` 前缀复用、LRU/相似度选槽、多序列 batched decode、speculative 槽位状态机）+ 原生 API `GET /health`·`GET /props`·`POST /completion(s)`·`POST /tokenize`·`POST /detokenize` + **OpenAI 兼容面**: `POST /v1/chat/completions`(+`/chat/completions`, 模板经 `chat::apply_str` 只读渲染、json_schema/json_object grammar 包裹+prefill、SSE `[DONE]`/usage 帧)、`POST /v1/completions`、`POST /embedding(s)`·`/v1/embeddings`（BERT `EncoderContext` 路径, bge-m3 逐位相同）、`GET /models`·`/v1/models`、`GET /slots`、rerank 族(参考同款 501)；`parity/run_server_parity_chat.sh` **24/24 检查全对齐**（chat/OAI/embeddings/models/slots, 2026-09-28）; ~~POST /slots save/restore~~ **已移**（PARITY:4583, `POST /slots/{id}` save/restore/erase）; **context shift 已补**(`--context-shift`/`--keep` + `Engine::context_shift`, `parity/run_server_parity_shift.sh` 260 token 双 server MATCH, 见 PARITY.md); 未移植（PARITY:2103-2111 逐条开档）: Responses/Anthropic/transcriptions、`/infill`、`/apply-template`+`*/input_tokens` 计数族、`/metrics`、`/v1/chat/completions/control`、`/lora-adapters`、`/v1/stream` 族、router 模式(server-models.cpp 全套)、API key/CORS/gzip、内置 UI、mtmd、rerank 打分、`n_cmpl>1` 子槽、idle sleep/purge; **已移(2026-09-28 代理TOOLS-5)**: `/tools` GET+POST(`--tools`/`--tools-runtime`, 7 内置工具 + x-tool-cwd/runtime/resp-type 头 + SSE 流式, `server_tools.rs`)、MCP 桥(`--mcp-servers-config/-json`, stdio 传输/warmup/懒重启, `server_mcp.rs`)、GCP/Vertex 兼容(AIP_* 四变量 + /predict 驼峰别名分发 + AIP 端口覆盖)、`/cors-proxy`(`--ui-mcp-proxy`/`-ag`; http:// 转发, https 答 500 已开档)、未启用时两者 403 feature_disabled; `parity/server_tools_mcp_parity.sh` 15/15; 仍未移(既有开档): Responses/Anthropic/transcriptions、`/infill`、`/apply-template`+计数族、`/metrics`、control、`/lora-adapters`、`/v1/stream` 族、router(server-models.cpp)、API key/CORS origins/gzip、内置 UI、mtmd、rerank 打分、`n_cmpl>1` 子槽、idle sleep/purge |
| tools/mtmd/clip.cpp（视觉路径 + 音频路径）| crates/llama/src/clip.rs（新）| ✅ 代理MM: `qwen3vl_merger` 端到端（hparams/KV、张量表、ViT 图、位置插值、M-RoPE、merger、预处理、encode）；实测见 PARITY.md 同名小节（cb 448 `-fa off` L2 rel 1.20e-4 / cos 0.9999999928；fixture PNG 全链路 2.66e-4 / 0.9999999714）；✅ 代理AUDIO2 (2026-09): build_vit 共享路径 + whisper-enc 音频图 6 arch（qwen2a/ultravox/voxtral/meralion/glma/musicflamingo）合成协议 **逐位同**（见 PARITY.md 音频图节）；**同步批次 C**: +`ling3vl`(f830688e9, qwen3vl 塔 + `mm.input_norm` norm-only merger + 顶层 mm.0/mm.2, 合成端到端测试 + `parity/ling3vl_parity.sh`); 四模型 pad+roll→`pad_ext` 六处(57b557cb9); conformer mel 修复(fcc891545, 对称窗/mel_floor_add/std_eps_after_sqrt); glm5v(649dcb103)不移植——glm4v 塔在 ~50 projector 缺口内; 缺口: 其余 ~50 projector(glm4v/glm5v 含)、视频、`clip_set_debug_output_embeddings` 之外的 audio-gen 输出 |
| tools/mtmd/mtmd-image.cpp | crates/llama/src/clip.rs（img_tool + clip::image_preprocess）| ➖ dyn_size / fixed_size 两个预处理 + Pillow 兼容 22-bit 定点点重采样（bilinear/bicubic/lanczos）；llava-uhd 系（internvl/minicpmv/deepseekocr/step3vl/lfm2/granite/muse-glimmer）未移植 |
| tools/mtmd/mtmd.cpp（视觉+音频半边）+ mtmd-helper.cpp | crates/llama/src/mtmd.rs（新）| ✅ marker 切分/tokenizer chunk/`mtmd_encode_chunk`（含 Audio 臂）/`tokenize_parts`(`mtmd_tokenize_from_parts`)/M-RoPE 解码位置/分批 eval; 音频预处理器臂扩到全 12 个（init_audio 家族含 granite/gemma4a/parakeet/mimo/qwen3tts_spk/pockettts, 代理AUDIO4）; 图像解码自实现 PNG（自写 inflate）/BMP/PPM（stb_image 的其余格式未移植）; 音频解码 WAV + **FLAC（代理AUDIO4: 自实现 spec 子集解码器, dr_flac 数值管线逐式照抄, 6 个 ffmpeg 产文件对参考嵌入逐位同, PARITY.md 音频第四轮 §C）**; mp3 保持点名报错; 缺口: 视频、lazy bitmap、`libmtmd` C ABI、交互 chat、`--jinja`、gen_audio 路径 |
| tools/mtmd/mtmd-cli.cpp | crates/tools/llama-mtmd-cli（新 crate）| ✅ `-m/--mmproj/--image/-p/-n/-t/-fa/--temp/-c` + `--dump-embd`/`--embd-file`/`--prompt-tokens`/`--logits-top`（parity 工具）；greedy 前缀 12/24 与参考一致，分歧点经 embedding 注入实验判定为文本侧（PARITY.md） |
| tools/mtmd/mtmd-audio.cpp（1557 行全量）| 1557 | crates/llama/src/mtmd_audio.rs | ✅ (代理AUDIO, 2026-10): 11 个预处理器 + FFT/DFT + mel 滤波器组 + 流式 ISTFT, 参考 probe 逐位校验; WAV 读取器(miniaudio 子集)在 mtmd.rs |
| tools/mtmd/models/whisper-enc.cpp | 137 | crates/llama/src/clip.rs（`build_whisper_graph`/`build_vit_audio`/`conv_1d_ph`/`build_stack`/`swiglu_swapped`/`pool_1d_avg2`） | ✅ 代理AUDIO2 (2026-09): 全 6 arch 投影器分支 + conv1d stem + build_vit 分离 qkv 路径, 合成协议对参考 **逐位同**（`ggml_gelu_erf` 已由 PERF4 补齐）; ~~POOL_1D/单张量 GLU 缺~~ POOL_1D 已由 AUDIO3 补齐并换字面构造（六投影器复跑逐位不变）; 单张量 GLU 仍为视图+swiglu_split 位同等价 |
| tools/mtmd/models/qwen3a.cpp + gemma4ua.cpp + conformer.cpp（lfm2a）| 88+19+216 | crates/llama/src/clip.rs（`build_qwen3a_graph`/`build_gemma4ua_graph`/`build_conformer_graph` + 融合 qkv 分支 + ExtraInput 注入） | ✅ 代理AUDIO3 (2026-09): qwen3a/gemma4ua 合成协议对参考**逐位同**（FA on/off 双路）; lfm2a 参考接受、嵌入差 ≤5.7e-6（ULP 级卷积模块残差, PARITY.md 开档）; 新算子 roll/conv_2d_dw_direct/conv_2d_direct/relu + POOL_1D 见 ggml 行 |
| tools/mtmd/models/granite-speech.cpp + gemma4a.cpp + parakeet.cpp + mimo-audio.cpp + qwen3tts-spkenc.cpp + pockettts-spkenc.cpp（+seanet） | 309+288+421+218+197+77+162 | crates/llama/src/clip.rs（`build_granite_speech_graph`/`build_gemma4a_graph`/`build_parakeet_graph`/`build_mimo_graph`/`build_qwen3tts_spkenc_graph`/`build_pockettts_spkenc_graph` + seanet 辅助 + hparams/加载/set_inputs/n_tokens/n_embd 全臂） | ✅ 代理AUDIO4 (2026-09): gemma4a/mimo_audio/qwen3tts_spkenc/pockettts_spkenc 合成协议对参考**逐位同**（FA on/off 双路）; granite_speech/parakeet 参考接受、嵌入差 ≤4.2e-4/1.4e-6——根因=参考自身 ssm_conv AVX-512 向量化（位级节点二分 + 反汇编 + 穷举 C 复现三重证明, PARITY.md 音频第四轮 §D）; 新算子 sub/sin/cos/sqr/mean/pad_reflect_1d/elu/pad_ext 见 ggml 行 |
| tools/mtmd/models/qwen3tts-gen.cpp + pockettts-gen.cpp | 770+291 | ⛔ 未移（输出生成器, llama-mtmd-cli 不驱动——装载走 `clip.has_gen_audio_encoder`（clip-impl.h:36）, 运行走 `mtmd_gen_audio_process`（mtmd.cpp:1922-2036）; 判据已开档; GEN_WAV 需 `ggml_col2im_1d` 亦未移）|
| tools/gguf-split（split/merge, 609 行）| crates/tools/gguf-split（新 crate）| ✅ 代理SPLIT (2026-09-24): `gguf_split`/`gguf_merge` 全参数(--split/--merge/--split-max-tensors/--split-max-size N(M|G)/--no-tensor-first-split/--dry-run/--delete-splits)与 kv 顺序(gguf_set_val_* remove+append 语义)、alignment=32 quirk、输出文本 逐字复刻; **跨工具逐字节**: qwen2.5-0.5b 三种模式(30 张量/100 张量+no-tensor-first/200M)共 17 个分片 与参考全部 cmp 相同、双方 merge 输出逐字节相同(含 7B 两分片 4.68GB)、参考可原样消费端口的分片(parity/gguf_split_parity.sh); 读取侧 split.* 支持见 gguf.cpp 行 |
| tools/completion（961 行, 直连交互工具: 会话循环/`common_chat_format_single`/in-prefix 交互）| crates/tools/llama-cli/src/interactive.rs | ✅ **已移(2026-09-28 代理TOOLS-5)**: 主环逐段(completion.cpp:545-940) + 会话格式化 + `--prompt-cache`(端口自有 token 框架); `common_prompt_batch_decode`/`common_replay_last_token` 的语义就位(分块 decode / n_past 记账), 参考侧 llama_state 会话字节格式不复刻(已开档) |
| tools/batched-bench（267 行并发基准）| — | ⛔ 未移（审计 2026-09-24 补录; llama-bench 已覆盖 `-b/-ub` 维度但无并发格）|
| tools/cvector-generator（878 行 + pca/mean.hpp）| — | ⛔ 未移（审计 2026-09-24 补录; 与行 68 "控制向量未移植"同族, 工具本身此前未点名）|
| tools/export-lora（439 行 LoRA 导出）| — | ⛔ 未移（审计 2026-09-24 补录; 端口只做 LoRA 加载侧）|
| tools/fit-params（83 行, common/fit.cpp 前端）| — | ⛔ 未移（审计 2026-09-24 补录）|
| tools/results（183 行 NMSE 对比）/ tools/tuning（1,089 行 fa-vec/bench 调参）| — | ⛔ 未移（审计 2026-09-24 补录, 开发工具）|
| tools/tts（214 行 TTS 生成驱动）| — | ⛔ 未移（审计 2026-09-24 补录; mtmd_audio.rs 预处理器已全量但无生成驱动）|
| tools/rpc（344 行远端后端 RPC）| — | ⛔ 范围外（GPU 分布式场景, 行 4 CPU-only 口径; 审计 2026-09-24 显式记录）|
| tools/mtmd/debug/mtmd-debug（311 行）| — | ➖ 工具本体未移（审计 2026-09-24 补录）; 其 `-p encode` dump 语义已由端口 `clip_cb_parity_dump` 复现（PARITY:1299 以参考侧该工具为真值源）|
| tools/mtmd/legacy-models/*.py（1,882 行权重转换脚本）| — | ⛔ 范围外（Python 转换器, 与 convert_hf_to_gguf.py 同类; 审计 2026-09-24 显式记录）|

## examples/ 审计汇总（审计 2026-09-24; 明细 parity/AUDIT_common_tools.md §3）

| example | 状态 |
|---|---|
| speculative-simple | ✅ 驱动环并入 speculative.rs（FILE_MAP:85） |
| gguf | ✅ 成 crates/tools/gguf |
| lookup-create/merge/stats | 📄 PARITY:3592（库层 ngram-cache 字节兼容, 工具未移） |
| parallel / embedding | ➖ 语义已由 server 多槽与 /embedding(-fe) 覆盖（run_embedding_parity/server slots parity）; example 本体未移 |
| simple / simple-chat / batched | ➖ 生成语义由 llama-cli 覆盖; example 本体未移 |
| diffusion（diffusion.cpp+cli, 731 行）| crates/tools/llama-cli/src/diffusion.rs | ✅ **已移(2026-09-28 代理TOOLS-5)**: 掩码迭代驱动(diffusion.cpp:103-408: 置信度 5 算法/双调度/gumbel/CFG/shift_logits) + `--diffusion-*` 旗标 + diffusion-cli 横幅/visual/收尾; `llm_arch_is_diffusion` 分发; in-port 单测 5 项(参考无生成二进制) |
| passkey / retrieval / lookahead / gguf-hash / idle / debug / eval-callback / speculative(旧) / training / convert-llama2c-to-ggml | ⛔ 未移（参考性示例; eval-callback 的 dump 语义已由 parity/ref_decode_dump 家族复现）|
| llama.android / llama.swiftui / llama.vim / batched.swift / sycl / test-cmake / deprecation-warning / gen-docs / *.py / *.sh | 范围外（平台绑定/GPU/构建/脚本）|

## 实测计划 (PARITY.md 待建)

1. GGUF 解析: qwen2.5-0.5b-instruct-q4_k_m.gguf ✅ (types+gguf 单元测试)
2. tokenizer: llama.cpp/models/ggml-vocab-*.gguf round-trip
3. 数值: 与参考构建同 prompt/temp=0/seed 逐 token 对比
4. 性能: pp512/tg128 基准对比 (llama-bench 等价物)


## 架构加载支持矩阵 (代理P, 2026-09-25 — load_model 级, 对真实文件张量数精确相等)
| arch | 本地文件 | 状态 | 剩余缺口 |
|---|---|---|---|
| qwen2 / llama | qwen2.5-0.5b / deepseek-coder-33b | ✅ 完整 + 前向 | — / 前向未跑(21.9GB) |
| qwen3 | Qwen3-Embedding-0.6B | ✅ 加载 | 前向 builder(per-head q/k norm) |
| gemma2 / gemma3 | (合成文件) | ✅ 加载+前向(PPL 2.41%/1.28%) | SWA 曲线已接(双 cache/mask), 无真实文件可跑长上下文 |
| phi3 | Phi-4-mini | ✅ 加载+前向 | **16/16 token 与参考一致**(P4 修复: rope_runtime 推导) |
| **gpt-oss** | gpt-oss-20b MXFP4 + Q4_K_M (459/459) | ✅ **加载+前向+真实推理** | **MXFP4 16/16 token 与参考一致**(FA 忠实语义: VKQ16+f16 Q点积+sinks src[4]); Q4_K_M 待重测 |
| **gemma4** | 12B-QAT (667/667) + 26B-A4B (658/658) | ✅ 加载(代理P, 含逐层 head/кv 数组+共享 rope) | 前向图 |
| **lfm2moe** | LFM2-8B-A1B (256/256) | ✅ 加载(代理P, shortconv+MoE 混合) | 前向图; dense_2 未测 |
| **granitehybrid** | tiny (666/666) + micro (506/506) | ✅ 加载(代理P, mamba2 mixer+MoE) | 前向图 |
| **qwen35** | 27B (851/851) + 27B-MTP (866/866) | ✅ 加载(代理P, GDN 线性注意力+MTP) | 前向图; qwen35moe 未移植 |

## 多模型实测矩阵 (代理L, 2026-09-25)
| 模型 | load_model | 前向 | 备注 |
|---|---|---|---|
| qwen2.5-0.5b Q4_K_M | ✅ | ✅ argmax=12095 " Paris" 与参考一致; 3-token 前缀 " Paris. It"; determinism 位精确 | 主对照 |
| Qwen3-Embedding-0.6B Q8_0 | ✅ (28层/310张量) | ✅ 16/16 token vs fresh 参考(FA on/off, CLI) | qwen3 builder 已接 CLI |
| Phi-4-mini Q6_K | ✅ (32层/196张量) | 手动测试(ignored) | partial arch |
| 62层/561张量 模型 | ✅ | — | |
| gemma-4-12B / gpt-oss-20b(MXFP4) / LFM2-8B / granite-h / qwen35-27B | ✅ 加载 | ✅ 各自 16/16 token vs fresh 参考(短 prompt) | 长上下文 SWA: gemma4 1164 token 16/16(见 PARITY) |
| bge-m3 (bert) / t5-xxl-encoder | ✅ 加载 | ✅ 编码器输出与参考 dump **100% 位精确**(全 T) | 新增 EncoderContext |
| llama-builder on qwen2 权重 | ✅ | 与 qwen2-builder argmax 一致 | 跨架构 builder 交叉验证 |

## SIMD 内核 + 线程模型（代理SIMD, 2026-09-27）
| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| ggml/src/ggml-cpu/arch/x86/quants.c（AVX2 量化内核: q4_0/q4_1/q5_0/q5_1/q8_0/q4_K/q5_K/q6_K + hsum/mul_sum/bytes_from_*/get_scale_shuffle 助手） | **crates/ggml/src/simd_x86.rs（新）** + vec_dot.rs 派发（标量 lane 移植保留为非 AVX2 兜底） | ✅ | `simd_x86::tests::simd_matches_scalar_bit_exact`（逐位 == 标量）× {量化器块, 随机原始块}；既有参考产物测试全绿（mulmat_ref.bin / *_real_ref.bin / vec_ref.bin）；真实模型 qwen 16/16、gpt-oss MXFP4 16/16 不变 |
| ggml/src/ggml-cpu/ggml-cpu.c:1404-1450 mul_mat chunk 网格 + :1727 mul_mat_id chunking | compute.rs::par_mul_mat（原来只在激活列并行 ⇒ 单 token=单线程） | ✅ | `vec_dot::thread_invariance_tests`（nth=1/2/3/4/8 逐位相同，含回归：顺序回退曾只跑第一个 chunk） |
| ggml-cpu.c threadpool + ggml_barrier（自旋等待） | compute.rs::Team / TeamShared / worker_loop（替换逐算子 rayon install: 45µs/区 → 自旋 barrier） | ✅ | 生成 65 → 77 t/s、sys 1.29s → 0.16s；`cargo test --workspace` 全绿 |

## qwen3 前向图（代理Q3, 2026-09-24）
| 参考文件 | 行数 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|---|
| src/models/qwen3.cpp:53-159 `llama_model_qwen3::graph::graph` | 159 | crates/llama/src/graph_arch.rs `build_qwen3_forward` + `Qwen3LayerWeights` / `Qwen3ModelWeights`（新） | ✅ 完成 | 真实文件 Qwen3-Embedding-0.6B-Q8_0：短提示 **16/16**（-fa on 与 -fa off 各自对照全新 server 首请求）；长提示（74 token）step 1 平局翻面，teacher-forced 残差 ≤0.477（详见 PARITY.md qwen3 节） |
| src/models/qwen3.cpp:3-46 `load_arch_hparams` / `load_arch_tensors` | 同上 | crates/llama/src/meta.rs QWEN3 分支 + crates/llama/src/model.rs:834-884 | ✅ 复用（未改动） | 既有 `meta.rs::qwen3_embedding_values` / `model.rs::qwen3_embedding_real_model`；本轮 e2e 另钉 hparams→AttnParams 接线（key_length=128 ⇒ n_rot=NEOX，freq_base 1e6） |
| crates/llama/tests/qwen3_e2e.rs（新） | — | 该 arch 的 e2e：默认跑 `qwen3_embedding_hparams_and_params`（hparams/张量接线/两提示词 tokenizer）；3 个 `#[ignore]`：`qwen3_embedding_0_6b_reference_parity`、`qwen3_embedding_0_6b_long_prompt_parity`、`qwen3_embedding_fa_path_spread` | ✅ | `cargo test --release -p llama --test qwen3_e2e -- --ignored --nocapture`（+`QWEN3_FA_OFF=1` 另一模式） |
| crates/llama/src/graph_arch.rs 测试：`toy_qwen3_matches_naive`（新）、`toy_flash_attn_smoke_all_archs` 增 qwen3 臂 | — | graph_arch.rs `#[cfg(test)] mod tests` | ✅ | 玩具前向 vs 朴素 ≤2e-2 相对误差；FA/非FA 差 1.3e-6 且每层恰好 1 个 FLASH_ATTN_EXT 节点 |

## JSON Schema → GBNF（代理A, 2026-09-24）

**落点**: `crates/llama/src/json_schema.rs`（新, 含 `common/json-schema-to-grammar.cpp` + `common/json-schema.cpp` + `common/json.cpp` 的所需子集）; `crates/llama/src/lib.rs` 加 `pub mod json_schema;`; 测试 `crates/llama/tests/json_schema_parity.rs`; 真值 `parity/{ref_json_schema_dump.cpp,gen_json_schema_ref.sh,schemas/,json_schema_ref.txt}`。

### C → Rust 符号对照（json-schema-to-grammar.cpp:line → Rust）

| C 位置 | C 符号 | Rust |
|---|---|---|
| :18 | `build_repetition` | `build_repetition` |
| :45 | `build_min_max_int`（含 `uniform_range`/`digit_range`/`more_digits` 局部 lambda） | `build_min_max_int`（内嵌同名 fn） |
| :229 | `SPACE_RULE` | `SPACE_RULE` |
| :236 / :251 | `PRIMITIVE_RULES` / `STRING_FORMAT_RULES` | `PRIMITIVE_RULES` / `STRING_FORMAT_RULES` |
| :260 | `is_reserved_name` | `is_reserved_name` |
| :275-285 | `INVALID_RULE_CHARS_RE`/`GRAMMAR_LITERAL_ESCAPE(S)_RE`/`NON_LITERAL_SET`/`ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` | `replace_invalid_rule_chars` / `format_literal` / `is_non_literal` / `escaped_in_regexps_but_not_in_literals`（`GRAMMAR_RANGE_LITERAL_ESCAPE_RE` 未使用未移植） |
| :287 / :305 / :313 | `replacePattern` / `format_literal` / `gbnf_format_literal` | `format_literal` / `pub fn gbnf_format_literal` |
| :315 | `gbnf_escape_length` | `gbnf_escape_length` |
| :342 | `class common_chat_schema_converter` | `struct Converter<'a>` |
| :356 :371 :390 :406 :641 :682 :699 :796 :820 :824 :828 :832 :877 :960-990 | `_add_rule` `_generate_union_rule` `_visit_pattern` `_pattern_to_rule` `_not_strings` `_resolve_ref` `_build_object_rule`(+`get_recursive_refs`) `_add_primitive` `add_schema` `_generate_constant_rule` `_visit_primitive` `_visit_all_of` `visit` `check_errors`/`format_grammar` | 同名 `Converter::*`（`_pattern_to_rule` 的 `transform`/`join_seq`/`get_dot` → `transform`/`join_seq`/`get_dot`; `literal_or_rule` → `LiteralOrRule`; 异常 → `PatternError::{Unsupported,Invalid}`） |
| :993 / :1008 | `json_schema_to_grammar(json, force_gbnf)` / `(document)` | `pub fn json_schema_to_grammar` / `json_schema_to_grammar_document` |
| :1015 | `build_grammar(cb, options)` | `pub fn build_grammar(&doc, GrammarOptions, cb)` + `GrammarBuilder::{add_rule,add_schema}` |
| json-schema.h:14-192 | `common_chat_schema*` 类族 / `type_set` / `value_type` / `string_format` | `SchemaKind`/`SchemaNode`/`SchemaProperty`/`SchemaDocument`（arena `Vec<SchemaNode>` + `NodeId`，替代裸指针以表达 `$ref` 环）/ `TypeSet`/`ValueType`/`StringFormat` |
| json-schema.cpp:12-344 | `common_chat_schema_builder` + `common_chat_schema_from_json` | `struct SchemaBuilder` + `pub fn schema_from_json` |
| json-schema.cpp:346-514 | `json_type` / `value_types` / `may_be_string` / `kind_name` / `type_name` | `json_type` / `SchemaDocument::{value_types,may_be_string}` / `SchemaNode::kind_name` / `type_name` |
| common/trie.h:12-50 (trie.cpp:43) | `common_trie` | `struct Trie`（`BTreeMap<u32,usize>` 子节点 = 码点序） |
| json.cpp:201/339/405 | `common_json::parse`/`dump`/`get<T>` | `Json::{parse,dump,get_i64,get_int,get_f64,get_str}`（`dump_float`/`format_buffer`/`append_exponent` 复刻 nlohmann Grisu2） |
| common.cpp:540-568 | `string_split`/`string_join`/`string_repeat` | 就地展开（`Vec::join`/`std::iter::repeat`） |

### 校验

`cargo test -p llama --test json_schema_parity` → **201/201 GBNF 逐字节**, 33 grammar / 248 判定全同, 197 JSON round-trip 同 nlohmann, 80 个参考套件 grammar 可解析, 186 两入口一致, 贪心生成 e2e; `cargo test --workspace` 全绿（llama 库 195 = 189 基线 + 6 新单测）。

## CLI 架构分发全覆盖 + json-schema 旗标（集成者, 2026-09-24）

**改动面**: `crates/llama/src/context.rs`（`ForwardWeights` 增 `Qwen3`/`GptOss` 两个变体；decode/decode_all 各补 2 个分支）与 `crates/tools/llama-cli/src/main.rs`（架构分发 + per-arch 参数推导 + 语法旗标）。`graph_arch.rs` / `json_schema.rs` **零改动**（builder 与转换器保持各自的 e2e 真值）。

| arch (`LlmArch`) | builder | CLI 臂 | CLI 对照（全新参考 server 首请求, temp=0） |
|---|---|---|---|
| qwen2 | `graph::build_qwen2_forward` | 已有 | 16/16（-fa on） |
| llama | `build_llama_forward` | 已有 | 本地无可用文件（21.9 GB） |
| phi3 | `build_phi3_forward` | 已有 | 16/16（Q6_K, -fa off） |
| gemma2 / gemma3 | `build_gemma2_forward` / `build_gemma3_forward` | 已有 | 仅合成 GGUF |
| qwen3 | `build_qwen3_forward` | **新增** | 16/16（-fa on 与 -fa off） |
| gpt-oss | `build_gpt_oss_forward` | **新增** | 16/16（MXFP4, -fa on） |
| gemma4 | `build_gemma4_forward` | **新增** | 16/16（12B-QAT, -fa off） |
| granitehybrid | `build_granite_forward` | **新增** | 16/16（h-tiny Q4_K_M, -fa off） |
| lfm2moe | `build_lfm2_forward` | **新增** | 16/16（8B-A1B Q4_K_M, -fa on） |
| qwen35 | `build_qwen35_forward` | **新增** | 16/16（27B Q4_K_M, -fa on） |
| 其余 | 无 builder | `other` 臂打印 arch 名并 exit 1 | — |

（上面"架构加载支持矩阵"里 gemma4/lfm2moe/granitehybrid/qwen35 行的"前向图"缺口语义已闭：builder 早已存在，本轮补齐的是 CLI 可达性 —— 见 `PARITY.md` 同名小节。）

**语法旗标映射**（参考 `common/arg.cpp`）:

| CLI | C 位置 | `params.sampling.grammar` |
|---|---|---|
| `--grammar GBNF` | :2265-2271 | `{USER, value}` |
| `--grammar-file FNAME` | :2273-2277 | `{USER, read_file(value)}` |
| `-j, --json-schema SCHEMA` | :2279-2283 | `{OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))}` |
| `-jf, --json-schema-file FILE` | :2286-2300 | 同上（先读文件） |

四个旗标写同一个字段 ⇒ **命令行后者覆盖前者**；handler 异常按参考版包成 `error while handling argument "<arg>": <msg>\n\nusage:\n<opt usage>\n\nto show complete usage, run with -h`（arg.cpp:866-873），端口与参考 `diff` 三例逐字节相同。

**对照脚本**: `parity/run_cli_arch_parity.sh`（`tokens` / `schema` 两种模式；server 启动→首请求→CLI→比对）+ `parity/cli_parity_cmp.py`。

## BERT + T5-encoder 编码器架构（代理BERT, 2026-09-25）

**参考**: `src/models/bert.cpp`（LLM_ARCH_BERT 分支）、`src/models/t5.cpp`（`graph<true>` 编码器实例化）、`src/models/t5encoder.cpp`、共享 builder `src/llama-graph.cpp` 的 no-cache 变体、`llama_model::build_graph` 的 pooling 挂载。
**新增驱动**: `llama_encode` 语义 = `crates/llama/src/context.rs::EncoderContext::{build,run,encode}`（解码路径 `DecodeContext` 零改动）。
**校验**: `crates/llama/tests/bert_e2e.rs` / `t5_e2e.rs`（默认跑装载+图结构，`#[ignore]` 跑真实前向对照 `parity/encode_*.bin`）、`encode_gemm_probe.rs`（编码器形状的 GEMM 内核对照，真值 `parity/mulmat_q8_bert_ref.bin`）。

### 图节点/助手 C → Rust 对照

| C 位置 | C 符号 | Rust | 备注 |
|---|---|---|---|
| models/bert.cpp:68-221 | `llama_model_bert::graph::graph` | `graph_arch::build_bert_forward` + `BertLayerWeights`/`BertModelWeights` | token+type+pos 嵌入、post-LN 布局、GELU-SEQ FFN |
| models/bert.cpp:95 / :154 / :209 | `build_norm(..., LLM_NORM, il)` | `graph_arch::build_norm`（新增，LLM_NORM 专用） | eps = `hparams.f_norm_eps`（bert.cpp:4） |
| models/bert.cpp:139-141 | `build_attn(inp_attn, wo, wo_b, …, 1/sqrt(head))` | `graph_arch::attn_no_cache`（kq_scale = 1/sqrt(n_embd_head)） | 非 FA 分支 |
| models/t5.cpp:264-358 | `llama_model_t5::graph<true>` | `graph_arch::build_t5_encoder_forward` + `T5EncoderLayerWeights`/`T5EncoderModelWeights` | RMSNorm 前置、kq_scale=1.0、gated GELU |
| models/t5encoder.cpp:42-44 | `llama_model_t5encoder::build_arch_graph` | 走同一 builder（t5encoder 复用 t5 的 `graph<true>`） | — |
| llama-graph.cpp:2740-2763 | `build_attn_inp_no_cache` | `EncodeInputs{kq_mask}` + `EncoderContext::build` 填 0/-inf | 单序列全 0（:414-441） |
| llama-graph.cpp:2451-2464 | `build_inp_pos` | `EncodeInputs.pos`（bert） | `llm_graph_input_pos::set_input` = 0..n-1 |
| llama-graph.cpp:2555-2567 | `build_inp_pos_bucket_enc` | `EncodeInputs.pos_bucket`（t5） | 填法见下 |
| llama-graph.cpp:3890-3923 | `llama_relative_position_bucket` | `graph_arch::relative_position_bucket` | 表 `parity/t5_bucket_ref.txt` 433/433 同 |
| llama-graph.cpp:2587-2599 | `build_pos_bias` | `graph_arch::build_pos_bias` | reshape→get_rows→reshape→permute(2,0,1,3)→cont |
| llama-graph.cpp:2480-2494 | `build_inp_out_ids` | `EncodeInputs.out_ids` | encode 恒为 0..n-1 恒等行选（llama-context.cpp:1518-1521） |
| llama-graph.cpp:2495-2511 / :2513-2528 | `build_inp_mean` / `build_inp_cls` | `EncodeInputs.mean` / `.cls` + `graph_arch::build_pooling` | 单序列：mean=1/n，cls=0（LAST=n-1） |
| llama-graph.cpp:3677-3772 | `build_pooling` | `graph_arch::build_pooling` | NONE/MEAN/CLS/LAST；RANK 未实现（panic 带行号） |
| llama-model.cpp:2764-2766 | `llama_model::build_graph` 挂 pooling | `EncoderContext::build` 末尾调用 | 与 C 同序 |
| llama-context.cpp:1466-1659 | `llama_context::encode` | `EncoderContext::{build,run}` / `encode` | n_ubatch≥n_tokens、全 token 为 output、强制非因果 |
| llama-context.cpp:964-1001 | `llama_get_embeddings` / `_seq` | `EncodeEmbeddings{n_embd_out,n_rows,values}` | NONE→平铺，pooled→单行 |
| llama-context.cpp:216-222 | pooling_type 解析 | `context::resolve_pooling` | UNSPECIFIED→hparams→NONE |
| llama-graph.cpp:1583-1613 | `build_norm` | `graph_arch::build_norm`（LLM_NORM）+ 既有 `build_norm_rms`（t5） | — |
| models/bert.cpp:23-62 | `llama_model_bert::load_arch_tensors` | `model.rs::load_arch_tensors` `LlmArch::BERT` 臂 | 389/389 张量（bge-m3） |
| models/t5encoder.cpp:9-40 | `llama_model_t5encoder::load_arch_tensors` | `model.rs::load_arch_tensors` `LlmArch::T5ENCODER` 臂 | 219/219 张量 |
| models/t5.cpp:3-13 / t5encoder.cpp:3-7 | `load_arch_hparams` | `model.rs::load_arch_hparams_t5` | `f_norm_rms_eps` + `n_rel_attn_bkts`（必需）、dec_* 只读 |
| llama-model.cpp:1257 | `LLM_KV_ATTENTION_CAUSAL` | 既有 `meta.rs`（`hparams.causal_attn`） | bge-m3 = false ⇒ 非因果 mask |
| llama-vocab.cpp:1947 | `tokenizer.ggml.token_type_count` | `ModelLoader::token_type_count` | bert.cpp:26-28 的 `n_token_types` |
| — | `llama_model::bert_weights` / `t5_encoder_weights`（端口新增，C 里是 `model.layers[il].*` 直读） | `model.rs::LlamaModel::{bert_weights,t5_encoder_weights}` | 权重装配集中一处 |

### 新增张量成员（`LayerTensors` / `ModelTensors` / `LlamaModel`）

`attn_out_norm(_b)`、`layer_out_norm(_b)`（bert.cpp:51-60）；`enc_attn_norm`、`enc_attn_rel_b`、`enc_w{ q,k,v,o }`、`enc_ffn_norm`、`enc_ffn_gate`、`enc_ffn_{down,up}`（t5encoder.cpp:27-38）；模型级 `token_types`、`position_embd`、`token_embd_norm(_b)`、`cls`、`cls_b`、`cls_out_b`（bert.cpp:30-38）。

### 校验

`cargo test --workspace` → **365 passed / 0 failed / 40 ignored**（ggml 99、llama 库 195 与基线一致，无回归）。
`cargo test -p llama --release --test bert_e2e --test t5_e2e -- --ignored --nocapture` → BERT T=1 99.02% 位精确（max|Δ|/rms 1.1e-7）、T5 T=1 **100% 位精确**；T≥2 尾见 `PARITY.md` 同名小节（mean|Δ|/rms 1.3-2.6e-2，根因为 tinyBLAS-F32 / f32 vec_dot 尾 + 激活量化放大，已由 `encode_gemm_probe.rs` 逐形状钉死）。
`bash parity/gen_encode_ref.sh` 重生成全部 11 个真值档；`parity/ref_t5_bucket.c` 重生成 433 行 bucket 表。

## tinyBLAS 生产 GEMM 路径（2026-09-25 追加）

| C 出处 | C 符号 | Rust 位置 | 说明 |
|---|---|---|---|
| llamafile/sgemm.cpp:625-671 | `tinyBLAS<KN,D,V,TA,TB,TC>`（`gemm_bloc`/`mnpack`/`gemm`） | `tinyblas::f::{panel,panel_bf16}` | AVX512 实例: `KN=16,__m512`（F32/F16）、`KN=32,__m512bh`（BF16，VDPBF16PS）；`hsum`=`_mm512_reduce_add_ps` 8/4/2/1 树；主循环 `l+KN<=k` + **K 尾段**（#29806: `load_partial` 掩码载入 + 每 acc 一次 `madd`, :653-668） |
| sgemm.cpp:387-461 | `load_partial{,_u16}`（`__m512`/`__m512bh`/`__m256i` 掩码载入） | `tinyblas::f::{load_partial16,load_partial_u16_512,bf16_load_partial}` | K 尾段专用（#29806）；掩码零车道积贡献恰为 0 |
| sgemm.cpp:568-612 | `tinyBLAS::matmul` 门控 | `tinyblas::Op::eligible` | x86 无 `k % KN != 0` bail（:569-572, #29806 移除）→ `m%16/%8/%4` 选 BM → `m%4!=0` bail |
| sgemm.cpp:1351-1795 | `tinyBLAS_Q0_AVX`（`gemm`/`gemm4xN`/`gemmMx4`/`load`/`updot`） | `tinyblas::q0::{tile,tile_tail,panel_decode,load_q8_0,load_q4_0,load_q5_0}` | 4 行板 × NCOL 列块；`updot` = 端口既有 `simd_x86::mul_sum_i8_pairs_float`（C 在本机走 `dpbusd` 分支，同值） |
| sgemm.cpp:3805-4149 | `llamafile_sgemm` 派发（含 `n<2`/`Ctype`/各 case 的 `Btype` 守卫） | `tinyblas::{Op,resolve,accepts,gemm}` | 谓词表见 `tinyblas.rs` 头注与 PARITY.md |
| ggml-cpu.c:1303-1325 / :1386-1409 | 两次 `llamafile_sgemm` 尝试（#29806 起均带 `!params->use_ref` 守卫, :1308/:1387）+ `goto UseGgmlGemm1/2` | `compute.rs::Cpu::{sgemm_planes,forward_mul_mat}` | 尝试 1 = `src1` 连续（`ggml_is_contiguous` 逐字复刻）且 `Btype=src1->type`；尝试 2 = `src1->type != vec_dot_type` 且 `Btype=vec_dot_type`；`use_ref` 的端口对应物 = `LLAMA_RUST_NO_TINYBLAS`（`tb_off`） |
| ggml-cpu.c:1385-1399 | wdata 行序 `[i13][i12][i11]`、`ldb = row_size/type_size` | `compute.rs::sgemm_planes`（`b_off2 = ne11*b_row`、`b_off3 = ne12*ne11*b_row`） | F16/BF16 由端口就地转换（`halff::f16::from_f32`），Q0 用既有 wdata |
| vec.cpp:455-528 | `ggml_vec_cvar_f32`（AVX512 分支） | `compute.rs::forward_norm` | 每 16 元素 `_mm512_reduce_add_ps(val*val)` + f64 累加，尾巴逐个 f64 |
| vec.cpp `ggml_vec_dot_f32` SIMD 尾巴（`.so` 反汇编 +0x1da..0x2ee） | 16 宽未融合块 / 8 宽块 / 标量 `vfmadd231ss` | `vec_dot::vec_dot_f32_c` | 尾巴累加在 f32（`float sumf`），非 f64 |
| sgemm.h / 导出的 `llamafile_sgemm` | dump 工具直接调用参考内核 | `parity/ref_tinyblas_dump.c` → `parity/tinyblas_ref.bin` | 6360 例路由+逐位真值；需 `ggml_backend_cpu_init()` 填 f16 查表 |

## 架构批次 6 个: gpt2 / phi2 / starcoder2 / command-r / gptneox / olmo2（代理BATCH, 2026-09-24）

合成 GGUF 对照 **12/12 全 16/16**（每格全新参考 server + 首请求，`-fa on`/`-fa off` 各 16 token；详见 `PARITY.md` 同名小节）。

| arch | C 文件 | Rust builder | 加载器 | CLI 臂 | 合成对照 |
|---|---|---|---|---|---|
| gpt2 | src/models/gpt2.cpp:15-148 | `graph_arch::build_gpt2_forward` | `model.rs` GPT2 臂 + `load_arch_hparams_batch` | `llama-cli` GPT2 臂（`Gpt2Params`） | 16/16（off/on），Δlogprob 1e-4 |
| phi2 | src/models/phi2.cpp:13-142 | `build_phi2_forward` | PHI2 臂 | `Phi2Params` | 16/16，1e-4 |
| starcoder2 | src/models/starcoder2.cpp:16-157 | `build_starcoder2_forward` | STARCODER2 臂 | `StarCoder2Params` | 16/16，1e-4 |
| command-r | src/models/command-r.cpp:13-143 | `build_command_r_forward` | COMMAND_R 臂 | `CommandRParams`（logit_scale） | 16/16，6e-4 / 1.3e-3 |
| gptneox | src/models/gptneox.cpp:54-219 | `build_gptneox_forward` | GPTNEOX 臂 | `GptNeoxParams`（use_par_res） | 16/16（par_res true 与 false 变体），1e-4 |
| olmo2 | src/models/olmo2.cpp:26-197 | `build_olmo2_forward` | OLMO2 臂（含 SWA 三键） | `Olmo2Params` | 16/16，1e-4 |

* 新增共享 helper（`graph_arch.rs`）：`build_ffn_gelu_seq`（LLM_FFN_GELU+SEQ，llama-graph.cpp:1795-1800/:1836-1852）、`qkv_fused`（:1649-1686）、`qkv_separate_parts`（:1687-1746）、`rope_qk`。`build_norm`（LLM_NORM）/`build_norm_rms`/`build_ffn_silu_par`/`attn_kv_cached` 未改动。
* `model.rs` 新增 `output_norm_b`（llama-model.h:632）到 `ModelTensors`/`LlamaModel`；`arch_tensors_support` 6 个 arch 标 `Partial`（已 CLI 可达，见 `context.rs::ForwardWeights::{Gpt2,Phi2,StarCoder2,CommandR,GptNeox,Olmo2}`）。
* 测试：`crates/llama/tests/arch_batch_e2e.rs`（9 个，默认 8 跑：每 arch 一个 loader/张量集/hparams/前向钉 + SWA 元数据 + writer round-trip；1 个 `#[ignore]` 造文件）。
* 合成文件生成器就在该测试里（`build_file`/`tensors_for`），可用 `cargo test -p llama --test arch_batch_e2e -- --ignored arch_batch_write_synth` 重建到 `/tmp/arch-batch/`；对照驱动 `parity/arch_batch_parity.sh` + `parity/arch_batch_cmp.py`。

## 架构批次 3 个: baichuan / bloom / mpt / starcoder / refact / plamo / stablelm / granite(dense) / minicpm — **ALiBi 机制 + 合成 GGUF 26/26 格全对齐**（代理BATCH3, 2026-09-27）

架构计数 **26/156 → 35/156**。协议沿用批次 1/2；合成文件在 `/tmp/arch-batch3/`、测试
`crates/llama/tests/arch_batch3_e2e.rs`（默认 11 跑 + 2 `#[ignore]`）、对照驱动
`ARCH_BATCH3=1 ./parity/arch_batch_parity.sh`（端口侧走 env 驱动的 `arch3_cli_driver`，
打印 llama-cli 同格式 debug 行 ⇒ `arch_batch_cmp.py` 零改动；含 baichuan-13B 的 72-token
长提示格）。**ALiBi 机制**（本批解锁整族）: `attn_kv_cached` 新 `max_bias` 参数
（llama-graph.cpp:2643/:2705，非 alibi 调用点全显式 0.0）+ KQ mask kept 值 `-|p0-p1|`
（llama-kv-cache.cpp:1692-1697 → `graph::fill_kq_mask_alibi{,_f16}`）+ `use_alibi` 在
`load_arch_hparams_batch` 尾部重判（llama-model.cpp:1419）。starcoder 在本 pinned
revision **无 alibi**（上游已删，旧队列标注作废）。

| arch | C 文件 | Rust builder | 加载器 | 合成对照 |
|---|---|---|---|---|
| baichuan | src/models/baichuan.cpp:3-121（13B=40 层 alibi 无 rope；7B=32 层 rope NORM 无 alibi） | `build_baichuan_forward` | BAICHUAN 臂 | 16/16（off/on 两变体 + 72-token 长提示格），Δlogprob ≤2.1e-3 |
| bloom | src/models/bloom.cpp:3-151（alibi 8、fused qkv+bias、token_embd_norm） | `build_bloom_forward` | BLOOM 臂 | 16/16，0 |
| mpt | src/models/mpt.cpp:3-171（**alibi 走 GGUF KV**、clamp、可选 pos_embd/bias/act-scales） | `build_mpt_forward` | MPT 臂 | 16/16（最小 + `-full` 两文件），≤1.9e-3 |
| starcoder | src/models/starcoder.cpp:3-154（pos_embd 必需、无 alibi） | `build_starcoder_forward` | STARCODER 臂 | 16/16，0 |
| refact | src/models/refact.cpp:3-160（alibi 8、rope NONE、rope_freqs 装而不读） | `build_refact_forward` | REFACT 臂（MoE 文件显式拒） | 16/16，0 |
| plamo | src/models/plamo.cpp:3-136（FFN 吃 norm 后输入、双残差） | `build_plamo_forward` | PLAMO 臂 | 16/16，≤1e-4 |
| stablelm | src/models/stablelm.cpp:3-172（12B: per-head q/k LN + parallel residual；3B: ffn_norm + partial rope） | `build_stablelm_forward` | STABLELM 臂 | 16/16（两变体各一份文件），≤1e-4 |
| granite(dense) | src/models/granite.cpp:3-320（logit/residual/embedding/attention scale、rope_finetuned 开关、MoE 分支） | 复用 `build_granite_forward` + `GraniteParams::dense` | GRANITE 臂 | 16/16，0 |
| minicpm | src/models/minicpm.cpp:3-96（graph = granite::graph，默认 scale 三连） | 同 granite | MINICPM 臂（与 granite 共臂） | 16/16，0 |

* CLI/DecodeContext 未接（context.rs 归并行代理；9 arch 由 `arch_batch3_e2e.rs::Batch3Model` 驱动，
  集成者接线时记得 mask 侧选 `fill_kq_mask_alibi*`）。mpt 的 Q/K 全宽 norm 分支在**参考里也不可达**
  （fused qkv 必需 + reshape 连续性断言，mpt.cpp:43 FIXME），端口同语义 panic。

## 架构批次 6b 个: nemotron(dense) / grok / chameleon / deci / jais / falcon-h1 / plamo2（代理BATCH6B, 2026-09-24）

批次 7 队列的 dense/ALiBi/hybrid-mamba 条目。架构计数（`arch_tensors_support` 口径）**54 → 61**；
任务账面 **63 → 70/156**。协议沿用批次 1-6，N=48 token/格 + `-long` 长提示格；ForwardWeights/CLI
臂与图同批落地，parity 驱动 llama-cli 本体（`ARCH_BATCH6B=1 ./parity/arch_batch_parity.sh`）。
合成文件在 `/tmp/arch-batch6b/`（10 个）、测试 `crates/llama/tests/arch_batch6b_e2e.rs`
（默认 11 跑 + 1 `#[ignore]` 造文件）。**28/28 格全 48/48 对齐**（9 arch × fa off/on + 5 long 格），
Δlogprob 大多为 0（grok FA 格 0.13-0.19 数值尾巴, token 不分叉）。

| arch | C 文件 | Rust builder | 加载器 | 合成对照（48 token, off/on） |
|---|---|---|---|---|
| nemotron (dense) | src/models/nemotron.cpp:50-150（LN+偏置、relu² MLP、必需 output） | `build_nemotron_forward` | NEMOTRON 臂 | 48/48 ×2, 0 |
| grok | src/models/grok.cpp:86-223（post-norm attn、GELU MoE、logit_scale、**GROK kq softcap 仅非FA** llama-graph.cpp:2682-2689、**f_embedding_scale=78.38 走共享 build_inp_embd** llama-graph.cpp:2433-2437） | `build_grok_forward` + `attn_kv_cached_grok` + `build_moe_ffn_gelu` | GROK 臂（默认值+可选键 :3-33） | 48/48 ×2 + long ×2 |
| grok-dense | grok.cpp:171-184 dense 分支 √2/2 + :75-78 `layer_output_norm` 名 | 同上 | 同臂 | 48/48 ×2 |
| chameleon | src/models/chameleon.cpp:53-204（全宽 q/k LN、图像 token logit 钳制 [4,8196)→-FLT_MAX — set_1d 打在 inp_out_ids 剪枝**后**的行上） | `build_chameleon_forward`（视图+concat 组合） | CHAMELEON 臂（swin_norm 可选、qk eps 1e-5） | 48/48 ×2, 0 |
| deci | src/models/deci.cpp:80-191（逐层 head/ffn 数组、四种层型、get_rope_factors） | `build_deci_forward` | DECI 臂 | 48/48 ×2, 0 |
| deci-mixed | deci.cpp:107-155（linear/attention-free/FFN-free 层型 + **FFN-free 层 continue 丢弃 attention 输出** 的字面语义） | 同上 | 同臂（数组 KV） | 48/48 ×2 + long ×2, 0 |
| jais | src/models/jais.cpp:55-132（LN+偏置、fused qkv+bias、无 rope、ALiBi 走 GGUF KV、kq_scale=1/n_embd_head） | `build_jais_forward` | JAIS 臂 | 48/48 ×2 + long ×2, 0 |
| falcon-h1 | src/models/falcon-h1.cpp:112-209（llm_build_mamba_base: 每层 attention∥mamba2 双支、attn_norm 两次 norm、双残差） | `build_falcon_h1_forward`（复用 `build_mamba2_mixer`） | FALCON_H1 臂（ssm 五键+eps 必需、is_recr 全 1） | 48/48 ×2 + long ×2, 0 |
| plamo2 | src/models/plamo2.cpp:110-426（自有 mamba mixer: 逐头 z/x 切分、bcdt x-proj、dt/B/C RMS；V 头宽≠Q/K；SWIGLU-SEQ FFN = 单张量对半 GLU） | `build_plamo2_forward` + `build_plamo2_{attn,mamba}_layer` | PLAMO2 臂（ssm 五键+eps 必需、is_recr=n_head_kv==0、dt_dim=max(64,n_embd/16)） | 48/48 ×2 + long ×2 |

* 无新增 ggml 算子：grok softcap 用 scale+tanh 组合、chameleon 的 `ggml_set_1d` 用视图+concat
  字节等价组合（ggml.c:3546 的 GGML_OP_SET 构造器留给集成者）、plamo2 的 `ggml_swiglu` 用
  半宽视图+`swiglu_split`（内核同元素, 逐位等价）。
* 排障记录（4 个坑: grok 共享 build_inp_embd 的 embedding_scale / 两个 hybrid 的 RMS eps 必需读
  / plamo2 `let mut cur` 遮蔽 / chameleon set_1d 的行选择）+ chameleon-swin 变体未对齐的诚实
  记录见 PARITY.md 批次 6b §2/§4。
* 顺手修复: 批次 5 `mamba_state_vs_full_recompute` 的并发文件竞态（私有 `-state` 后缀）。
* 回归: 批次 1-6 parity 串行全对齐; qwen2.5 双 FA 锚点 16/16; `cargo test --workspace` **564/0**
  （基线 553 + 11 新默认测试）。

### 下一批队列（批次 8，按成本排序；C file:line = pinned bd4f514db1）— 历史快照

> 批次 6b 已消掉批次 7 队列的 nemotron/grok/chameleon/deci/jais/falcon-h1/plamo2；
> 批次 7 已消掉 deepseek4+dsv4（见下）；批次 8（本节下方的批次 8 小节）已消掉
> hunyuan-moe/dots1/glm4-moe/bailingmoe/bailingmoe2/minimax-m2/cohere2moe/exaone-moe
> （最新队列在"下一批队列（批次 9）"）。

| arch | C 文件:行 | 需要什么（相对已有 helper） |
|---|---|---|
| **deepseek4 + dsv4** | src/models/deepseek4.cpp（1502）+ src/llama-kv-cache-dsv4.cpp（2253） | 批次 6 的 MLA 机制件可复用; 剩: dsv4 四 cache（iswa raw + csa/hca 两个 `llama_dsv4_comp_state` 压缩 kv_b 状态 + lid）、`ggml_dsv4_hc_comb/hc_pre` 新算子（ggml.c:6457-6541, **集成者加**）、hyper-connection、o_group lora、compress_ratios+压缩 rope、swiglu clamp、SQRT_SOFTPLUS 门控、SWA_FULL 非因果窗、hash 层（完整清单 PARITY.md 批次6 §4） |
| hunyuan-moe / dots1 / glm4-moe / bailingmoe2/3 / llada-moe / minimax-01/m2/m3 / cohere2moe / exaone-moe / smallthinker / graniteswitch 等 | 各 src/models/*.cpp | 全部是 `build_moe_ffn` 家族 + 既有 norm/attn helper 的组合; 按下游用户优先级排（glm4-moe/hunyuan-moe 先）。glm4-moe 注意 per-layer SWA 模式; minimax-01 有线性注意力层（n_embd_head_la 状态, 批次6b 的 hybrid recurrent 基建可复用） |
| 线性注意力 | plamo3.cpp（195）/ kimi-linear.cpp / qwen3next / delta-net-base 等 | delta-net/RWKV 新算子族（GATED_DELTA_NET 已有 = qwen35 用）, kimi-linear 的 KDA/PLE 需要新 cache 行, 成本高 |
| jais2 / eurobert / arcee / dream / wicked-fast 等小众 | 各自 models/*.cpp | 大多是既有 helper 的直接组合, 逐个按需 |
| 多模态（clip/mtmd 系新 projector）、稀疏注意力（glm-dsa）、RWKV 全家 | — | 需要算子或 cache 层新机制, 不在"下一批"档 |

其余多模态（clip/mtmd 系新 projector）、稀疏注意力（glm-dsa）、RWKV 全家需要算子或 cache 层新机制，不在"下一批"档。

## 架构批次 8 个（MoE 长尾家族）: hunyuan-moe / dots1 / bailingmoe / bailingmoe2 / glm4-moe / minimax-m2 / cohere2moe / exaone-moe — **合成 GGUF 32/32 格全对齐**（代理BATCH8, 2026-09-30）

上面队列第二行的 `build_moe_ffn`-composition 条目。架构计数（`arch_tensors_support`）
**62 → 70**; 任务账面 **71 → 79/156**。协议沿用批次 1-7 + `-long` 长提示格；ForwardWeights/
CLI 臂与图同批落地，parity 驱动 llama-cli 本体（`ARCH_BATCH8=1 ./parity/arch_batch_parity.sh`，
`-long` 后缀 = >64 token 格；cohere2moe/exaone-moe 带 n_swa=64，`-long` 格真实越窗 ——
llama-cli 的 DecodeContext 构造处本批补了 llama-model.cpp:2687-2690 的 iswa 分裂判定）。
合成文件在 `/tmp/arch-batch8/`（9 个）、测试 `crates/llama/tests/arch_batch8_e2e.rs`
（默认 11 跑 + 1 `#[ignore]` 造文件）。**32/32 格（8 arch × fa off/on × 短/长提示）全 16/16**，
worst |dlogprob| 大多 0.0000（cohere2moe/exaone-moe 越窗 FA 格 0.0010/0.0007）。

| arch | C 文件 | Rust builder | 加载器 | 合成对照（16 token, off/on） |
|---|---|---|---|---|
| hunyuan-moe | src/models/hunyuan-moe.cpp:52-187（rope **后** q/k norm；MoE+共享专家 MLP 双分支；专家张量在稠密 n_ff 宽） | `build_hunyuan_moe_forward` | HUNYUAN_MOE 臂 | 16/16 ×2 + long ×2, 0.0000 |
| dots1 | src/models/dots1.cpp:70-193（norm-先-rope；MHA；胖共享专家；exp_probs_b 可选） | `build_dots1_forward` | DOTS1 臂 | 16/16 ×2 + long ×2, 0.0000 |
| bailingmoe | src/models/bailingmoe.cpp:58-180（qkv 宽度取 n_rot；kq_scale=1/√n_rot；softmax 定死） | `build_bailingmoe_forward` | BAILINGMOE 臂 | 16/16 ×2 + long ×2, 0.0000 |
| bailingmoe2 | src/models/bailingmoe2.cpp:86-211（直接建融合 attn_qkv；gating 必需键；n_ff_shexp 覆盖共享宽） | `build_bailingmoe2_forward` | BAILINGMOE2 臂 | 16/16 ×2 + long ×2, 0.0000 |
| glm4-moe | src/models/glm4-moe.cpp:291-444（attn_post_norm 当 FFN norm；gating 缺省→SIGMOID；exp_probs_b 必需；355B 可选 q/k norm；mrope 分支未移） | `build_glm4_moe_forward` | GLM4_MOE 臂 | 16/16 ×2 + long ×2, 0.0000 |
| minimax-m2 | src/models/minimax-m2.cpp:43-168（全宽 q/k norm；部分 rope 64<128；SIGMOID+必需偏置；专家在稠密 n_ff） | `build_minimax_m2_forward` | MINIMAX_M2 臂 | 16/16 ×2 + long ×2, 0.0000 |
| cohere2moe | src/models/cohere2moe.cpp:145-289（iswa pattern 4 dense-first；仅 SWA+dense lead rope；FFN 吃 norm 后输入；(moe+shexp)×0.5；双残差；logit_scale；融合 gate_up_exps 优先） | `build_cohere2moe_forward` + `build_moe_ffn_silu_gu`（融合 gate_up 分支, llama-graph.cpp:2167-2190） | COHERE2MOE 臂（RMS/LN 二选一 norm 类型） | 16/16 ×2 + long（越窗 64）×2, 0.0010(FA 尾巴) |
| exaone-moe | src/models/exaone-moe.cpp:104-239（**仅 SWA 层 rope**（pattern 4 非 dense-first）；output.weight 必需；shexp 恒建） | `build_exaone_moe_forward` | EXAONE_MOE 臂 | 16/16 ×2 + long（越窗 64）×2, 0.0007(FA 尾巴) |

* 新增共享 helper：`build_moe_ffn_silu_gu`（融合 `ffn_gate_up_exps` 的一次 mul_mat_id + 双
  view 分支）。MTP 图类（glm4-moe/cohere2moe graph_mtp、bailingmoe2 nextn 块）张量按
  NOT_REQUIRED 移植、图未移（deepseek4 同款口径）。
* **CLI iswa 接线**（llama-cli main.rs）：DecodeContext 构造处补 llama-model.cpp:2687-2690
  的分裂判定（`swa_type != NONE && is_swa_any()` → `new_with_swa`，与 llama-server 同款，
  但排除自建 cache 的 `Deepseek32`/`Deepseek4` —— 否则 deepseek4 被路由进普通 iswa 对、
  绕过 `KvCache::new_dsv4`，批次 7 parity 实测翻车; **llama-server 的同款判定没有该排除,
  server 侧潜在错路由留给集成者**）。
* 未移（集成者条目）：**minimax-01**（lightning attention 需 ggml crate 的 `ggml_exp`
  GGML_UNARY_OP_EXP + `llm_graph_input_la` decay 输入族 + n_embd_head_la recurrent state）、
  **bailingmoe3**（KDA delta-net 递归层 + MLA + swiglu_clamp, 成本高）。详见 PARITY.md 批次8 §4。
* 回归：批次 1-7 parity 串行全对齐; qwen2.5 双 FA 锚点 16/16; `cargo test --workspace`
  **580/0**（基线 569 + 11 新默认测试）。

### 下一批队列（批次 9，按成本排序；C file:line = pinned bd4f514db1）

> 批次 8 已消掉 MoE 长尾队列的 hunyuan-moe/dots1/glm4-moe/bailingmoe(2)/minimax-m2/cohere2moe/exaone-moe。
> **批次 9 已消掉本表的线性注意力整行**（plamo3 / kimi-linear / qwen3next / delta-net-base +
> bailingmoe3 行 —— 零新算子, 见「架构批次 9」节; 最新队列在「下一批队列（批次 10）」）。

| arch | C 文件:行 | 需要什么（相对已有 helper） |
|---|---|---|
| **smallthinker** | src/models/smallthinker.cpp:66-189 | `build_moe_ffn` 的 RELU（`ggml_reglu_split`）+ `probs_in`（router 吃**原始 inpL**, 层内先算 logits）+ 可选 SWA 图模板 — helper 加 probs_in/RELU 两参数即可, 便宜 |
| **graniteswitch** | src/models/granite-switch.cpp（427） | granite 家族的 switch 变体, 既有 granite/nemotron-h 机制件组合 |
| **llada-moe** | src/models/llada-moe.cpp:54-163 | **非因果** diffusion attention（`build_attn_inp_no_cache`, hparams.causal_attn=false）— 需 no-cache 解码路径（bert EncoderContext 基建可挂）, 中等 |
| **minimax-m3 / minimax-01** | src/models/minimax-m3.cpp / minimax-01.cpp | 01: lightning attention（**先等 ggml crate 补 `ggml_exp`**, 集成者条目; 批次 9 盘点修正: EXP 已可用则照 minimax-01 的 `n_embd_head_la` recurrent-state 挂法落地）; m3 需读源定 |
| ~~bailingmoe3~~ | ✅ 批次 9 已落（融合 GDN K=1 + RecurrentState KDA 单元, 无需新算子） | — |
| ~~线性注意力~~ | ✅ 批次 9 已落 plamo3 / kimi-linear / qwen3next（delta-net 非默认 chunking 路径未移, PARITY.md 批次9 §4） | — |
| jais2 / eurobert / arcee / dream / wicked-fast 等小众 | 各自 models/*.cpp | 大多是既有 helper 的直接组合, 逐个按需 |
| 多模态（clip/mtmd 系新 projector）、稀疏注意力（glm-dsa）、RWKV 全家 | — | 需要算子或 cache 层新机制, 不在"下一批"档 |

### 下一批队列（批次 10，按成本排序；C file:line = pinned bd4f514db1）

> 批次 9 已消掉线性注意力家族（plamo3 / qwen3next / kimi-linear / bailingmoe3, 零新算子）。
> **批次 10 已消掉本表的 smallthinker / graniteswitch / llada-moe / minimax-01 四行**
> （ggml_exp + 三 unary 件全落, 见「架构批次 10」节; 最新队列在「下一批队列（批次 11）」）。

| arch | C 文件:行 | 需要什么（相对已有 helper） |
|---|---|---|
| ~~smallthinker~~ | ✅ 批次 10 已落（probs_in + ReGLU MoE, iswa 图模板） | — |
| ~~graniteswitch~~ | ✅ 批次 10 已落（in-graph adapter 路由 + switched LoRA + pad/round 算子） | — |
| ~~llada-moe~~ | ✅ 批次 10 已落（非因果 no-cache 图, in-port 验证 —— 参考侧 memory=nullptr 拒绝生成） | — |
| ~~minimax-01~~ | ✅ 批次 10 已落（lightning attention + ggml_exp + la 衰减输入 + n_embd_head_la 状态）; minimax-m3 仍待读源定 | — |
| **kimi-k3** | src/models/kimi-k3.cpp | KDA 家族 + rs-rollback（`llm_arch_supports_rs_rollback` 名单, llama-arch.cpp:1116-1132）—— K>1 快照路径（delta-net-base.cpp:546-606）, 需 gdn.rs 扩 K>1 |
| **qwen4exp / qwen35moe** | qwen4exp.cpp / qwen35moe.cpp | qwen35 机制件 + 混合 idx cache（`llama_memory_hybrid_idx`）, 中等 |
| **minimax-m3** | src/models/minimax-m3.cpp | 读源定（minimax-01 的 lightning 件已位同可用） |
| jais2 / eurobert / arcee / dream / hrm-text / laguna / maple / rnd1 等小众 | 各自 models/*.cpp | ~~全部已落~~（批次 11a/11b/12 —— hrm-text/laguna/maple 见「架构批次 12」节, **架构队列清空**） |
| 多模态（clip/mtmd 系新 projector）、glm-dsa、RWKV 全家、arwkv7 | — | 需要算子或 cache 层新机制, 不在"下一批"档 |

### 下一批队列（批次 11，按成本排序；C file:line = pinned bd4f514db1）

> 批次 10 已消掉 small-arch + EXP-op 整行（smallthinker / graniteswitch / llada-moe /
> minimax-01 —— `ggml_exp` GGML_UNARY_OP_EXP + pad/round/reglu 三件也随之入仓, 见
> PARITY.md「架构批次 10」节）。

| arch | C 文件:行 | 需要什么（相对已有 helper） |
|---|---|---|
| **kimi-k3** | ✅ 批次 11a 已落（见「架构批次 11a」节; K>1 rs-rollback 仍为集成者条目, 默认 `n_rs_seq=0` 语义） | — |
| **qwen4exp / qwen35moe** | ✅ 批次 11a 已落（见「架构批次 11a」节; qwen4exp 的 QSA 块压缩与 PLE 未移 —— 验证文件走合法稠密/无 PLE 配置; qwen35moe 的 graph_mtp 未移） | — |
| **minimax-m3** | ✅ 批次 11a 已落（见「架构批次 11a」节; MSA 块 top-k 稀疏注意 + idx cache —— fa=off 双提示词格全对齐, fa=on 为参考服务器/库自分歧格, 端口对参考库位精确, 见 PARITY.md 批次 11a §2; 多流 MSA ns>1 未移） | — |
| ~~minimax-m2/m3 之外的 MoE 长尾~~: ~~grovemoe/apertus/dots3note~~ | ✅ 批次 11a 已落（见「架构批次 11a」节; grovemoe 的双 MoE sigmoid 选择, apertus 的 xIELU, dots3note 的 DSA-over-isowa） | — |
| ~~jais2 / eurobert / arcee / dream / rnd1 / talkie / nanbeige~~ | ✅ 批次 11b 已落（见「架构批次 11b」节; dream/rnd1 是 llada 家族 —— 参考 memory=nullptr 拒绝生成, in-port 验证; eurobert 走 EncoderContext 与参考 llama_encode 位同） | ~~hrm-text / laguna / maple~~ ✅ 批次 12 已落（见「架构批次 12」节 —— **队列清空**, src/models/*.cpp 再无未移植图） |
| 多模态（clip/mtmd 系新 projector）、glm-dsa、RWKV 全家、arwkv7 | — | 需要算子或 cache 层新机制, 不在"下一批"档 |

## 架构批次 6 个（MLA 家族）: deepseek2 / deepseek2-lite / deepseek2-legacy / deepseek2-ocr / deepseek32（DSA lightning indexer）/ deepseek（非 MLA v2 base）— **合成 GGUF 全格对齐**（代理BATCH6, 2026-09-24）

架构计数 **+4**（`arch_tensors_support` 非 Unsupported 集 50 → 54；任务侧账面 59/156 → 63/156）。
协议沿用批次 1-5 + `-long` 长提示格；ForwardWeights/CLI 臂与图同批落地，parity 直接驱动
llama-cli 本体（`ARCH_BATCH6=1 ./parity/arch_batch_parity.sh`，`-long` 后缀 = ~100 token 格）。
合成文件在 `/tmp/arch-batch6/`（真实 MLA 比例缩小: kv_lora_rank 32 / qk_rope 16 / qk_nope 24 /
v_mla 20 / key_length 48）、测试 `crates/llama/tests/arch_batch6_e2e.rs`（默认 9 跑 + 1 `#[ignore]`）。

| arch | C 文件 | Rust builder | 加载器 | 合成对照（16 token, fa off/on） |
|---|---|---|---|---|
| deepseek2 | src/models/deepseek2.cpp:417-713（**默认 MLA 吸收路径** :540-592: wk_b 吸收、压缩 K-only cache、wv_b 解压; V3 式 MoE）+ **graph_mtp :170-415（MTP 图类, 见 PARITY.md 的 MTP 节）** | `build_deepseek2_forward` + `attn_k_cached_mla` + `build_deepseek2_mtp_forward` | DEEPSEEK2 臂（hparams 在 meta.rs 既有） | 16/16 ×2 + `-long` 16/16 ×2; MTP draft parity 16/16 ×2 |
| deepseek2-lite | deepseek2.cpp:109/:498（`q_lora_rank=0` → 直接 wq） | 同上 | 同臂 | 16/16 ×2 |
| deepseek2-legacy | deepseek2.cpp:593-634（无 mla 键 → 未拆分 wkv_b 解压 MHA + 尾部 rope `ggml_rope_set_offset`） | 同上 | 同臂 | 16/16 ×2 + `-long` 16/16 ×2 |
| deepseek2-ocr | deepseek2ocr.cpp:23-75 + deepseek2.cpp:473-494（MHA + NEOX rope 定死 10000） | 同上（`is_ocr`） | DEEPSEEK2OCR 臂（本批新增, meta.rs + model.rs） | 16/16 ×2 |
| deepseek32 | src/models/deepseek32.cpp:162-484（MLA 图 + **DSA lightning indexer**: NEOX rope 的 indexer q/k、Hadamard(k_rot)、`ggml_lightning_indexer` 融合打分、top-k 合成 KQ mask）+ **graph_mtp :495-725** | `build_deepseek32_forward` + `build_deepseek32_mtp_forward` | DEEPSEEK32 臂（hparams 既有） | 16/16 ×2 + `-long` 16/16 ×2; MTP draft parity 16/16 ×2 |
| deepseek（v2 base） | src/models/deepseek.cpp:74-194（非 MLA: build_qkv GQA + dense-lead MoE） | `build_deepseek_forward` + `build_qkv_deepseek` | DEEPSEEK 臂（本批新增） | 16/16 ×2 |

* 本批新机制件（可复用）: ggml 的 `rope_set_offset`（ggml.c:4528; 内核 n_offs 支持早已有）、
  `fill`（ggml.c:5407）、`lightning_indexer`（ggml.c:6423 + ops.cpp:12124）、
  **`top_k` = libstdc++ partial_sort 堆选择的逐算子复刻**（ops.cpp:8550-8604; indexer 分数
  ReLU 并列 0.0 时 top-k 的**并列集合**决定 DSA mask 放行哪些行, argsort 的不稳定并列序不够,
  oracle 测试钉死）; kv_cache.rs 的第三种 cache 形态 `KvLidCache`/`new_dsa`
  （llama-kv-cache-dsa.cpp:14-54 的 MLA K-only + lid 索引键双 cache）+ `gen_hadamard`
  （llama-kv-cache.cpp:23-53）+ `DecodeContext` 的 lid step 输入（F16 mask/k_rot/row_idx）。
* deepseek4 + dsv4 未做（批次 7 队列第一项, 清单在 PARITY.md 批次6 §4）; MTP 图类（两个 arch 的
  graph_mtp）与 mistral-large 温度缩放未移（assert 拒绝）。
* 回归: 批次 1-5 parity 全对齐（批次 4 曾一格假错 = 并行 parity 串了 8790 端口, 串行重跑全绿）;
  qwen2.5 双 FA 锚点 16/16; `cargo test --workspace` 通过。

## 架构批次 5 个: mamba / mamba2 / jamba / nemotron-h — **mamba 家族（recurrent-state 架构）, 合成 GGUF 20/20 格全对齐**（代理BATCH5, 2026-09-24）

架构计数 **+4**（`arch_tensors_support` 非 Unsupported 集 46 → 50；任务侧账面 55/156 → 59/156）。
协议沿用批次 1-4，两处加强：**N=48**（递归越过 prompt 充分展开）+ `-long` 长提示格（~100 token，
覆盖 conv-state 滑窗 T>1 写回）。合成文件在 `/tmp/arch-batch5/`、测试
`crates/llama/tests/arch_batch5_e2e.rs`（默认 8 跑 + 2 `#[ignore]`：造文件、env 驱动 parity driver
`arch5_cli_driver`）、对照驱动 `ARCH_BATCH5=1 ./parity/arch_batch_parity.sh`（`ARCH_BATCH5_CLI=1`
+ 前者 = llama-cli 本体）。CLI/DecodeContext 接线同步落地（`ForwardWeights::{Mamba,Jamba,
NemotronH}` + llama-cli 4 arch 臂）。`recurrent-gemma` 在本 pinned revision 不存在（models/ 无此文件）。

| arch | C 文件 | Rust builder | 加载器 | 合成对照（48 token + -long 格） |
|---|---|---|---|---|
| mamba | src/models/mamba.cpp:35-112（mamba1: in_proj 切 x/z、conv1d(+bias,silu)、x_proj dt\|B\|C、dt_proj、逐通道 A、`silu(z)·y` 门控） | `build_mamba_forward` + `build_mamba1_mixer` | MAMBA 臂 + `load_arch_hparams_batch`（ssm 四元组, 可选 dt_b_c_rms） | 47/47 ×2（参考少报 1 byte token 既有现象）+ long 48/48 ×2，Δlogprob ≤1e-4 |
| mamba2 | src/models/mamba2.cpp:35-90（d_in_proj=d_inner+conv_dim+dt_rank、{1,n_head} A/D、分组 B/C、ssm_norm、conv bias 必需; 图复用 mamba 的 models.h:942） | 同 builder（`MambaLayerMixer::Mamba2`→`build_mamba2_mixer`，granite 的 `build_mamba2_layer` 改为委托同一函数体） | MAMBA2 臂（五元组） | 48/48 ×2 + long 48/48 ×2，≤1e-4 |
| jamba | src/models/jamba.cpp:32-198（hybrid mamba1+**无 rope** attention; 逐层 dense\|MoE FFN; mamba 层 dt/B/C RMS 三元组必需） | `build_jamba_forward` | JAMBA 臂（is_recr=n_head_kv==0） | 48/48 ×2，≤1e-4 |
| jamba-moe | jamba.cpp:107-128（逐层 ffn_gate_inp 存在性分支） | 同上（`build_moe_ffn_silu` 复用） | 同臂 | 48/48 ×2，≤8e-4 |
| nemotron-h | src/models/nemotron-h.cpp:49-340（三态层 is_recr=n_head_kv==0&&n_ff==0; 无 rope attention + f_attention_scale; relu² dense\|MoE(sigmoid/路由偏置/latent/共享专家); MTP 头加载 1:1） | `build_nemotron_h_forward` + `build_ffn_relu_sqr`(clamp+mul) + `build_moe_ffn_relu_sqr` | NEMOTRON_H 臂（LN eps 必需/RMS 回退 + MoE 键） | 48/48 ×2 + long 48/48 ×2（on 1.1e-3），≤1e-4 |
| nemotron-h-moe | nemotron-h.cpp:111-130/291-333（latent 投影路径） | 同上 | 同臂（moe_latent_size） | 48/48 ×2，≤2e-4 |

* recurrent 驱动 = granite-hybrid 的 `RecurrentState` 泛化（`recurrent_dims()` 三个新臂;
  mamba1 不读 group_count ⇒ n_embd_r=(d_conv-1)*d_inner，与 `llama_hparams` 同式）。
  batch-vs-step 状态一致性测试 `mamba_state_vs_full_recompute`（6 spec 全跑）。
* RELU_SQR 无独立算子的组合实现与 granite `granite_state_vs_full_recompute` 既有失败的归因
  （原始函数体逐位复现 ⇒ 与本批无关，PARITY.md 既有开档）见 PARITY.md 批次5 §4/§5。
* 回归: 批次 1/2/3/4 parity 全对齐; qwen2.5 双 FA 锚点 16/16; granite 参考对照锚点通过
  （文档带内）; `cargo test --workspace` **522/0**。


## 架构批次 4 个: qwen2moe / qwen3moe / granite-moe / phimoe / arctic / olmoe / ernie4-5-moe + smollm3 / seed-oss / openelm — **MoE 家族, 合成 GGUF 22/22 格全对齐**（代理BATCH4, 2026-09-28）

架构计数 **35/156 → 45/156**。协议沿用批次 1-3；合成文件在 `/tmp/arch-batch4/`、测试
`crates/llama/tests/arch_batch4_e2e.rs`（默认 11 跑 + 2 `#[ignore]`：造文件、env 驱动的
parity driver）、对照驱动 `ARCH_BATCH4=1 ./parity/arch_batch_parity.sh`（端口侧
`arch4_cli_driver` 打印 llama-cli 同格式 debug 行，`arch_batch_cmp.py` 零改动复用；脚本
把批次 3/4 的公共格逻辑合并成 `run_env_driver_cell`）。每个 MoE 文件 n_expert=4 /
n_expert_used=2, 路由真实 exercised。

| arch | C 文件 | Rust builder | 加载器 | 合成对照 |
|---|---|---|---|---|
| qwen2moe | src/models/qwen2moe.cpp:3-194（norm_w=false + sigmoid 门控 shexp :147-165） | `build_qwen2moe_forward` | QWEN2MOE 臂（+`ffn_gate_inp_shexp`） | 16/16（off/on），Δlogprob 1e-4 |
| qwen3moe | src/models/qwen3moe.cpp:3-179（per-head q/k norm + norm_w=true, tie 头） | `build_qwen3moe_forward` | QWEN3MOE 臂 | 16/16，0 |
| granite-moe | src/models/granite-moe.cpp（graph = granite 的, models.h:1680; 专家=稠密 n_ff） | 复用 `build_granite_forward` + `GraniteParams::dense` | GRANITE_MOE 臂 | 16/16，0 |
| phimoe | src/models/phimoe.cpp + phi3.cpp:67-192（graph = phi3 的, models.h:661; 偏置 RMS 四对） | `build_phimoe_forward`（+`build_norm_rms_b`） | PHIMOE 臂 | 16/16，2e-4 |
| arctic | src/models/arctic.cpp:3-180（双 FFN: 稠密方阵 + inpSA 上的 MoE :136-139） | `build_arctic_forward` | ARCTIC 臂（+`ffn_norm_exps`） | 16/16，1e-4 |
| olmoe | src/models/olmoe.cpp:3-173（全宽 q/k norm, reshape=false, MHA） | `build_olmoe_forward` | OLMOE 臂 | 16/16，3e-4 |
| ernie4-5-moe | src/models/ernie4-5-moe.cpp:3-133（dense-lead+step 切层, exp_probs_b, 可选 shexp; hparams 在 meta.rs ERNIE4_5_MOE 臂） | `build_ernie45_moe_forward` | ERNIE4_5_MOE 臂（ernie4-5.cpp 的 arch 分支） | 16/16 + `-nosh` 变体格 16/16，3e-4 |
| smollm3 | src/models/smollm3.cpp:3-152（nope: 每 4 层第 4 层不 rope） | `build_smollm3_forward` | SMOLLM3 臂 | 16/16，4e-4 |
| seed-oss | src/models/seed-oss.cpp:3-151（attn_post_norm 即 FFN norm） | `build_seed_oss_forward` | SEED_OSS 臂 | 16/16，0 |
| openelm | src/models/openelm.cpp:3-171（逐层头数/FFN 宽数组 + 融合 qkv 头段切分 + per-head q/k norm） | `build_openelm_forward` | OPENELM 臂（per-layer KV 宽度 `KvCache::new_with_dims` 先例） | 16/16，1.3e-3（fa on） |

* CLI/DecodeContext 未接（context.rs 归并行代理; 10 arch 由 `arch_batch4_e2e.rs::Batch4Model`
  驱动，集成者接线时注意 openelm 的 per-layer KV 宽度与 olmoe 的 MHA 约束——[n_embd] 的
  k-norm 只在 n_head_kv==n_head 时对齐，GQA 文件在参考里也会 abort）。
* 回归: 批次 1/2/3 parity 全对齐; qwen2.5 双 FA 锚点 16/16; `cargo test --workspace` 503/0。
* 缺口: phimoe 真实文件的 rope_factors long/short 选择未对照（合成两侧都不带）; nemotron
  因 RELU_SQR 算子（ggml crate）缺位移入批次 5。

## 架构批次 2 个: codeshell / orion / olmo / xverse / internlm2 / exaone / gemma(v1) / falcon（代理BATCH2, 2026-09-25）

合成 GGUF 对照 **16/16 格全部对齐**（每格全新参考 server + 首请求，`-fa on`/`-fa off` 各 16 token；
falcon 的 15/15 + 1 个参考少报的 byte token 见 PARITY.md §4）。架构计数 **18/156 → 26/156**
（第三批 2026-09-27 再到 35/156，见下节）。

| arch | C 文件 | Rust builder | 加载器 | CLI 臂 | 合成对照 |
|---|---|---|---|---|---|
| codeshell | src/models/codeshell.cpp:12-153 | `graph_arch::build_codeshell_forward` | `model.rs` CODESHELL 臂 + `load_arch_hparams_batch` | `llama-cli` CODESHELL 臂（`codeshell_weights`） | 16/16（off/on），Δlogprob 1e-4 |
| orion | src/models/orion.cpp:12-141 | `build_orion_forward` | ORION 臂 | `OrionParams` | 16/16，1e-4 |
| olmo | src/models/olmo.cpp:15-142 | `build_olmo_forward` | OLMO 臂（`attention.clamp_kqv` 可选） | `OlmoParams`（`f_clamp_kqv`） | 16/16，0 |
| xverse | src/models/xverse.cpp:14-136 | `build_xverse_forward` | XVERSE 臂 | `XverseParams` | 16/16，1e-4 |
| internlm2 | src/models/internlm2.cpp:13-139 | `build_internlm2_forward` | INTERNLM2 臂（output 必填） | `Internlm2Params` | 16/16，1e-4 |
| exaone | src/models/exaone.cpp:12-136 | `build_exaone_forward` | EXAONE 臂（`rope_freqs` 模型级名字） | `ExaoneParams` | 16/16，1e-4 |
| gemma(v1) | src/models/gemma.cpp:13-139 | `build_gemma1_forward` | GEMMA 臂（hparams 在 meta.rs 已是 Full） | `Gemma1Params`（attention_scale） | 16/16，0 |
| falcon | src/models/falcon.cpp:13-161 | `build_falcon_forward` | FALCON 臂（`attn_norm_2` 可选对） | `FalconParams` | 15/15 对齐（参考少报 1 个 byte token，PARITY.md §4） |

* 本批**未改任何共享 helper**：全部落在既有 `qkv_separate_parts`/`qkv_fused`/`rope_qk`/`build_norm`/`build_norm_rms`/
  `build_ffn_silu_par`/`build_ffn_gelu_seq`/`build_ffn_gelu_par`/`attn_kv_cached` 上；新增的共享件只有
  `context.rs::ForwardWeights` 的 8 个变体与 `tok_embd()/set_tok_embd()` 两个穷尽访问器（后者用于 mtmd.rs 两处
  match 的 2 行兼容通配臂，见 PARITY.md §7）。
* `model.rs`：`LayerTensors` 新增 `attn_norm_2`/`attn_norm_2_b`；`load_arch_tensors` 新增 8 个臂；
  `arch_tensors_support` 8 个 arch 标 `Partial`（已 CLI 可达，见 `context.rs::ForwardWeights::*`）。
* 测试：`crates/llama/tests/arch_batch2_e2e.rs`（12 个：默认 10 跑——每 arch 一个 loader/张量集/hparams/前向钉
  + 变体（codeshell 反向 tie、olmo head+clamp、exaone norope、falcon 40B/tie）+ 元数据 round-trip + writer 类型检查；
  2 个 `#[ignore]`——造文件、falcon context probe）。
* 合成文件生成器就在该测试里（`build_file`/`tensors_for`），`cargo test --release -p llama --test arch_batch2_e2e -- --ignored arch_batch2_write_synth`
  重建到 `/tmp/arch-batch2/`；对照驱动 `ARCH_BATCH2=1 ./parity/arch_batch_parity.sh` + `parity/arch_batch_cmp.py`
  （后者新增参考 byte-token 少报的子序列对齐）。
* falcon 归因工具：`parity/falcon_probe.sh` + `parity/falcon_probe_cmp.py` + 测试里的 `falcon_teacher_force_probe`。

## 投机解码接线 + 批次 3 CLI/server 分发（集成者INTEG, 2026-09-28）

* `common/arg.cpp:4135-4253, 4382-4395` 的 `--spec-*`/`-md`/已移除旗标 → `llama-cli/src/main.rs` 与
  `llama-server/src/main.rs` 两处同名 parse 臂（映射表见 PARITY.md 同名节 §1）; ngram 子结构旗标
  （:4255-4376, :4719）与 draft 设备/线程/HF 旗标（:3971-4235）未接。
* `common_speculative_init_from_params`（speculative.cpp:2523-2604）→ CLI `run_speculative` / server
  `load_engine`（`has_dft()` 才装 draft, `ctx_dft.n_ctx = llama_n_ctx(ctx_tgt)`）;
  `examples/speculative-simple/speculative-simple.cpp:17-377` 的循环 → CLI 直用
  `speculative_simple_generate`; `tools/server/server-context.cpp:1109-1300, 2995-3075, 3742-3757,
  3897-4017` → `llama-server/src/engine.rs`（spec_draft/spec_i_batch 槽位状态机 + 计数/日志）。
* `common/sampling.cpp:678-715` 归属钩子 → `llama/src/sampling.rs::SamplingContext::sample_and_accept_n`（委托）;
  `llama-context.cpp:1046-1048`/`common.cpp:1583-1620` → `DecodeContext::{n_ctx, can_seq_rm}`。
* 批次 3 分发（PARITY.md 批次 3 节 §5 的集成清单落地）: `context.rs::ForwardWeights` 新增
  `Baichuan/Bloom/Mpt/Starcoder/Refact/Plamo/Stablelm` 7 变体（granite-dense/minicpm 复用 `Granite`）+
  `max_alibi_bias()`; `fill_mask`/`fill_mask_seq` 的 `alibi` 参接 `graph::fill_kq_mask_alibi{,_f16}`;
  `llama-cli` 的 9 个 arch 臂 + 权重组装 helper（`baichuan_weights`…`granite_dense_weights`）。

## 批次 4 CLI 接线: `ForwardWeights` 9 变体 + llama-cli 9 arch 臂（集成者INTEG, 2026-09-28）

* `context.rs`: `ForwardWeights::{Qwen2Moe,Qwen3Moe,Phimoe,Arctic,Olmoe,Ernie45Moe,Smollm3,
  SeedOss,Openelm}` + `forward()`/`n_layer()`/`output()`/`tok_embd()`/`set_tok_embd()` 臂;
  `kv_dims()` 增 Openelm 逐层 KV 宽度臂（`KvCache::new_with_dims` 路径, gemma4/qwen35 同款）。
  granite-moe 复用既有 `Granite` 变体。
* `llama-cli/src/main.rs`: `forward_weights` 9 个新 arch 臂（granite-moe 并入 `GRANITE|MINICPM`
  臂走 `granite_dense_weights` + `GraniteParams::dense`）+ `qwen2moe_weights`…`openelm_weights`
  9 个组装 helper（逐字段 = `arch_batch4_e2e.rs` 的同名函数）; olmoe 臂带 MHA 守卫
  （`n_head_kv != n_head` 显式报错, olmoe.cpp:29）; phimoe 的 rope factors 按参考
  `get_rope_factors`（llama-model.cpp:2259-2272）解析, `forward_weights` 为此新增 `n_ctx` 参。
* `parity/arch_batch_parity.sh`: 批次 4 增 `ARCH_BATCH4_CLI=1` 模式——llama-cli 本体经
  `run_cli_arch_parity.sh` 对照（批次 1 协议）; 默认仍走 `arch4_cli_driver`。
* 对照: CLI 本体 **22/22**（11 arch × fa off/on, 逐 token 16/16）; env 驱动复验 22/22;
  批次 1/2/3 与 qwen2.5 双 FA 锚点不回归; `cargo test --workspace` **507/0**。

## 预填收口: FA/elementwise AVX512 内核（代理SIMD 第 2 轮, 2026-09-24）
| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| vec.h:1172-1208 `ggml_v_expf/ggml_v_silu(__m512)` + vec.cpp:380-413 `ggml_vec_silu_f32` + vec.cpp:531-597 `ggml_vec_soft_max_f32` + vec.h:703-766 `ggml_vec_scale_f32` + vec.h:89-99/108/128 `ggml_vec_add/acc/mul_f32`（`__AVX512F__ && __AVX512DQ__` 分支） | **crates/ggml/src/simd_x86.rs** AVX512 节（`v_expf_16`/`v_silu_16`/`vec_silu_f32`/`vec_soft_max_f32{,_inplace}`/`vec_scale_f32`/`vec_mul1_f32`/`vec_add_f32`/`vec_acc_f32`/`vec_mul_f32`） | ✅ | `simd_x86::tests::avx512_*` 4 个测试（逐位 == 标量 lane 移植，含指数 -140..140 扫描与下溢区）；无 AVX2 中间档（vec.h:1215 是另一多项式，不位同 AVX512 参考）→ 非 AVX512 主机回落标量 |
| simd-gemm.h:24-131（ukernel RM=RN=4/KN=16 + 驱动 + jj 标量尾）+ :134-145 标量 `#else` | `simd_x86::{gemm_ukernel, simd_gemm_avx512_impl}` + `flash_attn::{simd_gemm 派发, simd_gemm_scalar}`（tiled 的 KQ 与 VKQ 两处调用） | ✅ | `avx512_gemm_matches_scalar_bit_exact`（含 3×5×7 / 非整块 M,N 尾）；每元素独立 kk 升序 fma 链 ⇒ 位同按构造；`ref_dump_bitexact` 11/11 tiled 用例对参考逐位不变 |
| ops.cpp:8852-9140 `..._flash_attn_ext_tiled` 的逐 lane 循环（scale/add/softmax/1/S/`ggml_cpu_fp16_to_fp32` V 打包 ops.cpp:9088/`memcpy` 存储行 :8815） | `flash_attn::tiled_impl`（`tiled` = AVX512 派发；`use_simd=false` 强制标量）+ `simd_x86::{fp16_to_f32_row}` + `Dst::write_row` 字节拷贝 | ✅ | `tiled_simd_matches_scalar_bit_exact`（5 形状 × F16/F32 KV 逐位）；dump 11/11 不变；pp64 82→2.1 ms |
| binary-ops.cpp `vec_binary_op_contiguous`（op_add/op_mul/op_div）F32×F32→F32 热路径 + `ne10==1` 广播行 | `compute.rs::forward_bin_op`（BinOp 枚举 + 切片快路径） | ✅ | `compute::tests::mul_broadcast_dim0`（回归：广播乘曾把未初始化 dst 当源）；`add_mul_broadcast_and_inplace` 既有 |
| compute.rs 的 silu/soft_max/norm 尾、flash_attn 的 KQ 定标与 VKQ 重缩放调用点 | 同上调 `simd_x86` 内核 | ✅ | 锚点：qwen fa on/off 16/16、gpt-oss 16/16（\|Δ\|0.158）、gemma4-12B SWA 16/16；`cargo test --workspace` **514/0**（基线 507 + 新 7） |
| （临时测量设施 `LLAMA_RUST_PROF` per-op 采样器, 前节引入） | 已移除（任务要求测量后清理） | — | 前后 per-op 表见 PARITY.md §预填收口 |

## 增量（2026-09-30）: tools/server 的 chat tools 接线（上表 tools/server 行的更新, append-only）

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| server-common.cpp:1151-1420 `oaicompat_chat_params_parse`（jinja 路径）+ server-context.cpp:1448-1455 模板装载 + server-schema.cpp:262-380（grammar 顺序/grammar_type/preserved/触发器升级） | `llama-server/src/{chat,main,api}.rs`（`oaicompat_chat_params_parse` 重写为 `chat_templates_apply` 链; `ChatTemplatesInit` 于 `load_engine`; `eval_llama_cmpl_schema` 带 vocab 的 grammar/triggers/preserved handler） | ✅ | `parity/run_server_parity_chat.sh` **29/29 MATCH**（tools 非流式×2 + required + refusal + 流式 tool_calls 增量 18 帧） |
| server-task.{h,cpp} `task_result_state`/`update_chat_msg`/`server_chat_msg_diff_to_json_oaicompat`（server-chat.cpp:621-649）+ server-task.cpp:414-525 三个 to_json | `llama-server/src/engine.rs`（`ChatStreamState` + `chat_msg_diff_to_json` + `send_partial`/`send_final` OaiChat 臂） | ✅ | 同上（`finish_reason:"tool_calls"`/`"stop"` 两态; message.tool_calls 数组） |
| llama-grammar.cpp:1398-1455 `accept_impl` awaiting 支路 + sampling.cpp:594-676 lazy 采样 + :222-268 触发器装配 | `llama-server/src/engine.rs`（`LazyTriggers`/`sample_with_lazy_grammar`/`grammar_accept_lazy`, WORD 字节最早匹配 + 跨界 token 尾段重放） | ✅ | 同上（qwen2.5 `<tool_call>\n` WORD 触发, 触发前自由采样） |
| server-context.cpp:3800-3802 `accept_special_token` | `llama-server/src/engine.rs process_token` | ✅ | preserved 标记特殊 token 不再丢文本 |

## 增量（2026-09-26）: 多 token 预填数值分歧闭合（rope FMA + get_n_kv 256 填充）

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| ops.cpp:6063-6079 `rotate_pairs`（GCC `-ffp-contract=fast` 把 :6076-6077 收缩成 vmulss+vfmsub/vfmadd, `.so` 反汇编 + dump 全量 8192/8192 元素实证） | `crates/ggml/src/compute.rs rotate_pairs_f32`（`x0.mul_add(cos, -(x1*sin))` / `x0.mul_add(sin, x1*cos)`） | ✅ | qwen3 Q-rope 0/8192 失配（plain 1584）; gpt-oss 同向; 节点 dump 位同 |
| llama-kv-cache.cpp:1250-1263 `get_n_kv`（`min(size, max(256, PAD(used,256)))`） | `crates/llama/src/kv_cache.rs n_kv/n_kv_swa/n_kv_lid`（+诊断用 `used_cells`） | ✅ | kq/softmax/kqv 的形状与 GEMM 路由随参考（softmax 全 lane 走 v_expf 多项式; KQ·V k=256 进 tinyBLAS）; 节点 dump 0 分歧 |
| ggml-backend.cpp:1798-1835 回调协议（`cb_eval`, 每节点 ask/data 两段） | `crates/ggml/src/compute.rs EvalNode`（补节点自身 `data`/`nb`, 即 `t->data/t->nb`） | ✅ | imatrix 用途不回归; 新 dump 测试消费 |
| （工具）`parity/ref_decode_dump.c` + `parity/decode_dump_cmp.py` + `crates/llama/tests/qwen3_prefill_dump.rs`（qwen3/gpt-oss 两 dump） | 参考侧逐节点流 dump ↔ 端口 eval-callback 镜像, DECDMP1 同格式 | ✅ | qwen3 FA off 336 节点 / FA on 280 节点 **0 分歧**; gpt-oss 剩余分歧定位到 FA 节点 316/20480 元素（1-2 ulp, ≥4 lane 行） |
| （口径）测试 harness `n_kv_prev+n` 模式（arch_e2e/batch3/4/5/gpt_oss/phi3/qwen3/graph_arch 内置） | 改为 step_inputs 顺序: 先 `assign` 再取填充 `n_kv`; mask 用 `cells[..n_kv].pos` | ✅ | batch3 baichuan alibi A/B 恢复; `cargo test --workspace` 564/0 |
| （口径）`phi3_diff.rs` rope 位检查 / `speculative_e2e.rs` greedy 门限 / `gpt_oss_e2e.rs` 门限与头注 | 与实现同步（收缩公式; 近平局判据; FA f16 尾巴带 13/16 sets + 0.235, 见 PARITY.md §多 token 预填） | ✅ | gpt-oss teacher-forced 16/16 ref-in-top5; Q4_K_M greedy 16/16 |

## 增量（2026-09-26）: 架构批次 7 — deepseek4 + 压缩 DSV4 KV cache（上表 deepseek4/dsv4 行的落地, append-only）

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| src/models/deepseek4.cpp（1502 行: hyper-connection 流、三比压缩注意力、hash 层、MTP）+ src/llama-kv-cache-dsv4.cpp（2253 行） | `crates/llama/src/graph_arch.rs`（`build_deepseek4_forward` 一族 + `build_deepseek4_mtp_forward`）/ `kv_cache.rs`（`Dsv4Cache`/`Dsv4CompState`/`Dsv4CompCache`/`Dsv4Plan`/`new_dsv4` + `apply_ubatch_lid`）/ `context.rs`（`dsv4_step_inputs` + `new_mtp`/`MtpForward`/`embeddings_nextn` 抽头）/ `meta.rs`+`model.rs`（DEEPSEEK4 臂, TENSOR_ALLOW_RESHAPE）/ llama-cli | ✅ 合成 parity 16/16 ×2FA + `-long` 16/16 ×2; MTP 三连（draft-mtp）11/12 格 16/16（deepseek4 fa=on 15/16 = 0.0147-nat FA 尾巴翻面） | `ARCH_BATCH7=1 ./parity/arch_batch_parity.sh`; `tests/arch_batch7_e2e.rs`; `parity/mtp_parity.sh` + `tests/mtp_e2e.rs`; 未移: 多序列压缩流/rs 回滚/序列化（清单 PARITY.md 批次7 §3-4 + MTP 节 §5） |
| ggml.c:6457-6607 `ggml_dsv4_hc_{comb,pre,post}` + ggml.h:641 SWIGLU_CLAMP + GGML_OP_SQRT + `ggml_rope_ext_back`（ggml.c:4485） | `crates/ggml/src/ops.rs::{dsv4_hc_comb,dsv4_hc_pre,dsv4_hc_post,swiglu_clamp,sqrt,rope_ext_back}` + `compute.rs` 四内核（hc comb 的 affine/+eps 步骤 mul_add 收缩对齐 gcc -O2; pre/post 的累加**不**收缩 —— 探针实证） | ✅ 逐位 | `compute::dsv4_ops_match_reference_dump` ↔ `parity/ref_dsv4_dump.c`→`parity/dsv4_ops_ref.bin`（12 段全位同） |
| ggml-cpu.c:1262-1266 HINT_SRC0_IS_HADAMARD → ops.cpp:12027-12101 快速 Walsh-Hadamard 变换（非 gemm!）; llama-impl.h:57-73 `llama_mul_mat_hadamard` | `ops.rs::mul_mat_set_hint_hadamard` + `compute.rs::forward_fwht`（蝶形 u±v; 16-lane SIMD 与标量逐元素等价） | ✅ 逐位 | `parity/ref_hadamard_mm_probe.c`（参考图 1067/1536 元素异于其自身 gemm ⇒ 走 fwht）; 修复后节点 dump 0 差 |
| （修）ggml-cpu.c:1519 mul_mat_id one_chunk 的 `ggml_vec_dot_f32` | `vec_dot.rs::vec_dot_row` F32 臂 → `vec_dot_f32_c`（原 f64 标量累加 ulp 漂移, deepseek4 F32 专家第一 token 即翻 argmax） | ✅ 逐位 | `parity/ref_mmid_probe.c` 隔离重放 0/288; 节点 dump logits 0/192000 |
| （工具）`parity/ref_dsv4_nodes.c`（ref_decode_dump.c 的 dsv4 孪生: n_batch 按提示定长 —— 参考的 reserve 图复用在 dsv4 计划尺寸输入上会触发 ggml-backend.cpp:283 越界断言; embeddings 路径同因断言, 走 DSV4_NOEMB） | 节点 dump bisection（DECDMP1 协议） | ✅ | `arch_batch7_e2e.rs::dsv4_prefill_node_dump`（env `DSV4_DUMP_OUT`/`DSV4_FA_OFF`） |

架构计数 **61 → 62**（`arch_tensors_support`）; 任务账面 **70 → 71/156**。

## 架构批次 9 个（线性注意力家族）: plamo3 / qwen3next / kimi-linear / bailingmoe3 — **合成 GGUF 12/12 格全对齐**（代理BATCH9, 2026-10-01, append-only）

架构计数（`arch_tensors_support`）**70 → 74**；任务账面 **79 → 83/156**。
**零新 ggml 算子** —— 参考 CPU 构建默认启用融合 GDN（`fused_gdn_ar/ch` 默认 true,
llama-context.cpp:233-234）且 `n_rs_seq` 默认 0，KDA/GDN 层的默认路径就是批次 6 位同的
`GGML_OP_GATED_DELTA_NET`（K=1, gdn.rs 的 kda 臂 = g->ne[0]==S_v 逐键衰减）；plamo3 在本修订版
是纯 SWA 注意力（无 ssm 键）。协议沿用批次 1-8 + N=48 token/格 + `-long` 长提示格；
ForwardWeights/CLI 臂与图同批落地，parity 直接驱动 llama-cli（`ARCH_BATCH9=1
./parity/arch_batch_parity.sh`）。合成文件在 `/tmp/arch-batch9/`（6 个：4 主格 +
qwen3next-legacy / kimi-linear-legacy 变体）、测试 `crates/llama/tests/arch_batch9_e2e.rs`
（默认 4 跑 + 1 `#[ignore]` 造文件）。全细节（算子映射表 / 逐 arch C 行号 / 状态-cache 映射 /
未移清单 / 排障 4 条）见 PARITY.md「架构批次 9」节。

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| src/models/plamo3.cpp（195 行: iswa 8 图模板、SWA 层 *_swa 频率、融合 qkv 三 view（head_q≠head_v）、q/k norm 先 rope、双后 norm、swiglu-SEQ FFN；post norm 张量**无 .weight 后缀** :50/:53） | `graph_arch.rs::{build_plamo3_forward, Plamo3Params}` / `meta.rs`+`model.rs` PLAMO3 臂 / `context.rs` Plamo3（kv_dims 默认 + shift 的 *_swa 频率交换）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2（越 SWA 窗 64; worst 0.0009 = FA f16 尾巴） | `ARCH_BATCH9=1 ./parity/arch_batch_parity.sh plamo3` |
| src/models/qwen3next.cpp（822 行: GDN 层（packed ssm_ba、conv 状态、H_k≠H_v 交织 repeat、融合 GDN、silu(z) 门控 norm）、门控全注意力（Q 投影 2 倍宽带 per-head gate）、MoE + sigmoid 门控共享专家（ffn_gate_inp_shexp）、融合 gate_up_exps 优先；expert_weights_scale 从不读） | `graph_arch.rs::{build_qwen3next_forward + attn/gdn/qkvz/ffn 子构建}` / QWEN3NEXT 臂（nextn NOT_REQUIRED）/ `context.rs` Qwen3Next（recurrent_dims + 逐层 kv_dims + shift）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2；`-legacy`（融合 ssm_in + 分离专家）in-port 双 FA | `ARCH_BATCH9=1 ./parity/arch_batch_parity.sh qwen3next` |
| src/models/kimi-linear.cpp（562 行: KDA 层（融合 attn_qkv 拆分、逐流 conv1d 状态单元（conv_step+channel·(d_conv-1) 布局）、两级 softplus 衰减 f_a/f_b、sigmoid o_norm 门）、**无 rope** MLA 层（wk_b 吸收 + [kv_lora\|rope] 压缩 K-only cache, 或 legacy wkv_b MHA）、MoE norm+probs_b+scale） | `graph_arch.rs::{build_kimi_linear_forward, kimi_causal_conv1d}` / KIMI_LINEAR 臂 / `context.rs` KimiLinear（KDA 分支的 n_embd_r=3·(d_conv-1)·n_head·head_dim / n_embd_s=head²·n_head 单元, rope NONE ⇒ 无 K-shift）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2；`-legacy`（wkv_b MHA 缓存, head_count_kv=n_head）in-port 双 FA | `ARCH_BATCH9=1 ./parity/arch_batch_parity.sh kimi-linear` |
| src/models/bailingmoe3.cpp（540 行: KDA 安全门 gate=lower_bound·sigmoid(a·(f_a·x+dt))、分离 wq/wk/wv、MLA（q_lora 压缩 + 尾部 NORM rope + sigmoid 输出门）、swiglu_clamp_exp/shexp 双键走通用 clamp 分支（llama-graph.cpp:1822-1846/:2210-2240））+ delta-net-base.cpp 的 build_conv_state / build_recurrent_attn（n_rs_seq==0 臂） | `graph_arch.rs::{build_bailingmoe3_forward, bailingmoe3_causal_conv1d, silu_par_clamped, build_bailingmoe3_moe}` / BAILINGMOE3 臂 / `context.rs` BailingMoe3（KDA 单元同 kimi + shift 的 MLA 尾部 rope nope 偏移）/ `attn_k_cached_mla` 的 wo 参数改 `Option`（bailingmoe3 传 None, bailingmoe3.cpp:343-350）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2 | `ARCH_BATCH9=1 ./parity/arch_batch_parity.sh bailingmoe3` |

未移（集成者条目）：delta-net 非默认 chunking/自回归路径（`-fgdn 0` 才可达, 需 CUMSUM/TRI/
SOLVE_TRI/DIAG/SET_INPLACE/PAD 六算子）、K>1 回滚快照（`n_rs_seq>0`）、qwen3next/bailingmoe3 的
`graph_mtp`（trunk-only 文件加载已就绪）、llama-server 的 forward_weights 未接本批四 arch。
参考侧已知问题：kimi-linear 的 legacy wkv_b 分支不可达（is_mla 强制 true ⇒ :482 解引用空
inp_attn_kv, 加载即 SIGSEGV —— 端口字面移植 + in-port 验证, PARITY.md 批次9 §6.5）。
qwen3next-legacy 显式格 48/48 ×2FA（0.0000）。
回归：`cargo test --workspace` **593/0**（基线 590 + 本批 4 默认测试）；批次 1-8 parity 串行
复跑全绿；qwen2.5 双 FA 锚点 16/16 不变。

## 架构批次 10 个（small-arch + EXP-op 批）: smallthinker / llada-moe / minimax-01 / graniteswitch — **合成 GGUF 12/12 格全对齐**（代理BATCH10, 2026-10, append-only）

架构计数（`arch_tensors_support`）**74 → 78**；任务账面 **83 → 87/156**。
本批新 ggml 件：**`ggml_exp`**（ggml.c:2892 → GGML_UNARY_OP_EXP=13, 逐元素 `expf`, 无 SIMD ——
vec.h:956 `ggml_vec_exp_f32` 就是标量 libm 循环; C-probe `parity/ref_exp_dump.c` →
`parity/exp_ref.bin` 位同钉死, F32/F16 双节 + raw-expf 交叉核对）、`ggml_pad`（ggml.c:5212 →
GGML_OP_PAD, ops.cpp:8196 内核）、`ggml_round`（ggml.c:2963 → GGML_UNARY_OP_ROUND=20）、
`ggml_reglu_split`（ggml.c:3019, vec.h:1401 标量循环）。协议沿用批次 1-9 + N=48 + `-long`
长提示格；ForwardWeights/CLI 臂与图同批落地（`ARCH_BATCH10=1 ./parity/arch_batch_parity.sh`）。
合成文件在 `/tmp/arch-batch10/`（5 个）, 测试 `crates/llama/tests/arch_batch10_e2e.rs`
（默认 5 跑 + 1 `#[ignore]` 造文件）。细节（算子映射 / 逐 arch C 行号 / 未移清单）见
PARITY.md「架构批次 10」节。

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| src/models/smallthinker.cpp（188 行: probs_in —— router 吃**原始 inpL** 层内先算 logits; LLM_FFN_RELU → reglu_split; 可选 iswa 图模板（window 键在 ⇒ n_swa 重钉 4096 + pattern 4 dense-first + *_swa 频率）否则 n_no_rope_layer_step=n_layer 全层 rope; expert_weights_scale 从不读） | `graph_arch.rs::{build_smallthinker_forward, build_moe_ffn_relu_probs, SmallthinkerParams}` / SMALLTHINKER 臂（model.rs）/ `context.rs` Smallthinker（shift 的 *_swa 频率交换）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2（0.0000）; `-swa` iswa 模板 in-port 双 FA（层 0 不 rope + iswa cache 对） | `ARCH_BATCH10=1 ./parity/arch_batch_parity.sh smallthinker` |
| src/models/llada-moe.cpp（163 行: `hparams.causal_attn=false`（:8）+ `build_attn_inp_no_cache`（:68）—— [T,T] 全可见 mask, 解码步只 attend 自己 ubatch 的 token; per-head q/k norm 先 rope; n_ff_exp 回落 n_ff/n_expert_used） | `graph_arch.rs::{build_llada_moe_forward, attn_no_cache_fa（F32→F16 cast 后 flash_attn 的 no-cache FA 臂, llama-graph.cpp:2635-2641）, LladaMoeParams}` / LLADA_MOE 臂 / `context.rs` LladaMoe（cache 存在但图不读 —— 同参考 llama_kv_cache）/ llama-cli | ⚠️ in-port 双 FA + 100-token 长格（参考侧**无生成路径**: llama-model.cpp:2289-2295 diffusion 家族 memory=nullptr, server-context.cpp:3177-3180 / llama-cli 均拒绝 memory-less 生成 —— 图 1:1 移植, 无参考输出可对） | `arch_batch10_e2e.rs::{pin_and_smoke, long_prompt_cells}` |
| src/models/minimax-01.cpp（484 行: `llm_graph_input_la`（:67-190, slopes=2^(-2^(-(log2(H)-3)))·ratio^h / q_decay=-slope·(pos_rel+1) / k_decay=-slope·(T-1-pos_rel) / diag_decay 三角含 -inf）+ 三处 `ggml_exp(scale(·, slope_scale))` + block_decay=exp(-slope·T)（:326-369）; slope_scale=1-il/(n_layer-1)+1e-5; n_embd_head_la²·n_head 状态 + attn_norm_2 全宽 RMS + sigmoid 输出门 + residual_scale 双缩放; 专家坐**dense n_ff**, n_ff_exp/weights_scale 从不读） | `graph_arch.rs::{build_minimax01_forward, Minimax01Params}`（la 输入由 builder 从 inp.pos 现算, 单序列 = set_input 的 n_seqs=1 切片）/ MINIMAX_01 臂 / `context.rs` Minimax01（recurrent_dims=(is_recr, 0, n_embd_s), lightning 层 0 宽 kv 行 —— llama-model.cpp:2559-2564 的 filter_attn）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2（0.0000/0.0001 = FA f16 尾巴）; 80 步递归 in-port（la 状态 + block decay 演化） | `ARCH_BATCH10=1 ./parity/arch_batch_parity.sh minimax-01` |
| src/models/granite-switch.cpp（427 行: adapter 路由表（:37-54 activate→slot/substitute）; **in-graph 单头路由层**（:263-287: pad 到 head 宽 → 因果 softmax 读回 → clamp+round+cast 成 I32 ids）; 每投影 switched LoRA delta（:193-219, 两次 mul_mat_id）; router_layer=n_layer 复用 n_layer_nextn（:58-71, K-shift 跳过 :2023-2025）; granite 标量族 + NORM rope） | `graph_arch.rs::{build_graniteswitch_forward, GraniteSwitchParams}`（switch 输入由 builder 从 inp.tokens 现算）/ GRANITE_SWITCH 臂（model.rs, adapter 表存 LlamaModel）/ `context.rs` GraniteSwitch（kv_dims = trunk 行 + 路由层 1 头行; n_layer=trunk+1）/ llama-cli | ✅ 48/48 ×2FA + `-long` 48/48 ×2（0.0000）; adapter-token 解码 in-port（激活 token 换 substitute 嵌入 + slot 路由, logits 随 slot 改变） | `ARCH_BATCH10=1 ./parity/arch_batch_parity.sh graniteswitch` |
| ggml.c:2892 `ggml_exp` + unary-ops.cpp:273/vec.h:956; ggml.c:5212 `ggml_pad` + ops.cpp:8196-8258; ggml.c:2963 `ggml_round`; ggml.c:3019 `ggml_reglu_split` + vec.h:1401 | `ggml/ops.rs::{exp,exp_inplace,round,pad,reglu_split}` + `compute.rs::{forward_exp,forward_round,forward_pad,glu REGLU 臂}`（GGML_UNARY_OP_EXP=13/ROUND=20, GGML_GLU_OP_REGLU=0） | ✅ 逐位 | `compute::tests::{exp_bit_exact_vs_reference, exp_f16_bit_exact_vs_reference}` ↔ `parity/ref_exp_dump.c` → `parity/exp_ref.bin`（8154 F32 + 600 F16 元素全位同, kind-3 raw expf 交叉核对证无 SIMD/libmvec） |

未移（集成者条目）：llama-server 的 forward_weights 未接本批四 arch（llama-cli 已接）;
minimax-01 的多序列 equal-seqs ubatch 切分（单序列口径）; graniteswitch 的
`llm_graph_input_switch::can_reuse`（端口每步重建图, 无复用路径）; llada-moe 参考侧无生成
驱动（详见 PARITY.md 批次10 §4）。
回归：`cargo test --workspace` **600/0**（基线 593 + 本批 5 默认测试 + ggml EXP 2 项）; 批次
1-9 parity 串行复跑全绿; qwen2.5 双 FA 锚点 MATCH。

## 架构批次 11b 个（长尾队列后半）: arcee / jais2 / talkie / nanbeige / dream / rnd1 / eurobert — **零新 ggml 算子, 合成 GGUF 全格对齐**（代理BATCH11B, 2026-10, append-only）

架构计数（`arch_tensors_support`）**85 → 92**（含批次 11a 的 +7）; 任务账面 **87 → 94/156**。
全部是既有 helper 的直接组合（xverse 体 / nemotron 的 relu² / llada-moe 的 no-cache 注意力 /
bert 的 EncoderContext / phimoe 的 rope_freqs 因子）。协议沿用批次 1-10 + `-long` 长提示格;
ForwardWeights/CLI 臂与图同批落地, parity 直接驱动 llama-cli（`ARCH_BATCH11B=1
./parity/arch_batch_parity.sh`）。合成文件在 `/tmp/arch-batch11b/`（8 个）、测试
`crates/llama/tests/arch_batch11b_e2e.rs`（默认 5 跑 + 2 `#[ignore]`：造文件、eurobert 参考对照）。
细节（逐 arch C 行号 / 未移清单 / 排障）见 PARITY.md「架构批次 11b」节。

| arch | C 文件 | Rust builder | 加载器 | 对照 |
|---|---|---|---|---|
| arcee | src/models/arcee.cpp:48-157（llama 骨架 + RELU_SQR/SEQ + 逐层 rope_freqs 因子; f_attention_scale 从不读 ⇒ 恒 1/√head 回落） | `build_arcee_forward` | ARCEE 臂 | 16/16 ×2FA + `-long` ×2, dlogprob 0.0000 |
| jais2 | src/models/jais2.cpp:54-155（LN+偏置三连, rope, up→relu²→down 双偏置 MLP） | `build_jais2_forward`（norm_eps=f_norm_eps 接线同 nemotron） | JAIS2 臂 | 16/16 ×2FA + `-long` ×2, 0.0000 |
| talkie | src/models/talkie.cpp:40-149（**weightless RMS 全家** + rope 后 [1,n_head] q-norm 广播 + embd_skip·layer_out_scale 残差 + logit_scale; ffn_down_s 死参数传 None） | `build_talkie_forward`（`ctx.rms_norm` 裸算子即 build_norm(x,nullptr)；LlamaModel::output_norm 槽别名 tok_embd（BERT 先例）） | TALKIE 臂（out_scale 复用既有 LAYER_OUT_SCALE 字段） | 16/16 ×2FA + `-long` ×2, 0.0000 |
| nanbeige | src/models/nanbeige.cpp:80-185（**num_loops 层展开** + loop 边界共享 output_norm + 指针别名 loop 槽; res->t_layer_inp 抽头未移） | `build_nanbeige_forward`（hparams.rs 增 nanbeige 三键, graniteswitch 先例） | NANBEIGE 臂（LayerTensors Copy 别名 :67-73） | 默认 14/14 ×2FA（参考少报 byte-token 既有现象）+ `-long` 16/16 ×2; `-loops2`（4→8 逻辑层）16/16 ×2FA + `-long` ×2, 0.0000; loop-norm 分叉 in-port 断言 |
| dream | src/models/dream.cpp:52-138（qwen2 体 + causal_attn=false + **build_attn_inp_no_cache**） | `build_dream_forward`（复用 llada-moe 的 attn_no_cache_fa 双 FA 臂） | DREAM 臂 | **in-port 双 FA + 100-token 长格**（参考 memory=nullptr 拒绝生成, llada-moe 先例） |
| rnd1 | src/models/rnd1.cpp:65-177（qwen3moe 转 diffusion: q/k norm 先 rope + softmax MoE + no-cache） | `build_rnd1_forward` | RND1 臂（n_ff_exp 回落; “for QWEN3MOE” 错误文案逐字保留） | **in-port 双 FA + 100-token 长格**（同上） |
| eurobert | src/models/eurobert.cpp:38-124（编码器: RMS + NEOX rope + 门控 SwiGLU, res->t_embd 无输头） | `build_eurobert_forward` + `EurobertRope`（EncoderParams::euro_rope Option; EncoderWeights::Eurobert; EncoderContext 的 pos 输入臂） | EUROBERT 臂（output 槽 tok_embd 别名, BERT 先例）+ `eurobert_weights()` | **与参考 llama_encode dump 100% 位精确**（T=12×128, `parity/ref_encode_dump --fa off`, `#[ignore]` 测试） |

* 新增共享件: `graph_arch.rs::{qkv_fused_plain}`（build_qkv 融合分支的无 AttnParams 形, eurobert 用）;
  `hparams.rs` 的 nanbeige 三键; `EncoderParams.euro_rope`（bert/t5 填 None, 三处构造点同步）。
* 未移（集成者条目）: build_cvec（各 arch no-op 既有口径）、nanbeige 的 t_layer_inp 抽头、
  llama-server 的 forward_weights 六 arch 未接（批次 9/10 同款条目; llama-cli 已接）、
  eurobert 仅 pooling NONE 格（MEAN/CLS 走既有 build_pooling, 未另设格）。
* 回归: 批次 1-10 parity 串行全绿（批次 5 首跑一格假错 = 并行 agent 串 8790 端口, 端口隔离复跑全过）;
  qwen2.5 双 FA 锚点 16/16; `cargo test --workspace` 全绿（批次 11b +5 默认测试）。

## 架构批次 11a 个（长尾队列前半）: apertus / grovemoe / qwen35moe / kimi-k3 / dots3note / minimax-m3 / qwen4exp — **合成 GGUF 18/18 对照格全对齐（minimax-m3 fa=on 两格参考服务器/库自分歧, 跳过并记录, 其余全 48/48）**（代理BATCH11A, 2026-10, append-only）

* 对应 C 源: models/{apertus,grovemoe,qwen35moe,kimi-k3,dots3note,minimax-m3,qwen4exp}.cpp（~4100 行）。
* 落点: `graph_arch.rs` 批次 11a 段（builders + `build_moe_ffn_silu_probs` / `KimiK3ResStack` /
  `attn_k_cached_mla_iswa` / `build_attn_msa_fa` + `MinimaxMsaCache/Step`）、`meta.rs`/`model.rs`
  各臂、`context.rs` ForwardWeights + `n_pos_per_embd` 补 IMROPE 两臂 + `build_msa_step`、
  llama-cli 六函数接线; 测试 `crates/llama/tests/arch_batch11a_e2e.rs`。
* 新 ggml 件: `ggml_xielu`、dup I32→F32（ops.cpp:570-575）、repeat 的 2 字节形态、
  `pool_2d`; 修复 pool_2d 平面写偏移、xielu FMA 收缩序、三处 fused wqkv 视图字节偏移、
  MSA cache 视图 stride / wo 漏乘 / F16 回转（详见 PARITY.md 批次 11a §3）。
* 未移（集成者条目）: K>1 rs-rollback（kimi-k3/qwen35moe/qwen4exp, delta-net-base.cpp:546-606）、
  qwen4exp 的 QSA 块压缩与 PLE n-gram（验证文件走合法稠密/无 PLE 配置）、minimax-m3 多流 MSA
  （ns>1）、qwen35moe/qwen4exp 全注意层 fused attn_qkv 布局（端口实现必需三分量布局）、
  qwen35moe 的 graph_mtp、llama-server 的 ForwardWeights 接线（批次 9/10 同款）。
* 新调试工具: `parity/ref_decode_dump.c` 增 `--decode-tail/--decode-ids`;
  `crates/llama/tests/arch_batch11a_dump.rs`（decode 逐步节点位精确对照, 后续批次可复用）。
* 细节（逐 arch C 行号 / minimax-m3 fa=on 的参考自分歧论证 / 排障）见 PARITY.md「架构批次 11a」节。

## 架构批次 12 个（终局长尾）: hrm-text / laguna / maple — **架构队列清空, 零新 ggml 算子, 合成 GGUF 全格对齐**; llama-server 补齐批次 9-12 全部 arch（代理BATCH12, 2026-10, append-only）

架构计数（`arch_tensors_support`）**92 → 95**；任务账面 **94 → 97/156**。src/models/ 的
*.cpp 图文件至此全部有端口 builder。协议沿用批次 1-11 + `-long` 长提示格; ForwardWeights/
CLI/**llama-server** 臂与图同批落地, parity 直接驱动 llama-cli（`ARCH_BATCH12=1
./parity/arch_batch_parity.sh`）。合成文件在 `/tmp/arch-batch12/`（4 个 + maple-allswa
in-port 变体）、测试 `crates/llama/tests/arch_batch12_e2e.rs`（默认 4 跑 + 1 `#[ignore]`
造文件）。细节（逐 arch C 行号 / 未移清单 / 服务器接线证据）见 PARITY.md「架构批次 12」节。

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| src/models/hrm-text.cpp（213 行: 交替低/高栈 —— 缓存槽 lps×h×(l+1) 别名 2×lps 物理块（:21-23/:60-67）; 学习态 `hrm.z_l_init` 裸名 [n_embd] 经 zH+zL 广播穿线（:46/:182-196）; 全 weightless RMS + 每栈末尾 norm（:107/:144/:162）; sigmoid 注意力门在 o_proj 前（:110-137）; embedding_scale 进共享 build_inp_embd（:8）） | `graph_arch.rs::{build_hrm_text_forward, HrmTextParams}`（build_stack 子函数 1:1）/ HRM_TEXT 臂（model.rs, 槽位别名同 nanbeige; `LlamaModel::hrm_z_l_init`）/ `context.rs` HrmText / llama-cli + llama-server | ✅ 16/16 ×2FA + 100-token 长格, dlogprob 0.0000 | `ARCH_BATCH12=1 ./parity/arch_batch_parity.sh hrm-text` |
| src/models/laguna.cpp（331 行: sigmoid MoE + `ffn_exp_probs.bias` 选中偏置（:129）+ 常开 shexp（独立尺寸键 :23-28）; **softplus 输出门双宽度**按张量宽度探测（:99-122: per-head reshape 广播 vs per-element 直乘）; 门吃注意力前 hidden（:201-210）; 逐层头数（:85）; **逐层类型 RoPE**（:179-191: SWA 层 plain rope —— ext/beta=0, attn_factor=1, n_ctx_orig=n_ctx_train, n_rot_swa —— 全注意力层 cparams YaRN）; SWA 可选（:30-49, 模式 4 dense-first; M.1 无键全注意力）） | `graph_arch.rs::{build_laguna_forward, LagunaParams}`（逐层 n_head 覆写（openelm 先例）; attn_kv_cached 自动 iswa/普通）/ LAGUNA 臂 / `context.rs` Laguna（K-shift 逐层类型臂）/ llama-cli + llama-server | ✅ XS.2 形（hybrid SWA + 逐层头数 + 双门宽 + θ 500000/10000 + rope_swa 24 维）16/16 ×2FA; M.1 形 `-full`（无窗全注意力, per-element 门）16/16 ×2FA; 两形 logits 互异 in-port 断言 | `ARCH_BATCH12=1 ./parity/arch_batch_parity.sh laguna laguna-full` |
| src/models/maple.cpp（150 行: softmax MoE over iswa 对（:73）; **只对 SWA 层 rope**（:90-99, n_rot(il)=n_rot_swa + *_swa 频率 —— 全注意力层从不旋转）; q/k norm 在 rope 前（:85-86）; pattern 数组 get_arr 必需（:10）; `swiglu_clamp_exp` 必需键但图不消费（:16/:121-131）; output 必需（:33）） | `graph_arch.rs::{build_maple_forward, MapleParams}` / MAPLE 臂 / `context.rs` Maple（K-shift 逐层类型: 全注意力层照 base 转 —— rope_pattern 恒 1, C 同款字面移植）/ llama-cli + llama-server | ✅ 16/16 ×2FA; `-long`（>64 过窗）16/16 ×2FA（0.0000/0.0009 FA 尾巴）; `maple-allswa`（全层 rope）in-port `maple_swa_rope_changes_logits` | `ARCH_BATCH12=1 ./parity/arch_batch_parity.sh maple maple-long` |
| src/models/eagle3.cpp（339 行: `graph<true>` 编码器 :103-146 —— 3 层目标特征 [3*n_embd_tgt, T] → 可选 RMS(norm_before_fc) → fc → **t_h_nextn**; `graph<false>` 解码器 :151-326 —— get_rows(tok_embd) + g 输入, attn_norm/attn_norm_2 双 RMS, concat(dim=il=0) 进 [2*n_embd] 的 wq/wk/wv, NORM rope（可选 rope_freqs）, 残差基 = norm_before_residual ? g_norm : 原始 g, eagle3_prenorm 即 t_h_nextn, output_norm → lm_head, 可选 d2t scatter（fill -inf + setrows, :307-320）; 加载 :3-101 —— target_layers 恰 3 / target_hidden_size / 两 norm 开关默认 false, own tok_embd/output/d2t 可选） | `graph_arch.rs::{build_eagle3_encoder_forward, build_eagle3_decoder_forward, Eagle3Weights, Eagle3Params}` + `eagle.rs::load_eagle3_head`（自带 mini-ModelLoader —— model.rs 通用加载器无 EAGLE3 臂, 该文件归其他批次; **ctx_other = 在头 Context 建同名张量、外置存储指向目标 mmap** = llama-context.cpp:156-163 的零拷贝对应物）/ `context.rs::{DecodeContext::new_eagle3, eagle 字段, encode_eagle3, forward 的 eagle 分支}` + `build_llama_forward` 的 layer_inp 记录（llama.cpp:127）/ llama-cli + llama-server 的 eagle 分支 | ✅ 合成对（llama 目标 + own/other 两头变体）`parity/eagle_parity.sh` **4 格全 PASS**: (a) trunk-unchanged 四流一致 ×2FA, (b) draft parity 16/16 ×2FA（ref acceptance `0 accepted / 39 generated, mean len 1.00`）; 服务器侧 16 token + 草稿计数 39/39 与参考逐字同; in-port `eagle_e2e` 3 测试 ×2FA | `cargo test --release -p llama --test eagle_e2e eagle_write_synth_files -- --ignored` 后 `bash parity/eagle_parity.sh` |
| tools/server 的 forward_weights（批次 9-12 集成） | `crates/tools/llama-server/src/main.rs`（22 arch 分发臂; `attn_params`/`first_attn_layer` 的逐 arch 覆写同 CLI）+ `src/weights.rs`（llama-cli 同名捆绑函数的逐字副本）+ eurobert 的 `EncoderContext` 分支（`EncoderWeights::Eurobert` + `euro_rope`, bert 先例）; 附带: /completion 的 `prompt` 字段改 `tokens.detokenize(ctx, true)`（server-context.cpp:2111, SPM 词表 BOS 回显对齐） | ✅ `run_server_parity.sh` 全字段 0 diff: maple / smallthinker MATCH ×2, bailingmoe3 首请求 MATCH（其 n_probs 第二请求格为既有循环态槽位复用缺口, PARITY.md 批次 12 §4）; eurobert 服务器嵌入 8×128 逐位同（max\|Δ\|=0）; 端口 server==端口 CLI 25 格 ×2FA = 50/50 全 16/16; qwen2.5 锚点双 FA MATCH + chat 29/29 | `MODEL=<synth> TAG=… bash parity/run_server_parity.sh` |

未移（集成者条目）: build_cvec（三 arch 的 no-op 既有口径）、hrm 的 prefix_lm prefill
（:14-15 round-trip 保留）、maple 的 swiglu_clamp_exp 图侧消费（本 pinned 版无 clamp 参）、
hrm/laguna/maple 的 LLM_TYPE 表（描述字符串）、末层 inp_out_ids get_rows（qwen3 既有口径）、
bailingmoe3 服务器复用槽位的 KDA 态重置（rs_z 语义, PARITY.md 批次 12 §4）、llama-server
仍不接批次 1-8 的 arch（gpt-oss/gemma4/granite/lfm2/qwen35/deepseek/mamba 混合/批次 8 MoE
长尾 —— `forward_weights` other 臂报错指引）。
回归：`cargo test --workspace` 全绿（批次 12 +4 默认测试）; 批次 1/11b parity 复跑 all runs
matched; qwen2.5 CLI 双 FA 32/32 ×2; `ARCH_BATCH12` 默认集 10 格 all matched; server 脚本
（run_server_parity / chat）MATCH。

## 补记（代理PERF2, 2026-09-30）— Q4_0 8x8 repack + 解码 FA AVX512（性能第二轮）

* 上表 repack 行的"未做 q4_0/iq4_nl"作废一半: **Q4_0 8x8 已移植**（`crates/ggml/src/repack.rs`
  的 Q4_0 节 + compute.rs 2D mul_mat/3D mul_mat_id 接线，拦截点=ggml-cpu.c:1751 的
  extra-buffer 先于 llamafile；真值 `parity/q4_0_repack_ref.bin`，来自新探针
  `parity/ref_repack_q4_0dump.cpp`）。Q2_K（AVX512 门, 本机模型不可达）与 IQ4_NL
  （端口无该类型量化器, 本地 GGUF 全扫描无 iq4_nl 张量）仍未移。
* `crates/ggml/src/simd_x86.rs` 新增 `vec_dot_f16`/`vec_mad_f16`/`vec_scale_f16`
  （vec.cpp:264 / vec.h:439 / vec.h:769 的 AVX512 F32Cx16 体）——`flash_attn.rs::attn_row`
  （one_chunk, T<64 的解码/verify 路径）改调它们 + Q 行一次性转换（ops.cpp:8746）+ 连续 K/V
  行免装配。位同证据: `flash_attn::ref_dump_*`（11/11 用例逐位, 含 mode 8/13/14 one_chunk）
  + 新增 `simd_x86::avx512_f16_vec_kernels_match_scalar_bit_exact`（16/32/64/尾长全同）。

## 补记（代理PERF3, 2026-09-27）— `inp_out_ids` 输出行裁剪（lm_head 前的 get_rows）

* 上文 llama-graph.cpp 行与 context.rs 行补一档: **`build_inp_out_ids`（llama-graph.cpp:2480-2496）+
  `llm_graph_input_out_ids::set_input`（:199-223）已移植** —— `DecodeInputs::out_ids`
  （graph.rs, `None`=n_outputs==n_tokens 恒等情形）+ `DecodeContext::set_out_rows`（context.rs）+
  共享 helper `graph::out_rows`（= 各 models/*.cpp 的尾款 `ggml_get_rows(cur, inp_out_ids)`:
  gemma4.cpp:416-417 / hrm-text.cpp:200-201 / hy-v4.cpp:583-585 / ernie4-5.cpp:124-125 /
  openai-moe.cpp:160-162 / glm4-moe.cpp:271-272 / deepseek2.cpp:395-397 / deepseek32.cpp:714）,
  全部 88 个 builder 头改走它（eagle3 头除外, eagle3.cpp 不 gather）。各 arch 批次注记里的
  "inp_out_ids get_rows 未移/skipped（值恒等）"条目自本轮起**整体关闭**。
* 测量与位同证据见 PARITY.md 性能第三轮（qwen pp64 414→497 t/s; gemma-12B lm_head 235→16 ms;
  锚点全绿: qwen 双 FA 16/16、gemma-12B 双 FA+SWA 长上下文 16/16、gpt-oss 16/16、embd-rows 全行位同、
  ppl 逐 chunk 同）。
* 性能第四轮 + gelu_erf（PARITY.md 性能第四轮）:
  * **arch/x86/repack.cpp:641 `gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>` 补齐**（ggml/src/repack.rs
    `simd_x86_q4_0::{gemm → gemm_avx512<VNNI> + gemm_256_section + rhs_512/lhs_512/acc_rows_512 +
    rhs_256/lhs_256/acc_rows_256 + msda16<VNNI>}`）: AVX512BW/DQ 2x2 网络（:660-1105, VNNI
    `vpdpbusd`）+ C 的 AVX2 共享段（:1100-1445, 含 `nc%16` 尾列 `xstart`）; 运行时门
    `simd_x86::{avx512bw, avx512vnni}`。gemma-12B pp64 repack=1 24.5→**47.3-48.8 t/s**（>repack=0 的
    28.7, 恢复 C 取向）; 位同: q4_0_ref_tests + 新增 shape-grid 测试（75600 槽位）全绿。
  * **ggml_gelu_erf 移植**（ggml.c:2796/2802 → ops.rs `Context::gelu_erf(_inplace)`;
    vec.h:1010/979 → ops.rs `ggml_vec_gelu_erf_f32/_f16`（标量 `erff`, extern "C" 绑 libc）;
    ops.cpp:2436-2453 → compute.rs `forward_gelu_erf`（F32/F16 分派 = 类型表）+ UNARY 臂
    params[0]=16）。dump: parity/ref_gelu_erf_dump.c → parity/gelu_erf_ref.bin（8152 F32 +
    600 F16 逐位同, 含裸公式 kind-3 交叉核对证无 SIMD erf）。whisper-enc/clip 音频图
    （clip.rs:993 BLOCKED 项）自此可建图。

<!-- AUDIT-MODELS 2026-09-24: src/models/ 155 文件逐符号审计（parity/AUDIT_models.md）。
     上表"src/llama-model.cpp/.h + src/models/*"行的真实覆盖率: 102/155 已移植
     (87 全符号 + 7 装载-only 组合 + 8 trunk-only 部分移植), 9 documented-skip
     (PARITY:381/:638/:4577), 43 MISSED (llama4/qwen3vl/rwkv*/gemma3n/glm4/
     glm-dsa/chatglm/mistral3/cohere2/ernie4-5 稠密 等; 全表见 AUDIT_models.md)。
     llama-arch.cpp 三表 (ARCH/TENSOR/KV NAMES) 153+273+243 条全 parity。 -->

## 架构批次 13 个（P0 标准注意力队列）: llama4 / qwen3vl(+moe) / qwen2vl / glm4 / glm-dsa / chatglm / mistral3 / cohere2 / minicpm3 / exaone4 / bitnet / dbrx / ernie4-5 + nemotron-h-moe 装载臂 — **批次13/14, 前代理中断后收尾**（append-only）

AUDIT_models.md 的 P0 拒载队列（真实公开 GGUF 的 arch; 本节含批次 14 的 RWKV/gemma3n
行——两批同一收尾轮闭环）。架构计数（`arch_tensors_support`）
**95 → 115**（批次 13 的 14 个图 arch + 批次 14 的 rwkv6/rwkv6qwen2/rwkv7/arwkv7/gemma3n;
nemotron-h-moe 批次 5 已计; 并行批次 15 在飞时树为 136）。协议同批次 1-12：合成 GGUF
（`/tmp/arch-batch13|14/`, 20+8 文件含变体）+ 全新参考 server 首请求 16/48 token ×`-fa on/off`
+ `-long` 长提示格; ForwardWeights/CLI/**llama-server** 臂同批落地。**收尾轮修复**：
llama4 的 MoE 专家权重须**前乘**（`weight_before_ffn`, llama-graph.cpp:2018 的
LLAMA_ARCH_LLAMA4 特例 —— repeat(hidden)×sigmoid 权重在专家 matmul 前、down 后不乘;
`build_moe_ffn_silu_gu` 加形参, 其余 7 调用点 false 不变）; llama4-noswa 移出默认集
（**pinned 参考自身 abort**: llama-graph.cpp:161 断言 f_attn_temp_scale!=0, 该变体无
温度键 —— in-port + server==CLI 验证）。细节（逐 arch C 行号 / 节点级根因 / 服务器
证据）见 PARITY.md「架构批次 13/14」节。

| 参考文件 | Rust 位置 | 状态 | 校验 |
|---|---|---|---|
| src/models/llama4.cpp（273 行: iswa CHUNKED / no-rope 层温度缩放 Q / rope 后无权 L2 qk-norm / sigmoid MoE+共享专家（权重前乘）/ 逐层 rope 频率） | `graph_arch.rs::{build_llama4_forward, llama4_ffn, Llama4Params, build_inp_attn_scale}` / LLAMA4 臂 / `context.rs` Llama4 / llama-cli + llama-server | ✅ 16/16 ×2FA 0.0000; -long 16/16 ×2FA; DECDMP1 节点级修复证据 | `ARCH_BATCH13=1 ./parity/arch_batch_parity.sh llama4 llama4-long` |
| src/models/qwen3vl.cpp / qwen3vlmoe.cpp / qwen2vl.cpp（IMRoPE/MRoPE sections + deepstack 零填充 + q/k norm; 文本侧图） | `graph_arch.rs::{build_qwen3vl_forward, build_qwen2vl_forward, …Params}` / QWEN3VL/QWEN3VLMOE/QWEN2VL 臂 / 两工具 | ✅ 各 16/16 ×2FA 0.0000 | `… qwen2vl qwen3vl qwen3vlmoe` |
| src/models/glm4.cpp / glm-dsa.cpp（SWIGLU-SEQ / 可选 rope_multi; DSA 闪电索引器 + MLA K-only cache + 共享索引器层 + sigmoid MoE; graph_mtp 已档批次 glm-dsa） | `graph_arch.rs::{build_glm4_forward, build_glm_dsa_forward}` / GLM4/GLM_DSA 臂（GLM_DSA 的 n_embd_head_k=kv_lora+qk_rope, n_head_kv=1 深seek2 约定） | ✅ glm4/-mrope/dsa/-shared 各 16/16 ×2FA; dsa-long 16/16 ×2FA | `… glm4 glm4-mrope glm-dsa glm-dsa-shared glm-dsa-long` |
| src/models/{chatglm,mistral3,cohere2,minicpm3,exaone4,bitnet,dbrx}.cpp + ernie4-5 稠密 | 各 `build_*_forward` + 臂（mistral3 温度缩放 f64 set_input 公式; cohere2 仅 SWA 层 rope+logit_scale; minicpm3 半-MLA n_head_kv=n_head; exaone4 QK-norm 先于 rope; bitnet 子量化+绑定头; dbrx 钳位 QKV; ernie4-5 复用 ernie45moe 稠密分支） | ✅ 全部 16/16 ×2FA 0.0000（含 mistral3-temp / exaone4-swa 变体与 cohere2/exaone4-swa -long） | `ARCH_BATCH13=1 ./parity/arch_batch_parity.sh`（默认集 all matched） |
| src/models/rwkv6.cpp / rwkv6qwen2.cpp / rwkv6-base.cpp / rwkv7.cpp / arwkv7.cpp / rwkv7-base.cpp（纯递归: token-shift conv 态 + WKV 态; 融合算子未移 → 组合扫描, FMA 舍入差为 integrator 项） | `graph_arch.rs`（批次 14 节: rwkv_token_shift_load/store, rwkv_wkv6_scan, rwkv_gla_scan, rwkv_wkv7_scan, build_rwkv6_time_mix/channel_mix, build_rwkv7_time_mix/channel_mix, build_rwkv6/qwen2/7/arwkv7_forward）/ RWKV6/RWKV6QWEN2/RWKV7/ARWKV7 臂 + `recurrent_dims` 的 RWKV 臂（RecurrentState 自动配） | ✅ rwkv6/rwkv6qwen2/rwkv7 **48/48 ×2FA**; rwkv6-long 48/48 ×2FA; rwkv6-legacy / rwkv7 变体 in-port; rwkv7-long/arwkv7 移出默认集（组合扫描 FMA 近平局 / 参考自身线程噪声, 融合算子后重开） | `ARCH_BATCH14=1 ./parity/arch_batch_parity.sh`（默认集 all matched） |
| src/models/gemma3n.cpp（465 行: 每层嵌入 + altup 栈(tanh 路由 predict/correct) + laurel + 高斯 top-k 稀疏 + KV-reuse 层（≥20 读 n_layer_kv_from_start-(is_swa?2:1) 层缓存行）+ 4:1 SWA） | `graph_arch.rs`（批次 14 节: build_gemma3n_forward + 8 helper + `gemma3n_attn`（**收尾修复: 补 wo 投影**, FA/非 FA 两分支））/ GEMMA3N 臂 / 两工具 | ✅ 47/47 ×2FA 0.0000; DECDMP1 节点级 1161/1117 计算 0 发散; -long fa-off 48/48（fa-on 0.008-nat 近平局 FA 尾巴, 非默认格） | `ARCH_BATCH14=1 ./parity/arch_batch_parity.sh gemma3n` |
| tools/server 的 forward_weights（批次 13/14 集成） + parity/run_server_arch_parity.sh 的批次 9-14 扩表 | `crates/tools/llama-server/src/main.rs`（19 arch 臂; gemma3n 的 SwaCacheSpec iswa 对 + RWKV 的 recurrent_dims 走通用构造）+ `src/weights.rs`（CLI 逐字副本）+ parity 脚本 BATCHDIR/model_of/cell_n | ✅ 端口 server==端口 CLI: 批次 13 **40/40**（含 llama4-noswa）、批次 14 **8/8** 全同; WITH_REF 全字段 0 diff 样本 22 格（覆盖每种驱动形态, gemma3n 唯一 diff=content 的 U+FFFD 渲染、token 逐位同） | `ARCH_BATCH=13|14 [WITH_REF=1] bash parity/run_server_arch_parity.sh <cell…>` |

未移（集成者条目, PARITY.md 批次 13/14 §5）: 三个融合 WKV 算子（RWKV_WKV6/
GATED_LINEAR_ATTN/RWKV_WKV7 的 C 行号在档）、gemma3n 的 per-level 多模态半边
（文字路径已移）、服务器 content 的非法 UTF-8 渲染差（共享面）。回归: 批次 13 默认集
46 格 + 批次 14 默认集 10 格 all matched; DECDMP1 镜像测试（arch_batch13/14_
prefill_node_dump, ignored）留作 bisect 工具。
| src/models/{qwen,maincoder,pangu-embed,plm,cogvlm,spark2-5,muse-glimmer,llada}.cpp（批次 15 稠密队列: v1 融合 QKV 半宽 FFN / rope 后 QK-norm / 必需 wo 偏置 / 共享 k_pe MLA-lite + rope_set_offset + relu² / visexp 双权重 + 朴素 rope / 逐头 sigmoid attn 门 / 反转 rope 规则 + 双 1e-8 后置 norm + softcap 头 / 无 cache 扩散注意力） | `graph_arch.rs`（批次 15 节: build_{qwen1,maincoder,pangu_embed,plm,cogvlm,spark25,muse_glimmer,llada}_forward + …Params/…Weights）/ QWEN/MAINCODER/PANGU_EMBED/PLM/COGVLM/SPARK2_5/MUSE_GLIMMER/LLADA 臂 / 两工具（llada 无 cache → 端口内） | ✅ qwen/maincoder/pangu/plm(fa off)/spark/muse 各 16/16 ×2FA; cogvlm 与参考**库**位精确（服务器自身分歧, minimax-m3 先例）; llada 端口内; plm fa-on 参考装载断言（skip 注释在档） | `ARCH_BATCH15=1 ./parity/arch_batch_parity.sh qwen maincoder pangu-embed plm spark2-5 muse-glimmer` |
| src/models/{hunyuan-vl,hunyuan-dense,granite-swa,afmoe,mellum,paddleocr}.cpp（批次 15 长尾: 可选 M-RoPE + XDRoPE alpha 重基 + rope 后 QK-norm / iswa+sinks+三 scale+SWIGLU 融合 shexp+deepstack / MuP 缩放+(il+1)%4 rope 规则+宽 shexp / 全 MoE+SWA 层朴素 rope（**因果解码器**, 审计"编码器"系笔误）/ M-RoPE 文本图） | `graph_arch.rs`（批次 15 节: build_{hunyuan_vl,granite_swa,afmoe,mellum,paddleocr}_forward; HUNYUAN_DENSE 共 HunyuanVl; GRANITE_SWA_MoE 走 build_moe_ffn_silu_probs + build_ffn_swiglu_seq）/ HUNYUAN_VL(\|HUNYUAN_DENSE)/GRANITE_SWA/AFMOE/MELLUM 臂 + PADDLEOCR/MISTRAL4/LLAMA_EMBED 共臂（ernie4_5/deepseek2/llama） | ✅ hunyuan-vl/-mrope/hunyuan-dense/granite-swa/-moe/afmoe/mellum/paddleocr 各 16/16 ×2FA（granite-swa-long/spark2-5-long 亦 16/16）; afmoe -noswa 与 mellum -swa 端口内 | `ARCH_BATCH15=1 ./parity/arch_batch_parity.sh hunyuan-vl hunyuan-vl-mrope hunyuan-dense granite-swa granite-swa-moe afmoe mellum paddleocr` |
| src/models/{hy-v3,mimo2,step35}.cpp（批次 15 iswa-MoE 队列; graph_mtp 各自开档 🟡 — MTP 张量随装载臂加载）/ hy-v4.cpp（iHC 2*hc 系数 + 门控 MLA 带 sinks + 可选 DSA 索引器（复用 lid cache 对与 top-k 掩码合成）+ sigmoid MoE 宽 shexp; 6 个 C helper 以 hy_v4_* 本地函数落地） | `graph_arch.rs`（批次 15 节: build_{hy_v3,mimo2,step35,hy_v4}_forward + hy_v4_hc_pre/post/head/reduce + hy_v4_attn_mla; **hy_v3 的 exp_probs_b 是裸名**（hy-v3.cpp:68 tn(TENSOR,i) 无后缀）; step35 半维 rope + 无 blk 槽共享 rope_freqs; hy_v4 的 out_ids 修剪一次于 HC 块尾（无二次 gather））/ HY_V3/MIMO2/STEP35/HY_V4 臂（后三者含 MTP nextn 张量; hy-v4 hparams 臂补 MLA+iHC+索引器全键与校验） | ✅ hy-v3/mimo2/step35/hy-v4/hy-v4-dsa/hy-v4-dsa-long 各 16/16 ×2FA; step35-long/mimo2-long 跨窗近并列翻转（0.013, rwkv7-long 先例）默认集外端口内; graph_mtp 三件 PARITY.md 批次 15 开档 | `ARCH_BATCH15=1 ./parity/arch_batch_parity.sh hy-v3 mimo2 step35 hy-v4 hy-v4-dsa` |
| src/models/{mistral4,llama-embed}.cpp + gemma-embedding.cpp / gemma4-assistant.cpp（批次 15 共臂与编码器: mistral4=deepseek2 图（models.h:1393）; llama-embed=graph<true> 无 cache 无头（**llama_encode 强制非因果**, llama-context.cpp:1526-1529）; gemma-embedding=对称 SWA 无 cache 编码器（EncodeInputs 扩 kq_mask_swa 双掩码）; gemma4-assistant=§integrator 已移 ctx_other 草稿（gemma4_assistant.rs, 见下） | `graph_arch.rs`（build_llama_embed_encoder + build_gemma_embedding_forward + qkv_fused_enc/qkv_separate_parts_enc）/ MISTRAL4/LLAMA_EMBED 共臂（DEEPSEEK2/LLAMA）/ GEMMA_EMBEDDING 臂 + EncoderWeights::{LlamaEmbed,GemmaEmbedding} + EncoderParams::{gemma_swa,causal} + 两工具 -fe 分支 | ✅ mistral4（MLA 配方）16/16 ×2FA; llama-embed 参考 llama_encode dump **位精确**; gemma-embedding 参考自身崩溃（GGML_ASSERT(buffer), ggml-backend.cpp:205）→ 端口内; gemma4-assistant §integrator 后已移（真机对位精确 ×2FA, 见批次 15 节） | `cargo test -p llama --test arch_batch15_e2e -- --ignored arch_batch15_encoder_reference_parity` |
| 批次 15 集成（context/CLI/server/脚本/测试） | `context.rs`（ForwardWeights 新变体 17 + 分发 + shift_rope 六新臂 + n_pos_per_embd 的 hunyuan-vl/paddleocr mrope; KvCache lid 的 HY_V4 接线; EncoderContext 的 kq_mask_swa/causal）+ `llama-cli`/`llama-server`（同形臂 + 批次 15 weights 装配 + gemma_swa/causal 装配; llama-embed 走 -fe 编码器分支）+ `parity/arch_batch_parity.sh` 的 `ARCH_BATCH15=1`（b15_model 候选解析 + plm fa-on skip 注释 + cogvlm/step35-long/mimo2-long 默认集外说明）+ `crates/llama/tests/arch_batch15_e2e.rs`（5 常规 + 2 ignored + DECDMP1 节点 dump 探针） | ✅ 默认集 19 格 ×2FA + 3 -long 格 all matched; `cargo test --workspace` 全绿; qwen2.5 锚点 16/16 ×2FA | `ARCH_BATCH15=1 bash parity/arch_batch_parity.sh` |

批次 15 §integrator 收口: gemma4-assistant 的 ctx_other 草稿驱动 **已移** — `crates/llama/src/gemma4_assistant.rs`（装载臂 + graph + `DecodeContext::attach_gemma4_assistant`/`decode_gemma4_assistant` = ctx_other 的端口内倒置: 头张量物化进主干 Context（mmap 外储, eagle.rs 先例）, 草稿图视图 share 映射层（llama-model.cpp:2698-2703）的主干 iswa KV）; `crates/llama/src/speculative.rs` 的 `CommonSpeculativeImplDraftMtp` 增 `is_mem_shared` 后端（process 跳过追平 decode :1519, 每步 pos0 :1718-1722, trait 的 draft()/process() 现携带 &mut 目标 ctx）; CLI/server 的 -md arch 识别接线; `crates/llama/tests/gemma4_assistant_e2e.rs`（真机对位, ignored）+ `parity/gguf_pin_gemma4_assistant.py`（头文件重钉 pinned 命名）+ `parity/ref_gemma4_assistant_dump.c`（参考库探针: 手工 ctx_other 配对重放 draft-mtp 步形, --nodes 节点流）。验证: 26B-A4B-QAT 主干 + 重钉名头 5 步草稿位精确 ×2FA。同轮: mimo2/step35 的 iswa kq ~3e-3 修复（路由偏置 post-gating 化 + swiglu 标量尾道 libm expf 化, `simd_x86::vec_swiglu_f32` 新增）并晋升 ARCH_BATCH15 默认集
（dflash/eagle 式模块）、hy-v3/mimo2/step35 的 graph_mtp（🟡 开档）、plm fa-on 的
参考侧 set_rows 断言、cogvlm 的参考服务器/库自分歧、gemma-embedding 的参考
llama_encode 崩溃、step35-long/mimo2-long 的跨窗近并列（0.013-logprob 翻转,
rwkv7-long 先例）。回归: 批次 1-14 未动; qwen2.5 锚点双 FA 16/16。

## ggml 审计更正（审计 agent, 2026-09-27, 只读; 详表见 parity/AUDIT_ggml.md）

上文 ggml 节若干行已过时, 更正如下（原行保留作历史, 以本段为准）:

| 原行（本文件） | 原状态 | 更正后 |
|---|---|---|
| ggml/src/ggml-alloc.c → crates/ggml/src/alloc.rs | ⏳ | **文件不存在**; 实际=tensor.rs::Context arena + `reset_graph_to`（watermark 重建, 等价简化 📄） |
| ggml/src/ggml-backend.cpp → crates/ggml/src/backend.rs | ⏳ | **文件不存在**; 实际=compute.rs（graph_compute:255 + set_eval_callback:229）+ llama/context.rs; CPU-only 简化 📄 |
| ggml/src/ggml-cpu/ggml-cpu.c → crates/ggml/src/cpu/mod.rs | ⏳ | **文件不存在**; 实际=compute.rs（Team threadpool:441 + mul_mat 全路由:876 + type_traits_cpu 对应 vec_dot_type/vec_dot_row）🔨→✅ |
| ggml/src/ggml-cpu/unary-ops.cpp, binary-ops.cpp → crates/ggml/src/cpu/*_ops.rs | ⏳ | **文件不存在**; 实际=compute.rs::forward_bin_op:630（✅）+ forward_{tanh,sigmoid,relu,elu,gelu,gelu_erf,silu,exp,softplus,round,xielu}（11/22 unary, 其余 11 个全部零 llama 层调用或归入 PLE/后端采样/llava 的 documented-skip） |
| ggml/src/ggml-cpu/ggml-cpu.cpp → crates/ggml/src/cpu/mod.rs | ⏳ | **文件不存在**; 实际=llama crate util（cpu_info/props, PARITY:680）✅ |
| ggml/src/ggml-cpu/llamafile/sgemm.cpp → 暂不移植 | ⛔ | **已移植**: crates/ggml/src/tinyblas.rs（1423 行; 6 组合 F32/F16/BF16/Q8_0/Q4_0/Q5_0 路由+尾块 tile 与参考 switch 逐位, tinyblas_ref.bin 6360 例, PARITY:301-305）✅ |
| ggml/src/ggml-cpu/arch/x86（SIMD 变体） | 🔨 | 基本落地: simd_x86.rs（8 量化族 AVX2 lane, 与标量逐位互证）+ vec_dot.rs AVX512 f32 lane + repack.rs 的 Q4_K AVX2/AVX512BW+DQ lane ✅（仅 q2_K_8x8/iq4_nl_8x8 两实例未移, PARITY:1662 在档） |
| ggml/src/ggml-cpu/repack.cpp 行内"未做: q4_0/q2_k/iq4_nl 8x8" | — | **q4_0_8x8 已做**（repack.rs:3700/3722 + compute.rs:3980 repacked_q4_0 + mul_mat_q4_0_repack, q4_0_repack_ref.bin）; q2_k/iq4_nl 仍缺（在档） |
| （COVERAGE.md:21 repack 行"Q4_0/Q4_K/Q2_K/IQ4_NL 8x8 待做"） | 🔨 | Q4_K/Q4_0 已完成（q4k_repack_ref.bin / q4_0_repack_ref.bin）; 剩 Q2_K/IQ4_NL |

新档案行（此前无行）:

| 参考文件 | 行数 | Rust 对应 | 状态 |
|---|--|--|---|
| ggml/src/ggml-opt.cpp | 1094 | — | 📄 训练 API; llama 层零调用（CROSS_ENTROPY_LOSS/OPT_STEP_* 只进 opt）; COVERAGE >1500 行截断遗漏 |
| ggml/src/ggml-backend-meta.cpp | 2517 | — | 📄 多 GPU 张量切分元后端; CPU 单后端不可达（COVERAGE:42 ⏳ 行改归档 ➖） |
| ggml/src/ggml-cpu/iqp.cpp/.h | 1253+39 | — | ➖ IQ panel gemm 性能 lane（IQ 权重 mul_mat 快路, ggml-cpu.c:1377, batch≥8 时替代行式 vec_dot; 端口 2026-09-27 起有 IQ vec_dot ⇒ 解码路径等价, 预填求和序差异 = 同 tinyBLAS 类结构间隙） |
| ggml/src/ggml-cpu/hbm.cpp/.h | 55+8 | — | 📄 Fujitsu A64FX HBM buffer（GGML_USE_CPU_HBM off） |
| ggml/src/ggml-cpu/traits.cpp/.h + arch-fallback.h + simd-mappings.h + simd-gemm.h | 36+38+377+1319+226 | vec_dot_type/vec_dot_row + *_generic/_scalar 兜底 + lane 注释 | ✅ |

审计发现的 MISSED（dispatch 级, 供 integrator 排队; 详单+file:line 见 parity/AUDIT_ggml.md §5）:
vec_dot 缺 13 类型（NVFP4/Q1_0/Q2_0/IQ×9/TQ×2, compute.rs:905-911 显式 panic; NVFP4 与 Q1_0/Q2_0 推理侧
此前无记录）; quants.rs:1082 dequantize_row 缺 IQ/TQ dispatch 臂（9 个 IQ 函数已在 quants_k.rs:2135-2393）;
bin_op 量化变体（唯一消费者 cvec 已 documented-skip）。死代码: tensor.rs 的 SquaredMulMat/MulView/
Conv2dDirect 判别值不在 pinned 枚举或无 dispatch; lib.rs:4 文档头 revision 号过时。

**→ 2026-09-27 审计修复轮已全部处置**（详 PARITY.md §AUDIT-ggml-修复轮）: 12 类型 vec_dot lane 已移
并位同参考（vecdot3_ref.bin; TQ×2 为显式 documented-refusal）; dequantize_row 的 9 个 IQ 臂已接
（quants.rs + iq_dispatch_tests）; bin_op 量化 src0 的 skip 理由已落在 compute.rs::forward_bin_op 注释
（ops.cpp:578 add_q_f32, 消费者 cvec）; 死枚举变体保留原因已注释（crates/llama 的名字映射按 exhaustively
match 引用, 归另一组移植 agent 所有）; lib.rs revision 已改 pinned bd4f514db1。

## src/ 顶层审计更正（代理AUDIT-SRC-CORE, 2026-09-27 — 明细 parity/AUDIT_src_core.md; 行号指本文件旧行）

对 src/ 表（行 49-71）的修正（32 文件汇总 = ✅25 + 🟡5 + ⛔1 + 未移 1; 明细见 parity/AUDIT_src_core.md）:

| 旧行 | 原文要点 | 更正 |
|---|---|---|
| 69 | `src/llama-cparams/impl/io/mmap quant ~800 ⏳` | **quant 实为 ✅**（quant.rs 1211 行 + quantize/pipeline.rs, 25/25+19/19 逐字节——与行 96/97 自相矛盾）; cparams→`LLAMA_MAX_SEQ` const + DecodeContext 字段（composed）; io→StateWriter/StateReader; mmap→memmap2（mlock 未移, 无档）。行内唯一真缺口: llama-impl.cpp 的 log 路由 4 函数（:28-62） |
| 70 | `llama-model-saver.cpp ~200 saver.rs ⏳` | 状态（未移）对; 但 `crates/llama/src/saver.rs` **不存在**（幻影引用）, 参考文件 505 行非 ~200。C 消费者 = llama_model_save_to_file（llama.cpp:498）+ 未移的 finetune 示例 + tests |
| 71 | `src/llama-cvntr.cpp 等新文件 ⏳` | **幻影条目**: pinned bd4f514db1 无 `llama-cvntr.cpp`（`ls src/*.h` 证实 32 个 .cpp 之外无此名）。行删除 |
| 60 | 行尾 "DSA 变体后续" | **过期**: DSA/DSA-iswa/MSA/dsv4/lid 均已移（行 37 三批追加 + kv_cache.rs new_dsa/KvLidCache/Dsv4CompState 实物） |
| 61 | "RWKV 记忆未移" | **过期**: 批次 14 已移 RWKV6/7（build_rwkv6/7_time_mix + token_shift 帮手 graph_arch.rs:32799/32811） |
| —（缺行） | `src/llama.cpp`（620 行, 公开 API + 装载分发） | **补行**: 🟡 — version/split_path/split_prefix/chat_apply_template/sampler_chain_default_params/装载五入口 composed 进 Gguf::open+load_model+chat.rs; backend_init/numa/supports_* = CPU-only no-op; MISSED: time_us/print_system_info/flash_attn_type_name/load_mode 文案（显示） |

其余 src 表行（arch 53 / hparams 54 / vocab 55 / unicode 56 / loader 57 / model 58 / graph 59 / context 62 / batch 63 /
sampler 64 / grammar 66 / chat 67 / adapter 68）经逐符号核对**与端口现状一致**, 无矛盾。
