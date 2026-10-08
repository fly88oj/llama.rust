# AUDIT: src/ 顶层 32 个 .cpp 逐文件逐符号迁移审计（pinned bd4f514db1）

审计日期: 2026-09-27。审计人: 代理AUDIT-SRC-CORE。
**同步批次 B 增记（2026-10-01, NEW 基线 def4d406a, 代理SYNC-B-SRCCORE）**: 本审计的 pinned 基线已由同步批次 B 推进到 def4d406a — src/ 顶层 delta 的逐文件落点见 PARITY.md "同步批次 B"。与本审计判定相关的变化: ① llama-batch 的 `llama_batch_ext`/`llama_batch_compat`/`batch_idxs` 已 1:1 落 batch.rs（原 ✅ 判定保持, 新符号全覆盖）; ② llama-context 的状态恢复失败清理（state_clear 族）/`get_causal_attn`/RANK-causal cls/batch_idxs 抽取置换已落 context.rs（原 🟡 的缺口不变 — 训练/opt 符号在新基线仍不移植）; ③ llama-memory-hybrid-idx 维持 ⛔（上游自家 TODO "pending complete reimplementation"; 新增的 kpool 半边 649dcb103 随之不移植, kpool 布局输入 `seq_pos_get` 已备）; ④ llama-model 新增 `llama_prec_policy`（CPU 无 op-param 消费面, 按名寻址保 saver 往返）; ⑤ llama-mmap/llama-impl 的 prefetch/clear_tensor_data 维持不移植（性能面, 端口无 lazy/direct-io 基础设施）。
范围: `/home/jeffrey/llm/llama.cpp-pinned/src/` 的 **32 个顶层 `.cpp`**（`ls src/*.cpp | wc -l` = 32；不含 `src/models/`（已由 AUDIT_models.md 审计））。`unicode-data.cpp` 按任务口径只核 `unicode.cpp` 的导出面（数据表为生成物）。
端口落点: `crates/llama/src/`（model.rs / meta.rs / context.rs / graph.rs / graph_arch.rs / kv_cache.rs / vocab.rs / sampling.rs / grammar.rs / batch.rs / chat.rs / adapter.rs / hparams.rs / arch.rs / quant.rs / unicode.rs）+ `crates/ggml/src/gguf.rs`（split/file 侧）+ 各工具 crate。

## 判定图例

| 码 | 含义 |
|---|---|
| ✅ | 全符号 1:1 移植（或含少量 composed 包装 accessor，逐个点名） |
| 🟡 | 主体移植，明确列出的子符号未移植（有 file:line） |
| ⛔ | documented-skip / documented-⏳：PARITY.md 或 FILE_MAP.md 有明文 |
| ❌ | **MISSED**：无端口符号、无文档记录 |

**汇总: 32 文件 = ✅26 + 🟡5 + ⛔0 + 未移 1**（🟡5 = llama-impl/llama-mmap/llama-model/llama.cpp/llama-context，缺的全是显示/日志/训练符号，见 MISSED 清单 triage；~~⛔1 = llama-memory-hybrid-idx~~ **批次 19 全量移入**（kv_cache.rs HybridIdxCache, glm5/qwen4exp 位比验收）; 未移 1 = llama-model-saver 整档（FILE_MAP:70 有 ⏳ 行，非静默））。符号级总计 ~1,415 个定义（不含 unicode-data 表）。

## MISSED 清单（按 triage）

### P0 — 行为相关（调用即缺）
| 符号 | file:line | triage | 备注 |
|---|---|---|---|
| `llama_model_saver` 全档（`add_kv*`×9 / `add_tensor` / `add_kv_from_model` / `add_tensors_from_model` / `save`×2 / `supports_arch`） | llama-model-saver.cpp:16-503 | **save 路径** | ~~整档未移~~ **→ 已移植 2026-09-28**: `crates/llama/src/saver.rs` 全档 1:1；byte 对照 492,021,632 字节仅 1 字节差（EOT id，vocab.rs:863 已档的扫描序分歧）；参考装载往返 16/16 同。见 PARITY.md §"src-core 审计缺口移植" |
| `llama_context::opt_init` / `opt_epoch_iter` / `opt_epoch` + `llama_set_param`(static) + C API `llama_opt_param_filter_all`/`llama_opt_init`/`llama_opt_epoch` | llama-context.cpp:3466,3482,3532,3643; 4375-4405 | **训练环** | **→ documented-skip 2026-09-28**（PARITY.md §六）: ggml-opt.cpp 依赖 ggml 的 backward/梯度算子面（端口 ggml 无 autodiff 且本轮冻结）+ sched/图复用机件 + 参考消费者为零（finetune 未移、仅测试用） |
| `llama_context::set_abort_callback` + C API `llama_set_abort_callback` | llama-context.cpp:1202 / 3912 | 行为（中断） | **→ 已移植 2026-09-28**: `DecodeContext::set_abort_callback`（图边界协作检查，同 GGML_STATUS_ABORTED 语义；逐节点检查在 ggml 侧为 integrator 项）。测试 `abort_callback_cancels_decode` |

### P1 — 公开 API/行为面较窄
| 符号 | file:line | triage | 备注 |
|---|---|---|---|
| **→ 已移植 2026-09-28**（display.rs `model_print_info` + model.n_elements/n_bytes; banner_parity.sh 68 行逐字节同, 除 EOT 行——vocab 扫描序分歧） | 显示 | 模型装载详情 dump（n_bytes/n_elements 统计随 load_stats 一并缺）。参考 server/CLI 启动横幅用 |
| `llm_type_name`（LLM_TYPES 名表）+ `type_name()`/`desc()`/`desc_str` | llama-model.cpp:868 / 1896 / 1900 | 显示 | **→ 已移植 2026-09-28**（display.rs 全表 + `llm_type_of` 检测, 覆盖表见其文档; llama-bench model_info.rs 同源已有）。端口只有 `arch.name()`（arch.rs）；`model_desc` 字符串（"llama 8B Q4_K"式）无对应物。量化侧只需 `LLM_TYPE_70B` 一个布尔（quant.rs:301-339 已按此口径移植并有注释） |
| **→ 已移植 2026-09-28**（DecodeContext/LlamaModel::memory_breakdown + KvCache::total_size/size_k/v_bytes 含 lid/dsv4; recurrent 半边 size_r/s/p 为 graph_arch 冻结, integrator 项） | 显示 | 显存/内存分解报表（参考 server "load" 横幅、perplexity、fit.cpp 用）。端口无任何消费者 |
| **→ 已移植 2026-09-28**（display.rs `vocab_print_info`, model_print_info 尾部调用同 :2174） | 显示 | 词表 dump |
| **→ 已移植 2026-09-28**（display.rs `loader_print_info` + `guess_ftype` GUESSED 位） | 显示 | 装载器 dump |
| **→ 已移植 2026-09-28**（`llama::time_us` / display::print_system_info / display::flash_attn_type_name） | 显示/杂 | 工具用自己的时钟（llama-bench util::get_time_ns 已验逐字节同输出）；system_info 横幅未移 |
| **→ 已移植 2026-09-28**（SamplerChain::perf_sampler{,_print,_reset} + perf 计数在 apply; perf_context 族同批落 context.rs） | 显示 | 采样计时。perf_context 族已由 llama-bench 自己的计时 composed（该工具 5 种输出格式已验逐字节同）；sampler 半边没有 |
| `llm_arch_all` | llama-arch.cpp:1030 | 仅测试 | （2026-09-28 注: port 的 arch 表即全量 const, 等价可得） 唯一消费者 `tests/test-llama-archs.cpp` |
| `llama_file_version_name` | llama-model-loader.cpp:21 | 内部错误串 | （2026-09-28: display.rs 已移） fver 校验失败时的文案；端口错误路径自述 |
| **→ 已移植 2026-09-28**（impl_log.rs 全族 + LLAMA_LOG_* 宏; quant.rs log_warn/error 已改路由） | 日志路由 | 端口只有 `log_warn`/`log_error`（quant.rs:1205,1209）直出 stderr；日志回调重定向（`--log-file` 族）无对应物。FILE_MAP:80 只记了 common/log.cpp 的旗标面，src 侧这 4 个未记 |
| **→ 已移植 2026-09-28**（mlock.rs; llama-cli `-lm mlock`/`mmap+mlock` 已接） | 内存钉扎 | `--mlock` 工作流；端口 memmap2 无 mlock。无 PARITY 记录（PARITY:748 只记了 load_mode 恒 mmap） |

### 已有文档、本审计核实其准确性
| 符号 | 出处 | 核实结论 |
|---|---|---|
| 后端采样族（`llama_sampler_backend_*` probe/graph/begin/n_nodes/copy_state + 11 个采样器的 `backend_init/apply/set_input/accept/reset` ~40 符号; `llm_graph_input_sampling`+`build_sampling` llama-graph.cpp:1270/:3777; `llama_context::set_sampler` :1275 + `llama_get_sampled_*` 8+ 方法/C API） | PARITY:1041 "base/backend sampling …端口只有 CPU 采样链" | 准确。GPU 采样卸载机制整族未移，端口采样全 CPU |
| `llama_sampler_init_grammar_lazy`/`_lazy_patterns` + `llama_grammar_trigger_pattern::find` + lazy 触发缓冲重放 | grammar.rs:36-41 注释 + sampling.rs:3194-3196 + PARITY:76 | 准确。字段保留、路径不可达（需 std::regex） |
| `llama_adapter_cvec` 全族 + `build_cvec` + `set_adapter_cvec` | FILE_MAP:68 "控制向量 llama_adapter_cvec 未移植" + adapter.rs:23-25 注释 | 准确 |
| T5 decoder 半边（`llm_graph_input_cross_embd`/`build_inp_cross_embd`/`build_attn_inp_cross`/`build_attn(cross)`） | PARITY:378-380（AUDIT_models t5 行） | 准确 |
| `llama_memory_hybrid_idx`（QWEN4EXP 的 QSA 稀疏注意力 + PLE） | graph_arch.rs:23160-23167 + PARITY 批次 11a | 准确。索引器张量按 NOT_REQUIRED 移植、图/记忆类型未移；合法无 compress_ratios/PLE 键的文件已验 |
| dsv4 剩余缺口（压缩 cells 的 seq_add/seq_div 行滚动、ON_DEVICE blob） | FILE_MAP:37 | 准确 |
| `llama_memory_recurrent` 的多序列 cells | FILE_MAP:61 + context.rs:1815,1872-1877 注释 | 准确（单活 cell 组合 + rs_zero 规则 + 快照环；mamba2 blob 与参考逐字节同已验） |
| `-lm/--load-mode` 非 auto/mmap 值 | PARITY:748 | 准确（`llama_load_mode_name/from_str` 随之失效为文案函数） |

## 逐文件审计表

### 1) 数据/小基建（6 文件）

| 文件 | 判定 | 符号核算 |
|---|---|---|
| **llama-cparams.cpp** (1 def) | ✅ | `llama_max_parallel_sequences` → `batch.rs:16 LLAMA_MAX_SEQ` const（composed accessor）。cparams 结构本身溶解进 `DecodeContext` 字段（context.rs:1895-1909 有逐字段 C 出处注释） |
| **llama-io.cpp** (2 defs) | ✅ | `llama_io_write_i::write_string`/`read_i::read_string` → `kv_cache.rs StateWriter/StateReader`（:2512/:2574，write/read u32/i32/u64/tensor + 字符串帧） |
| **llama-impl.cpp** (11 defs) | 🟡 | ported: `replace_all`/`format`（Rust `format!` 等价）、`llama_format_tensor_shape`×2（model.rs:959 `format_shape` + arch.rs:1890）、`gguf_kv_to_str`（adapter.rs:197 的 gguf_kv 全量 map 端 + dflash.rs:293 受限版）。**MISSED**: `llama_log_get/set/internal_v/callback_default` :28-62（见 P1） |
| **llama-arch.cpp** (11 defs + 3 表) | ✅ | 三表 153/153 名字逐字同（AUDIT_models 已核）。函数: `LLM_KV::operator()`→`kv_name`、`LLM_TN_IMPL::str`→`tensor_name{,_xid,_suffix}`、`llm_arch_name`→`name`、`llm_arch_from_string`→`from_name`、`is_recurrent/is_hybrid/is_diffusion/supports_rs_rollback/supports_sm_tensor` 五个谓词同名。`llm_tensor_info_for`→`LlmTensor::layer()`（layer 半；op 半 defer 有注释 arch.rs:1565-1566）。**MISSED(仅测试)**: `llm_arch_all` :1030 |
| **llama-hparams.cpp** (37 defs) | ✅ | **0 missed**。36 个方法 + `is_masked_swa` 全在 hparams.rs:822-1300（set_swa_pattern/set_recr_pattern/is_swa_any/n_head*/n_ff*/n_expert_used*/n_gqa/n_rot/n_embd_*/is_*_variable/*_max/n_embd_r/n_embd_s/is_recr/ple_conv_state/is_ple/n_pos_per_embd/is_swa/is_mla/is_indexer_full/*_mla/has_kv/has_rope/n_layer/use_mrope） |
| **llama-mmap.cpp** (20 defs) | 🟡 | `llama_mmap`（pimpl/NUMA prefault/ranges_complement/unmap_fragment）→ memmap2::Mmap 组合（model.rs/gguf.rs `Arc<Mmap>` 零拷贝；split 多分片路由 = weights_map/files[] 语义，PARITY:5370 已记）。`llama_file` → std::fs + mmap 读取（read_u32/write_raw 语义在 gguf 解析/StateWriter）。`llama_path_max` → OS PATH_MAX。**MISSED**: `llama_mlock` :815（P1，无文档） |

### 2) 词法/交互（6 文件）

| 文件 | 判定 | 符号核算 |
|---|---|---|
| **llama-batch.cpp** (19 defs) | ✅ | **0 missed**。`llama_batch_allocr` 全 16 方法（init/ubatch_reserve/get_batch/get_n_tokens/get_n_outputs/get_n_used/get_out_ids/seq_pos_min/seq_pos_max/split_reset/split_simple/split_equal/split_seq/clear/ubatch_add/ubatch_print）+ `llama_batch_get_one/init/free` = batch.rs（`LlamaBatch::get_one/add/clear` + `BatchAllocr`，10 默认测试） |
| **llama-chat.cpp** (5 defs) | ✅ | **0 missed**。`trim`(内联)/`llm_chat_template_from_str`/`llm_chat_detect_template`/`llm_chat_apply_template`/`llama_chat_builtin_templates` → chat.rs（apply/apply_named/apply_str + builtin_templates:266 + 检测; 43/43 模板测试 + minja 引擎 131/131 位同） |
| **llama-vocab.cpp** (135 defs) | ✅ | impl::load（1939，含 1258 rwkv unescape）→ `Vocab::load`+`load_inner`；7 个 tokenizer（spm/bpe/wpm/ugm/rwkv/plamo2/test）→ `enum Tokenizer` + 各 struct（vocab.rs:228）；`tokenizer_st_partition`/`init_tokenizer`/`token_to_piece_for_cache`/`escape_whitespace`/`unescape_whitespace`/`decode_text`/`tokenize`/`token_to_piece`×3/`detokenize` 全在；C API ~75 个 accessor → Vocab 方法/`pub` 字段（special token 族、fim 族、flags 族、suppress、max_token_len、precompiled_charsmap、find_bpe_rank、get_bpe_merges、text_to_token、byte_to_token）。`token_prefix/middle/suffix` = fim 别名（fim_*_id 字段）。`llama_vocab_cls` = deprecated token_bos 别名。**MISSED**: `print_info`×2（显示，P1） |
| **unicode.cpp** (26 defs) | ✅ | 12 个导出全在: `unicode_len_utf8`/`cpt_from_utf8`/`cpt_to_utf8`/`cpts_normalize_nfd`/`cpts_from_utf8`/`cpt_flags_from_cpt`/`cpt_flags_from_utf8`/`byte_to_utf8`/`utf8_to_byte`/`tolower`/`cpt_is_han`/`regex_split`（unicode.rs:110-1396）。static 族（cpt_flags_array/两 map/byte_encoding_process/7 个 regex_split_custom/stl 版）同名对应。**0 missed** |
| **unicode-data.cpp** (纯数据) | ✅ | unicode_data.rs 生成物（unicode_gen.py），范围外按约定只核导出面 |
| **llama-grammar.cpp** (44 defs) | ✅ | 解析器（get_symbol_id/generate_symbol_id/add_rule/parse_alternates/parse_sequence/parse_rule/parse/print/c_rules）+ 匹配器（is_end_of_sequence/match_char/match_partial_char/advance_stack/reject_candidates_for_stack/reject_candidates/detect_left_recursion/accept_chr/accept）+ init/apply/accept/accept_str/accept_token/free + decode_utf8×2 + parse_*×8 全在 grammar.rs（8 gbnf 逐元素 + 63 步 stacks 位同 + piece 表 151936 全同）。`clone_impl` → `#[derive(Clone)]`（有注释 :40-41）。⛔ 有档: `trigger_pattern::find` :378-409 + lazy 触发重放 :1403-1443（std::regex） |
| **llama-sampler.cpp** (191 defs) | ✅ | 20 个采样器类型（empty/greedy/dist/top_k/top_p/min_p/typical/temp/temp_ext/xtc/top_n_sigma/mirostat/mirostat_v2/penalties/dry/adaptive_p/logit_bias/infill/grammar/chain）各 name/apply/accept/reset/free/clone → Rust trait `Sampler` + Drop/构造器（vtable 样板 ~60 符号 composed 进 trait 语义）。基础设施: `llama_sampler_init`(:355 vtable 构造→trait object)、partial_sort×2→`std_partial_sort_by`、`llama_sample_dist`→`sample_dist`、`llama_log_softmax`→内联于 typical、`llama_sampler_sample`→`SamplerChain::sample(logits)`（ctx+idx 形态 composed: 调用方持 logits）、`llama_sampler_get_seed`→trait `get_seed`、`get_overlapping_token_sequences`→dry 内部。`llama_sampler_copy`(:4299)→composed（server 每槽重建链）。⛔ 有档: backend 采样族 ~40 符号 + lazy grammar 2 个。**MISSED**: `llama_perf_sampler`×3（显示，P1） |
| **llama-adapter.cpp** (15 defs) | ✅ | `llama_adapter_cvec::tensor_for/apply_to/init/apply` ⛔ 有档（FILE_MAP:68 + adapter.rs:23-25）。其余全移: `get_weight`/`llama_adapter_lora_init_impl`(151)/`init`/`init_from_file_ptr`/`meta_val_str`/`meta_count`/`meta_key_by_index`/`meta_val_str_by_index`/`free` + aLoRA（`adapter.alora.invocation_tokens` 装载 :224-241 → `alora_invocation_tokens` pub 字段；C accessor×2 = 字段读取）。注意: aLoRA 的**运行时激活门**（server `lora_all_alora`，server-common.cpp:150）随 `/lora-adapters` 端点未移而不可达（FILE_MAP:102 已档该端点） |

### 3) 模型装载（4 文件）

| 文件 | 判定 | 符号核算 |
|---|---|---|
| **llama-model-loader.cpp** (23 defs) | ✅ | `create_tensor`(:1116)→model.rs:1027（flags/reshape/dup/partial 全语义 + n_created 计数）、`check_tensor_dims`(:880)→:975、`done_getting_tensors`(:1392)→:1125、`llama_ftype_name`→quantize 工具表（25/25 位同验证）、`llama_get_list_splits`→gguf.rs split.* 装配、`get_arch(_name)`/`get_weight`/`require_weight`/`get_tensor_meta`/`require_tensor_meta`→`Gguf::arch`/`find_tensor` 组合。composed: `init_mappings`/`get_mapping_range`/`unmap_weight`/`load_data_range`/`load_all_data`/`lazy_read::buft/add` → 恒 mmap 零拷贝（PARITY:748 有档）+ 外置存储（eagle.rs/dflash.rs mini-loader 同款）。`weight_buft_supported`/`select_weight_buft`(:927/:1067) ⛔ CPU-only（GPU buffer-type 选择）。**MISSED**: `print_info` :1803（显示）、`llama_file_version_name` :21（错误串） |
| **llama-model.cpp** (90 defs) | 🟡 | 装载主干全移: `load_hparams`(:1231 共享段)→meta.rs:409 `load_hparams_generic`（逐 KV/断言/默认值/HUNYUAN_VL/WAVTOKENIZER_DEC 特例核对）；`load_arch_hparams` 各 arch 臂在 meta.rs（覆盖面归 AUDIT_models）；`load_vocab`/`load_tensors`/`create_tensor`×2/`create_tensor_qkv`/`create_tensor_gate_up_exps`/`load_swa_pattern` → model.rs ModelLoader + load_arch_tensors；`llama_model_mapping`/`create`(:43/:353/:366)→load_model + arch_tensors_support 分发；`llama_meta_device_get_split_state`(:375)→eagle.rs/dflash.rs 外置 mmap（PARITY:3794）；`create_memory`(:2274)→**composed** 进 context.rs new_impl（swa/dsa/dsa_iswa/dsv4/msa/plain 分支 + hybrid recurrent 半，context.rs:2263-2400，:2741 注明 dispatched like create_memory）；`build_graph`(:2762)→ForwardWeights+graph_arch builder 分发；`llama_model_rope_type`→meta.rs:2042；`get_rope_freq_base/scale/factors`→context.rs:1182 起逐层解析；`llama_model_target_layer_ids`×2→eagle3 target_layers（PARITY:3793）；`llama_model_get_tok_embd`(:3326)→mtmd.rs:1158 `read_tok_embd_row`（消费侧）；C API ~35 个 accessor（n_ctx_train/n_embd*/n_layer*/n_head*/n_swa/n_cls_out/cls_label/meta_*/ftype/size/n_params/has_encoder/has_decoder/decoder_start_token/is_recurrent/is_hybrid/is_diffusion/get_vocab/free*）→ LlamaModel pub 字段/方法（n_cls_out/cls_labels 字段在）。GPU 族（`dev_layer`/`dev_output`/`select_buft`/`buft_supported`/`make_cpu_buft_list`/`make_gpu_buft_list`/`tensor_split`/`n_gpu_layers`/`split_mode`/`n_devices`/`get_device`/`has_tensor_overrides`/`get_tensor`(:2239 按名取张量→by_name map)）⛔ CPU-only 或 composed。**MISSED**: `print_info`/`load_stats`/`llm_type_name`/`type_name`/`desc`/`memory_breakdown`/`llama_rope_scaling_type_name`(:1024)/`llama_expert_gating_func_name`(:1008)（后两者唯二消费者是 print_info 与 model-saver，均显示/save 路径）。`llm_ffn_op_type_from_string`(:1052) 唯一消费者是 modern-bert.cpp:20（该文件 ⛔ documented-skip, PARITY:381）——枚举 `LlmFfnOpType` 已移（hparams.rs:118），字符串解析未移（一致跳过） |
| **llama-model-saver.cpp** (15 defs) | **未移** | **整档未移**（见 P0 表）。FILE_MAP:70 有 ⏳ 行（非静默）但指向不存在的 saver.rs 且行数错 |
| **llama-quant.cpp** (24 defs) | ✅ | **0 missed（语义级）**。`llama_model_quantize_impl`→quantize pipeline.rs（19/19+25/25 逐字节验证）；`zeros/remap_layer/remap_imatrix`→pipeline.rs 前处理（`--prune-layers` 缺口 FILE_MAP:96 已档）; `tensor_name_match_*`/`tensor_get_category`/`category_is_attn_v`/`tensor_allows_quantization`/`tensor_type_fallback`/`llama_tensor_get_type_impl`/`llama_tensor_get_type`/`tensor_requires_imatrix`/`init_quantize_state_counters`/`llama_ftype_get_default_type`→quant.rs 同名；`llama_tensor_dequantize_impl`/`llama_tensor_quantize_impl`→quantize 工具（imatrix 19/19）；`llama_model_quantize_default_params`→`QuantizeParams::new`；llama-ext.h 三件套: `llama_quant_init/free`→`QuantizeState::new`/Drop、`llama_quant_model_from_metadata`→`QuantModelInfo::from_hparams`（:312，LLM_TYPE_70B 推断同构）、`llama_quant_tensor_allows_quantization`→pub fn :527、`llama_quant_compute_types`→composed（reset_counters+init_counters+get_type 三段皆 pub，调用方拼装） |
| **llama.cpp** (28 defs) | 🟡 | ported: `llama_version`→lib.rs `version()`、`llama_split_path`/`llama_split_prefix`→gguf.rs:26/:33（含分片装配消费）、`llama_chat_apply_template`→chat.rs `apply_str/apply_str_simple`（jinja）+ builtin 路径 `apply_named`、`llama_sampler_chain_default_params`→`SamplingParams` 默认+`default_chain`、装载五入口（`llama_model_load_from_file`/`_from_splits`/`_from_file_ptr`/deprecated `llama_load_model_from_file`/static `llama_model_load`+`_from_file_impl`+`llama_prepare_model_devices`）→composed 进 `Gguf::open`+`load_model`（单/多分片/FILE* 语义合并）；`llama_supports_mmap/mlock/gpu_offload/rpc`+`llama_max_devices`+`llama_max_tensor_buft_overrides`→CPU-only 常量语义（mmap=true 其余 false）composed；`llama_backend_init/free`/`llama_numa_init`→no-op（单 CPU 后端常驻，rayon）。**MISSED**: `llama_time_us`/`llama_print_system_info`/`llama_flash_attn_type_name`（P1）；`llama_load_mode_name/from_str`（PARITY:748 的恒-mmap 设计使其为死文案） |

### 4) 图与上下文（2 文件）

| 文件 | 判定 | 符号核算 |
|---|---|---|
| **llama-graph.cpp** (123 defs) | ✅ | 26 个 `llm_graph_input_*` 类（embd/embd_h/pos/attn_temp/pos_bucket/pos_bucket_kv/out_ids/mean/cls/rs/cross_embd/attn_no_cache/attn_kv/attn_k/attn_kv_msa/attn_k_dsa/attn_k_dsa_iswa/attn_kv_iswa/attn_k_iswa/dsv4_raw/dsv4/attn_cross/mem_hybrid/mem_hybrid_k/mem_hybrid_iswa/sampling/k_shift）→ composed 进 `DecodeInputs`（graph.rs:64: tokens/pos/kq_mask/row_idx/out_ids）+ `EncodeInputs`（pos_bucket/mean/cls/out_ids）+ MSA 步进输入（context.rs:5298-5348）+ dsv4 comp inputs（kv_cache.rs:912 `Dsv4CompInputs`）+ MTP 的 embd_h（graph_arch.rs:14456）+ recurrent r/s/p plane（graph_arch `RecurrentState`）。`set_input` 语义 = mask 填充族 `fill_kq_mask{,_alibi,_f16,...}`（graph.rs:391-505，对应 :1557-1788 的 5 个模板实例）。helpers: `build_norm`→`build_norm`+`build_norm_rms`+gemma 变体; `build_qkv`×2→`qkv_fused`/`qkv_separate_parts`(+clamp/plain 变体); `build_ffn`→`build_ffn_silu_par/gelu_par/gelu_seq/relu_sqr/swiglu_seq/...`; `build_moe_ffn`×2→`build_moe_ffn_{silu,gelu,gemma4,deepseek2,dsv4,relu_sqr,relu_probs,silu_gu,silu_probs,silu_swigluoai,gpt_oss,deepseek2_like}`; `build_lora_mm(_id)`→adapter.rs:596-607; `build_cvec` ⛔; `build_attn_mha`→`attn_kv_cached`+`flash_attn_core(_sinks)`（sinks/softcap/alibi 逐点有 C 行号注释）; `build_attn` 8 个重载→`attn_kv_cached`/`attn_no_cache(_fa)`/`attn_k_cached_mla(_iswa)`/`attn_dsv4`/`build_attn_msa_fa`; `build_attn_inp_*` 10 个→对应 cache view 构造; `build_inp_*` 12 个→DecodeInputs/EncodeInputs 构造（attn_scale :2465→graph_arch.rs:26107 断言 0 路径）; `build_pos_bias`/`relative_position_bucket`→:3740/:3789（433/433 表同）; `build_pooling`→:3873（RANK panic 有档）; `build_dense_out`→`out_rows`+mul_mat 组合（graph.rs:84-105 有值恒等证明注释）; `build_rs`×2/`build_rs_inp`→内联 reshape（:32706 注释 s_copy 退化）; `build_rwkv_token_shift_load/store`→:32799/:32811; `build_inp_mem_hybrid*`→RecurrentState plane 直读; `llm_graph_result`（get_max_nodes/reset/set_inputs/set_outputs/can_reuse/add_input/add_fused_node/set_params）+ `resolve_fused_ops`（context 侧）→composed 为端口"每步重建图 + 组合算子"设计（FILE_MAP:39 有档; 融合算子 integrator 项 PARITY:85 有档）。⛔ 有档: `llm_graph_input_sampling`+`build_sampling`（backend 采样）、cross_embd 半边（t5 decoder）。**0 新 missed** |
| **llama-context.cpp** (168 defs) | 🟡 | `llama_context` ~60 方法 → `DecodeContext`（decode/decode_all/decode_batch/encode/encode_eagle3/new_*/set_embeddings(_nextn/_layer_inp)/set_causal_attn/seq_rm/seq_cp/seq_add/seq_div/seq_keep/seq_pos_min/max/n_ctx/can_seq_rm/get_can_shift/logits_ith/get_embeddings(_nextn/_layer_inp)/state_{get,set}_data/state_seq_{get,set}_data/recurrent_state_live_seq/reset_sequence）+ `EncoderContext`（encode/build/run，BERT/T5 e2e 位精确已验）。`process_ubatch`/`encode`/`decode` 主循环→decode 驱动（多序列 ubatch 切分/KQ mask/out_ids/selection-sort reorder = output_reorder :2293 内联 :5466/:5630）; `output_reserve`/`output_resolve_row`→内联（:1924/:2220 注释）; `extract_layer_inputs`→eagle3 layer_inp 抽头（PARITY:3793）; `state_write/read_data`/`state_seq_write/read_data`/io_magic 帧→StateWriter/StateReader + `state_seq_{get,set}_data`（dsv4/dsa/msa/recurrent 四类 blob 与参考逐字节同，FILE_MAP:37）; 文件态 4 个（state_load/save_file、state_seq_load/save_file :3215-3348）→engine.rs /slots save/restore 消费侧组合（:1092 注明 llama_state_seq_save_file 格式）; `llama_init_from_model`/`default_params`→DecodeContext::new*+工具参数结构; memory C API 10 个（llama_memory_clear/seq_*/can_shift）→DecodeContext 同名方法; getter C API ~45 个→字段/方法。⛔ 有档: `set_sampler`+`get_sampled_*` 8 方法+C API（backend 采样, PARITY:1041）、`set_adapters_lora`/`adapters_lora_are_same`→adapter.rs 线程局部（FILE_MAP:68）、`set_adapter_cvec`、sched_reserve/graph_reserve/graph_max_nodes/get_gf_res_*/graph_params/graph_compute/graph_get_cb/copy_tensor_async_rows/needs_raw_logits/ubatch_prepare_reserve/ctx_type_to_graph_type/llama_graph_n_input_tensors（sched/复用机件→每步重建设计）。**MISSED**: `opt_init/opt_epoch_iter/opt_epoch/llama_set_param`+`llama_opt_*` C API（P0 训练环）、`memory_breakdown`+`llama_get_memory_breakdown`（P1）、`set_warmup`（P1, 参考内无消费者）、`set_abort_callback`（P0）、`perf_get_data/perf_reset`+`llama_perf_context*`（composed 进 llama-bench 自计时，输出已验; `perf_context_print` 无对应）、`attach/detach_threadpool`/`set_n_threads`（rayon/Team 组合）、`llama_get_ctx_other`（composed: dflash/eagle3 独立上下文） |

### 5) KV cache 与 memory（11 文件）

| 文件 | 判定 | 符号核算 |
|---|---|---|
| **llama-kv-cache.cpp** (86 defs) | ✅ | `ggml_is_power_of_2`/`ggml_gen_hadamard`→kv_cache.rs:1221/:1247; cells/slot 主干（clear/seq_rm/seq_cp/seq_keep/seq_add/seq_div/seq_pos_min/max/prepare/update/find_slot/apply_ubatch）+ get_k/get_v/cpy_k/cpy_v/build_input_{k,v}_idxs/build_input_{k,v}_rot/set_input_{k,v}_idxs/set_input_k_shift/set_input_kq_mask（5 模板实例→fill 族）/set_input_pos_bucket/set_input_{k,v}_rot + context 包装 22 个 → kv_cache.rs KvCache + graph.rs fill 族（iswa 双 cache/mask、位置搬移族+K-shift、16/16 长上下文与 mamba2 blob 位同均已验）。state 5 件（state_write/read/state_read_sinfo/state_write_meta/state_write_data/state_read_meta/state_read_data）→`state_seq_write/read`（raw+swa 半, has_kv 过滤修正 RSSTATE）。`get_prev_tokens`/`has_cell_ext`(:1838/:1833)→未移但唯一消费者是 qwen4exp PLE（本身 ⛔，一致跳过）。`memory_breakdown`/`total_size`/`size_k_bytes`/`size_v_bytes`→MISSED（显示, P1）。trivial getter（get_size/get_n_stream/get_has_shift/type_k/type_v/get_layer_ids/get_k_storage/get_cells/get_n_kv/get_can_shift）→字段/内联 |
| **llama-kv-cache-dsa.cpp** (24 defs) | ✅ | 复合体（mla+lid 两 llama_kv_cache）→kv_cache.rs `new_dsa`+`KvLidCache`（cells/size/head/layers/n_embd_head_k/hadamard/nrot）+ `assign_lid`/`apply_ubatch_lid`/`get_k_lid` + state 半（`lid_state_{write,read}`，minimax-m3/deepseek32 blob 逐字节同已验）。context 8 个方法→DecodeContext 步进。**0 missed** |
| **llama-kv-cache-dsa-iswa.cpp** (24 defs) | ✅ | dots3note 组装：iswa 对 + lid 池（context.rs:2364-2385 注释 llama-kv-cache-dsa-iswa.cpp:17-81 语义; lid 与 base 逐 cell 同步）。state 顺序（:236-242）已按 dflash2 报告移植。**0 missed** |
| **llama-kv-cache-iswa.cpp** (24 defs) | ✅ | `KvSwaCache`/`new_swa`/`layer_is_swa`/`find_slot_swa`/`apply_ubatch_swa`（purge 语义）/`n_kv_swa`/`swa_cache`。state→raw+swa 双半 `state_seq_write/read`。**0 missed** |
| **llama-kv-cache-msa.cpp** (27 defs) | ✅ | `msa` 复合（base+idx）+ `msa_state_seq_{write,read}`（context.rs:2899/:2963）+ 步进输入 `set_input_cell_pos/pos_slot/pos_mask`（:275/:304/:348→context.rs:5298-5348，MSA 步进输入逐条注释）。**0 missed** |
| **llama-kv-cache-dsv4.cpp** (104 defs) | ✅ | static 22 个→kv_cache.rs（dsv4_comp_size/stream_offset/state_n_used_k_rows/src|dst_stream_range/write|read_tensor_streams/write|read_k_cache/plan_positions/build_comp_plan→`Dsv4Plan::build`/build_seq…）; `llama_dsv4_comp_state` 15 方法→`Dsv4CompState`（clear/seq_cp/apply_copies→snapshot_shift_seq/restore_seq/get_*/state_write/read/get_kv*/cpy_*）; `llama_kv_cache_dsv4` 30 方法→`new_dsv4`/`clear_dsv4(_seq)`/`seq_cp_dsv4`/`reset_rs_idx_for_ubatches`/raw iswa 对+三压缩 cache（FILE_MAP:37 的 DSV4RS/NGRAM/DSV4STATE 三批验证）; context 30 方法→`Dsv4Step`/`Dsv4Cache`。⛔ 有档: 压缩 cells 的 seq_add/seq_div 行滚动、ON_DEVICE blob（FILE_MAP:37 尾注） |
| **llama-memory.cpp** (2 defs) | ✅ | `llama_memory_status_combine`/`is_fail`→composed 进 `Result<>` 错误路径（无独立符号，语义溶解）。虚接口 `llama_memory_i` 本身→具体类型分发 |
| **llama-memory-hybrid.cpp** (23 defs) | ✅ | attn 半+recurrent 半复合 → DecodeContext 的 kv + `RecurrentState` 组合（filter_attn/filter_recr 逐 arch 判定在 context.rs:1014-1096 注释带 C 行号）。state 半→`state_write_attn_half`+recurrent 半（mamba2 blob 逐字节同已验）。**0 missed** |
| **llama-memory-hybrid-iswa.cpp** (23 defs) | ✅ | 同上 iswa 变体（context.rs:1773-1777 注释 :195-203）。**0 missed** |
| **llama-memory-hybrid-idx.cpp** | **✅ (批次 19, 2026-10-02)** | 全量移入 crates/llama/src/kv_cache.rs `HybridIdxCache`（MSA 先例形态: 索引器 = 注意力 cells 之上的旁侧缓冲）: 构造/hparams_idx（MQA 单头 ×(kpool>0?3:1) 宽/MLA 伪装 K-only）、mem_idx_stale 族、kpool_layout(+update)、kpool_state(+build_sizes/build_state/pad/cur)、set_input_kpool、set_input_qsa（ns=1 特化, mrope-2D ranked 路径端口无输入面）、seq_* 钩子、state_write/read（锁步布局校验）/state_drop——context.rs 记忆选择（glm5-next/qwen4exp）+ Glm5KpoolStep/Qwen4QsaStep 逐步输入; 验收: glm5-next 与 qwen4exp(QSA+PLE) 合成文件各 13 图节点流 vs NEW 参考位同 0 分歧（PARITY 批次 19） |
| **llama-memory-recurrent.cpp** (38 defs) | ✅ | clear/seq_rm/seq_cp/seq_keep/seq_add/seq_div/seq_pos_min/max/set_rs_idx/prepare/find_slot/init_* → `RecurrentState`+单活 cell+快照环（`recurrent_seq`/`RecurrentSnapshots`，rs_zero 规则; find_slot 的 per-seq cells 简化为单活+PARITY RSSTATE 有档）; state 6 件→`recurrent_state_seq_{write,read}`（:3119/:3196，blob 逐字节同已验）; context 13 方法→DecodeContext。`memory_breakdown`/`total_size`/`size_r/s/p_bytes`→MISSED（显示, P1） |

## FILE_MAP.md 对照结论（矛盾/过期）

1. **行 69 `src/llama-cparams/impl/io/mmap quant ~800 对应小模块 ⏳` — 严重过期**。quant 实为 ✅（quant.rs 1211 行 + pipeline.rs，25/25+19/19 逐字节验证，行 96/97 自己也是这么说的——同表自相矛盾）；cparams/impl/io 已 composed（见上表）；mmap 已 composed 为 memmap2。该行应整体重写。
2. **行 70 `llama-model-saver.cpp ~200 crates/llama/src/saver.rs ⏳`** — 状态对（未移），但 `saver.rs` 不存在（幻影文件引用）、行数错（505）。
3. **行 71 `src/llama-cvntr.cpp 等新文件 ⏳ 逐个补` — 幻影条目**。pinned 参考无 `llama-cvntr.cpp`（`ls src/*.h` 证实）。该行应删。
4. **行 60 kv-cache 行尾 "DSA 变体后续" — 过期**。DSA/DSA-iswa/MSA/dsv4/lid 均已移（行 37 的三批追加记录 + 本审计核实）；行 60 与行 37 内部矛盾。
5. **`src/llama.cpp`（620 行公开 API 文件）在 FILE_MAP src 表中没有自己的行** —— 审计补上（判定 🟡，见逐文件表）。
6. 行 61 "RWKV 记忆未移" — 过期：批次 14 已移 RWKV6/7（AUDIT_models 行 29-31 + token_shift 帮手 graph_arch.rs:32799）。

## 抽检记录（验证口径）

每主文件 ≥10 符号双树 grep 核对（摘）: hparams 36/36 方法名对齐; batch 19/19; arch 11 函数+表; vocab load/tokenize/detokenize/token_to_piece×3/special 族/fim 族/bpe_ranks/charsmap; unicode 12 导出; grammar parse_*×8/match_*/reject_*/detect_left_recursion/accept 族; sampler 20 类型+init_*(20); adapter meta_*/init/free/alora; quant 全 static 名; model-loader create_tensor/check_dims/done_getting; kv-cache seq_*/find_slot/apply_ubatch/state_*; dsv4 Dsv4Plan/Dsv4CompState 方法; graph 26 输入类映射+helper 族; context decode/encode/state_*/seq_*; llama.cpp split_path/prefix/version/chat_apply。计数以 `awk` 汇总 /tmp/cpp_symbols.txt（1415 行定义记录）。

## 结论

32 文件中 **31 个的行为语义面完整**（✅25 + 🟡5 的 trunk），1 个 ⛔ 有档（hybrid-idx，另有 backend 采样/cvec/t5-decoder 半边作为子符号 ⛔，PARITY/代码注释双录），1 个整档未移（model-saver）。真正的缺口集中在:
- **save 路径**（model-saver 整档）;
- **训练路径**（opt_epoch 族，无档）;
- **显示/报表面**（print_info/desc/llm_type_name/memory_breakdown/perf_sampler/size getters，无档）;
- **日志路由与 mlock**（llama_log_set 族、llama_mlock，无档）。

其中只有 opt_epoch 族与 model-saver 是"如果有人要写对应工具会直接撞上"的；其余全部是终端输出层面的缺失。

---

## 修复轮记录（2026-09-28, 代理SAVE-DISPLAY）

上表 MISSED 清单已全部处置（P0×3 + P1 全部移植或 documented-skip），详见 PARITY.md §"src-core 审计缺口移植"。
新文件: `crates/llama/src/{saver,display,impl_log,mlock}.rs`; 增量: model.rs(name/classifier_labels/ftype/load_stats
字段)、meta.rs(read_classifier_labels)、context.rs(abort/perf/memory_breakdown)、kv_cache.rs(total_size/size_k/v_bytes)、
sampling.rs(perf_sampler 族)、tools/llama-cli(banner 走新面 + `-lm/--load-mode`)。测试: saver_e2e.rs、
abort_callback_cancels_decode、display/impl_log/mlock 单元组; parity/{ref_model_saver.c,gen_model_saver_ref.sh,banner_parity.sh}。
锚点不回归: qwen2.5 16/16 ×2FA。
