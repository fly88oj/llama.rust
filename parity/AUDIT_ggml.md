# AUDIT: `ggml/src/` CPU 相关子集 — 逐文件逐符号迁移核对

- **审计基线**: llama.cpp pinned `bd4f514db1`（`/home/jeffrey/llm/llama.cpp-pinned`），对照 Rust 端口
  `/home/jeffrey/works/personal/github/fly88oj/llama.rust`（只读审计，未改任何代码）。
- **日期**: 2026-09-27（审计 agent）。范围: `ggml/src/` 顶层 + `ggml-cpu/` + 头文件；GPU 后端目录
  （cuda/hip/sycl/vulkan/metal/cann/musa/opencl/openvino/rpc/webgpu/virtgpu/zdnn/zendnn/et/hexagon/blas）
  不在 CPU 口径内（COVERAGE.md 的 ⛔ 规则）。
- **分类**: ✅ported · 🔗composed（等价构造，给出构造点）· 📄documented-skip（PARITY.md/FILE_MAP.md/
  COVERAGE.md 已记录）· **MISSED（无对应且无记录）**。
- 复验锚点: 本文所有 file:line 均为两侧实际行号，可直接 grep（见 §6 抽查清单）。

---

## 0. 结论摘要

| 维度 | 数字 |
|---|---|
| GGML_OP_* 枚举（NONE..GLU, COUNT=101） | **57 有 Rust 内核** · **6 composed** · **37 documented-skip（llama 层零调用）** · **0 个"已移植架构会用而缺失"的 op** |
| ggml_unary_op（22 个） | 11 ported（tanh/sigmoid/relu/elu/gelu/gelu_erf/silu/exp/softplus/round/xielu）· 11 skip（abs/sgn/neg/step/gelu_quick/hardswish/hardsigmoid/expm1/floor/ceil/trunc，全部零 llama 层调用或已归入 PLE/后端采样/llava 的 documented-skip） |
| ggml_glu_op（7 个） | 7/7 全 ported（REGLU/GEGLU/SWIGLU/SWIGLU_OAI/GEGLU_ERF/GEGLU_QUICK/SWIGLU_CLAMP；GEGLU_ERF/GEGLU_QUICK 走 unary gelu_erf/gelu——llama 层只用 split 变体） |
| type_traits（ggml.c:632） | 43 类型 blck/size/is_quantized/name 全对（types.rs 测试钉死）；`to_float`=quants::dequantize_row 差 11 个 IQ/TQ dispatch 臂（内核函数已存在，见 §4/§5-A） |
| type_traits_cpu（ggml-cpu.c:215） | vec_dot/vec_dot_type 覆盖 16 类型（F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K-Q6K/MXFP4）+ Q8K from_float；**缺 13 类型（Q1_0/Q2_0/NVFP4/IQ×9/TQ×2）**；nrows 恒 1 == x86 参考口径 ✓ |
| MISSED（真实缺口） | **3 项 dispatch 级**（§5-A，均"内核在、接线无"或"量化侧有、推理侧无"）+ 4 项文档级（§5-B） |

---

## 1. 顶层文件清单（`ggml/src/`）

| 文件 | 行数 | Rust 对应 | 状态 | 符号级结论 |
|---|--|--|---|---|---|
| ggml.c | 8167 | types.rs + tensor.rs + graph.rs + ops.rs + quants*.rs | 🔨→✅ | 见 §1.1/§3/§4 |
| ggml.cpp | 38 | lib.rs（Rust panic 机制替代 terminate handler） | 📄 | cosmetic |
| ggml-common.h | 1911 | blocks.rs | ✅ | **0 missed**：26 个 block 结构体 + QK 常量全 1:1（含 QK1_0=128/QK2_0=64/QK_NVFP4_SUB=16）；查找表 kvalues_iq4nl/kvalues_fp4 在 quants_k.rs:647 / quants.rs:658，IQ2/IQ3/IQ1 网格表在 parity/iq_tables.rs（ harness 侧，与"IQ vec_dot 未移"政策一致） |
| ggml-impl.h | ~850 | 分散（fp16 转换 ops.rs:239/266、softplus ssm.rs:22、GGML_PAD 等） | ✅ | softplus/gelu 表/exp 多项式位级复刻（PARITY:32） |
| ggml-quants.c | 5638 | quants.rs + quants_k.rs + quants_impl.rs | ✅ | dequant 全类型位精确（quants_ref*.bin 三档）；quantize: 简单型+Q1_0/Q2_0+Q2K..Q6K+TQ? **无 TQ 量化器**（TQ1_0/TQ2_0 from_float_ref 未移——ternary 模型不可造不可跑，与"IQ 家族未移"同类，PARITY:1227 只点名 IQ1/IQ2/IQ3）；IQ4_NL/IQ4_XS 量化器在 quants_impl.rs:952-1160（imatrix 侧，19/19 ftype 与参考逐字节） |
| ggml-alloc.c | 1249 | tensor.rs::Context arena + reset_graph_to | 📄 | CPU-only 简化（COVERAGE:78"arena 分配 等价简化"）；gallocr 的复用规划数值不可观测 |
| ggml-backend.cpp | 2513 | compute.rs + llama/context.rs | 📄 | CPU-only；行为面（eval_callback ggml-backend.cpp:2045 → compute.rs:229 set_eval_callback、buffer_clear=arena 零初始化、cpu device description → util::cpu_info）已核（PARITY:680/1141-1196） |
| ggml-backend-meta.cpp | 2517 | — | 📄 | 多 GPU 张量切分元后端（llama-model.cpp:375 消费）；COVERAGE:42 ⏳ 未映射，CPU 单后端不可达 |
| ggml-backend-reg.cpp / -dl.cpp | 18.5k+1k | — | 📄 | 后端注册/dlopen 动态加载；端口无 GPU 后端（PARITY:680 backends 恒 "CPU"） |
| ggml-threading.cpp/.h | 250+198 | compute.rs::Team | ✅ | critical_section → Rust 互斥/spin Team；ggml_barrier 自旋语义复刻（PARITY:125/139/174） |
| ggml-opt.cpp | 1094 | — | 📄(弱) | 训练 API；llama 层零调用（CROSS_ENTROPY_LOSS/OPT_STEP_* 只进 opt）。**COVERAGE 无行**（1094 行未入 >1500 行清单）——见 §5-B |
| gguf.cpp | 1714 | gguf.rs + gguf_write.rs | ✅ | 见 §1.2 |
| ggml.h | 3022 | types.rs/tensor.rs | ✅ | 枚举判别值 = GGUF wire 值（类型表 §4）；op 判别值 internal-only 合法（不落盘） |
| ggml-feats.h | 4.8k | —（编译期特性开关） | 📄 | aarch64/s390 特性探测；x86 侧等价 = simd_x86.rs 的 is_x86_feature_detected（运行期） |

### 1.1 ggml.c 符号级（builder 面）

端口 `ops.rs`（2646 行）+ `tensor.rs`。401 个函数定义中推理相关 builder ~120 个；
下表只列**有语义内容**的组（`*_inplace`/`*_impl` 与同名同核，不重复计）：

| ggml.c 组 | 端口 | 状态 |
|---|---|---|
| dup/cpy/cont(1-4d)/cast/set_zero | `Context::{dup,cpy,cont,cont_2d/3d/4d,cast}` ops.rs:467-1749 | ✅（cast=GGML_OP_CAST 复用 dup，COVERAGE:197 同参考 ops.cpp） |
| add/mul/div/sub (+inplace) | ops.rs:478-540 + compute.rs:630 forward_bin_op | ✅ F32/F16/BF16；**量化 src0/dst（binary-ops.cpp 的 *_q_f32）未移**——唯一消费者 = 控制向量 cvec（llama 层 ⏳ documented-skip，FILE_MAP:120） |
| add_id | ops.rs:1048 + compute.rs:1486 | ✅ |
| scale/scale_bias | ops.rs:1096/1105 + compute.rs:1821 | ✅ |
| clamp | ops.rs:542 + compute.rs:3518 | ✅ |
| norm/rms_norm | ops.rs:1065/1081（合并为 GgmlOp::Norm + OP_FLAG_NORM_IS_RMS）+ compute.rs:1701/1771 | ✅ 合并设计（判别值 internal-only） |
| soft_max(_ext)/add_sinks | ops.rs:1122-1176 + compute.rs:1863（scale/max_bias=alibi/sinks 全） | ✅ |
| rope 家族（rope/custom/ext/ext_back/multi/multi_back/set_offset/yarn_corr_dims） | ops.rs:1543-1670 | ✅ rope_set_offset 批次6 补齐（COVERAGE:11）；rope_multi ✅ |
| get_rows/get_rows_back/set_rows | ops.rs:1671/1693/557 + compute.rs:2743/2798/3162 | ✅ |
| repeat/repeat_4d | ops.rs:1987/2254 + compute.rs:3258 | ✅（2 字节形态 F16/BF16/I16 批次 11a 补齐，COVERAGE:368） |
| concat/sum_rows/mean/argmax/argsort(_top_k) | ops.rs:1959/1979/702/1925/1934/1199 + compute.rs 对应 | ✅ |
| reshape/view/permute/transpose | ops.rs:1756-1923（layout-only，compute no-op） | ✅ |
| unary 22 个 | 11 个见 §3 UNARY 行 | 🔨 见 §0 |
| glu 家族（7 op × {fused,split,swapped}） | ops.rs:68-77 常量 + 2196-2252 split builders + 818 swiglu_clamp + 1178 swiglu_oai | ✅ 单张量 fused 型（ggml.c:1890 llama-graph.cpp 的 LLM_FFN_*_PAR 单张量）= 视图+split 位同等价（documented，PARITY:330） |
| im2col/conv_2d/conv_2d_direct/conv_2d_dw_direct/roll/pad(_ext/_reflect_1d)/pool_1d/pool_2d/arange/interpolate(=upscale)/fill/top_k/sqrt/sin/cos/sqr | ops.rs:2337-2534 等 | ✅（conv_2d_direct 是 composed builder——不落 CONV_2D 节点，见 §5-C）；conv_1d/conv_1d_dw(_ph) 复合 builder 在 llama crate clip.rs:3144-3160/4056（audio 家族消费，COVERAGE:193） |
| flash_attn_ext(+sinks/prec/n_kv_max) | ops.rs:2010-2078 + flash_attn.rs | ✅（sinks=src[4] 语义 PARITY:36） |
| ssm_conv/ssm_scan/gated_delta_net/lightning_indexer/dsv4_hc_{comb,pre,pre_gated,post} | ops.rs:2079-996/609/2272 | ✅（pre_gated ops.rs:924；ds v4 oracle dsv4_ops_ref.bin） |
| rwkv_wkv6/gated_linear_attn/rwkv_wkv7 | crates/ggml/src/wkv.rs 融合内核（spec + AVX512 双体; graph_arch.rs time-mix 直接调用） | ✅（2026-09-28: parity/wkv_ref.bin 11 节 94768 元素位一致, S≥64 wkv7 梯队; 参考自身 sub-64 非确定性已证 parity/wkv7_oob_proof.c; gla 标量尾的 GCC 收缩方向已按反汇编复刻） |
| 训练/未用 builder: add1/acc/log/set(除 set_1d 一处)/out_prod/diag/diag_mask_zero/win_(un)part/get_rel_pos/add_rel_pos/leaky_relu/timestep_embedding/conv_transpose_{1d,2d}/conv_3d/im2col_3d/col2im_1d/cumsum/count_equal/sum/repeat_back/silu_back/rms_norm_back/soft_max_ext_back/flash_attn_back/group_norm/l2_norm/solve_tri/tri/map_custom*/custom/cross_entropy_loss*/opt_step* | — | 📄 见 §3 逐 op 行（llama 层零调用，除注明者） |
| set_1d | chameleon 的 img_logits 置 -FLT_MAX（chameleon.cpp:198） | 🔗 graph_arch.rs:11703-11707 组合 |
| l2_norm | `build_gdn_l2_norm` = rms_norm(eps/n)·scale(1/√n)（models.h:14-19 同式）+ rwkv7 kk | 🔗 graph_arch.rs:3415 |
| quantize_chunk/quantize_init/quantize_free/requires_imatrix | llama crate quant.rs + ggml quants_impl.rs | ✅（19/19 ftype 逐字节；IQ2/3/1 家族显式报错文案，PARITY:1227） |
| 图: build_forward_expand/graph_cpy/graph_* | graph.rs（拓扑 DFS）+ Context::mark/reset_graph_to（watermark 重建） | ✅ 等价简化（节点顺序无关数值） |
| 类型/杂项: type_name/type_size/blck_size/row_size/nbytes/is_contiguous*/n_dims/op_name… | types.rs/tensor.rs/ops.rs:2537+ | ✅（`ggml_type_name` 小写表在 llama crate params，PARITY:667 已记录） |

### 1.2 gguf.cpp 符号级

| gguf.cpp 符号 | 端口 | 状态 |
|---|---|---|
| gguf_init_from_file(_ptr) / init_empty / init_from_buffer / init_from_callback / reader 族 | gguf.rs::{open, open_single, from_bytes, part_storage} | ✅ 读：qwen2.5 真文件交叉验证 + **split.* 多分片装配**（FILE_MAP:21）；init_from_callback（流式）📄 未移（端口直接 mmap，行为面不缺） |
| gguf_find_key/find_tensor/get_n_kv/get_n_tensors/get_tensor_{name,ne,offset,size,type}/get_{version,alignment,data_offset,meta_data,meta_size} | gguf.rs 对应 pub fn | ✅ |
| gguf_get_val_{u8..u64,i8..i64,f32,f64,bool,str} + get_kv_type | Value::as_* + get_* | ✅（数值数组隐式加宽与参考 loader 同，gguf.rs:194 注释） |
| gguf_get_arr_{n,type,data,str} | Value::as_array → (GgufType, &[Value]) + as_f32_slice | ✅ |
| gguf_writer_{base,buf,file} + write_to_{buf,file,file_ptr} + write_out | gguf_write.rs::{new,write,write_meta} | ✅ **写与参考逐字节一致**（parity/gguf_write_ref.bin, ref_gguf_write.c）；magic "GGUF"/v3/alignment 32/general.alignment KV 全对（gguf.rs:15-18, gguf_write.rs:39-96） |
| gguf_set_val_*(19 型)/set_arr_{data,str}/set_kv/set_tensor_{data,type}/add_tensor/remove_key/check_reserved_keys | gguf_write.rs::{set_kv,add_tensor,set_tensor_type}（写侧）+ llama crate 载入侧 | ✅ 已用面全；remove_key/single 型 setter 无消费者 |
| gguf_type/type_size/type_name | GgufType | ✅ |

---

## 2. `ggml-cpu/` 文件清单

| 文件 | 行数 | Rust 对应 | 状态 | 符号级结论 |
|---|--|--|--|---|
| ggml-cpu.c | 3944 | compute.rs | 🔨→✅ | 主循环 graph_compute:255（含 eval_callback:229）、mul_mat 全路由:876（wdata[962-990] → llamafile 两次尝试 → repack 拦截[925/938，序=参考 ggml-cpu.c:1751 先于 sgemm] → vec_dot）、mul_mat_id:1279、type_traits_cpu 对应 = vec_dot_type(vec_dot.rs:1070)+compute.rs:905-911+mxfp4_vec_dot_type:5438、threadpool=Team:441 par_rows；`ggml_cpu_init`（f16 表）→ ops.rs:222 惰性表；`ggml_cpu_has_*` → simd_x86.rs:38-52 运行期探测 |
| ops.cpp | 12206 | compute.rs(8210)+ssm.rs+flash_attn.rs+gdn.rs+ops.rs | 🔨→✅ | 见 §3 op-inventory（247 个唯一 `ggml_compute_forward_*` 函数名 → 57 个参考 op 有 Rust 内核 + 组合；参考侧行号锚点：dup_bytes:326 mean:1513 repeat:1698 fill:2230 norm:3827 rms_norm:3924 scale:4697 get_rows:5105 get_rows_back:5361 set_rows:5221 soft_max:5584 rope:6082 conv2d_dw:7603 pool2d:7769 pad:8196 roll:8328 arange:8386 argsort:8489 top_k:8557 fa_one_chunk:8614 ssm_conv:9703 ssm_scan:9773 gdn:11049 hc_comb:11142 fwht:12027 lightning_indexer:12126） |
| quants.c | 1339 | vec_dot.rs + simd_x86.rs + compute.rs(mxfp4) | ✅(x86 口径) | 通用内核 q4_0/q4_1/q5_0/q5_1/q8_0/q2_K..q6_K 8 族 ✅（simd_x86.rs AVX2 lane + 标量兜底，逐位互证 + vecdot_ref.bin/mulmat_ref.bin 真值）；mxfp4_q8_0 ✅ compute.rs:5451（generic 形）；**q1_0/q2_0/nvfp4/iq×9/tq×2 未移**（§5-A） |
| vec.h/.cpp | 1570+613 | vec_dot.rs + ops.rs + compute.rs | ✅ | vec_dot_f32=AVX512 lane 复刻（vec_dot_f32_c, vec_ref.bin）；cvar（norm 方差）AVX512 分支 compute.rs:1701（PARITY:303）；gelu f16 查表 ops.rs:222；tanh 标量 ops.rs:349；exp 多项式 ops.rs:115/148；silu ops.rs:190；softmax 求和 vec_dot.rs:1369；mad/scale（gdn 消费）gdn.rs |
| repack.cpp | 5253 | repack.rs(6127) | ✅(5 族中 3 族) | **MXFP4 8x8** ✅（repack:165/gemv:626/gemm:676，repack_ref.bin）；**Q4_K 8x8** ✅（repack:975/gemv:2645/gemm:2670 = AVX2/AVX512BW+DQ lane 复刻，q4k_repack_ref.bin）；**Q4_0 8x8** ✅（gemv:3700/gemm:3722 + compute.rs:3980 repacked_q4_0 + mul_mat_q4_0_repack，q4_0_repack_ref.bin）——**COVERAGE:21 行"Q4_0/Q4_K 待做"已过时**；q2_K_8x8/iq4_nl_8x8 未移（PARITY:1662/3982 已记录"本机模型不可达"） |
| llamafile/sgemm.cpp | 4164 | tinyblas.rs(1423) | ✅ | 6 组合（F32/F16/BF16/Q8_0/Q4_0/Q5_0）路由表+尾块 tile_tail 与参考 switch 逐位（tinyblas_ref.bin 6360 例, PARITY:301-305）；FILE_MAP:46 的 ⛔ 行**已过时** |
| binary-ops.cpp | 154 | compute.rs::forward_bin_op:630 | ✅ | add/sub/mul/div（F32/F16/BF16；add1/add_q 训练/cvec 侧 📄） |
| unary-ops.cpp | 337 | compute.rs::forward_{tanh,sigmoid,relu,elu,gelu,gelu_erf,silu,exp,softplus,round,xielu} | 🔨 | 11/22（§0） |
| traits.cpp/.h | 36+38 | （并入 vec_dot_type/vec_dot_row） | ✅ | ggml_get_type_traits_cpu 的消费面 |
| repack.h / ops.h / quants.h / vec.h / common.h / ggml-cpu-impl.h | 头 | 类型/常量并入各 rs | ✅ | — |
| ggml-cpu.cpp | 716 | llama crate util（cpu_info/线程参数） | ✅ | device description/props（PARITY:680） |
| iqp.cpp/.h | 1253+39 | — | 📄 | IQ panel gemm（IQ 权重专用 mul_mat/mul_mat_id 快路，ggml-cpu.c:1377 接线）；端口无 IQ 权重路径 ⇒ 不可达。**无 COVERAGE 行**（§5-B） |
| hbm.cpp/.h | 55+8 | — | 📄 | Fujitsu A64FX HBM buffer（GGML_USE_CPU_HBM off） |
| arch-fallback.h | 377 | （标量兜底=vec_dot.rs *_generic/_scalar） | ✅ | 兜底内核与 lane 内核互证（simd_matches_scalar_bit_exact 测试族） |
| simd-mappings.h | 1319 | simd_x86.rs + vec_dot.rs lane 注释 | ✅ | F32_STEP 64/EPR 16/ARR 4（AVX512 口径）逐处引用 |
| simd-gemm.h | 226 | （tinyblas 复刻内联） | ✅ | — |

### 2.1 lane 目录分类（政策=复刻 x86 参考构建实际派发的 lane）

| 目录 | 参考内容 | 分类 |
|---|---|---|
| arch/x86/{quants,repack,cpu-feats}.cpp | AVX2/AVX512 lane | **ported**：quants→simd_x86.rs+vec_dot.rs（8 量化族 AVX2 lane 位级）；repack→repack.rs（Q4_K AVX2 + AVX512BW+DQ gemm lane、Q4_0/MXFP4 外积 lane）；cpu-feats→运行期探测。覆盖检查：x86 参考构建在 mul_mat 上实际派发的量化 lane = 上述 3 族的全部实例（K-quant 中仅 Q4_K 有 x86 repack 实例，repack.cpp:5006；Q5_K/Q6_K 无[repack.cpp:5050-5071] → 行式 vec_dot，两侧一致）——**无未覆盖的 x86 派发** |
| arch/arm | quants(4319)+repack(5465)+cpu-feats | 📄 documented-skip（COVERAGE:20/26"仅 x86 lane"；nrows=2 的 ARM MATMUL_INT8 分支随之豁免） |
| arch/riscv / s390 / loongarch / powerpc / wasm | quants(+repack) | 📄 documented-skip（COVERAGE:14/47/48/68） |
| amx/ | mmq.cpp 2511 | 📄 documented-skip（COVERAGE:44） |
| kleidiai/ | 1919 | 📄 documented-skip（COVERAGE:54） |
| spacemit/ | ime*/rvv/spine ~9k | 📄 documented-skip（COVERAGE:17/32/58/60 ⏳ 行——建议 PARITY 归档为 ➖） |

---

## 3. Op-inventory（**最高价值表**）：GGML_OP_*（参考 ggml.h:490-604，NONE=0 … GLU=100，COUNT=101）

判别：✅=端口有 builder+内核；🔗=composed；📄=documented-skip/llama 层零调用（给出依据）。
端口 enum `crates/ggml/src/tensor.rs:17`（判别值 internal-only，与 wire 无关）。
对账口径: 端口 enum 57 个变体 = 54 个活变体 + 3 个死变体（§5-C）；57 个"✅参考 op"映射到 54 个活变体
是因为三处合并——RMS_NORM→`GgmlOp::Norm`+`OP_FLAG_NORM_IS_RMS`（ops.rs:1084）、GET_ROWS_BACK→
`GgmlOp::GetRows`+`OP_FLAG_GET_ROWS_BACK`（ops.rs:1705 设, compute.rs:574 判）、CONT→`GgmlOp::Dup`（参考 forward_cont==dup）。

| # | GGML_OP_* | 端口 | 依据（ref ops.cpp → port compute.rs） |
|---|---|---|---|
| 0 | NONE | ✅ | GgmlOp::None |
| 1 | DUP | ✅ | dup_bytes:326/dup_flt/dup_same_cont/dup_to_q/dup_from_q → forward_dup:2831（含 I32→F32 dup，批次 11a） |
| 2 | ADD | ✅ | binary-ops.cpp → forward_bin_op:630（F32/F16/BF16；量化变体随 cvec 📄） |
| 3 | ADD_ID | ✅ | :563 → forward_add_id:1486 |
| 4 | ADD1 | 📄 | 训练；src/ 零调用 |
| 5 | ACC | 📄 | 0 调用 |
| 6 | SUB | ✅ | audio 轮次4 → forward_bin_op Sub（vec.h:112 标量） |
| 7 | MUL | ✅ | forward_bin_op（AVX512 lane 热路径） |
| 8 | DIV | ✅ | forward_bin_op（vdivps==标量除法） |
| 9 | SQR | ✅ | :281 → forward_sqr:4662（mimo/qwen3tts-spkenc） |
| 10 | SQRT | ✅ | :285 → forward_sqrt:4598（deepseek4 SQRT_SOFTPLUS） |
| 11 | LOG | 📄 | 仅 opt/perplexity CPU 直调，图面 0 调用 |
| 12 | SIN | ✅ | :289 → forward_sin_cos:4629 |
| 13 | COS | ✅ | :293 → forward_sin_cos(true) |
| 14 | SUM | 📄 | 仅后端采样（llama-sampler.cpp:1290+；端口 CPU 采样已位同，后端采样族 documented-skip PARITY:34） |
| 15 | SUM_ROWS | ✅ | → forward_sum_rows:3236（双累加器 ith==0） |
| 16 | CUMSUM | 📄 | 后端采样 + qwen3tts-gen(⛔) + delta-net-base(融合 GDN 替代) |
| 17 | MEAN | ✅ | :1513 → forward_mean:4691 |
| 18 | ARGMAX | ✅ | → forward_argmax:3105 |
| 19 | COUNT_EQUAL | 📄 | 仅后端采样 |
| 20 | REPEAT | ✅ | :1698 → forward_repeat:3258 |
| 21 | REPEAT_BACK | 📄 | 0 调用 |
| 22 | CONCAT | ✅ | concat_any → forward_concat:3359 |
| 23 | SILU_BACK | 📄 | 训练 |
| 24 | NORM | ✅ | :3827 → forward_norm:1701（含 ggml_vec_cvar AVX512 分支） |
| 25 | RMS_NORM | ✅(合并) | :3924 → GgmlOp::Norm+OP_FLAG → forward_rms_norm:1771（rms_norm_mul_fused 语义在图侧 mul 组合） |
| 26 | RMS_NORM_BACK | 📄 | 训练 |
| 27 | GROUP_NORM | 📄 | 仅 wavtokenizer-dec（⛔ 音频生成器族） |
| 28 | L2_NORM | 🔗 | rms_norm(eps/n)·scale(1/√n)（graph_arch.rs:3415 = models.h:14-19 同式；rwkv7 kk 用之） |
| 29 | MUL_MAT | ✅ | ggml-cpu.c:1182+ → forward_mul_mat:876（wdata/llamafile×2/repack/vec_dot 全路由） |
| 30 | MUL_MAT_ID | ✅ | ggml-cpu.c:1599+ → forward_mul_mat_id:1279（含 Q4K 3D repack gemv） |
| 31 | OUT_PROD | 📄 | 训练 |
| 32 | SCALE | ✅ | :4697 → forward_scale:1821（含 scale_bias） |
| 33 | SET | 🔗 | 唯一图面消费者 chameleon.cpp:198 set_1d → graph_arch.rs:11703 组合 |
| 34 | CPY | ✅ | → forward_dup（cpy==dup） |
| 35 | CONT | ✅ | → forward_dup（cont==dup）+ cont_{1..4}d builders |
| 36 | RESHAPE | ✅ | layout-only |
| 37 | VIEW | ✅ | layout-only（view_1d..4d ops.rs:1756-1841） |
| 38 | PERMUTE | ✅ | layout-only |
| 39 | TRANSPOSE | ✅ | layout-only |
| 40 | GET_ROWS | ✅ | :5105 → forward_get_rows:2743（f32/f16/bf16/q 四型） |
| 41 | GET_ROWS_BACK | ✅ | :5361 → forward_get_rows_back:2798 |
| 42 | SET_ROWS | ✅ | :5221 → forward_set_rows:3162 |
| 43 | DIAG | 📄 | 0 调用 |
| 44 | DIAG_MASK_INF | ✅ | diag_mask_f32 → forward_diag_mask_inf:3308 |
| 45 | DIAG_MASK_ZERO | 📄 | 0 调用（同核 diag_mask_zero 值 0） |
| 46 | SOFT_MAX | ✅ | :5584 → forward_soft_max:1863（scale/max_bias/sinks/多线程归约） |
| 47 | SOFT_MAX_BACK | 📄 | 训练 |
| 48 | ROPE | ✅ | :6082 → forward_rope:2578（全部 mode/section/theta 缩放；mrope_cache indep_sects 修复 PARITY:199） |
| 49 | ROPE_BACK | ✅ | → forward_rope（rope_ext_back ops.rs:794） |
| 50 | CLAMP | ✅ | clamp_f16/f32 → forward_clamp:3518 |
| 51 | CONV_TRANSPOSE_1D | 📄 | 0 调用（旧 sd/vision 路径） |
| 52 | IM2COL | ✅ | im2col_f32/f16 → forward_im2col:3554 |
| 53 | IM2COL_BACK | 📄 | 训练 |
| 54 | IM2COL_3D | 📄 | conv_3d 族 0 调用 |
| 55 | COL2IM_1D | 📄 | 仅 qwen3tts-gen（⛔，FILE_MAP:111 且点名 "GEN_WAV 需 ggml_col2im_1d 亦未移"） |
| 56 | CONV_2D | 🔗(direct)/✅(composite) | conv_2d_direct=im2col+mul_mat+permute 位同（ops.rs:2428-2486，conformops_ref.bin）；composite conv_2d ops.rs:2378 |
| 57 | CONV_3D | 📄 | 0 调用 |
| 58 | CONV_2D_DW | ✅ | :7603(whcn) → forward_conv_2d_dw:4986（whcn/cwhn 两 layout 参考侧只有 whcn 被 llama 层喂） |
| 59 | CONV_TRANSPOSE_2D | 📄 | 0 调用 |
| 60 | POOL_1D | ✅ | → forward_pool_1d:4842（audio 轮次2 起为字面 op，pool1d_ref.bin） |
| 61 | POOL_2D | ✅ | :7769 → forward_pool_2d:5065（max/avg；批次 11a 平面写偏移修复） |
| 62 | POOL_2D_BACK | 📄 | 训练 |
| 63 | UPSCALE | ✅ | → forward_upscale:3629（nearest/bilinear/bicubic + ALIGN_CORNERS，interp_ref.bin） |
| 64 | PAD | ✅ | :8196 → forward_pad:4790（pad_ext 含 circular 旗标 ops.rs:755） |
| 65 | PAD_REFLECT_1D | ✅ | :8282 → forward_pad_reflect_1d:4720 |
| 66 | ROLL | ✅ | :8328 → forward_roll:4933 |
| 67 | ARANGE | ✅ | :8386 → forward_arange:5140 |
| 68 | TIMESTEP_EMBEDDING | 📄 | 0 调用（sd 遗产） |
| 69 | ARGSORT | ✅ | :8489 → forward_argsort:3127（ASC/DESC tie 语义；argsort_top_k 组合） |
| 70 | TOP_K | ✅ | :8557 → forward_top_k:4459（partial_sort 堆选择逐算子复刻，oracle 钉并列集合） |
| 71 | LEAKY_RELU | 📄 | 0 调用 |
| 72 | TRI | 📄 | delta-net-base（融合 GDN 替代）+ qwen3tts-gen(⛔) |
| 73 | FILL | ✅ | :2230 → forward_fill:4270（DSA top-k mask） |
| 74 | FLASH_ATTN_EXT | ✅ | :8614 one_chunk + tiled/reduce_partials → flash_attn.rs + forward:3460（f16 累加/sinks/mask 广播全，fa_ref.bin） |
| 75 | FLASH_ATTN_BACK | 📄 | 训练 |
| 76 | SSM_CONV | ✅ | :9703 → ssm.rs + forward_ssm_conv:3737 |
| 77 | SSM_SCAN | ✅ | :9773 → ssm.rs + forward_ssm_scan:3785（AVX512 STEP64 舍入序复刻） |
| 78 | WIN_PART | 📄 | 0 调用（deepseekocr 注释提及未用） |
| 79 | WIN_UNPART | 📄 | 0 调用 |
| 80 | GET_REL_POS | 📄 | 仅 llava（⛔ projector 族） |
| 81 | ADD_REL_POS | 📄 | 仅 llava |
| 82 | RWKV_WKV6 | ✅ | wkv.rs rwkv_wkv6_f32（ops.cpp:10413-10603; wkv_ref.bin 位一致; graph_arch.rs rwkv6 time-mix 调用） |
| 83 | GATED_LINEAR_ATTN | ✅ | wkv.rs gla_f32（ops.cpp:10623-11418; S=20 标量尾 = GCC 的 v·k-融合收缩, 位一致; rwkv6qwen2 调用） |
| 84 | RWKV_WKV7 | ✅ | wkv.rs rwkv_wkv7_f32（ops.cpp:11422-11617; S≥64 位一致——参考 sub-64 不可复证 wkv7_oob_proof.c; S<64 单线程确定性派发） |
| 85 | SOLVE_TRI | 📄 | 仅 delta-net-base 非融合路径（端口走融合 GDN） |
| 86 | GATED_DELTA_NET | ✅ | :11049/one_chunk → gdn.rs（K=1 切片，端口图只造 K=1）+ forward:2363 |
| 87 | LIGHTNING_INDEXER | ✅ | :12126 → forward_lightning_indexer:4379 |
| 88 | DSV4_HC_COMB | ✅ | :11142 → forward_dsv4_hc_comb:5192 |
| 89 | DSV4_HC_PRE | ✅ | → forward_dsv4_hc_pre:5287（+pre_gated builder） |
| 90 | DSV4_HC_POST | ✅ | → forward_dsv4_hc_post:5349 |
| 91 | UNARY | ✅(11/22) | 见 §0；缺的 11 个：abs/sgn（仅 qwen4exp PLE，PARITY 批次 11a documented-skip）、neg（rwkv6/7-base → scale(-1) 位等，graph_arch.rs:33194 注释）、step/cumsum（后端采样族）、gelu_quick/hardswish/hardsigmoid/expm1/floor/ceil/trunc（0 调用） |
| 92-94 | MAP_CUSTOM1/2/3 | 📄 | 自定义算子宿主 API，llama 层 0 调用 |
| 95 | CUSTOM | 📄 | 同上 |
| 96 | CROSS_ENTROPY_LOSS | 📄 | 仅 ggml-opt |
| 97 | CROSS_ENTROPY_LOSS_BACK | 📄 | 训练 |
| 98 | OPT_STEP_ADAMW | 📄 | 仅 ggml-opt |
| 99 | OPT_STEP_SGD | 📄 | 仅 ggml-opt |
| 100 | GLU | ✅ | glu 家族 7 op 全（forward_glu:1538；split/oai/clamp builders） |

**0 个 op 属于"已移植架构会派发但端口缺失"**（最高价值结论：37 个 📄 全部核过 llama 层调用面——
src/ + common/ + tools/mtmd/models/ 的 `ggml_*` 构造 grep；_rwkv/_gla/_tri/_solve_tri/_cumsum 的
消费者全部走端口的组合/融合替代路径并有 16/16 或 48/48 parity 记录）。

---

## 4. 类型表核对（两表逐行）

### 4.1 ggml.c:632 `type_traits`（43 个有效类型 + 7 个 DEPRECATED 槽位）

- **blck_size/type_size/is_quantized/type_name**: 端口 types.rs:72-114 `traits()` 全 43 行对齐
  （types.rs 测试 `type_table_matches_reference` 逐类型断言 bpw；本审计用参考表 632-1006 行逐行复核
  QK 常量与 sizeof：Q1_0=128/18、Q2_0=64/18、TQ1_0=256/54、TQ2_0=256/66、MXFP4=32/34、NVFP4=64/… 全同）。
  DEPRECATED 槽位（4/5/31-33/36-38）：端口 from_u32 → None（等价拒绝）✓。
- **to_float**: 参考表 = dequantize_row_*（Q 系全类型 + F16/BF16 row + IQ 全类型）。
  端口 = quants.rs:1082 `dequantize_row` **缺 11 臂**：IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/
  IQ4_NL/IQ4_XS/TQ1_0/TQ2_0 —— 其中 9 个 IQ 的函数**已存在**（quants_k.rs:2135-2393）只是未接 dispatch
  （quants.rs:1110 `unimplemented!("IQ/ternary coming")`）。参考侧 TQ dequant 在 ggml-quants.c:2428/2467，端口无。
  → §5-A.2。
- **from_float_ref**（量化参考实现）: 简单型+Q1_0/Q2_0+Q2K..Q6K ✅（quants.rs/quants_k.rs:1685-2130，
  逐字节钉死）；IQ4_NL/IQ4_XS ✅（quants_impl.rs:952-1160，imatrix 消费）；IQ1/IQ2/IQ3 家族 📄
  （PARITY:1227 显式记录：需 4608 u16 运行时表 + kmap/neighbours，写文件前显式报错）；
  **TQ1_0/TQ2_0 量化器未移且无记录**（→ §5-B）。

### 4.2 ggml-cpu.c:215 `type_traits_cpu`

| 类型 | ref from_float | ref vec_dot | ref vec_dot_type | ref nrows | 端口 | 状态 |
|---|---|---|---|---|---|---|
| F32 | cpu_fp32_to_fp32 | vec_dot_f32 | F32 | 1 | wdata 直拷 + vec_dot_f32_c（AVX512 lane） | ✅ |
| F16 | cpu_fp32_to_fp16 | vec_dot_f16 | F16 | 1 | mul_mat 分支内转换 + vec_dot_f16 | ✅ |
| BF16 | cpu_fp32_to_bf16 | vec_dot_bf16 | BF16 | 1 | 同上 | ✅ |
| Q4_0/Q5_0/Q8_0 | quantize_row_q8_0 目标 | q*_q8_0 | Q8_0 | 1(x86) | vec_dot_type ✓ + simd_x86 lane | ✅ |
| Q4_1/Q5_1 | q8_1 | q*_q8_1 | Q8_1 | 1 | ✓ | ✅ |
| Q2K..Q6K | q8_K | q*_K_q8_K | Q8_K | 1 | ✓（compute.rs:976-980 wdata Q8K 转换 = ref from_float） | ✅ |
| MXFP4 | quantize_row_mxfp4 | mxfp4_q8_0 | Q8_0 | 1 | mxfp4_vec_dot_type（compute.rs:5438）+ vec_dot_mxfp4_q8_0（compute.rs:5451）+ repack 路径 | ✅ |
| NVFP4 | quantize_row_nvfp4 | nvfp4_q8_0 | Q8_0 | 1 | **缺 vec_dot/vec_dot_type**（dequant 有 quants.rs:733） | **MISSED §5-A.1** |
| Q1_0/Q2_0 | quantize_row_q1/2_0 | q1_0/q2_0_q8_0 | Q8_0 | 1 | **缺 vec_dot 侧**（量化/反量化已有） | **MISSED §5-A.1** |
| IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M | NULL/注释 | iq*_q8_K | Q8_K | 1 | 缺（iqp.cpp 快路同缺） | 📄+§5-A.1 |
| IQ4_NL | quantize_row_iq4_nl | iq4_nl_q8_0 | Q8_0 | 1 | 缺推理侧（量化器有） | 📄(PARITY:493)+§5-A.1 |
| IQ4_XS | quantize_row_iq4_xs | iq4_xs_q8_K | Q8_K | 1 | 缺推理侧 | 同上 |
| TQ1_0/TQ2_0 | quantize_row_tq* | tq*_q8_K | Q8_K | 1 | 缺 | §5-B |
| Q8_K | quantize_row_q8_K | — | — | — | quantize_row_q8_K ✓（wdata） | ✅ |
| I32 | cpu_fp32_to_i32 | — | — | — | dup/cast 路径覆盖 | ✅ |
| nrows=2（ARM MATMUL_INT8: Q4_0/Q4_1/Q8_0/Q4_K/Q6_K） | | | | | 端口恒 1 = **x86 参考构建口径** ✓（arm lane documented-skip 的必然推论） | ✅ |

---

## 5. MISSED 清单（triage：op-family / helper / lane）

### A. 真实缺口（"会在已支持类型上 panic"或"量化侧有、推理侧无"，建议接 integrator 队列）

> **→ 2026-09-27 修复轮已全部处置**（详 PARITY.md §AUDIT-ggml-修复轮）：A.1 的 12 类型 lane 已移并
> 位同参考（parity/vecdot3_ref.bin + ref_vecdot_dump3.c；TQ×2 转为 B.4 同款显式 refusal）；A.2 的
> 9 个 IQ dispatch 臂已接（quants.rs::dequantize_row + iq_dispatch_tests）；A.3 维持 documented-skip、
> 理由已落在 compute.rs::forward_bin_op 注释（ops.cpp:578 add_q_f32 / 消费者 cvec）。

1. **[helper-family] vec_dot/vec_dot_type 缺 13 类型** — `compute.rs:905-911` `vec_dot_type().or(mxfp4…)`
   之后 panic `"mul_mat: no vec_dot"`：
   - NVFP4（参考 ggml-cpu.c:292-296 注册 nvfp4→Q8_0/`ggml_vec_dot_nvfp4_q8_0` = quants.c generic）——
     端口有 dequantize（quants.rs:733, 量化工具可把 NVFP4 当源类型转换）但**无 vec_dot 也无量化器**，
     NVFP4 权重不可跑，且无文档记录；
   - Q1_0/Q2_0（quants.c generic 内核；端口量化器已位级修复 PARITY:79 但只服务 llama-quantize 输出）；
   - IQ×9 + IQ4_NL/IQ4_XS（IQ 家族有 PARITY:1227/493 的量化器侧记录，**推理侧 panic 无记录**）；
   - TQ1_0/TQ2_0（完全无记录，见 B.4）。
   **影响面**: 加载这些类型的 GGUF 后首个 mul_mat panic（未定义行为面=显式 panic，非静默错值）。
2. **[helper] `dequantize_row` dispatch 缺 IQ/TQ 臂** — quants.rs:1082-1111；9 个 IQ dequant 函数已在
   quants_k.rs:2135-2393（pub），get_rows/dup/量化转换路径对 IQ 张量 panic。与 A.1 同根（"IQ 权重
   不支持"），但状态是"函数在、接线无"，修复成本一行 match 臂 ×9。
3. **[helper] `bin_op` 无量化 src0/dst 变体**（binary-ops.cpp `*_q_f32`，参考为 add 专用）— 唯一
   消费者控制向量 cvec 已是 llama 层 documented-skip（FILE_MAP:120）；若未来移 cvec 需连带。

### B. 文档级缺口（建议补 COVERAGE.md/PARITY.md 行，不是代码缺口）

> **→ 2026-09-27 修复轮已归档**：B.4 的 TQ 链已入 PARITY/本文件（显式 refusal + 测试钉死 panic 面）；
> B.5/B.6 的 ggml-opt/iqp/hbm 行已入 COVERAGE；B.7 的过时行（COVERAGE:21 repack、FILE_MAP 幻影
> 指针/sgemm ⛔/arch-x86 🔨）已全部修正。

4. **TQ1_0/TQ2_0 全链无记录**: blocks.rs 结构体+types.rs 表行有，quantize/dequant/vec_dot 三侧全无，
   COVERAGE/PARITY/FILE_MAP 均无 ternary 行。参考侧消费者 = 仅测试 fixture。
5. **ggml-opt.cpp（1094 行）无 COVERAGE 行**（>1500 行截断遗漏）；建议按"训练 API ⛔"归档。
6. **iqp.cpp/hbm.cpp 无行**：建议按"IQ panel gemm 性能 lane ➖ / A64FX ➖"归档（本审计已给出定性）。
7. **COVERAGE/FILE_MAP 过时行**: COVERAGE:21 repack 行（Q4_K/Q4_0 已完成）、FILE_MAP:23/24/30/43/44
   指向不存在的 `alloc.rs/backend.rs/cpu/mod.rs/cpu/*_ops.rs`、FILE_MAP:46 sgemm ⛔（tinyblas.rs 已在）、
   FILE_MAP:47 arch/x86 🔨（simd_x86.rs 已落）。已列入下方 FILE_MAP 更正段。

### C. 死代码（建议清理，非 parity 问题）

> **→ 2026-09-27 修复轮处置**：变体**保留 + 现场注释**——crates/llama 的 op 名字映射（tests/
> arch_batch*_e2e.rs、tests/*_dump.rs、src/clip.rs）对枚举 exhaustive match，该目录归另一组移植
> agent 所有，删除会破坏其构建；注释已写明"pinned 枚举无此 op/无 dispatch/仅名字映射引用"。
> lib.rs:4 的 revision 引用已改为 pinned bd4f514db1。

- `GgmlOp::SquaredMulMat = 8` / `GgmlOp::MulView = 53`（tensor.rs:25/45）：**pinned 枚举无此二 op**
  （旧 revision 遗产），仅测试名映射引用；`GgmlOp::Conv2dDirect = 76`（tensor.rs:142）声明后无
  forward_op 分发（builder 是 composed，不落此节点）。
- `crates/ggml/src/lib.rs:4` 文档头仍写 "Reference: @ 7b50d589 (master)" — 与 pinned bd4f514db1 不符。

---

## 6. 复验抽查清单（≥15 op + ≥10 kernel，两侧 grep 命令直接可用）

| 抽查 | 参考锚 | 端口锚 |
|---|---|---|
| MUL_MAT 路由 | ggml-cpu.c:1182/1273/1382-1401 | compute.rs:876（wdata:948-990, sgemm:1000+, repack:925/938） |
| MUL_MAT_ID | ggml-cpu.c:1599+ | compute.rs:1279 |
| FLASH_ATTN_EXT | ops.cpp:8614(one_chunk)/9348 | flash_attn.rs（one_chunk/reduce_partials）+ compute.rs:3460 |
| SSM_SCAN | ops.cpp:9773 | ssm.rs + compute.rs:3785 |
| SSM_CONV | ops.cpp:9703 | ssm.rs + compute.rs:3737 |
| GATED_DELTA_NET | ops.cpp:11049 | gdn.rs + compute.rs:2363 |
| ROPE | ops.cpp:6082 | compute.rs:2578 |
| SOFT_MAX | ops.cpp:5584 | compute.rs:1863 |
| NORM/RMS_NORM | ops.cpp:3827/3924 | compute.rs:1701/1771 |
| TOP_K | ops.cpp:8557 | compute.rs:4459 |
| ARGSORT | ops.cpp:8489 | compute.rs:3127 |
| GET_ROWS/BACK | ops.cpp:5105/5166 | compute.rs:2743/2798 |
| SET_ROWS | ops.cpp:5221 | compute.rs:3162 |
| FWHT（HINT_SRC0_IS_HADAMARD） | ops.cpp:12027 | compute.rs:4307 |
| DSV4_HC×3 | ops.cpp:11142+ | compute.rs:5192/5287/5349 |
| LIGHTNING_INDEXER | ops.cpp:12126 | compute.rs:4379 |
| vec_dot_f32 AVX512 lane | vec.h/`.so` 反汇编 | vec_dot.rs:1229 vec_dot_f32_c |
| Q4_K 8x8 gemm lane | arch/x86/repack.cpp:2042 | repack.rs:2661 |
| Q4_0 8x8 | arch/x86/repack.cpp | repack.rs:3700/3722 + compute.rs:3980 |
| MXFP4 8x8 | repack.cpp(generic)+arch | repack.rs:626/676 + compute.rs:5451 |
| tinyBLAS 6 组合路由 | sgemm.cpp:3820/4131 | tinyblas.rs:100-190 + 测试表 |
| gguf 写字节 | gguf.cpp:1425+ | gguf_write.rs:96+（gguf_write_ref.bin） |
| 类型表 43 行 | ggml.c:632-1006 | types.rs:72-114（测试） |
| cpu traits 表 | ggml-cpu.c:215-416 | vec_dot.rs:1070-1102 + compute.rs:5438 |

---

## 7. 各文件"0 missed"声明

以下文件在本审计口径下 **0 MISSED**（全部符号有对应或已归档的 documented-skip）：
`ggml-common.h`(blocks.rs)、`ggml.cpp`、`ggml-threading.cpp`、`gguf.cpp`（读+写+split）、
`ggml-cpu/vec.h/.cpp`、`ggml-cpu/binary-ops.cpp`、`ggml-cpu/repack.cpp`（5 族中 2 族 documented-skip
带行号）、`ggml-cpu/llamafile/sgemm.cpp`（6/6 组合）、`ggml-cpu/traits.cpp`、`ggml-cpu/ggml-cpu.cpp`、
`arch-fallback.h`、`simd-mappings.h`、`ggml.h`。
带 MISSED 的文件: `ggml.c`（§4.1/§5-A.2 dispatch 臂）、`ggml-cpu/ggml-cpu.c`+`quants.c`（§5-A.1）、
`ops.cpp`（UNARY 11 子项 + §5-A.3）、`ggml-quants.c`（TQ/IQ 量化器，§5-B.4）。
