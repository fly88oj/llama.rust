//! phi3_diff.rs — P4 侦查：实模型 Phi-4-mini Q6_K 上 phi3 前向与参考的数值分歧定位。
//!
//! 背景（代理 L 既得）：prompt `"The capital of France is"` 上参考稳定输出
//! `" Paris. What is the capital of Germany? ..."`，本移植在第 7 个 token 分叉
//! （参考 ` Germany`，本移植 ` France`）。qwen2 在同协议下 64/64 对齐，所以分歧
//! 只能来自 phi3 特有的图构建逻辑。
//!
//! # 结论（2026-09-24，代理 N）
//! **根因 = 移植缺少 `cparams.yarn_attn_factor *= hparams.rope_attn_factor`
//! （参考 src/llama-context.cpp:214）**。Phi-4-mini 的 GGUF 带
//! `phi3.rope.scaling.attn_factor = 1.190238118171692`，参考把它乘进 rope 的
//! `mscale`（llama-graph.cpp:1476 `attn_factor (cparams.yarn_attn_factor)`
//! → `ggml_rope_ext(...)` 的 attn_factor → ops.cpp:5959 `rope_yarn`
//! `*cos_theta = cosf(theta) * mscale`），于是参考把旋转通道的 Q/K 放大 1.1902 倍；
//! 本移植传 attn_factor = 1.0（`hp.yarn_attn_factor`，未乘 `hp.rope_attn_factor`），
//! 旋转通道幅度差 19%、注意力分数里旋转部分的贡献差 mscale² = 41.7%。
//!
//! ## 证据链（两条独立路线，互相闭合）
//! 1. `phi3_ref_gap_trace`（现配置，attn_factor = 1.0）：第 7 步复现分叉，
//!    France − Germany = **+1.79** logits（参考 −1.35，即参考 Germany 高 1.35），
//!    pair-gap 最大偏差 6.53，轨迹 9/10。
//! 2. `phi3_attn_factor_probe`（attn_factor = 1.190238118171692，并把 ext_factor
//!    从 -1.0 归 0.0、n_ctx_orig 从 131072 归 4096，即完全照参考 cparams 接线）：
//!    **逐 token 复现参考的 10 个 token**（含 ` Germany`），pair-gap 偏差降到
//!    0.01..0.56。
//! 3. 反向对照（参考侧，单字节改动）：把同一 GGUF 的
//!    `phi3.rope.scaling.attn_factor` 改成 1.0（`cp --reflink` 不适用 → 全量
//!    cp 到 /tmp + patch 文件偏移 109 的 4 字节；`cmp` 确认全文件只有这 3 字节不同），
//!    参考 server 的输出翻转为 ` Paris. What is the capital of France? The`
//!    —— 与本移植现输出一致，且本移植（attn_factor = 1.0）与该参考轨迹 10/10 相同
//!    （见 `phi3_attn1_reference_trace` 与 `REF_TOP6_ATTN1`）。两侧在同一位点上
//!    互换了行为，只有 attn_factor 这一个变量。
//! 4. 参考侧自身稳定：`-t 1` 与 `-t 8` 的 top-20 logprob 逐位一致（0.000 差），
//!    两个 fresh 实例的 capture 完全一致 → 残差不是参考的抖动。
//!
//! ## 逐疑点结论（全部排除，附 C 行号）
//! 1. **部分 rope（n_rot=96 < head_dim=128）**：与 C 一致。`phi3_rope_mscale_check`
//!    在 ne0=128/n_rot=96/NEOX 下把移植的 rope op 与 C 公式
//!    （`ggml_rope_cache_init` ops.cpp:5975 + `rope_yarn` ops.cpp:5959 +
//!    `rotate_pairs` ops.cpp:6063 + 非旋转通道回填 ops.cpp:6217-6226）
//!    **逐位对齐（0/768 位差）**；通道映射为 pairs (j, j+48)（rotate_pairs 的
//!    scale=2 → `ic = i0/2`），非旋转通道 96..127 原样复制。
//! 2. **ffn_up 两半切分**：与 C 一致。参考走 phi3.cpp:149
//!    `build_ffn(up, NULL, NULL, NULL /*gate*/, ..., LLM_FFN_SWIGLU, LLM_FFN_SEQ)`
//!    → llama-graph.cpp:1885 `ggml_swiglu(cur)`，CPU 内核（ops.cpp:3189-3224，
//!    `nc = ne0/2`、`swapped = 0`）即 `dst = silu(前半) * 后半`
//!    （vec.cpp:417 `ggml_v_silu(x[i])*g[i]`）。`phi3_ffn_up_swiglu_split_check`
//!    （0/288 位差）验证移植的 views+silu+mul 组合与此逐位相同。
//! 3. **融合 qkv 切分**：与 C 一致。llama-graph.cpp:1667-1674 的三段 view_3d
//!    （偏移 0 / row_size(n_embd_q) / row_size(n_embd_q+n_embd_k)，
//!    `nb1 = row_size(type, head_dim)`，`nb2 = qkv->nb[1]`，reshape=true 见
//!    llama-graph.cpp:1619-1631）；`phi3_fused_qkv_split_check` 用分段标记权重
//!    在 24/8 头几何下验证三段偏移全部正确（0 失配）。
//! 4. **longrope / rope factors**：数值上是 no-op。llama-model.cpp:2259
//!    `get_rope_factors` 在 `n_ctx_seq <= n_ctx_orig_yarn`（512 <= 4096）时返回
//!    `rope_short`，作为 `ggml_rope_ext` 的 src2 → `freq_factors`；
//!    `rope_yarn(theta/ff)` 中 ff=1.0 时 `x/1.0f == x` 位精确。
//!    `phi3_rope_factors_are_all_one` 实测 rope_short 48 个值全 = 1.0
//!    （rope_long 才非 1）。顺带发现参考侧 quirk：C 的 cache 循环按 ne0=128
//!    访问 freq_factors[0..64]，而张量只有 n_rot/2=48 个元素（越界读 mmap 尾部，
//!    因前 48 个全是 1.0 且实际只用到 n_dims=96 个通道，无影响）。
//!
//! ## 修复建议（≤5 行，未施加 —— 本文件只读实现）
//! `crates/tools/llama-cli/src/main.rs` 的 `AttnParams`（唯一生产接线点）：
//! ```diff
//! -        ext_factor: hp.yarn_ext_factor,
//! -        attn_factor: hp.yarn_attn_factor,
//! +        // llama-context.cpp:173-175: 负值 = "未设置" → YARN ? 1.0 : 0.0
//! +        ext_factor: if hp.yarn_ext_factor < 0.0 { 0.0 } else { hp.yarn_ext_factor },
//! +        // llama-context.cpp:214 `cparams.yarn_attn_factor *= hparams.rope_attn_factor`
//! +        attn_factor: hp.yarn_attn_factor * hp.rope_attn_factor,
//! ```
//! 另需把 `crates/llama/src/hparams.rs:570` 的默认值 `rope_attn_factor: 0.0`
//! 改为 `1.0`（参考 llama-hparams.h:149 `float rope_attn_factor = 1.0f;`），
//! 否则缺该 key 的模型会整条 rope 归零。`n_ctx_orig` 建议同时改为
//! `hp.n_ctx_orig_yarn`（llama-graph.cpp:1486；仅在 ext_factor != 0 时有数值影响，
//! 属语义修正）。同一改法需同步 `crates/llama/tests/arch_e2e.rs:243` 的
//! `attn_params()` 辅助函数（测试口径）。
//!
//! ## 已知残留（新发现，超出 P4 范围，已定位到机制，见 `phi3_layerwise_probe`）
//! 修好 attn_factor 后，本移植与参考的 pair-gap 仍有 0.01..0.56 logits 的偏差
//! （10 步 greedy 轨迹不受影响，修好 attn_factor 后 10/10 一致）。这不是
//! attn_factor 造成的（两侧都归到 1.0 时偏差同量级），也不是本移植独有：
//!
//! * `phi3_layerwise_probe`（用参考 libllama 的 cb_eval dump 每层张量，见
//!   `/tmp/ref_dump.c`）显示 0/1/2 层全部张量在 **≤3e-6（1-2 ulp）** 内一致，
//!   **首个 >1e-3 的偏差出现在第 3 层的 ffn_out-3 / l_out-3**（且只出现在
//!   token 3/4 —— 即 attend 过更多 KV 的靠后位置），此后逐层放大到
//!   result_output 的 max|d| = 1.46 logits。这是 F16 KV cache 的“量化阈值翻转”
//!   放大：Kcur/Vcur 只有 1e-6 级差异，落到 F16（相对精度 4.9e-4）时个别元素
//!   跨过舍入边界 → 注意力输出跳 1e-4 → 残差流逐层放大。
//! * 同一机制在参考侧同样存在，量级相同：同一 binary、同一模型，只把 KV cache
//!   从 F16 换成 F32（`-ctk f32 -ctv f32`），参考**自己**的 pair-gap 就变了
//!   0.1..2.59 logits，第 7 步答案从 ` Germany` 变成 ` Italy`（这正是 PARITY.md
//!   记录的“参考自身双模态”的同一根源）。
//!
//! 结论：残差 = 已归档的 K-quant mul_mat 最后 1-2 ulp 差异（PARITY.md
//! “已知数值差异来源” 1）经 F16 KV cache 放大后的必然结果，< 0.6 logits 且不改
//! 变 10 步轨迹，不阻塞 P4。要做到更严格的一致性只能改 KV cache 精度
//! （两侧同时改 F32）或逐内核复刻 AVX512 lane 顺序。
//!
//! # 复现跑法
//! ```text
//! # 1) 参考侧（fresh server + 首请求，PARITY.md 协议；端口避开 8801/8802/8805）
//! REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
//! $REF/llama-server -m <Phi-4-mini-Q6_K.gguf> -c 512 -t 8 -fa off --port 8820 &
//! curl -s localhost:8820/completion -H 'Content-Type: application/json' \
//!   -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0,
//!        "top_k":1,"cache_prompt":false,"logprobs":20}' > /tmp/ref.json
//! # 1b) 反向对照：attn_factor = 1.0 的参考（把 /tmp 副本的 f32 元数据改掉）
//! python3 - <<'PY'
//! # 定位 phi3.rope.scaling.attn_factor 的值偏移并在副本上写 1.0f（偏移 109）
//! import struct; f=open('/tmp/phi4mini-attn1.gguf','r+b'); f.seek(109)
//! f.write(struct.pack('<f',1.0)); f.close()
//! PY
//! $REF/llama-server -m /tmp/phi4mini-attn1.gguf ... --port 8821 &   # -> " France"
//! # 1c) KV 精度对照：同一模型 -ctk f32 -ctv f32（第 7 步变 " Italy"，gap 动 0.1..2.59）
//! # 2) 本移植（release；调试构建在这个 3.8B 模型上太慢）
//! cargo test -p llama --release --test phi3_diff -- --ignored --nocapture
//! # 3) 逐层对照（可选）：参考侧 dump + phi3_layerwise_probe
//! gcc -O2 -I<sp>/llama.cpp-pinned/include -I<sp>/ggml/include /tmp/ref_dump.c \
//!     -o /tmp/ref_dump -L$REF -lllama -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$REF
//! /tmp/ref_dump <Phi-4-mini-Q6_K.gguf> /tmp/refdump
//! cargo test -p llama --release --test phi3_diff -- --ignored --nocapture phi3_layerwise_probe
//! ```
//! 参考的 top-6 已内置为 `REF_TOP6` / `REF_TOP6_ATTN1`（每步 (id, logprob)，由上面
//! 的 curl 捕获），所以前两个端到端测试不需要 server 也能跑。
//!
//! 只读约束：本文件不修改任何实现代码；`build_phi3_forward` 只通过 `AttnParams`
//! 参数化（attn_factor / ext_factor / n_ctx_orig 三个字段的接线差异即根因）。

use std::path::Path;
use std::sync::Arc;

use ggml::types::GgmlType;
use ggml::{Context, Gguf, Graph, TensorId};
use llama::graph::{AttnParams, DecodeInputs, ForwardResult};
use llama::graph_arch::{build_phi3_forward, Phi3LayerWeights, Phi3ModelWeights};
use llama::kv_cache::KvCache;
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const PHI4_MINI: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q6_K.gguf";

const PROMPT: &str = "The capital of France is";

/// Reference capture: fresh `llama-server` (bd4f514db1, `-c 512 -t 8 -fa off`,
/// port 8820), first request, `temperature=0, top_k=1, cache_prompt=false,
/// logprobs=20`. Per step the first 6 `top_logprobs` as (token id, logprob).
/// Chosen ids: 12650 ` Paris`, 13 `.`, 4614 ` What`, 382 ` is`, 290 ` the`,
/// 9029 ` capital`, 328 ` of`, 17237 ` Germany`, 30 `?`, 623 ` The`.
const REF_TOP6: [&[(i32, f32)]; 10] = [
    &[
        (12650, -0.307318),
        (64429, -2.977507),
        (723, -3.768935),
        (5542, -4.138571),
        (148812, -4.331694),
        (625, -4.450861),
    ],
    &[
        (13, -0.585195),
        (558, -1.370012),
        (11, -2.245878),
        (326, -3.672788),
        (364, -3.983087),
        (3692, -5.158636),
    ],
    &[
        (4614, -1.663613),
        (12650, -2.140946),
        (199999, -2.341611),
        (623, -2.659652),
        (6432, -3.128592),
        (3253, -3.263133),
    ],
    &[
        (382, -0.347691),
        (1078, -3.058140),
        (5030, -3.488747),
        (4931, -3.675087),
        (673, -4.071983),
        (6439, -4.144585),
    ],
    &[
        (290, -0.102439),
        (1617, -4.106040),
        (480, -4.381064),
        (3613, -4.522868),
        (261, -4.643874),
        (220, -5.104762),
    ],
    &[
        (9029, -0.088060),
        (1308, -4.070131),
        (10574, -4.076841),
        (11540, -4.489328),
        (9107, -5.582949),
        (13749, -5.613970),
    ],
    &[
        (328, -0.060317),
        (5030, -2.894324),
        (30, -6.987772),
        (885, -7.103560),
        (3901, -8.149787),
        (81835, -8.245784),
    ],
    &[
        (17237, -1.147846),
        (22384, -1.266163),
        (26350, -1.834084),
        (10128, -2.500591),
        (10198, -3.120243),
        (10351, -3.669399),
    ],
    &[
        (30, -0.224707),
        (3901, -1.984137),
        (1715, -3.346556),
        (16842, -4.572966),
        (59668, -4.946753),
        (63062, -6.242819),
    ],
    &[
        (623, -1.451924),
        (21230, -1.843495),
        (12650, -1.918179),
        (355, -3.498215),
        (10128, -3.524777),
        (17237, -3.611928),
    ],
];

/// Reference greedy ids (PHI4_MINI_REF16[..10] in arch_e2e.rs).
const REF_IDS: [i32; 10] = [12650, 13, 4614, 382, 290, 9029, 328, 17237, 30, 623];

/// Same protocol, same prompt, but on a copy of the GGUF whose only difference
/// is one patched metadata f32: `phi3.rope.scaling.attn_factor` 1.190238118171692
/// → 1.0 (the patch is 4 bytes at file offset 109, see the module doc). This is
/// the reference's *own* negative control for the root cause: with that field
/// neutralised the reference emits ` France` at index 7, i.e. the port's current
/// behaviour. Captured with a fresh server on port 8821, first request.
const REF_TOP6_ATTN1: [&[(i32, f32)]; 10] = [
    &[
        (12650, -0.391576),
        (64429, -3.537473),
        (5542, -3.614053),
        (625, -3.914964),
        (261, -4.059679),
        (21230, -4.179903),
    ],
    &[
        (13, -0.679871),
        (558, -1.166401),
        (11, -2.473828),
        (364, -3.733517),
        (326, -3.989399),
        (3692, -4.560627),
    ],
    &[
        (4614, -1.856971),
        (623, -2.207668),
        (12650, -2.258009),
        (4101, -3.088430),
        (199999, -3.238481),
        (3253, -3.318599),
    ],
    &[
        (382, -0.225289),
        (5030, -3.388367),
        (1078, -3.918816),
        (2226, -3.965981),
        (553, -4.014649),
        (4931, -4.064042),
    ],
    &[
        (290, -0.122838),
        (480, -3.524057),
        (220, -3.682123),
        (1617, -4.015520),
        (9029, -4.812089),
        (261, -4.873781),
    ],
    &[
        (9029, -0.066185),
        (10574, -4.143757),
        (11540, -4.525345),
        (1308, -4.661770),
        (152167, -5.720753),
        (7769, -6.171898),
    ],
    &[
        (328, -0.017038),
        (5030, -4.150819),
        (30, -8.235963),
        (885, -8.781900),
        (9029, -9.052476),
        (326, -9.133069),
    ],
    &[
        (10128, -0.424039),
        (17237, -2.168588),
        (22384, -2.870695),
        (10198, -2.937109),
        (26350, -3.431970),
        (24868, -4.107286),
    ],
    &[
        (30, -0.231026),
        (3901, -2.140221),
        (1715, -2.915467),
        (59668, -4.334572),
        (16842, -4.935997),
        (11, -6.090813),
    ],
    &[
        (623, -0.440564),
        (12650, -1.557015),
        (10128, -4.125802),
        (533, -4.347176),
        (23530, -4.566590),
        (199999, -5.178444),
    ],
];

/// `phi3.rope.scaling.attn_factor` in the GGUF — the reference's effective
/// `cparams.yarn_attn_factor` (llama-context.cpp:214; hparams.yarn_attn_factor
/// is 1.0 here). Also used as the fallback when the port's hparam is unset
/// (the port's default is 0.0, C's is 1.0 — see the report).
const REF_ATTN_FACTOR_FALLBACK: f32 = 1.190238118171692;

// ---------------------------------------------------------------------------
// harness (same protocol as arch_e2e.rs' RealHarness; local copy so this file
// is self-contained)
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
}

fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    match load_model(&gguf, mmap.clone()) {
        Ok(model) => Some(Loaded { model, gguf, mmap }),
        Err(e) => {
            eprintln!("SKIP: load_model({path}) failed: {e}");
            None
        }
    }
}

fn phi3_weights(m: &LlamaModel) -> Phi3ModelWeights {
    Phi3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .map(|x| Phi3LayerWeights {
                attn_norm: x.attn_norm.expect("attn_norm"),
                wqkv: x.wqkv.expect("fused wqkv"),
                wqkv_b: x.wqkv_b,
                wo: x.wo.expect("wo"),
                wo_b: x.wo_b,
                ffn_norm: x.ffn_norm.expect("ffn_norm"),
                ffn_down: x.ffn_down.expect("ffn_down"),
                ffn_up: x.ffn_up.expect("ffn_up"),
            })
            .collect(),
    }
}

/// Port-wiring AttnParams, verbatim from `crates/tools/llama-cli/src/main.rs:246`
/// (and arch_e2e.rs' `attn_params`) — i.e. what the port actually runs today.
fn attn_params_port(m: &LlamaModel) -> AttnParams {
    let hp = &m.hparams;
    AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: hp.n_ctx_train as i32,
        freq_base: hp.rope_freq_base_train,
        freq_scale: if hp.rope_freq_scale_train == 0.0 {
            1.0
        } else {
            hp.rope_freq_scale_train
        },
        ext_factor: hp.yarn_ext_factor,
        attn_factor: hp.yarn_attn_factor,
        beta_fast: hp.yarn_beta_fast,
        beta_slow: hp.yarn_beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        // anchor: the reference capture / the whole P4 investigation is `-fa off`
        use_flash_attn: false,
    }
}

/// Reference-wiring AttnParams: the three fields the port wires differently,
/// taken through the C chain
///   * `attn_factor` = cparams.yarn_attn_factor = hparams.yarn_attn_factor
///                     * hparams.rope_attn_factor       (llama-context.cpp:214)
///   * `ext_factor`  = cparams.yarn_ext_factor          (llama-context.cpp:173-175:
///                     `< 0` → `rope_scaling_type == YARN ? 1.0 : 0.0`; phi3's
///                     scaling type is "linear" → 0.0)
///   * `n_ctx_orig`  = cparams.n_ctx_orig_yarn          (llama-graph.cpp:1486;
///                     the phi3 GGUF carries rope.scaling.original_context_length=4096)
fn attn_params_ref(m: &LlamaModel) -> AttnParams {
    let hp = &m.hparams;
    let rope_attn = if hp.rope_attn_factor == 0.0 {
        REF_ATTN_FACTOR_FALLBACK // port default is 0.0; the GGUF carries the real value
    } else {
        hp.rope_attn_factor
    };
    AttnParams {
        n_ctx_orig: hp.n_ctx_orig_yarn as i32,
        ext_factor: 0.0,
        attn_factor: hp.yarn_attn_factor * rope_attn,
        ..attn_params_port(m)
    }
}

fn threads() -> usize {
    8
}

struct RealHarness {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
}

impl RealHarness {
    fn new(gctx: Context, kv: KvCache) -> Self {
        let watermark = gctx.mark();
        RealHarness {
            gctx,
            kv,
            watermark,
        }
    }

    /// Decode `tokens` at `pos`; returns last-token logits [n_vocab].
    fn decode(
        &mut self,
        w: &Phi3ModelWeights,
        ap: &AttnParams,
        tokens: &[i32],
        pos: &[i32],
    ) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        // assign first, then the (256-padded) n_kv — step_inputs order
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let kq_mask = self
            .gctx
            .new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| p.copy_from_slice(pos))
            .unwrap();
        {
            let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
            self.gctx
                .data_bytes_mut(row_idx)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&idxs));
        }
        {
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
            mask.fill(f32::NEG_INFINITY);
            // padded (empty) cells keep pos = -1 → stay masked
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            for (iq, &qp) in pos.iter().enumerate() {
                for (ik, &kp) in kv_pos.iter().enumerate() {
                    if 0 <= kp && kp <= qp {
                        mask[iq * n_kv as usize + ik] = 0.0;
                    }
                }
            }
        }
        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let result: ForwardResult =
            build_phi3_forward(&mut self.gctx, w, ap, &self.kv, &inputs, sinfo, n_kv, n);
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, threads());
        // (assign already happened before the graph build, step_inputs order)

        let n_vocab = self.gctx.ne(logits)[0] as usize;
        let all: &[f32] = bytemuck::cast_slice(self.gctx.data_bytes(logits).unwrap());
        all[n_vocab * (n - 1)..n_vocab * n].to_vec()
    }
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

fn topk(v: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.into_iter().take(k).map(|i| (i as i32, v[i])).collect()
}

fn piece(vocab: &Vocab, id: i32) -> String {
    format!("{:?}", vocab.token_to_piece(id))
}

fn text_of(vocab: &Vocab, ids: &[i32]) -> String {
    ids.iter()
        .map(|&t| vocab.token_to_piece(t).to_string())
        .collect::<Vec<_>>()
        .concat()
}

/// Greedy `n_steps` with one AttnParams configuration. Returns the per-step
/// last-token logits (step 0 = prefill of the 5-token prompt) and the greedy ids.
fn run_greedy(
    h: &mut RealHarness,
    w: &Phi3ModelWeights,
    ap: &AttnParams,
    prompt: &[i32],
    n_steps: usize,
) -> (Vec<Vec<f32>>, Vec<i32>) {
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let mut cur = h.decode(w, ap, prompt, &pos);
    let mut ids = Vec::new();
    let mut logits = Vec::new();
    let mut p = prompt.len() as i32;
    for step in 0..n_steps {
        let tok = argmax(&cur);
        ids.push(tok);
        logits.push(cur.clone());
        if step + 1 == n_steps {
            break;
        }
        cur = h.decode(w, ap, &[tok], &[p]);
        p += 1;
    }
    (logits, ids)
}

/// Per-step two-sided table: reference top-3 vs port top-3, plus the port's
/// pair gap for every id in the reference's top-6. Gaps are normalised to each
/// side's own argmax, so they are directly comparable (a log-softmax shift
/// cancels in differences).
fn print_comparison(
    vocab: &Vocab,
    label: &str,
    table: &[&[(i32, f32)]],
    logits: &[Vec<f32>],
    ids: &[i32],
) {
    println!("\n===== {label} =====");
    for (step, (lv, &tok)) in logits.iter().zip(ids).enumerate() {
        let port_top = topk(lv, 3);
        let p0 = port_top[0].1;
        let ref0 = table[step][0].1;
        println!("step {step}: port argmax {tok} {}", piece(vocab, tok));
        print!("   ref :");
        for (id, lp) in table[step].iter().take(3) {
            print!(" {id}{}={:+.3}", piece(vocab, *id), lp - ref0);
        }
        println!();
        print!("   port:");
        for (id, l) in &port_top {
            print!(" {id}{}={:+.3}", piece(vocab, *id), l - p0);
        }
        println!();
        print!("   pair gap (id: ref / port):");
        for (id, lp) in table[step].iter() {
            let pg = lv[*id as usize] - p0;
            print!(" {id}: {:+.3} / {:+.3}", lp - ref0, pg);
        }
        println!();
    }
    let ref_ids: Vec<i32> = table.iter().map(|t| t[0].0).collect();
    let same = ids.iter().zip(&ref_ids).filter(|(a, b)| a == b).count();
    println!(
        "MATCH {same}/10 vs reference ids; text = {:?}",
        text_of(vocab, ids)
    );
}

/// Worst |pair-gap delta| against the reference's top-6, over all steps.
fn worst_pair_gap_delta(table: &[&[(i32, f32)]], logits: &[Vec<f32>]) -> f32 {
    let mut worst = 0f32;
    for (step, lv) in logits.iter().enumerate() {
        let p0 = lv[argmax(lv) as usize];
        let ref0 = table[step][0].1;
        for (id, lp) in table[step].iter() {
            worst = worst.max(((lv[*id as usize] - p0) - (lp - ref0)).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// 1. the port as it runs today (production wiring) — reproduces the divergence
// ---------------------------------------------------------------------------

/// The port's production wiring (`attn_factor = hp.yarn_attn_factor` = 1.0,
/// `ext_factor = hp.yarn_ext_factor` = -1.0, `n_ctx_orig = n_ctx_train`) —
/// reproduces the OPEN P4 divergence and prints the two-sided logit-gap table.
///
/// Asserts what L reported: a 7-token prefix and the France/Germany flip.
#[test]
#[ignore = "manual: 3.8B real model, ~1-2 min in release; no server needed (REF_TOP6 is embedded)"]
fn phi3_ref_gap_trace() {
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let hp = l.model.hparams.clone();
    println!(
        "hparams: n_rot={} head_dim={} n_ctx_train={} n_ctx_orig_yarn={} rope_attn_factor={} \
         yarn_attn_factor={} yarn_ext_factor={} freq_scale_train={} rope_type={:?}",
        hp.n_rot(0),
        hp.n_embd_head_k(0),
        hp.n_ctx_train,
        hp.n_ctx_orig_yarn,
        hp.rope_attn_factor,
        hp.yarn_attn_factor,
        hp.yarn_ext_factor,
        hp.rope_freq_scale_train,
        hp.rope_type,
    );
    let ap = attn_params_port(&l.model);
    println!(
        "port AttnParams: ext_factor={} attn_factor={} n_ctx_orig={}",
        ap.ext_factor, ap.attn_factor, ap.n_ctx_orig
    );
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k = l.model.n_embd_k_gqa_max() as i64;
    let n_v = l.model.n_embd_v_gqa_max() as i64;
    let prompt = vocab.tokenize(PROMPT, true, true);
    println!("prompt ids: {prompt:?}");

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    let mut h = RealHarness::new(gctx, kv);
    let (logits, ids) = run_greedy(&mut h, &w, &ap, &prompt, 10);
    print_comparison(
        &vocab,
        "port (attn_factor = 1.0, ext_factor = -1.0)",
        &REF_TOP6,
        &logits,
        &ids,
    );
    println!(
        "worst |pair-gap delta| vs reference = {:.4}",
        worst_pair_gap_delta(&REF_TOP6, &logits)
    );

    let prefix = ids.iter().zip(&REF_IDS).take_while(|(a, b)| a == b).count();
    assert!(
        prefix >= 7,
        "expected the documented 7-token prefix, got {prefix}"
    );
    assert_ne!(
        ids[7], REF_IDS[7],
        "port must reproduce the idx=7 flip (France vs Germany)"
    );
    let l7 = &logits[7];
    let p0 = l7[argmax(l7) as usize];
    let (g_fr, g_de) = (l7[10128] - p0, l7[17237] - p0);
    let r_of = |id: i32| REF_TOP6[7].iter().find(|(i, _)| *i == id).unwrap().1 - REF_TOP6[7][0].1;
    println!(
        "step 7 port: France {:+.3}, Germany {:+.3} (France - Germany = {:+.3})",
        g_fr,
        g_de,
        g_fr - g_de
    );
    println!(
        "step 7 ref : France {:+.3}, Germany {:+.3} (France - Germany = {:+.3})",
        r_of(10128),
        r_of(17237),
        r_of(10128) - r_of(17237)
    );
}

// ---------------------------------------------------------------------------
// 2. the hypothesis: reference wiring (attn_factor *= rope_attn_factor)
// ---------------------------------------------------------------------------

/// Same run with the reference's cparams wiring
/// (`attn_factor = 1.0 * 1.190238118171692`, `ext_factor = 0.0`,
/// `n_ctx_orig = 4096`). If the missing `*= rope_attn_factor` is the root cause,
/// this reproduces the reference's 10 tokens exactly — including ` Germany` at
/// index 7 — and the pair gaps at every step collapse onto the reference values.
///
/// Cross-check (manual, no test contacts a server): copy the GGUF with
/// `cp --reflink=always`, patch the f32 at the `phi3.rope.scaling.attn_factor`
/// metadata slot to 1.0, start a fresh reference server on that copy and repeat
/// the curl above — the reference then emits ` France` at index 7, i.e. the two
/// implementations swap places exactly when this one field is neutralised.
#[test]
#[ignore = "manual: 3.8B real model, ~1-2 min in release; no server needed"]
fn phi3_attn_factor_probe() {
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let hp = l.model.hparams.clone();
    let ap = attn_params_ref(&l.model);
    println!(
        "ref AttnParams: ext_factor={} attn_factor={:.15} (rope_attn_factor={} * yarn_attn_factor={}) n_ctx_orig={}",
        ap.ext_factor, ap.attn_factor, hp.rope_attn_factor, hp.yarn_attn_factor, ap.n_ctx_orig
    );
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k = l.model.n_embd_k_gqa_max() as i64;
    let n_v = l.model.n_embd_v_gqa_max() as i64;
    let prompt = vocab.tokenize(PROMPT, true, true);

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    let mut h = RealHarness::new(gctx, kv);
    let (logits, ids) = run_greedy(&mut h, &w, &ap, &prompt, 10);
    print_comparison(
        &vocab,
        "port with reference attn_factor (1.190238118171692)",
        &REF_TOP6,
        &logits,
        &ids,
    );

    let same = ids.iter().zip(&REF_IDS).filter(|(a, b)| a == b).count();
    let worst = worst_pair_gap_delta(&REF_TOP6, &logits);
    let worst_patched = worst_pair_gap_delta(&REF_TOP6_ATTN1, &logits);
    println!("worst |pair-gap delta| over steps 0..9 = {worst:.4} logits (vs the real reference)");
    println!("  same measure against the attn_factor=1.0-patched reference = {worst_patched:.4}");
    assert_eq!(
        ids,
        REF_IDS.to_vec(),
        "reference wiring must reproduce the reference trajectory ({same}/10 matched)"
    );
    // The trajectory (the actual OPEN P4 symptom) is now exact. The residual
    // pair-gap delta is NOT an attn_factor artefact: it is ~0.1-0.6 against the
    // real reference here AND ~0.1-0.5 when the *unpatched* port config is
    // compared with the attn_factor=1.0-patched reference (see
    // phi3_attn1_reference_trace) — i.e. the port's general logit-level noise
    // floor, present for both attn_factor values. Kept as a loose guard.
    assert!(
        worst < 1.0,
        "pair gaps should track the reference (worst {worst:.4})"
    );
}

/// The port's *current* production wiring (attn_factor = 1.0) compared against
/// the reference on the attn_factor=1.0-patched file (`REF_TOP6_ATTN1`). Both
/// sides then use the same effective mscale, so this isolates the port's
/// residual logit-level difference from the root cause: it is ~0.1-0.5 logits,
/// the same band seen in `phi3_attn_factor_probe`, and does not change the
/// greedy trajectory (10/10 tokens match).
#[test]
#[ignore = "manual: 3.8B real model, ~1-2 min in release"]
fn phi3_attn1_reference_trace() {
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let ap = attn_params_port(&l.model);
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k = l.model.n_embd_k_gqa_max() as i64;
    let n_v = l.model.n_embd_v_gqa_max() as i64;
    let prompt = vocab.tokenize(PROMPT, true, true);

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    let mut h = RealHarness::new(gctx, kv);
    let (logits, ids) = run_greedy(&mut h, &w, &ap, &prompt, 10);
    print_comparison(
        &vocab,
        "port (attn_factor = 1.0) vs reference patched to attn_factor = 1.0",
        &REF_TOP6_ATTN1,
        &logits,
        &ids,
    );
    let worst = worst_pair_gap_delta(&REF_TOP6_ATTN1, &logits);
    println!("worst |pair-gap delta| over steps 0..9 = {worst:.4} logits");
    let ref_ids: Vec<i32> = REF_TOP6_ATTN1.iter().map(|t| t[0].0).collect();
    assert_eq!(
        ids, ref_ids,
        "the port must walk the attn_factor=1.0 reference's trajectory"
    );
    assert!(worst < 1.0, "same noise band expected (worst {worst:.4})");
}

/// Numerical conditioning probe: how much do the pair gaps move when the rope
/// cache is perturbed by ~1 ulp (the only effect of the port's
/// `ext_factor = -1.0` vs the reference's 0.0, `phi3_rope_mscale_check` measures
/// 1.19e-7 relative) and by an inert parameter change (n_ctx_orig only feeds
/// corr_dims, which are dead code when ext_factor == 0 — must be bit-identical).
///
/// This bounds how much of the residual in `phi3_attn_factor_probe` can be
/// blamed on the port's remaining parameter deltas: if a 1-ulp rope change moves
/// the gaps by ~0.1-0.5 logits, the residual against the reference is
/// conditioning-amplified noise, not a missing component.
#[test]
#[ignore = "manual: 3 runs x 10 steps on a 3.8B model (~2 min in release)"]
fn phi3_sensitivity_probe() {
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let base = attn_params_ref(&l.model);
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k = l.model.n_embd_k_gqa_max() as i64;
    let n_v = l.model.n_embd_v_gqa_max() as i64;
    let prompt = vocab.tokenize(PROMPT, true, true);

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    let mut h = RealHarness::new(gctx, kv);

    let run = |h: &mut RealHarness, ap: &AttnParams, label: &str| {
        // NOTE: without this the previous run's cells stay in the cache and are
        // still visible in the mask (their positions are <= the new query
        // positions), which silently changes the logits by O(1) — measured as a
        // 4.89-logit max |d| before this was added.
        h.kv.clear();
        let (logits, ids) = run_greedy(h, &w, ap, &prompt, 10);
        println!(
            "{label}: ids {:?} ({}/10 vs reference)",
            ids,
            ids.iter().zip(&REF_IDS).filter(|(a, b)| a == b).count()
        );
        logits
    };
    let a = run(
        &mut h,
        &base,
        "A ext=0, mscale=1.1902, n_ctx_orig=4096 (reference wiring)",
    );
    let b = run(
        &mut h,
        &AttnParams {
            ext_factor: -1.0,
            n_ctx_orig: 131072,
            ..base.clone()
        },
        "B ext=-1.0 (port's value), same mscale",
    );
    let c = run(
        &mut h,
        &AttnParams {
            ext_factor: 0.0,
            n_ctx_orig: 131072,
            ..base.clone()
        },
        "C ext=0, n_ctx_orig=131072 (inert, must equal A bit-for-bit)",
    );

    let (dc, dmax_c) = cmp_logits(&a, &c);
    println!("A vs C (inert param): {dc} bit-differing logits, max |d| = {dmax_c:e}");
    assert_eq!(
        dc, 0,
        "n_ctx_orig must be inert at ext_factor == 0 ({dc} differ, {dmax_c:e})"
    );
    let (db, dmax_b) = cmp_logits(&a, &b);
    println!(
        "A vs B (1 ulp rope perturbation from the port's ext_factor = -1.0): \
         {db} bit-differing logits, max |d| = {dmax_b:e}"
    );
    println!(
        "A-vs-B worst |pair-gap delta| = {:.4} logits",
        worst_delta_between(&REF_TOP6, &a, &b)
    );
    // Captured: BOTH are exactly 0 bit-differing logits (max |d| = 0). The rope
    // cache does differ by ~1.2e-7 relative in the ramp region, but every
    // downstream rounding step absorbs it — so sub-ulp input differences are
    // provably NOT the source of the residual in `phi3_attn_factor_probe`.
    assert!(
        dmax_b < 1e-3,
        "the port's ext_factor=-1.0 must be numerically inert ({dmax_b:e})"
    );
    println!(
        "=> the port's logits are bit-stable under sub-ulp perturbations, so the \
         0.01..0.56 pair-gap residual vs the reference is NOT conditioning noise: \
         it needs a difference of >= ~1e-6 relative somewhere (see module doc)."
    );
}

/// (bit-differing count, max |delta|) between two logit vectors.
fn cmp_logits(a: &[Vec<f32>], b: &[Vec<f32>]) -> (usize, f32) {
    let mut n = 0usize;
    let mut mx = 0f32;
    for (x, y) in a.iter().zip(b) {
        for (p, q) in x.iter().zip(y) {
            if p.to_bits() != q.to_bits() {
                n += 1;
            }
            mx = mx.max((p - q).abs());
        }
    }
    (n, mx)
}

// ---------------------------------------------------------------------------
// 4. layer-by-layer localisation against the reference's own intermediate
//    tensors (dumped with libllama's cb_eval by /tmp/ref_dump.c)
// ---------------------------------------------------------------------------

/// Strided f32 read of a (possibly non-contiguous) tensor into row-major order.
fn read_strided(ctx: &Context, id: TensorId) -> Vec<f32> {
    let t = *ctx.ne(id);
    let nb = *ctx.nb(id);
    let ne: Vec<i64> = t.iter().copied().take_while(|&x| x > 1).collect();
    let ne = if ne.is_empty() { vec![1] } else { ne };
    let total: i64 = ne.iter().product();
    let bytes = ctx.data_bytes(id).expect("tensor data");
    let mut out = vec![0f32; total as usize];
    let f32_at =
        |off: usize| -> f32 { f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) };
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..ne.len() {
            let idx = rem % ne[d];
            rem /= ne[d];
            off += idx as usize * nb[d] as usize;
        }
        *slot = f32_at(off);
    }
    out
}

/// A copy of `graph_arch::build_phi3_forward` (plus its private `attn_kv_cached`
/// helper) that records the tensors the reference's graph exposes through `cb()`
/// (phi3.cpp: attn_norm / Qcur / Kcur / Vcur / ffn_norm / ffn_out / l_out /
/// result_norm / result_output). The final logits are asserted bit-identical to
/// `build_phi3_forward`'s by the caller, so the taps are trustworthy.
fn build_phi3_forward_taps(
    ctx: &mut Context,
    w: &Phi3ModelWeights,
    p: &AttnParams,
    kv: &KvCache,
    inp: &DecodeInputs,
    n_kv: u32,
    n_tokens: usize,
    taps: &mut Vec<(String, TensorId)>,
) -> ForwardResult {
    let t = n_tokens as i64;
    let n_layer = w.layers.len();
    let mut graph = Graph::new(1024);
    let mut inp_l = ctx.get_rows(w.tok_embd, inp.tokens);

    for il in 0..n_layer {
        let lw = &w.layers[il];
        let residual = inp_l;

        let cur = build_norm_rms_local(ctx, inp_l, lw.attn_norm, p.norm_eps);
        taps.push((format!("attn_norm-{il}"), cur));

        let mut qkv = ctx.mul_mat(lw.wqkv, cur);
        if let Some(b) = lw.wqkv_b {
            qkv = ctx.add(qkv, b);
        }
        let n_embd_q = p.n_embd_head_k * p.n_head;
        let n_embd_k = p.n_embd_head_k * p.n_head_kv;
        let qkv_nb1 = ctx.nb(qkv)[1] as usize;
        let hd_k = p.n_embd_head_k as usize;
        let hd_v = p.n_embd_head_v as usize;
        let q = ctx.view_3d(qkv, p.n_embd_head_k, p.n_head, t, hd_k * 4, qkv_nb1, 0);
        let k = ctx.view_3d(
            qkv,
            p.n_embd_head_k,
            p.n_head_kv,
            t,
            hd_k * 4,
            qkv_nb1,
            n_embd_q as usize * 4,
        );
        let v = ctx.view_3d(
            qkv,
            p.n_embd_head_v,
            p.n_head_kv,
            t,
            hd_v * 4,
            qkv_nb1,
            (n_embd_q + n_embd_k) as usize * 4,
        );
        let rope = |ctx: &mut Context, x: TensorId| {
            ctx.rope_ext(
                x,
                inp.pos,
                None,
                p.n_rot as i32,
                p.rope_mode,
                p.n_ctx_orig,
                p.freq_base,
                p.freq_scale,
                p.ext_factor,
                p.attn_factor,
                p.beta_fast,
                p.beta_slow,
            )
        };
        let q = rope(ctx, q);
        let k = rope(ctx, k);
        taps.push((format!("Qcur_pre-{il}"), q));
        taps.push((format!("Kcur-{il}"), k));
        taps.push((format!("Vcur-{il}"), v));
        let q = ctx.scale(q, 1.0 / (p.n_embd_head_k as f32).sqrt());
        taps.push((format!("Qcur-{il}"), q));

        // --- attn_kv_cached (graph_arch.rs:602) verbatim
        let k_rows = {
            let nb2 = ctx.nb(k)[2] as usize;
            ctx.view_2d(k, p.n_embd_head_k * p.n_head_kv, t, nb2, 0)
        };
        let v_rows = {
            let nb2 = ctx.nb(v)[2] as usize;
            ctx.view_2d(v, p.n_embd_head_v * p.n_head_kv, t, nb2, 0)
        };
        let k_dst = ctx.set_rows(kv.layers[il].k, k_rows, inp.row_idx);
        let v_dst = ctx.set_rows(kv.layers[il].v, v_rows, inp.row_idx);
        graph.build_forward(ctx, k_dst);
        graph.build_forward(ctx, v_dst);
        let k_view = kv.get_k(ctx, il, p.n_embd_head_k, p.n_head_kv, n_kv);
        let v_view = kv.get_v(ctx, il, p.n_embd_head_v, p.n_head_kv, n_kv);
        let qp = ctx.permute(q, 0, 2, 1, 3);
        let k_view = ctx.permute(k_view, 0, 2, 1, 3);
        let kq = ctx.mul_mat(k_view, qp);
        let kq = ctx.soft_max_ext(kq, Some(inp.kq_mask), 1.0, 0.0);
        let v_view = ctx.permute(v_view, 0, 2, 1, 3);
        let v_t = ctx.transpose(v_view);
        let v_c = ctx.cont(v_t);
        let kqv = ctx.mul_mat(v_c, kq);
        let kqv = ctx.permute(kqv, 0, 2, 1, 3);
        let kqv = ctx.cont(kqv);
        let kqv = ctx.reshape_2d(kqv, p.n_embd_head_v * p.n_head, t);
        let mut cur = ctx.mul_mat(lw.wo, kqv);
        if let Some(b) = lw.wo_b {
            cur = ctx.add(cur, b);
        }
        let _ = lw.wo_b;

        let residual = ctx.add(cur, residual);
        let cur = build_norm_rms_local(ctx, residual, lw.ffn_norm, p.norm_eps);
        taps.push((format!("ffn_norm-{il}"), cur));

        let up2 = ctx.mul_mat(lw.ffn_up, cur);
        let n_ff = ctx.ne(lw.ffn_down)[0];
        let nb_row = ctx.nb(up2)[1] as usize;
        let gate = ctx.view_2d(up2, n_ff, t, nb_row, 0);
        let up = ctx.view_2d(up2, n_ff, t, nb_row, n_ff as usize * 4);
        let silu_gate = ctx.silu(gate);
        let prod = ctx.mul(silu_gate, up);
        let cur = ctx.mul_mat(lw.ffn_down, prod);
        taps.push((format!("ffn_out-{il}"), cur));

        inp_l = ctx.add(residual, cur);
        taps.push((format!("l_out-{il}"), inp_l));
    }

    let cur = build_norm_rms_local(ctx, inp_l, w.output_norm, p.norm_eps);
    taps.push(("result_norm".to_string(), cur));
    let mut logits = ctx.mul_mat(w.output, cur);
    if let Some(b) = w.output_b {
        logits = ctx.add(logits, b);
    }
    taps.push(("result_output".to_string(), logits));
    graph.build_forward(ctx, logits);
    ForwardResult {
        logits,
        embd: Some(cur),
        graph,
    }
}

fn build_norm_rms_local(ctx: &mut Context, x: TensorId, w: TensorId, eps: f32) -> TensorId {
    let n = ctx.rms_norm(x, eps);
    ctx.mul(n, w)
}

/// Source of the reference-side layer dumper, written to `/tmp/ref_dump.c` by
/// `phi3_layerwise_probe` when that file is missing. Build/run:
///
/// ```text
/// gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
///     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include /tmp/ref_dump.c -o /tmp/ref_dump \
///     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
///     -lllama -lggml -lggml-base -lggml-cpu -lm \
///     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
/// /tmp/ref_dump <Phi-4-mini-Q6_K.gguf> /tmp/refdump
/// ```
const REF_DUMP_C: &str = r#"// ref_dump.c — dump named intermediate tensors of the reference build for one
// Phi-4-mini prefill, for layer-by-layer comparison against the Rust port.
//
// build: gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/include \
//            -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//            /tmp/ref_dump.c -o /tmp/ref_dump \
//            -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//            -lllama -lggml -lggml-base -lggml-cpu \
//            -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run:   /tmp/ref_dump <model.gguf> <outdir> [n_predict]
//
// The callback protocol (ggml-backend.cpp:1798-1835): the sched asks with
// ask=true for every node; returning true forces that node to be computed and
// handed back with ask=false (data valid, backend synchronized).

#include "llama.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DUMPS 64

static const char * g_names[MAX_DUMPS];
static int          g_ndump = 0;
static char         g_dir[512];
static int          g_active = 0;

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!g_active) {
        return false; // ignore the graph_reserve/warmup passes
    }
    if (ask) {
        const char * name = ggml_get_name(t);
        for (int i = 0; i < g_ndump; i++) {
            if (strcmp(name, g_names[i]) == 0) {
                return true;
            }
        }
        return false;
    }

    const char * name = ggml_get_name(t);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", g_dir, name);

    const int64_t n = ggml_nelements(t);
    float * buf = (float *) malloc((size_t) n * sizeof(float));
    if (!buf) {
        fprintf(stderr, "OOM for %s\n", name);
        return true;
    }
    // strided read: views (e.g. Vcur) have nb[2] equal to the parent row size,
    // so a raw memcpy mixes tokens; walk the elements instead.
    {
        const int nd = ggml_n_dims(t);
        const int64_t * ne = t->ne;
        const size_t * nb = t->nb;
        for (int64_t flat = 0; flat < n; flat++) {
            int64_t rem = flat;
            size_t off = 0;
            for (int d = 0; d < nd; d++) {
                const int64_t idx = rem % ne[d];
                rem /= ne[d];
                off += (size_t) idx * nb[d];
            }
            const char * p = (const char *) t->data + off;
            buf[flat] = t->type == GGML_TYPE_F32
                ? *(const float *) p
                : ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        }
    }
    if (0) {
        fprintf(stderr, "unsupported type %d for %s\n", (int) t->type, name);
        free(buf);
        return true;
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        free(buf);
        return true;
    }
    fwrite(buf, sizeof(float), (size_t) n, f);
    fclose(f);

    double sum = 0.0, mn = INFINITY, mx = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        sum += buf[i];
        mn = fmin(mn, buf[i]);
        mx = fmax(mx, buf[i]);
    }
    printf("DUMP %-16s ne=[%lld,%lld,%lld,%lld] type=%d n=%lld sum=%.9g min=%.9g max=%.9g first=%.9g\n",
           name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
           (int) t->type, (long long) n, sum, mn, mx, (double) buf[0]);
    fflush(stdout);
    free(buf);
    return true;
}

static void add(const char * fmt_il, int il) {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt_il, il);
    g_names[g_ndump++] = strdup(buf);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <outdir> [n_predict]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    snprintf(g_dir, sizeof(g_dir), "%s", argv[2]);
    const int n_predict = argc > 3 ? atoi(argv[3]) : 0;

    // layers 0..3 + the last two, plus the model-level taps
    const int layers[] = {0, 1, 2, 3, 30, 31};
    for (size_t i = 0; i < sizeof(layers)/sizeof(layers[0]); i++) {
        const int il = layers[i];
        add("attn_norm-%d", il);
        add("Qcur-%d", il);
        add("Kcur-%d", il);
        add("Vcur-%d", il);
        add("ffn_norm-%d", il);
        add("ffn_out-%d", il);
        add("l_out-%d", il);
    }
    add("%s", 0); // placeholder, replaced below
    g_ndump--;
    g_names[g_ndump++] = strdup("result_norm");
    g_names[g_ndump++] = strdup("result_output");

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 512;
    cparams.n_batch          = 512;
    cparams.n_ubatch         = 512;
    cparams.n_threads        = 8;
    cparams.n_threads_batch  = 8;
    cparams.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval          = cb_eval;
    cparams.cb_eval_user_data = NULL;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    const char * prompt = "The capital of France is";
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token toks[16];
    int nt = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks, 16, true, true);
    if (nt <= 0) {
        fprintf(stderr, "tokenize failed: %d\n", nt);
        return 1;
    }
    printf("prompt tokens (%d):", nt);
    for (int i = 0; i < nt; i++) printf(" %d", toks[i]);
    printf("\n");
    fflush(stdout);

    struct llama_batch batch = llama_batch_init(nt + n_predict, 0, 1);
    for (int i = 0; i < nt; i++) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1; // all positions -> out_ids == identity, shapes match the port
    }
    batch.n_tokens = nt;
    g_active = 1;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    if (n_predict > 0) {
        // greedy: argmax over the last row of logits, then one more step
        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
        printf("argmax after prefill: %d\n", best);
        fflush(stdout);
        llama_batch b1 = llama_batch_get_one(&best, 1);
        b1.pos[0] = nt;
        if (llama_decode(ctx, b1) != 0) {
            fprintf(stderr, "decode2 failed\n");
            return 1;
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}"#;

/// Writes the dumper source to /tmp/ref_dump.c and prints the build line when the
/// file is absent (keeps this probe reproducible without extra repo files).
fn ensure_ref_dumper() {
    let p = std::path::Path::new("/tmp/ref_dump.c");
    if !p.exists() {
        if let Err(e) = std::fs::write(p, REF_DUMP_C) {
            eprintln!("cannot write /tmp/ref_dump.c: {e}");
            return;
        }
        println!("wrote /tmp/ref_dump.c — now build it (see the phi3_layerwise_probe doc)");
    }
}

/// Compares this port's per-layer tensors against the reference dump produced by
/// `/tmp/ref_dump.c` (libllama `cb_eval`; see that file for the build/run line)
/// and prints the first layer/tensor where they diverge, with the max |delta|
/// and the relative max |delta|.
///
/// Setup (one-off):
///   gcc -O2 -I<span>/llama.cpp-pinned/include -I<...>/ggml/include /tmp/ref_dump.c \
///       -o /tmp/ref_dump -L<span>/build-rust-ref/bin -lllama -lggml -lggml-base \
///       -lggml-cpu -lm -Wl,-rpath,<...>/build-rust-ref/bin
///   /tmp/ref_dump <Phi-4-mini-Q6_K.gguf> /tmp/refdump
/// then `cargo test -p llama --release --test phi3_diff -- --ignored --nocapture phi3_layerwise_probe`.
///
/// Reference tensor shapes are 4D [ne0, 5, 1, 1] with the 5 prefill positions;
/// the port's taps are read with their own strides so the comparison is
/// element-for-element in the same linear order.
#[test]
#[ignore = "manual: needs /tmp/refdump (see the doc comment) + a 3.8B real forward"]
fn phi3_layerwise_probe() {
    let dir = std::path::Path::new("/tmp/refdump");
    if !dir.exists() {
        ensure_ref_dumper();
        eprintln!("SKIP: build + run /tmp/ref_dump first (see the test doc comment)");
        return;
    }
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let ap = attn_params_ref(&l.model);
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k = l.model.n_embd_k_gqa_max() as i64;
    let n_v = l.model.n_embd_v_gqa_max() as i64;
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    let mut h = RealHarness::new(gctx, kv);

    // fresh prefill, exactly like the reference run (positions 0..4, all outputs)
    let sinfo = h.kv.find_slot(prompt.len() as u32).unwrap();
    h.gctx.reset_graph_to(h.watermark);
    let tokens_t = h.gctx.new_tensor_1d(GgmlType::I32, prompt.len() as i64);
    let pos_t = h.gctx.new_tensor_1d(GgmlType::I32, prompt.len() as i64);
    let kq_mask = h
        .gctx
        .new_tensor_2d(GgmlType::F32, prompt.len() as i64, prompt.len() as i64);
    let row_idx = h.gctx.new_tensor_1d(GgmlType::I64, prompt.len() as i64);
    for tid in [tokens_t, pos_t, kq_mask, row_idx] {
        h.gctx.arena_resize_tensor(tid);
    }
    h.gctx
        .with_i32_mut(tokens_t, |p| p.copy_from_slice(&prompt))
        .unwrap();
    h.gctx
        .with_i32_mut(pos_t, |p| p.copy_from_slice(&pos))
        .unwrap();
    {
        let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
        h.gctx
            .data_bytes_mut(row_idx)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&idxs));
    }
    {
        let mask: &mut [f32] = bytemuck::cast_slice_mut(h.gctx.data_bytes_mut(kq_mask).unwrap());
        mask.fill(f32::NEG_INFINITY);
        for q in 0..prompt.len() {
            for k in 0..=q {
                mask[q * prompt.len() + k] = 0.0;
            }
        }
    }
    let inputs = DecodeInputs {
        tokens: tokens_t,
        pos: pos_t,
        kq_mask,
        row_idx,
        out_ids: None,
    };

    let mut taps: Vec<(String, TensorId)> = Vec::new();
    let res = build_phi3_forward_taps(
        &mut h.gctx,
        &w,
        &ap,
        &h.kv,
        &inputs,
        prompt.len() as u32,
        prompt.len(),
        &mut taps,
    );
    let mut gf = res.graph;
    for (_, id) in &taps {
        gf.build_forward(&h.gctx, *id);
    }
    ggml::compute::graph_compute(&mut h.gctx, &mut gf, threads());
    h.kv.assign(sinfo, &pos, 0);

    // read every tap out before any further graph activity (decode() resets the arena)
    let dumped: Vec<(String, Vec<f32>)> = taps
        .iter()
        .map(|(n, id)| (n.clone(), read_strided(&h.gctx, *id)))
        .collect();

    // sanity: the tapped copy must equal the production builder bit-for-bit
    h.kv.clear();
    let prod_logits = h.decode(&w, &ap, &prompt, &pos);
    let n_vocab = prod_logits.len();
    let tap_out = dumped.last().expect("result_output tap").1.clone();
    let tap_last: Vec<f32> = tap_out[n_vocab * (prompt.len() - 1)..n_vocab * prompt.len()].to_vec();
    let (d, mx) = cmp_vec(&prod_logits, &tap_last);
    println!("tap-copy vs build_phi3_forward logits: {d} bit-differing, max |d| = {mx:e}");
    assert_eq!(
        d, 0,
        "the tapped copy must reproduce build_phi3_forward bit-exactly"
    );

    println!("\n=== layer-by-layer vs /tmp/refdump ===");
    let mut first_bad: Option<String> = None;
    for (name, pv) in &dumped {
        let path = dir.join(format!("{name}.f32"));
        if !path.exists() {
            println!("  {name:16} (no reference file)");
            continue;
        }
        let rb = std::fs::read(&path).expect("read ref dump");
        let rv: Vec<f32> = rb
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if rv.len() != pv.len() {
            println!(
                "  {name:16} LEN MISMATCH ref {} vs port {}",
                rv.len(),
                pv.len()
            );
            continue;
        }
        let mut mx = 0f32;
        let mut n_bitdiff = 0usize;
        let mut n_over_1e5 = 0usize;
        let mut first = None;
        for (i, (r, p)) in rv.iter().zip(pv).enumerate() {
            if r.to_bits() != p.to_bits() {
                n_bitdiff += 1;
            }
            let dd = (r - p).abs();
            if dd > 1e-5 {
                n_over_1e5 += 1;
                if first.is_none() {
                    first = Some((i, *r, *p));
                }
            }
            mx = mx.max(dd);
        }
        let flag = if mx > 1e-3 { " <<< DIVERGES" } else { "" };
        println!(
            "  {name:16} n={:7} max|d|={mx:.3e} bit-diff={n_bitdiff:6} >1e-5={n_over_1e5:6}{flag} first={first:?}",
            rv.len()
        );
        if mx > 1e-3 && first_bad.is_none() {
            first_bad = Some(name.clone());
        }
    }
    println!("\nfirst diverging tensor (max|d| > 1e-3): {first_bad:?}");
}

fn cmp_vec(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut n = 0usize;
    let mut mx = 0f32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
        }
        mx = mx.max((x - y).abs());
    }
    (n, mx)
}

/// Worst difference between the pair gaps of two runs of the same config family.
fn worst_delta_between(table: &[&[(i32, f32)]], a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut worst = 0f32;
    for step in 0..a.len() {
        let (pa, pb) = (
            a[step][argmax(&a[step]) as usize],
            b[step][argmax(&b[step]) as usize],
        );
        for (id, _) in table[step].iter() {
            let ga = a[step][*id as usize] - pa;
            let gb = b[step][*id as usize] - pb;
            worst = worst.max((ga - gb).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// 3. component checks (cheap, deterministic, no model needed)
// ---------------------------------------------------------------------------

/// Port rope op vs a naive transcription of the C reference
/// (`ggml_rope_cache_init` ops.cpp:5975 + `rope_yarn` ops.cpp:5959 +
/// `rotate_pairs` ops.cpp:6063 + the non-rotated pass-through loop
/// ops.cpp:6217-6226), at Phi-4-mini's partial-rope geometry
/// (ne0 = 128, n_rot = 96, NEOX) and at the reference's yarn parameters.
///
/// Also quantifies the two port/reference parameter differences:
///   * attn_factor (mscale) 1.0 vs 1.190238118171692 → 19% magnitude on the
///     rotated channels (the root cause);
///   * ext_factor -1.0 vs 0.0 (the port passes hparams.yarn_ext_factor = -1.0
///     straight through) → ~1 ulp reassociation only.
#[test]
fn phi3_rope_mscale_check() {
    // Phi-4-mini geometry
    let ne0 = 128usize;
    let n_rot = 96usize;
    let n_head = 2usize;
    let seq = 3usize;
    let base = 10000.0f32;

    let mut ctx = Context::new();
    let a = ctx.new_tensor_3d(GgmlType::F32, ne0 as i64, n_head as i64, seq as i64);
    ctx.arena_resize_tensor(a);
    let mut st = 0x1234_5678u32;
    let mut rnd = move || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        ((st >> 8) as f32 / 16777216.0) - 0.5
    };
    let x0: Vec<f32> = (0..ne0 * n_head * seq).map(|_| rnd()).collect();
    ctx.with_f32_mut(a, |p| p.copy_from_slice(&x0)).unwrap();
    let pos_data = [0i32, 3, 41];
    let pos = ctx.new_tensor_1d(GgmlType::I32, seq as i64);
    ctx.arena_resize_tensor(pos);
    ctx.with_i32_mut(pos, |p| p.copy_from_slice(&pos_data))
        .unwrap();

    // naive C transcription
    let naive = |ext_factor: f32, mscale: f32, n_ctx_orig: i32| -> Vec<f32> {
        let n_dims = n_rot;
        let theta_scale = base.powf(-2.0 / n_dims as f32);
        // ggml_rope_yarn_corr_dims (ggml.c:4469)
        let corr_dim = |n: f32| {
            n_dims as f32 * (n_ctx_orig as f32 / (n * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let corr = [
            corr_dim(32.0).floor().max(0.0),
            corr_dim(1.0).ceil().min((n_dims - 1) as f32),
        ];
        let mut out = vec![0f32; ne0 * n_head * seq];
        for i2 in 0..seq {
            let mut cache = vec![0f32; ne0];
            let mut theta = pos_data[i2] as f32;
            for i0 in (0..ne0).step_by(2) {
                // rope_yarn (ops.cpp:5959); ff == 1 (rope_short factors are all 1.0)
                let theta_interp = 1.0 * theta;
                let mut t = theta_interp;
                let mut ms = mscale;
                if ext_factor != 0.0 {
                    let (lo, hi) = (corr[0], corr[1]);
                    let ramp = ((i0 as f32 - lo) / (hi - lo)).clamp(0.0, 1.0) * ext_factor;
                    t = theta_interp * (1.0 - ramp) + theta * ramp;
                    ms *= 1.0 + 0.1 * (1.0f32).ln();
                }
                cache[i0] = t.cos() * ms;
                cache[i0 + 1] = t.sin() * ms;
                theta *= theta_scale;
            }
            for i1 in 0..n_head {
                let src = &x0[(i2 * n_head + i1) * ne0..][..ne0];
                let dst = &mut out[(i2 * n_head + i1) * ne0..][..ne0];
                dst.copy_from_slice(src);
                // rotate_pairs(n_dims, n_dims/2, cache, src, dst, 2) — with
                // the reference build's FMA contraction of ops.cpp:6076-6077
                // (right product rounded, left product fused; see
                // compute.rs rotate_pairs_f32)
                let mut i0 = 0;
                while i0 < n_dims {
                    let ic = i0 / 2;
                    let (ct, st_) = (cache[i0], cache[i0 + 1]);
                    let (xv0, xv1) = (src[ic], src[ic + n_dims / 2]);
                    dst[ic] = xv0.mul_add(ct, -(xv1 * st_));
                    dst[ic + n_dims / 2] = xv0.mul_add(st_, xv1 * ct);
                    i0 += 2;
                }
                // pass-through for channels >= n_rot (ops.cpp:6217, n_offs == 0)
                dst[n_dims..].copy_from_slice(&src[n_dims..]);
            }
        }
        out
    };

    fn run(
        ctx: &mut Context,
        a: TensorId,
        pos: TensorId,
        n_rot: i32,
        n_ctx_orig: i32,
        base: f32,
        ext_factor: f32,
        mscale: f32,
    ) -> Vec<f32> {
        let y = ctx.rope_ext(
            a, pos, None, n_rot, 2, n_ctx_orig, base, 1.0, ext_factor, mscale, 32.0, 1.0,
        );
        let mut g = Graph::new(4);
        g.build_forward(ctx, y);
        ctx.arena_resize_tensor(y);
        ggml::compute::graph_compute(ctx, &mut g, 1);
        ctx.f32s(y).unwrap().to_vec()
    }

    // 1) the port op must equal the naive C formula bit-for-bit at the reference
    //    parameters (validates the partial-rope pass-through + channel mapping)
    let ref_cfg = run(
        &mut ctx,
        a,
        pos,
        n_rot as i32,
        4096,
        base,
        0.0,
        REF_ATTN_FACTOR_FALLBACK,
    );
    let want = naive(0.0, REF_ATTN_FACTOR_FALLBACK, 4096);
    let mut bad = 0usize;
    let mut max_rel = 0f32;
    for (g, w) in ref_cfg.iter().zip(&want) {
        if g.to_bits() != w.to_bits() {
            bad += 1;
        }
        max_rel = max_rel.max((g - w).abs() / w.abs().max(1e-3));
    }
    println!(
        "port rope vs naive C formula (ext=0, mscale=1.1902381, n_rot=96/ne0=128): \
         {bad}/{} bit-differing, max rel {max_rel:e}",
        want.len()
    );

    // 2) quantify the two parameter deltas the port currently applies
    let port_cfg = run(&mut ctx, a, pos, n_rot as i32, 131072, base, -1.0, 1.0);
    let ext_only = run(&mut ctx, a, pos, n_rot as i32, 4096, base, 0.0, 1.0);
    let mscale_only = run(
        &mut ctx,
        a,
        pos,
        n_rot as i32,
        4096,
        base,
        0.0,
        REF_ATTN_FACTOR_FALLBACK,
    );
    let rel = |got: &[f32], want: &[f32]| -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1e-3))
            .fold(0f32, f32::max)
    };
    println!(
        "mscale 1.0 vs 1.1902381 (rotated channels only): rel diff = {:e}  (attention-score factor mscale^2 = {:.6})",
        rel(&ext_only, &mscale_only),
        REF_ATTN_FACTOR_FALLBACK * REF_ATTN_FACTOR_FALLBACK
    );
    println!(
        "ext_factor -1.0 (port) vs 0.0 (ref) at mscale=1.0: rel diff = {:e} (reassociation only)",
        rel(&port_cfg, &ext_only)
    );
    // non-rotated channels must be an exact copy in every configuration
    for cfg in [&port_cfg, &ext_only, &mscale_only] {
        for i1 in 0..n_head {
            for i2 in 0..seq {
                let row = &cfg[(i2 * n_head + i1) * ne0..][..ne0];
                for c in n_rot..ne0 {
                    assert_eq!(
                        row[c],
                        x0[(i2 * n_head + i1) * ne0 + c],
                        "pass-through channel {c} not copied"
                    );
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "port rope must match the C formula bit-for-bit ({bad} differ)"
    );
}

/// ffn_up emits [2*n_ff, T]; the reference applies `ggml_swiglu` to the whole
/// tensor (phi3.cpp:149 `build_ffn(..., up, NULL, NULL, NULL /*gate*/, ..., down,
/// ..., LLM_FFN_SWIGLU, LLM_FFN_SEQ, il)` → llama-graph.cpp:1885
/// `cur = ggml_swiglu(ctx0, cur)`), whose CPU kernel splits each row in half
/// (ops.cpp:3189-3224): `nc = src0->ne[0]/2`, `swapped = 0` → gate = first half,
/// up = second half, `dst = silu(gate) * up` (vec.cpp:417
/// `ggml_v_silu(x[i]) * g[i]`).
///
/// This test verifies the port's composed views + silu + mul produce exactly
/// that on a synthetic [2*n_ff, T] F32 tensor: silu on the FIRST half only, no
/// interleaving, a product (not a sum).
#[test]
fn phi3_ffn_up_swiglu_split_check() {
    let n_ff = 96i64;
    let t = 3i64;
    let mut ctx = Context::new();
    let up2 = ctx.new_tensor_2d(GgmlType::F32, 2 * n_ff, t);
    ctx.arena_resize_tensor(up2);
    let mut st = 0xdead_beefu32;
    let mut rnd = move || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        ((st >> 8) as f32 / 16777216.0) * 8.0 - 4.0
    };
    let vals: Vec<f32> = (0..(2 * n_ff * t)).map(|_| rnd()).collect();
    ctx.with_f32_mut(up2, |p| p.copy_from_slice(&vals)).unwrap();

    // exactly graph_arch.rs build_phi3_forward's phi3 swiGLU composition
    let nb_row = ctx.nb(up2)[1] as usize;
    let gate = ctx.view_2d(up2, n_ff, t, nb_row, 0);
    let up = ctx.view_2d(up2, n_ff, t, nb_row, n_ff as usize * 4);
    let silu_gate = ctx.silu(gate);
    let prod = ctx.mul(silu_gate, up);
    let mut g = Graph::new(4);
    g.build_forward(&ctx, prod);
    ctx.arena_resize_tensor(prod);
    ggml::compute::graph_compute(&mut ctx, &mut g, 1);
    let got = ctx.f32s(prod).unwrap().to_vec();

    // C: dst[i] = silu(first_half[i]) * second_half[i] with the AVX512 silu
    // x/(1+exp(-x)) the port's ggml_silu_f32 replicates
    let mut bad = 0usize;
    for r in 0..t as usize {
        for i in 0..n_ff as usize {
            let gv = vals[r * 2 * n_ff as usize + i];
            let uv = vals[r * 2 * n_ff as usize + n_ff as usize + i];
            let want = ggml::ops::ggml_silu_f32(gv) * uv;
            if got[r * n_ff as usize + i].to_bits() != want.to_bits() {
                bad += 1;
            }
        }
    }
    println!(
        "swiglu split: {bad}/{} bit-differing vs silu(first_half)*second_half",
        n_ff * t
    );
    assert_eq!(bad, 0, "swiglu split mismatch: {bad} values differ");
    // negative control: the second half must NOT be silu'ed, and it must not be
    // an interleaved (even/odd) split
    let uv0 = vals[n_ff as usize];
    assert!((got[0] / uv0 - ggml::ops::ggml_silu_f32(vals[0]) / vals[0]).abs() > 1e-3);
    assert!(got[0].is_finite() && got[(n_ff * t - 1) as usize].is_finite());
}

/// Fused qkv split (llama-graph.cpp:1667-1674): Q/K/V are views into the
/// `[n_embd, n_embd_q + n_embd_k + n_embd_v]` matmul output with
/// `nb1 = row_size(type, head_dim)`, `nb2 = qkv->nb[1]`, offsets
/// `0 / row_size(n_embd_q) / row_size(n_embd_q + n_embd_k)`; `reshape = true`
/// (llama-graph.cpp:1619-1631), so the views stay 3D and no `cont` follows.
/// Verified at Phi-4-mini head geometry (24 q-heads, 8 kv-heads, head_dim 128
/// scaled down to 4/2/8 here) against a naive segment split of a permuted weight.
#[test]
fn phi3_fused_qkv_split_check() {
    let n_embd = 64i64;
    let (n_head, n_head_kv, hd) = (4i64, 2i64, 8i64);
    let n_q = hd * n_head;
    let n_k = hd * n_head_kv;
    let n_v = hd * n_head_kv;
    let t = 2i64;

    let mut ctx = Context::new();
    let w = ctx.new_tensor_2d(GgmlType::F32, n_embd, n_q + n_k + n_v);
    ctx.arena_resize_tensor(w);
    // column j of w is constant = 1000 + j, so each output column identifies its
    // source column unambiguously
    let mut wv = vec![0f32; (n_embd * (n_q + n_k + n_v)) as usize];
    for j in 0..(n_q + n_k + n_v) as usize {
        for i in 0..n_embd as usize {
            wv[j * n_embd as usize + i] = 1000.0 + j as f32;
        }
    }
    ctx.with_f32_mut(w, |p| p.copy_from_slice(&wv)).unwrap();
    let x = ctx.new_tensor_2d(GgmlType::F32, n_embd, t);
    ctx.arena_resize_tensor(x);
    ctx.with_f32_mut(x, |p| p.fill(0.01)).unwrap();

    // exactly build_phi3_forward's split
    let qkv = ctx.mul_mat(w, x);
    let qkv_nb1 = ctx.nb(qkv)[1] as usize;
    let q = ctx.view_3d(qkv, hd, n_head, t, hd as usize * 4, qkv_nb1, 0);
    let k = ctx.view_3d(
        qkv,
        hd,
        n_head_kv,
        t,
        hd as usize * 4,
        qkv_nb1,
        n_q as usize * 4,
    );
    let v = ctx.view_3d(
        qkv,
        hd,
        n_head_kv,
        t,
        hd as usize * 4,
        qkv_nb1,
        (n_q + n_k) as usize * 4,
    );
    // materialise each view so the comparison does not depend on view strides
    let (qc, kc, vc) = (ctx.cont(q), ctx.cont(k), ctx.cont(v));
    let mut g = Graph::new(8);
    for tid in [qc, kc, vc] {
        g.build_forward(&ctx, tid);
    }
    for tid in [qc, kc, vc] {
        ctx.arena_resize_tensor(tid);
    }
    ggml::compute::graph_compute(&mut ctx, &mut g, 1);

    for (name, tid, heads, off) in [
        ("Q", qc, n_head as usize, 0usize),
        ("K", kc, n_head_kv as usize, n_q as usize),
        ("V", vc, n_head_kv as usize, (n_q + n_k) as usize),
    ] {
        let got = ctx.f32s(tid).unwrap().to_vec();
        // cont'd [hd, heads, t] is row-major: (d0, d1, d2) -> d0 + d1*hd + d2*heads*hd
        let mut bad = 0usize;
        let mut first_bad = None;
        for d2 in 0..t as usize {
            for d1 in 0..heads {
                for d0 in 0..hd as usize {
                    let want =
                        (1000.0 + (off + d1 * hd as usize + d0) as f32) * n_embd as f32 * 0.01;
                    let gv = got[d0 + d1 * hd as usize + d2 * heads * hd as usize];
                    if (gv - want).abs() > 1e-1 {
                        bad += 1;
                        if first_bad.is_none() {
                            first_bad = Some((d0, d1, d2, gv, want));
                        }
                    }
                }
            }
        }
        println!(
            "qkv split {name}: {bad} mismatching elements of {} ({first_bad:?})",
            got.len()
        );
        assert_eq!(
            bad, 0,
            "{name} segment mismatch — fused split offsets are wrong"
        );
    }
}

/// `rope_short`/`rope_long` factors: phi3.cpp:97 `model.get_rope_factors()`
/// (llama-model.cpp:2259) returns `rope_short` whenever
/// `n_ctx_seq <= n_ctx_orig_yarn` (512 <= 4096 here), and the reference feeds it
/// as `ggml_rope_ext`'s src2 → `freq_factors` (ops.cpp:6157, 5977:
/// `const float ff = freq_factors ? freq_factors[i0/2] : 1.0f; ... rope_yarn(theta/ff, ...)`).
/// `x / 1.0f == x` bit-exactly, so the factors are numerically inert iff every
/// stored value is exactly 1.0 — this test checks the real file's two tensors.
#[test]
fn phi3_rope_factors_are_all_one() {
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let mut short_non_one = usize::MAX;
    for (name, id) in [
        ("rope_short", l.model.layers[0].rope_short),
        ("rope_long", l.model.layers[0].rope_long),
    ] {
        let Some(id) = id else {
            println!("{name}: absent");
            continue;
        };
        let v: Vec<f32> = l
            .model
            .ctx
            .data_bytes(id)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let non_one = v.iter().filter(|&&x| x != 1.0).count();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "{name}: n={} min={mn} max={mx} non-1.0 entries = {non_one}",
            v.len()
        );
        if name == "rope_short" {
            short_non_one = non_one;
        }
    }
    let hp = &l.model.hparams;
    println!(
        "n_ctx_seq=512 <= n_ctx_orig_yarn={} -> reference uses rope_short (inert iff all 1.0)",
        hp.n_ctx_orig_yarn
    );
    // C reads freq_factors[i0/2] for i0 < ne0 (head_dim 128) -> index up to 63,
    // while the tensor is only n_rot/2 = 48 long: an out-of-bounds read of the
    // mmap'd file in the reference (llama.cpp quirk; harmless here because every
    // value is 1.0 for the first 48 and the tail is never used for rotation,
    // n_rot = 96 <= ne0)
    println!(
        "rope factor tensor len {} < head_dim/2 {} (C's cache loop reads up to head_dim/2-1)",
        hp.n_rot(0) / 2,
        hp.n_embd_head_k(0) / 2
    );
    if short_non_one != usize::MAX {
        assert_eq!(
            short_non_one, 0,
            "rope_short has {short_non_one} non-1.0 entries — factors matter"
        );
    }
}
