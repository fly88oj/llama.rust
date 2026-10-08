//! quant_strategy.rs — table tests for `llama::quant` (the port of
//! llama-quant.cpp's `llama_tensor_get_type`).
//!
//! Every expectation is hand-derived from the C source (line numbers quoted in
//! llama/src/quant.rs) for a Qwen2.5-0.5B-shaped model — the same shape as the
//! local `qwen2.5-0.5b-instruct-q4_k_m.gguf` used by the byte-parity test in
//! `crates/tools/quantize/tests/`.

use ggml::types::GgmlType;
use llama::arch::LlmArch;
use llama::hparams::LlamaHparams;
use llama::quant::*;

/// llama-quant.cpp:104-154 — category assignment, incl. the order of the
/// `find` checks.
#[test]
fn categories_match_reference_order() {
    use TensorCategory::*;
    let cases: &[(&str, TensorCategory)] = &[
        ("token_embd.weight", TokenEmbd),
        ("per_layer_token_embd.weight", TokenEmbd),
        ("output.weight", Output),
        ("blk.0.attn_qkv.weight", AttentionQkv),
        ("blk.0.attn_kv_b.weight", AttentionKvB),
        ("blk.0.attn_v.weight", AttentionV),
        ("blk.0.attn_k.weight", AttentionK),
        ("blk.0.attn_q.weight", AttentionQ),
        ("blk.0.attn_output.weight", AttentionOutput),
        ("blk.0.ffn_up.weight", FfnUp),
        ("blk.0.ffn_gate.weight", FfnGate),
        ("blk.0.ffn_down.weight", FfnDown),
        ("blk.0.ffn_gate_exps.weight", FfnGate),
        ("blk.0.ffn_gate_up_exps.weight", FfnGate),
        ("blk.0.attn_norm.weight", Other),
    ];
    for (name, want) in cases {
        assert_eq!(tensor_get_category(name), *want, "{name}");
    }
    assert!(category_is_attn_v(AttentionV));
    assert!(category_is_attn_v(AttentionQkv));
    assert!(category_is_attn_v(AttentionKvB));
    assert!(!category_is_attn_v(AttentionK));
}

/// llama-quant.cpp:847-890 — the ftype → default type table.
#[test]
fn ftype_default_types() {
    use Ftype::*;
    let cases: &[(Ftype, Option<GgmlType>)] = &[
        (MostlyQ4_0, Some(GgmlType::Q4_0)),
        (MostlyQ4_1, Some(GgmlType::Q4_1)),
        (MostlyQ5_0, Some(GgmlType::Q5_0)),
        (MostlyQ5_1, Some(GgmlType::Q5_1)),
        (MostlyQ8_0, Some(GgmlType::Q8_0)),
        (MostlyF16, Some(GgmlType::F16)),
        (MostlyBF16, Some(GgmlType::Bf16)),
        (AllF32, Some(GgmlType::F32)),
        (MostlyQ2_K, Some(GgmlType::Q2K)),
        (MostlyQ2_K_S, Some(GgmlType::Q2K)),
        (MostlyQ3_K_S, Some(GgmlType::Q3K)),
        (MostlyQ3_K_M, Some(GgmlType::Q3K)),
        (MostlyQ3_K_L, Some(GgmlType::Q3K)),
        (MostlyQ4_K_S, Some(GgmlType::Q4K)),
        (MostlyQ4_K_M, Some(GgmlType::Q4K)),
        (MostlyQ5_K_S, Some(GgmlType::Q5K)),
        (MostlyQ5_K_M, Some(GgmlType::Q5K)),
        (MostlyQ6_K, Some(GgmlType::Q6K)),
        (MostlyIQ3_XS, Some(GgmlType::Iq3S)),
        (MostlyIQ3_M, Some(GgmlType::Iq3S)),
        (MostlyIQ2_S, Some(GgmlType::Iq2Xs)),
        (MostlyIQ2_M, Some(GgmlType::Iq2S)),
        (MostlyQ1_0, Some(GgmlType::Q1_0)),
        (MostlyQ2_0, Some(GgmlType::Q2_0)),
        (Guessed, None),
    ];
    for (f, want) in cases {
        assert_eq!(f.default_type(), *want, "{f:?}");
    }
}

// -----------------------------------------------------------------------
// full-model simulation of the preliminary loop
// -----------------------------------------------------------------------

/// A hand-built Qwen2-0.5B-shaped model: 24 layers, n_head 14, n_head_kv 2
/// (n_gqa = 7), ffn 4864, n_embd 896, vocab 151936 — the same shape as the
/// local test file `qwen2.5-0.5b-instruct-q4_k_m.gguf`.
///
/// Tensor list order = the order `llama_model_quantize_impl` iterates it
/// (`ml.weights_map` with `weight_name_comparer`).
fn qwen2_0_5b_model() -> QuantModelInfo {
    let mut h = LlamaHparams::new();
    h.n_layer_all = 24;
    h.n_embd = 896;
    h.n_head_arr = vec![14; 24];
    h.n_head_kv_arr = vec![2; 24];
    h.n_ff_arr = vec![4864; 24];
    h.n_expert_used_arr = vec![1; 24];
    QuantModelInfo::from_hparams(LlmArch::QWEN2, &h, 151936)
}

fn qwen2_0_5b_tensors() -> Vec<(String, [i64; 4])> {
    let mut v: Vec<(String, [i64; 4])> = vec![
        ("output.weight".into(), [896, 151936, 1, 1]),
        ("output_norm.weight".into(), [896, 1, 1, 1]),
        ("token_embd.weight".into(), [896, 151936, 1, 1]),
    ];
    for il in 0..24i32 {
        let b = |n: &str| format!("blk.{il}.{n}");
        v.push((b("attn_k.bias"), [128, 1, 1, 1]));
        v.push((b("attn_k.weight"), [896, 128, 1, 1]));
        v.push((b("attn_norm.weight"), [896, 1, 1, 1]));
        v.push((b("attn_output.weight"), [896, 896, 1, 1]));
        v.push((b("attn_q.bias"), [896, 1, 1, 1]));
        v.push((b("attn_q.weight"), [896, 896, 1, 1]));
        v.push((b("attn_v.bias"), [128, 1, 1, 1]));
        v.push((b("attn_v.weight"), [896, 128, 1, 1]));
        v.push((b("ffn_down.weight"), [4864, 896, 1, 1]));
        v.push((b("ffn_gate.weight"), [896, 4864, 1, 1]));
        v.push((b("ffn_norm.weight"), [896, 1, 1, 1]));
        v.push((b("ffn_up.weight"), [896, 4864, 1, 1]));
    }
    v
}

/// Run the preliminary loop (llama-quant.cpp:1077-1112) over the model:
/// returns target types keyed by name.
fn compute_targets(
    info: &QuantModelInfo,
    ftype: Ftype,
    patterns: &[TensorTypeOverride],
    params_tweak: impl Fn(&mut QuantizeParams),
) -> (Vec<(String, GgmlType)>, i32) {
    let mut params = QuantizeParams::new();
    params.ftype = ftype;
    params_tweak(&mut params);
    let default_type = ftype.default_type().unwrap();

    let tensors = qwen2_0_5b_tensors();
    let mut metadata: Vec<TensorMetadata> = tensors
        .iter()
        .map(|(n, _)| TensorMetadata::new(n.clone()))
        .collect();
    let mut qs = QuantizeState::new(info, patterns);
    init_quantize_state_counters(&mut qs, &mut metadata);

    let mut out = Vec::with_capacity(tensors.len());
    for (i, (name, ne)) in tensors.iter().enumerate() {
        // `cur_type` only matters for the "not quantized" return path; use
        // F32 everywhere (the input file is q5_0/q4_K/...; the target type
        // does not depend on it)
        let cur = GgmlType::F32;
        let t = llama_tensor_get_type(&mut qs, &params, name, ne, cur, default_type, &metadata[i])
            .unwrap();
        metadata[i].target_type = t;
        out.push((name.clone(), t));
    }
    (out, qs.n_fallback)
}

fn get<'a>(v: &'a [(String, GgmlType)], name: &str) -> GgmlType {
    v.iter().find(|(n, _)| n == name).unwrap().1
}

/// Hand-derived from llama-quant.cpp for the local Qwen2.5-0.5B file.
/// `--allow-requantize` aside, this is exactly the type assignment the
/// reference `llama-quantize` produces (verified byte-for-byte in the
/// tool-level parity test).
#[test]
fn strategy_table_qwen2_0_5b() {
    let m = qwen2_0_5b_model();
    assert_eq!(m.n_gqa(), 7);
    assert!(!m.is_type_70b);

    // ---- Q4_0 ----
    let (t, n_fallback) = compute_targets(&m, Ftype::MostlyQ4_0, &[], |_| {});
    assert_eq!(get(&t, "output.weight"), GgmlType::Q8_0); // OUTPUT → Q6_K → fallback (896 % 256)
    assert_eq!(get(&t, "token_embd.weight"), GgmlType::Q4_0);
    assert_eq!(get(&t, "blk.0.attn_v.weight"), GgmlType::Q4_0);
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q4_0);
    assert_eq!(get(&t, "blk.0.attn_norm.weight"), GgmlType::F32); // 1D
    assert_eq!(n_fallback, 1); // only output.weight

    // ---- Q8_0 ----
    let (t, n_fallback) = compute_targets(&m, Ftype::MostlyQ8_0, &[], |_| {});
    assert!(t
        .iter()
        .filter(|(n, _)| !n.contains("_norm") && !n.ends_with(".bias"))
        .all(|(_, ty)| *ty == GgmlType::Q8_0));
    assert_eq!(n_fallback, 0);

    // ---- Q4_K_M (the interesting mixing + fallback cases) ----
    let (t, n_fallback) = compute_targets(&m, Ftype::MostlyQ4_K_M, &[], |_| {});
    // output.weight: OUTPUT → Q6_K → 896 % 256 → Q8_0
    assert_eq!(get(&t, "output.weight"), GgmlType::Q8_0);
    // token_embd: Q4_K → 896 % 256 → Q5_0
    assert_eq!(get(&t, "token_embd.weight"), GgmlType::Q5_0);
    // attn_v: i_wv 0 → use_more_bits(0,24) → Q6_K → Q8_0
    assert_eq!(get(&t, "blk.0.attn_v.weight"), GgmlType::Q8_0);
    assert_eq!(get(&t, "blk.1.attn_v.weight"), GgmlType::Q8_0); // i_wv 1 < 3
    assert_eq!(get(&t, "blk.2.attn_v.weight"), GgmlType::Q8_0); // i_wv 2 < 3
    assert_eq!(get(&t, "blk.3.attn_v.weight"), GgmlType::Q5_0); // (3-3)%3==0 → Q4_K → Q5_0
    assert_eq!(get(&t, "blk.5.attn_v.weight"), GgmlType::Q8_0); // (5-3)%3==2 → Q6_K → Q8_0
                                                                // attn_q/k/output, ffn_gate/up: Q4_K → Q5_0 (896 % 256)
    assert_eq!(get(&t, "blk.0.attn_q.weight"), GgmlType::Q5_0);
    assert_eq!(get(&t, "blk.0.attn_k.weight"), GgmlType::Q5_0);
    assert_eq!(get(&t, "blk.0.attn_output.weight"), GgmlType::Q5_0);
    assert_eq!(get(&t, "blk.0.ffn_gate.weight"), GgmlType::Q5_0);
    assert_eq!(get(&t, "blk.0.ffn_up.weight"), GgmlType::Q5_0);
    // ffn_down: 4864 % 256 == 0 → no fallback
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q6K); // use_more_bits(0,24)
    assert_eq!(get(&t, "blk.3.ffn_down.weight"), GgmlType::Q4K); // (3-3)%3==0
    assert_eq!(get(&t, "blk.5.ffn_down.weight"), GgmlType::Q6K); // (5-3)%3==2
    assert_eq!(get(&t, "blk.21.ffn_down.weight"), GgmlType::Q6K); // >= 7*24/8 = 21
    assert_eq!(get(&t, "blk.19.ffn_down.weight"), GgmlType::Q4K); // (19-3)%3 == 1
    assert_eq!(get(&t, "blk.20.ffn_down.weight"), GgmlType::Q6K); // (20-3)%3 == 2
                                                                  // fallback count: every tensor with ne[0] = 896 whose 256-block target is
                                                                  // demoted — 6 per layer (attn_q/k/v/output, ffn_gate/up) plus token_embd;
                                                                  // output.weight is NOT among them for Q4_K_M because the OUTPUT branch
                                                                  // already assigns Q8_0 (`nx % qk_k != 0`, llama-quant.cpp:466) — this is
                                                                  // the 145 the reference reports for this file.
    assert_eq!(n_fallback, 1 + 24 * 6);

    // ---- Q3_K_M ----
    let (t, _) = compute_targets(&m, Ftype::MostlyQ3_K_M, &[], |_| {});
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q5K); // i_layer < 24/16 = 1
    assert_eq!(get(&t, "blk.1.ffn_down.weight"), GgmlType::Q4K);
    // attn_v has ne[0] = 896, so every 256-block choice is demoted again
    assert_eq!(get(&t, "blk.0.attn_v.weight"), GgmlType::Q5_1); // i_wv 0 < 2 → Q5_K → Q5_1
    assert_eq!(get(&t, "blk.2.attn_v.weight"), GgmlType::Q5_0); // i_wv 2 → Q4_K → Q5_0
    assert_eq!(get(&t, "blk.0.attn_output.weight"), GgmlType::Q5_0); // Q4_K → Q5_0
    assert_eq!(get(&t, "blk.0.attn_k.weight"), GgmlType::Q4_0); // Q3_K → Q4_0
    assert_eq!(get(&t, "token_embd.weight"), GgmlType::Q4_0); // Q3_K → Q4_0
    assert_eq!(get(&t, "output.weight"), GgmlType::Q8_0); // OUTPUT: 896 % 256 → Q8_0 directly

    // ---- Q6_K: the impl never changes Q6_K; ne[0]=896 tensors still fall back ----
    let (t, _) = compute_targets(&m, Ftype::MostlyQ6_K, &[], |_| {});
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q6K);
    assert_eq!(get(&t, "blk.7.attn_v.weight"), GgmlType::Q8_0); // Q6_K → Q8_0 (896 % 256)

    // ---- F16: default type not quantized → no mixing, no fallback ----
    let (t, n_fallback) = compute_targets(&m, Ftype::MostlyF16, &[], |_| {});
    assert_eq!(get(&t, "blk.0.attn_v.weight"), GgmlType::F16);
    assert_eq!(get(&t, "output.weight"), GgmlType::F16);
    assert_eq!(n_fallback, 0);
}

/// llama-quant.cpp:683-739 — `--pure` skips the mixing logic; manual
/// `--tensor-type` overrides win over both; per-tensor-type flags.
#[test]
fn pure_and_manual_overrides() {
    struct M(&'static str);
    impl NameMatcher for M {
        fn is_match(&self, n: &str) -> bool {
            n.contains(self.0)
        }
    }
    let m = qwen2_0_5b_model();

    // --pure: blk.0.ffn_down would be Q6_K without it
    let (t, n_fallback) = compute_targets(&m, Ftype::MostlyQ4_K_M, &[], |p| p.pure = true);
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q4K);
    // fallback still applies with --pure (llama-quant.cpp:735)
    assert_eq!(get(&t, "blk.0.attn_q.weight"), GgmlType::Q5_0);
    assert!(n_fallback > 0);

    // --tensor-type ffn_down=q8_0
    let overrides = vec![TensorTypeOverride {
        pattern: Box::new(M("ffn_down")),
        ty: GgmlType::Q8_0,
    }];
    let (t, _) = compute_targets(&m, Ftype::MostlyQ4_K_M, &overrides, |_| {});
    assert_eq!(get(&t, "blk.0.ffn_down.weight"), GgmlType::Q8_0);
    assert_eq!(get(&t, "blk.0.ffn_gate.weight"), GgmlType::Q5_0);

    // --token-embedding-type / --output-tensor-type (llama-quant.cpp:687-706)
    let (t, _) = compute_targets(&m, Ftype::MostlyQ4_K_M, &[], |p| {
        p.token_embedding_type = Some(GgmlType::Q5_1);
        p.output_tensor_type = Some(GgmlType::F16);
    });
    assert_eq!(get(&t, "token_embd.weight"), GgmlType::Q5_1);
    assert_eq!(get(&t, "output.weight"), GgmlType::F16);

    // --leave-output-tensor: output.weight keeps its current type
    let (t, _) = compute_targets(&m, Ftype::MostlyQ4_0, &[], |p| {
        p.quantize_output_tensor = false
    });
    assert_eq!(get(&t, "output.weight"), GgmlType::F32); // == cur_type in the simulation
                                                         // COPY: nothing is quantized (llama-quant.cpp:289)
    let (t, _) = compute_targets(&m, Ftype::MostlyQ8_0, &[], |p| p.only_copy = true);
    assert!(t.iter().all(|(_, ty)| *ty == GgmlType::F32));
}

/// llama-quant.cpp:372-425 — shape fallback table.
///
/// `ncols = 96` is a multiple of 32 but not of 256, so it exercises the
/// 256→32 block demotion; `ncols = 100` is divisible by neither and hits
/// the "(WARNING: must use F16 due to unusual shape)" path
/// (llama-quant.cpp:411-421). `ncols = 896` is the local Qwen2.5-0.5B case.
#[test]
fn fallback_table() {
    let m = qwen2_0_5b_model();
    let cases: &[(GgmlType, i64, GgmlType)] = &[
        (GgmlType::Q6K, 896, GgmlType::Q8_0), // llama-quant.cpp:400
        (GgmlType::Q4K, 96, GgmlType::Q5_0),  // llama-quant.cpp:398
        (GgmlType::Q5K, 96, GgmlType::Q5_1),  // llama-quant.cpp:399
        (GgmlType::Q2K, 96, GgmlType::Q4_0),  // llama-quant.cpp:397
        (GgmlType::Q3K, 96, GgmlType::Q4_0),
        (GgmlType::Q2_0, 96, GgmlType::Q4_0),
        (GgmlType::Iq4Xs, 96, GgmlType::Iq4Nl), // llama-quant.cpp:392
        (GgmlType::Iq2Xxs, 96, GgmlType::Iq4Nl),
        // the demoted type is still incompatible → F16 (llama-quant.cpp:411-421)
        (GgmlType::Q4K, 100, GgmlType::F16),
        (GgmlType::Q6K, 100, GgmlType::F16),
        // a 32-block target with a bad shape → F16 (llama-quant.cpp:402-407)
        (GgmlType::Q4_0, 100, GgmlType::F16),
        (GgmlType::Q8_0, 1, GgmlType::F16),
        // 32-block targets whose shape is fine are left alone
        (GgmlType::Q4_0, 96, GgmlType::Q4_0),
    ];
    for (ty, ncols, want) in cases {
        let mut qs = QuantizeState::new(&m, &[]);
        let got = tensor_type_fallback(&mut qs, "t", *ncols, *ty).unwrap();
        assert_eq!(got, *want, "{ty:?} ncols {ncols}");
        // the counter only moves when the target block size does not divide ncols
        let expect_fallback = i32::from(*ncols % ty.blck_size() as i64 != 0);
        assert_eq!(qs.n_fallback, expect_fallback, "{ty:?} ncols {ncols}");
    }
    // non-quantized targets are never passed to the fallback by the caller
    let mut qs = QuantizeState::new(&m, &[]);
    assert_eq!(
        tensor_type_fallback(&mut qs, "t", 100, GgmlType::F16).unwrap(),
        GgmlType::F16
    );
}

/// llama-quant.cpp:893-907 + the weight ordering (llama-model-loader.h:54).
#[test]
fn counters_and_ordering() {
    let m = qwen2_0_5b_model();
    let mut meta: Vec<TensorMetadata> = [
        "blk.0.attn_v.weight",
        "output.weight",
        "blk.2.ffn_down.weight",
        "token_embd.weight",
        "blk.0.attn_kv_b.weight",
    ]
    .iter()
    .map(|s| TensorMetadata::new(s.to_string()))
    .collect();
    let mut qs = QuantizeState::new(&m, &[]);
    init_quantize_state_counters(&mut qs, &mut meta);
    assert_eq!(qs.n_attention_wv, 2); // attn_v + attn_kv_b
    assert!(!qs.has_tied_embeddings);
    assert_eq!(qs.n_ffn_down, 24);
    assert_eq!(qs.n_ffn_gate, 24);
    assert_eq!(qs.n_ffn_up, 24);

    let mut names: Vec<&str> = vec![
        "blk.10.a.weight",
        "blk.2.b.weight",
        "output.weight",
        "blk.0.attn_k.weight",
        "token_embd.weight",
        "output_norm.weight",
    ];
    names.sort_by(|a, b| weight_name_cmp(a, b));
    assert_eq!(
        names,
        vec![
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            "blk.0.attn_k.weight",
            "blk.2.b.weight",
            "blk.10.a.weight",
        ]
    );
}

/// llama-quant.cpp:287-365 — the name filters.
#[test]
fn allows_quantization_filters() {
    let p = QuantizeParams::new();
    let ne2 = [896i64, 151936, 1, 1];
    let ne1 = [896i64, 1, 1, 1];
    assert!(tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.attn_q.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.attn_q.weight",
        &ne1
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.attn_norm.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.ffn_gate_inp.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::BERT,
        "position_embd.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::BERT,
        "token_types.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.ssm_conv1d.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::RWKV7,
        "blk.0.time_mix_w1.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::T5,
        "blk.0.attn_rel_b.weight",
        &ne2
    ));
    assert!(!tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.shortconv.conv.weight",
        &ne2
    ));
    assert!(tensor_allows_quantization(
        &p,
        LlmArch::LLAMA,
        "blk.0.ffn_gate_exps.weight",
        &[4096, 14336, 8, 1]
    ));
    // --leave-output-tensor
    let mut p2 = QuantizeParams::new();
    p2.quantize_output_tensor = false;
    assert!(!tensor_allows_quantization(
        &p2,
        LlmArch::LLAMA,
        "output.weight",
        &ne2
    ));
    // only_copy
    let mut p3 = QuantizeParams::new();
    p3.only_copy = true;
    assert!(!tensor_allows_quantization(
        &p3,
        LlmArch::LLAMA,
        "blk.0.attn_q.weight",
        &ne2
    ));
}

/// llama-quant.cpp:822-841.
#[test]
fn requires_imatrix_table() {
    assert!(tensor_requires_imatrix(
        "blk.0.attn_q.weight",
        GgmlType::Iq2Xxs,
        Ftype::MostlyIQ2_XXS
    ));
    assert!(!tensor_requires_imatrix(
        "token_embd.weight",
        GgmlType::Iq2Xxs,
        Ftype::MostlyIQ2_XXS
    ));
    assert!(!tensor_requires_imatrix(
        "output.weight",
        GgmlType::Iq2Xxs,
        Ftype::MostlyIQ2_XXS
    ));
    assert!(tensor_requires_imatrix(
        "blk.0.attn_q.weight",
        GgmlType::Q2K,
        Ftype::MostlyQ2_K_S
    ));
    assert!(!tensor_requires_imatrix(
        "blk.0.attn_q.weight",
        GgmlType::Q2K,
        Ftype::MostlyQ2_K
    ));
    assert!(!tensor_requires_imatrix(
        "blk.0.attn_q.weight",
        GgmlType::Q4K,
        Ftype::MostlyQ4_K_M
    ));
}

/// MoE / 70B branches (llama-quant.cpp:559-569, 512-516, 438-452).
#[test]
fn moe_and_70b_branches() {
    // Mixtral-like: 32 layers, 8 experts, n_gqa = 4
    let mut h = LlamaHparams::new();
    h.n_layer_all = 32;
    h.n_embd = 4096;
    h.n_expert = 8;
    h.n_head_arr = vec![32; 32];
    h.n_head_kv_arr = vec![8; 32];
    h.n_ff_arr = vec![14336; 32];
    h.n_expert_used_arr = vec![2; 32];
    let moe = QuantModelInfo::from_hparams(LlmArch::LLAMA, &h, 32000);
    assert_eq!(moe.n_gqa(), 4);
    assert!(!moe.is_type_70b); // n_expert == 8 branch wins (models/llama.cpp:21)

    let p = {
        let mut p = QuantizeParams::new();
        p.ftype = Ftype::MostlyQ4_K_M;
        p
    };
    let mut meta = vec![TensorMetadata::new("blk.0.attn_v.weight".into())];
    let mut qs = QuantizeState::new(&moe, &[]);
    init_quantize_state_counters(&mut qs, &mut meta);
    let got = llama_tensor_get_type(
        &mut qs,
        &p,
        "blk.0.attn_v.weight",
        &[4096, 1024, 1, 1],
        GgmlType::Q4K,
        GgmlType::Q4K,
        &meta[0],
    )
    .unwrap();
    // n_expert == 8 → Q8_0 (llama-quant.cpp:565-569)
    assert_eq!(got, GgmlType::Q8_0);

    // moe layer_info parses the layer out of the name
    assert_eq!(
        layer_info(0, 32, "blk.7.ffn_down.weight", 8).unwrap(),
        (7, 32)
    );
    assert!(layer_info(0, 32, "blk.32.ffn_down.weight", 8).is_err());
    assert!(layer_info(0, 32, "ffn_down.weight", 8).is_err());
    // dense models never parse the name
    assert_eq!(
        layer_info(3, 24, "blk.7.ffn_down.weight", 1).unwrap(),
        (3, 24)
    );
    assert_eq!(parse_blk_layer("blk.12.ffn_down.weight"), Some(12));
    assert_eq!(parse_blk_layer("blk.1e5.x"), Some(1)); // sscanf stops at 'e'
    assert_eq!(parse_blk_layer("token_embd.weight"), None);

    // 70B: 80 layers, n_head != n_head_kv → Q3_K/Q4_K bumped to Q5_K
    let mut h = LlamaHparams::new();
    h.n_layer_all = 80;
    h.n_embd = 8192;
    h.n_head_arr = vec![64; 80];
    h.n_head_kv_arr = vec![8; 80];
    h.n_ff_arr = vec![28672; 80];
    h.n_expert_used_arr = vec![1; 80];
    let big = QuantModelInfo::from_hparams(LlmArch::LLAMA, &h, 128256);
    assert!(big.is_type_70b);
    let mut p = QuantizeParams::new();
    p.ftype = Ftype::MostlyQ2_K;
    let mut meta = vec![TensorMetadata::new("blk.0.attn_v.weight".into())];
    let mut qs = QuantizeState::new(&big, &[]);
    init_quantize_state_counters(&mut qs, &mut meta);
    // n_gqa = 8 ≥ 4 → Q4_K, then the 70B bump → Q5_K
    let got = llama_tensor_get_type(
        &mut qs,
        &p,
        "blk.0.attn_v.weight",
        &[8192, 1024, 1, 1],
        GgmlType::Q2K,
        GgmlType::Q2K,
        &meta[0],
    )
    .unwrap();
    assert_eq!(got, GgmlType::Q5K);

    // Q4_K_S + i_attention_wv < 4 → Q5_K, plus the 70B bump keeps Q5_K
    let mut p = QuantizeParams::new();
    p.ftype = Ftype::MostlyQ4_K_S;
    let mut qs = QuantizeState::new(&big, &[]);
    let got = llama_tensor_get_type(
        &mut qs,
        &p,
        "blk.0.attn_v.weight",
        &[8192, 1024, 1, 1],
        GgmlType::Q4K,
        GgmlType::Q4K,
        &meta[0],
    )
    .unwrap();
    assert_eq!(got, GgmlType::Q5K);
}
