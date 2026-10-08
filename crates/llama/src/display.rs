//! display.rs — the model-load banner / reporting family of the reference:
//!
//!   * `llm_type_name` + the `llm_type` enum (llama-model.h:22-159,
//!     llama-model.cpp:868-1005) and `llama_model::type_name`/`desc`
//!     (llama-model.cpp:1896-1902, :1415)
//!   * `llama_rope_scaling_type_name` (llama-model.cpp:1019-1027) and
//!     `llama_expert_gating_func_name` (:1008-1016)
//!   * `llama_ftype_name` (llama-model-loader.cpp:33-76) +
//!     `llama_file_version_name` (:21-28) + `llama_model_loader::print_info`
//!     (:1803-1813)
//!   * `llama_model::print_info` (llama-model.cpp:1956-2174) — includes
//!     `llama_vocab::print_info` (llama-vocab.cpp:3896-3930) at its end
//!   * `llama_flash_attn_type_name` (llama.cpp:38-49) and
//!     `llama_print_system_info` (llama.cpp:598-617)
//!
//! Every line goes through the ported log routing (`crate::llama_log_info!`),
//! so a callback set with [`crate::impl_log::log_set`] sees the banner.
//!
//! Model-type *detection*: the C sets `model.type` inside each
//! `models/<arch>.cpp::load_arch_hparams`. The port's loaders (meta.rs) do not
//! keep a type field, so [`llm_type_of`] recomputes it from the same
//! (n_layer/n_embd/n_ff/n_head/n_vocab) switches. Coverage: the archs listed
//! in [`llm_type_of`]'s match; every other arch returns `?B` (the C's default
//! for an unrecognized layer count) — see PARITY.md for the exact list.

use crate::arch::LlmArch;
use crate::hparams::{LlamaHparams, LlamaRopeScalingType};
use crate::model::LlamaModel;
use crate::vocab::{Vocab, VocabType};

// ---------------------------------------------------------------------------
// llm_type (llama-model.h:22-159) + llm_type_name (llama-model.cpp:868-1005)
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmType {
    #[default]
    UNKNOWN,
    T14M,
    T17M,
    T22M,
    T33M,
    T47M,
    T60M,
    T70M,
    T80M,
    T109M,
    T137M,
    T140M,
    T149M,
    T160M,
    T190M,
    T220M,
    T230M,
    T250M,
    T256M,
    T270M,
    T335M,
    T350M,
    T360M,
    T395M,
    T410M,
    T450M,
    T475M,
    T558M,
    T700M,
    T770M,
    T780M,
    T950M,
    B0_3,
    B0_5,
    B0_6,
    B0_8,
    B1,
    B1_2,
    B1_3,
    B1_4,
    B1_5,
    B1_6,
    B1_7,
    B1_8,
    B2,
    B2_6,
    B2_8,
    B2_9,
    B3,
    B4,
    B6,
    B6_9,
    B7,
    B8,
    B9,
    B11,
    B12,
    B13,
    B14,
    B15,
    B16,
    B20,
    B26,
    B27,
    B30,
    B31,
    B32,
    B34,
    B35,
    B36,
    B40,
    B65,
    B70,
    B120,
    B142,
    B236,
    B290,
    B314,
    B405,
    B456,
    B671,
    SMALL,
    MEDIUM,
    LARGE,
    XL,
    A1_7B,
    A2_7B,
    B8x7,
    B8x22,
    B16x12,
    B16x3_8,
    B10B_128x3_66,
    B57B_A14,
    B17B_16E,   // llama4 Scout
    B17B_128E,  // llama4 Maverick
    A13B,
    B1B_A400M,  // Granite3 MoE
    B3B_A800M,  // Granite3 MoE
    B7B_A1,
    B8B_A1,     // lfm2moe
    B7_9B_A1_3, // Ling-3.0-tiny
    B12B_A2_5,
    B16B_A1,
    B21B_A3, // Ernie MoE small
    B24B_A2, // lfm2moe
    B26B_A4, // Gemma4
    B30B_A3,
    B31B_A3_5,
    B32B_A9, // Granite4 Hybrid
    B35B_A3, // Qwen3.5
    B48B_A3, // Kimi Linear
    B75B_A9, // Nemotron 3 Puzzle
    B80B_A3, // Qwen3 Next
    A3B,     // Qwen3.8 Flash Next
    B100B_A6,
    B102B_A12, // Solar-Open
    B106B_A12, // GLM-4.5-Air
    B118B_A8,  // Laguna-S-2
    B120B_A12, // Nemotron 3 Super
    B122B_A10, // Qwen3.5
    B124B_A5_1, // Ling-3.0-flash
    B196B_A11, // Step3.5-Flash
    B230B_A10, // Minimax M2
    B428B_A23, // Minimax M3
    B235B_A22,
    B288B_A19, // dots3-note
    B300B_A47, // Ernie MoE big
    B310B_A15, // /MiMo-V2-Flash
    B355B_A32, // GLM-4.5
    B397B_A17, // Qwen3.5
    B685B_A37, // DeepSeek V3.2
    B744B_A40, // GLM-5
    B2_8T_A50, // Kimi-K3
    B320B_A18, // GLM-5.3-Flash
    E2B,
    E4B,
}

/// `llm_type_name` (llama-model.cpp:868-1005).
pub fn llm_type_name(t: LlmType) -> &'static str {
    use LlmType::*;
    match t {
        UNKNOWN => "?B",
        T14M => "14M",
        T17M => "17M",
        T22M => "22M",
        T33M => "33M",
        T47M => "47M",
        T60M => "60M",
        T70M => "70M",
        T80M => "80M",
        T109M => "109M",
        T137M => "137M",
        T140M => "140M",
        T149M => "149M",
        T160M => "160M",
        T190M => "190M",
        T220M => "220M",
        T230M => "230M",
        T250M => "250M",
        T256M => "256M",
        T270M => "270M",
        T335M => "335M",
        T350M => "350M",
        T360M => "360M",
        T395M => "395M",
        T410M => "410M",
        T450M => "450M",
        T475M => "475M",
        T558M => "558M",
        T700M => "700M",
        T770M => "770M",
        T780M => "780M",
        T950M => "950M",
        B0_3 => "0.3B",
        B0_5 => "0.5B",
        B0_6 => "0.6B",
        B0_8 => "0.8B",
        B1 => "1B",
        B1_2 => "1.2B",
        B1_3 => "1.3B",
        B1_4 => "1.4B",
        B1_5 => "1.5B",
        B1_6 => "1.6B",
        B1_7 => "1.7B",
        B1_8 => "1.8B",
        B2 => "2B",
        B2_6 => "2.6B",
        B2_8 => "2.8B",
        B2_9 => "2.9B",
        B3 => "3B",
        B4 => "4B",
        B6 => "6B",
        B6_9 => "6.9B",
        B7 => "7B",
        B8 => "8B",
        B9 => "9B",
        B11 => "11B",
        B12 => "12B",
        B13 => "13B",
        B14 => "14B",
        B15 => "15B",
        B16 => "16B",
        B20 => "20B",
        B26 => "26B",
        B27 => "27B",
        B30 => "30B",
        B31 => "31B",
        B32 => "32B",
        B34 => "34B",
        B35 => "35B",
        B36 => "36B",
        B40 => "40B",
        B65 => "65B",
        B70 => "70B",
        B120 => "120B",
        B142 => "142B",
        B236 => "236B",
        B290 => "290B",
        B314 => "314B",
        B405 => "405B",
        B456 => "456B",
        B671 => "671B",
        SMALL => "0.1B",
        MEDIUM => "0.4B",
        LARGE => "0.8B",
        XL => "1.5B",
        A1_7B => "A1.7B",
        A2_7B => "A2.7B",
        B8x7 => "8x7B",
        B8x22 => "8x22B",
        B16x12 => "16x12B",
        B16x3_8 => "16x3.8B",
        B10B_128x3_66 => "10B+128x3.66B",
        B57B_A14 => "57B.A14B",
        B17B_16E => "17Bx16E (Scout)",
        B17B_128E => "17Bx128E (Maverick)",
        A13B => "A13B",
        B1B_A400M => "1B.A400M",
        B3B_A800M => "3B.A800M",
        B7B_A1 => "7B.A1B",
        B8B_A1 => "8B.A1B",
        B7_9B_A1_3 => "7.9B.A1.3B",
        B12B_A2_5 => "12B.A2.5B",
        B16B_A1 => "16B.A1B",
        B21B_A3 => "21B.A3B",
        B24B_A2 => "24B.A2B",
        B26B_A4 => "26B.A4B",
        B30B_A3 => "30B.A3B",
        B31B_A3_5 => "31B.A3.5B",
        B32B_A9 => "32B.A9B",
        B35B_A3 => "35B.A3B",
        B48B_A3 => "48B.A3B",
        B75B_A9 => "75B.A9B",
        B80B_A3 => "80B.A3B",
        A3B => "A3B",
        B100B_A6 => "100B.A6B",
        B102B_A12 => "102B.A12B",
        B106B_A12 => "106B.A12B",
        B118B_A8 => "118B.A8B",
        B120B_A12 => "120B.A12B",
        B122B_A10 => "122B.A10B",
        B124B_A5_1 => "124B.A5.1B",
        B196B_A11 => "196B.A11B",
        B230B_A10 => "230B.A10B",
        B428B_A23 => "428B.A23B",
        B235B_A22 => "235B.A22B",
        B288B_A19 => "288B.A19B",
        B300B_A47 => "300B.A47B",
        B310B_A15 => "310B.A15B",
        B355B_A32 => "355B.A32B",
        B397B_A17 => "397B.A17B",
        B685B_A37 => "685B.A37B",
        B744B_A40 => "744B.A40B",
        B2_8T_A50 => "2.8T.A50B",
        B320B_A18 => "320B.A18B",
        E2B => "E2B",
        E4B => "E4B",
    }
}

/// The `models/<arch>.cpp` type switches, recomputed from the loaded hparams
/// (the C sets `model.type` inside `load_arch_hparams`). The archs below are
/// ported 1:1; every other arch returns the C's `default: LLM_TYPE_UNKNOWN`
/// ("?B") — the exact coverage list lives in PARITY.md.
pub fn llm_type_of(arch: LlmArch, h: &LlamaHparams, n_vocab: u32) -> LlmType {
    use LlmType::*;
    let nl = h.n_layer();
    match arch {
        // models/llama.cpp:8-31 (n_expert == 8 = the Mixtral-style tables)
        LlmArch::LLAMA | LlmArch::DECI => {
            if h.n_expert == 8 {
                match nl {
                    32 => B8x7,
                    56 => B8x22,
                    _ => UNKNOWN,
                }
            } else {
                match nl {
                    16 | 22 => B1,
                    26 | 28 => B3,
                    30 => T256M,
                    32 => {
                        if n_vocab == 49152 {
                            B3
                        } else if n_vocab < 40000 {
                            B7
                        } else {
                            B8
                        }
                    }
                    36 => B8,
                    40 => B13,
                    48 => B34,
                    60 => B30,
                    80 => if h.n_head(0) == h.n_head_kv(0) { B65 } else { B70 },
                    _ => UNKNOWN,
                }
            }
        }
        LlmArch::QWEN2 => match nl {
            24 => if h.n_embd == 1024 { B0_5 } else { B1 },
            28 => if h.n_embd == 1536 { B1_5 } else { B7 },
            32 => B7,
            36 => B3,
            40 => if h.n_head(0) == 20 { B4 } else { B13 },
            48 => B14,
            64 => B32,
            80 => B70,
            _ => UNKNOWN,
        },
        LlmArch::QWEN2MOE => match nl {
            24 => A2_7B,
            28 => B57B_A14,
            _ => UNKNOWN,
        },
        LlmArch::QWEN3 => match nl {
            28 => if h.n_embd == 1024 { B0_6 } else { B1_7 },
            36 => if h.n_embd == 2560 { B4 } else { B8 },
            40 => B14,
            64 => B32,
            _ => UNKNOWN,
        },
        LlmArch::QWEN3MOE | LlmArch::QWEN3VLMOE => match nl {
            48 => B30B_A3,
            94 => B235B_A22,
            _ => UNKNOWN,
        },
        LlmArch::QWEN35 | LlmArch::QWEN35MOE => match nl {
            24 => if h.n_embd == 1024 { B0_8 } else { B2 },
            32 => if h.n_embd == 2560 { B4 } else { B9 },
            64 => B27,
            _ => UNKNOWN,
        },
        LlmArch::BERT => match nl {
            3 => T17M,   // bge-micro
            6 => T22M,   // MiniLM-L6
            12 => T33M,  // MiniLM-L12 / bge-small
            _ => UNKNOWN,
        },
        LlmArch::GEMMA => match nl {
            18 => B2,
            28 => B7,
            _ => UNKNOWN,
        },
        LlmArch::GEMMA2 => match nl {
            26 => B2,
            42 => B9,
            46 => B27,
            _ => UNKNOWN,
        },
        LlmArch::GEMMA3 => match nl {
            18 => T270M,
            26 => B1,
            32 => B8,  // Rnj-1
            34 => B4,
            48 => B12,
            62 => B27,
            _ => UNKNOWN,
        },
        LlmArch::PHI2 => match nl {
            24 => B1,
            32 => B3,
            _ => UNKNOWN,
        },
        LlmArch::PHI3 => match nl {
            24 => B1,
            32 => B3,
            40 => B14,
            _ => UNKNOWN,
        },
        LlmArch::GPT2 => match nl {
            12 => SMALL,
            24 => MEDIUM,
            36 => LARGE,
            48 => XL,
            _ => UNKNOWN,
        },
        LlmArch::MPT => match nl {
            32 => B7,
            48 => B30,
            _ => UNKNOWN,
        },
        LlmArch::FALCON => match nl {
            32 => B7,
            60 => B40,
            _ => UNKNOWN,
        },
        LlmArch::REFACT => match nl {
            32 => B1,
            _ => UNKNOWN,
        },
        LlmArch::STARCODER => match nl {
            24 => B1,
            36 => B3,
            42 => B7,
            40 => B15,
            _ => UNKNOWN,
        },
        LlmArch::STARCODER2 => match nl {
            30 => B3,
            32 => B7,
            40 => B15,
            52 => B20, // granite
            88 => B34, // granite
            _ => UNKNOWN,
        },
        LlmArch::DEEPSEEK => match h.n_ff_exp(0) {
            1408 => B16,
            1792 => B20,
            _ => UNKNOWN,
        },
        LlmArch::DEEPSEEK2 => match nl {
            27 => B16,
            47 => B30B_A3,
            60 => B236,
            61 => B671,
            _ => UNKNOWN,
        },
        LlmArch::DEEPSEEK2OCR => match nl {
            12 => B3,
            _ => UNKNOWN,
        },
        LlmArch::DEEPSEEK32 => match nl {
            61 => B685B_A37,
            _ => UNKNOWN,
        },
        LlmArch::GLM_DSA => match nl {
            78 => B744B_A40,
            _ => UNKNOWN,
        },
        LlmArch::GLM5_NEXT => match nl {
            45 => B320B_A18, // GLM-5.3-Flash (glm5-next.cpp:55-58)
            _ => UNKNOWN,
        },
        LlmArch::MINICPM => match nl {
            52 => B1,
            40 => B2,
            _ => UNKNOWN,
        },
        LlmArch::GLM4 => match nl {
            17 => B1, // GLM-OCR
            40 => B9,
            61 => B32,
            _ => UNKNOWN,
        },
        LlmArch::COHERE2 => match nl {
            32 => B8,
            _ => UNKNOWN,
        },
        LlmArch::OLMO => match nl {
            22 => B1,
            32 => B7,
            80 => B70,
            _ => UNKNOWN,
        },
        LlmArch::DBRX => match nl {
            40 => B16x12,
            _ => UNKNOWN,
        },
        LlmArch::STABLELM => match nl {
            24 => B1,
            32 => B3,
            40 => B12,
            _ => UNKNOWN,
        },
        LlmArch::INTERNLM2 => match nl {
            32 => B7,
            48 => B20,
            _ => UNKNOWN,
        },
        LlmArch::ORION => match nl {
            40 => B14,
            _ => UNKNOWN,
        },
        LlmArch::EXAONE => match nl {
            32 => B8,
            _ => UNKNOWN,
        },
        LlmArch::NEMOTRON => match nl {
            32 => B4,
            _ => UNKNOWN,
        },
        LlmArch::JAIS => match nl {
            24 => B1_3,
            40 => B13,
            _ => UNKNOWN,
        },
        LlmArch::BLOOM => match nl {
            24 => B1,
            30 => B3,
            _ => UNKNOWN,
        },
        LlmArch::ARCTIC => match nl {
            35 => B10B_128x3_66,
            _ => UNKNOWN,
        },
        LlmArch::OPENAI_MOE => match nl {
            24 => B20,
            36 => B120,
            _ => UNKNOWN,
        },
        LlmArch::LFM2MOE => match nl {
            24 => B8B_A1,
            40 => B24B_A2,
            _ => UNKNOWN,
        },
        LlmArch::GEMMA4 => match nl {
            30 => B26B_A4,
            35 => E2B,
            42 => E4B,
            60 => B31,
            _ => UNKNOWN,
        },
        _ => UNKNOWN,
    }
}

// ---------------------------------------------------------------------------
// small display-string helpers
// ---------------------------------------------------------------------------

/// `llama_rope_scaling_type_name` (llama-model.cpp:1019-1027).
pub fn rope_scaling_type_name(t: LlamaRopeScalingType) -> String {
    match t {
        LlamaRopeScalingType::NONE => "none",
        LlamaRopeScalingType::LINEAR => "linear",
        LlamaRopeScalingType::YARN => "yarn",
        LlamaRopeScalingType::LONGROPE => "longrope",
        LlamaRopeScalingType::UNSPECIFIED => "unspec", // .at() throws in C; unreachable for loaded models
    }
    .to_string()
}

/// `llama_expert_gating_func_name` (llama-model.cpp:1008-1016).
pub fn expert_gating_func_name(t: u32) -> &'static str {
    match t {
        1 => "softmax",  // LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX
        2 => "sigmoid",  // LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
        4 => "sqrtsoftplus", // SQRT_SOFTPLUS
        _ => "unknown",  // NONE / SOFTMAX_WEIGHT (llama-model.cpp:1015)
    }
}

/// `llama_flash_attn_type_name` (llama.cpp:38-49).
pub fn flash_attn_type_name(t: i32) -> &'static str {
    match t {
        0 => "auto",
        1 => "disabled",
        2 => "enabled",
        _ => "unknown",
    }
}

/// `LLAMA_FTYPE_GUESSED` (include/llama.h: 1 << 31).
const LLAMA_FTYPE_GUESSED: i32 = 1 << 31;

/// `llama_ftype_name` (llama-model-loader.cpp:33-76). The raw ftype is the
/// `general.file_type` KV; the GUESSED bit prepends "(guessed) ".
pub fn ftype_name(ftype: i32) -> String {
    let base = match ftype & !LLAMA_FTYPE_GUESSED {
        0 => "all F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K - Medium",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        21 => "Q2_K - Small",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        24 => "IQ1_S - 1.5625 bpw",
        25 => "IQ4_NL - 4.5 bpw",
        26 => "IQ3_S - 3.4375 bpw",
        27 => "IQ3_S mix - 3.66 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        31 => "IQ1_M - 1.75 bpw",
        32 => "BF16",
        36 => "TQ1_0 - 1.69 bpw ternary",
        37 => "TQ2_0 - 2.06 bpw ternary",
        38 => "MXFP4 MoE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        _ => "unknown, may not work",
    };
    if ftype & LLAMA_FTYPE_GUESSED != 0 {
        format!("(guessed) {base}")
    } else {
        base.to_string()
    }
}

/// `llama_model_loader`'s ftype resolution (llama-model-loader.cpp:754-790):
/// the `general.file_type` KV when present; otherwise guessed from the
/// majority tensor type with `LLAMA_FTYPE_GUESSED` set. The same mapping the
/// verified llama-bench `model_info::guess_ftype` carries.
pub fn guess_ftype(gguf: &ggml::Gguf) -> i32 {
    use ggml::types::GgmlType as T;
    // weights_map is a std::map keyed by tensor name -> name order
    let mut tensors: Vec<&ggml::gguf::TensorInfo> = gguf.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    let mut type_max = T::F32;
    let mut n_type_max = 0u32;
    for t in tensors {
        let c = counts.entry(t.ty as u32).or_insert(0);
        *c += 1;
        if n_type_max < *c {
            n_type_max = *c;
            type_max = t.ty;
        }
    }
    let base: i32 = match type_max {
        T::F32 => 0,
        T::F16 => 1,
        T::Bf16 => 32,
        T::Q1_0 => 40,
        T::Q2_0 => 41,
        T::Q4_0 => 2,
        T::Q4_1 => 3,
        T::Q5_0 => 8,
        T::Q5_1 => 9,
        T::Q8_0 => 7,
        T::Q2K => 10,
        T::Q3K => 12,
        T::Q4K => 15,
        T::Q5K => 17,
        T::Q6K => 18,
        T::Tq1_0 => 36,
        T::Tq2_0 => 37,
        T::Iq2Xxs => 19,
        T::Iq2Xs => 20,
        T::Iq2S => 28,
        T::Iq3Xxs => 23,
        T::Iq1S => 24,
        T::Iq1M => 31,
        T::Iq4Nl => 25,
        T::Iq4Xs => 30,
        T::Iq3S => 26,
        T::Nvfp4 => 39,
        _ => 0, // the C warns and falls through to ALL_F32
    };
    match gguf.get_u32("general.file_type") {
        Some(v) => v as i32,
        None => base | LLAMA_FTYPE_GUESSED,
    }
}

/// `llama_file_version_name` (llama-model-loader.cpp:21-28).
pub fn file_version_name(fver: u32) -> &'static str {
    match fver {
        1 => "GGUF V1 (support until nov 2023)",
        2 => "GGUF V2",
        3 => "GGUF V3 (latest)",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// C printf-compatible float formatting for the banner lines
// ---------------------------------------------------------------------------

/// `%.<prec>e` with C's two-digit exponent ("1.0e-06", "-2.5e+08").
pub fn cfmt_e(v: f64, prec: usize) -> String {
    let s = format!("{:.*e}", prec, v);
    let (mant, exp) = s.split_once('e').expect("exp");
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("+", exp),
    };
    format!("{mant}e{sign}{digits:0>2}")
}

/// `%g` (precision 6): fixed for exponents in [-4, 6), scientific otherwise,
/// trailing zeros stripped.
pub fn cfmt_g(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    if (-4..6).contains(&exp) {
        let decimals = (5 - exp).max(0) as usize;
        let mut s = format!("{v:.decimals$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    } else {
        // C %g strips trailing zeros from the mantissa too
        let mut m = cfmt_e(v, 5);
        let (mant, exp) = m.split_once('e').unwrap();
        let mut mant = mant.to_string();
        if mant.contains('.') {
            while mant.ends_with('0') {
                mant.pop();
            }
            if mant.ends_with('.') {
                mant.pop();
            }
        }
        m = format!("{mant}e{exp}");
        m
    }
}

/// `print_f` (llama-model.cpp:1964-1988): a per-layer value printed as the
/// scalar when every layer agrees, else a bracketed comma list.
fn print_f(_h: &LlamaHparams, n: u32, f: impl Fn(usize) -> i64) -> String {
    let n = n as usize;
    let v: Vec<i64> = (0..n).map(&f).collect();
    let is_var = v.iter().any(|x| *x != v[0]);
    if v.is_empty() {
        return "0".to_string(); // n_layer_all == 0 never reaches print_info
    }
    if is_var {
        let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
        format!("[{}]", items.join(", "))
    } else {
        v[0].to_string()
    }
}

// ---------------------------------------------------------------------------
// print_info family
// ---------------------------------------------------------------------------

/// `llama_model_loader::print_info` (llama-model-loader.cpp:1803-1813).
pub fn loader_print_info(fver: u32, ftype: i32, n_bytes: u64, n_elements: u64) {
    crate::llama_log_info!("print_info: file format = {}\n", file_version_name(fver));
    crate::llama_log_info!("print_info: file type   = {}\n", ftype_name(ftype));
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if (n_bytes as f64) < GIB {
        crate::llama_log_info!(
            "print_info: file size   = {:.2} MiB ({:.2} BPW) \n",
            n_bytes as f64 / 1024.0 / 1024.0,
            n_bytes as f64 * 8.0 / n_elements as f64
        );
    } else {
        crate::llama_log_info!(
            "print_info: file size   = {:.2} GiB ({:.2} BPW) \n",
            n_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            n_bytes as f64 * 8.0 / n_elements as f64
        );
    }
}

/// The loader totals from a parsed GGUF (what `llama_model_loader`'s
/// n_elements/n_bytes hold at print time).
pub fn loader_stats(gguf: &ggml::Gguf) -> (u64, u64) {
    let mut n_elements = 0u64;
    let mut n_bytes = 0u64;
    for t in &gguf.tensors {
        n_elements += t.ne.iter().product::<i64>() as u64;
        n_bytes += t.size_bytes();
    }
    (n_elements, n_bytes)
}

/// `llama_model::print_info` (llama-model.cpp:1956-2174) — hparams dump +
/// `vocab.print_info()` at the end (`llama_model::print_info` :2174).
pub fn model_print_info(model: &LlamaModel, vocab: &Vocab) {
    use LlmArch::*;
    let h = &model.hparams;
    let rope_scaling_type = rope_scaling_type_name(h.rope_scaling_type_train);
    let arch = model.arch;
    let n_vocab = vocab.id_to_token.len() as u32;
    let ty = llm_type_of(arch, h, n_vocab);

    // hparams (llama-model.cpp:1981-2044)
    crate::llama_log_info!("print_info: arch                  = {}\n", model.arch.name());
    crate::llama_log_info!("print_info: vocab_only            = {}\n", h.vocab_only as i32);
    crate::llama_log_info!("print_info: no_alloc              = {}\n", h.no_alloc as i32);

    if !h.vocab_only {
        crate::llama_log_info!("print_info: n_ctx_train           = {}\n", h.n_ctx_train);
        crate::llama_log_info!("print_info: n_embd_inp            = {}\n", h.n_embd_inp());
        crate::llama_log_info!("print_info: n_embd                = {}\n", h.n_embd);
        crate::llama_log_info!("print_info: n_embd_out            = {}\n", h.n_embd_out());
        crate::llama_log_info!("print_info: n_layer               = {}\n", h.n_layer());
        crate::llama_log_info!("print_info: n_layer_all           = {}\n", h.n_layer_all);
        crate::llama_log_info!(
            "print_info: n_head                = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_head(il) as i64)
        );
        crate::llama_log_info!(
            "print_info: n_head_kv             = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_head_kv(il) as i64)
        );
        crate::llama_log_info!("print_info: n_rot                 = {}\n", h.n_rot_full);
        crate::llama_log_info!("print_info: n_swa                 = {}\n", h.n_swa);
        crate::llama_log_info!("print_info: is_swa_any            = {}\n", h.is_swa_any() as i32);
        crate::llama_log_info!(
            "print_info: non_causal_type       = {}\n",
            h.non_causal_type as i32
        );
        crate::llama_log_info!("print_info: n_embd_head_k         = {}\n", h.n_embd_head_k_full);
        crate::llama_log_info!("print_info: n_embd_head_v         = {}\n", h.n_embd_head_v_full);
        crate::llama_log_info!(
            "print_info: n_gqa                 = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_gqa(il) as i64)
        );
        crate::llama_log_info!(
            "print_info: n_embd_k_gqa          = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_embd_k_gqa(il) as i64)
        );
        crate::llama_log_info!(
            "print_info: n_embd_v_gqa          = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_embd_v_gqa(il) as i64)
        );
        crate::llama_log_info!("print_info: f_norm_eps            = {}\n", cfmt_e(h.f_norm_eps as f64, 1));
        crate::llama_log_info!(
            "print_info: f_norm_rms_eps        = {}\n",
            cfmt_e(h.f_norm_rms_eps as f64, 1)
        );
        crate::llama_log_info!("print_info: f_clamp_kqv           = {}\n", cfmt_e(h.f_clamp_kqv as f64, 1));
        crate::llama_log_info!(
            "print_info: f_max_alibi_bias      = {}\n",
            cfmt_e(h.f_max_alibi_bias as f64, 1)
        );
        crate::llama_log_info!("print_info: f_logit_scale         = {}\n", cfmt_e(h.f_logit_scale as f64, 1));
        crate::llama_log_info!(
            "print_info: f_attn_scale          = {}\n",
            cfmt_e(h.f_attention_scale as f64, 1)
        );
        crate::llama_log_info!(
            "print_info: f_attn_value_scale    = {:.4}\n",
            h.f_attn_value_scale
        );
        crate::llama_log_info!(
            "print_info: n_ff                  = {}\n",
            print_f(h, h.n_layer_all, |il| h.n_ff(il) as i64)
        );
        crate::llama_log_info!("print_info: n_expert              = {}\n", h.n_expert);
        crate::llama_log_info!("print_info: n_expert_used         = {}\n", h.n_expert_used(0));
        crate::llama_log_info!("print_info: n_expert_groups       = {}\n", h.n_expert_groups as i32);
        crate::llama_log_info!("print_info: n_group_used          = {}\n", h.n_group_used as i32);
        crate::llama_log_info!("print_info: causal attn           = {}\n", h.causal_attn as i32);
        crate::llama_log_info!("print_info: pooling type          = {}\n", h.pooling_type as i32);
        crate::llama_log_info!("print_info: rope type             = {}\n", h.rope_type as i32);
        crate::llama_log_info!("print_info: rope scaling          = {}\n", rope_scaling_type);
        crate::llama_log_info!(
            "print_info: freq_base_train       = {:.1}\n",
            h.rope_freq_base_train
        );
        crate::llama_log_info!("print_info: freq_scale_train      = {}\n", cfmt_g(h.rope_freq_scale_train as f64));
        if h.swa_type != crate::hparams::LlamaSwaType::NONE {
            crate::llama_log_info!(
                "print_info: freq_base_swa         = {:.1}\n",
                h.rope_freq_base_train_swa
            );
            crate::llama_log_info!(
                "print_info: freq_scale_swa        = {}\n",
                cfmt_g(h.rope_freq_scale_train_swa as f64)
            );
            crate::llama_log_info!("print_info: n_embd_head_k_swa     = {}\n", h.n_embd_head_k_swa);
            crate::llama_log_info!("print_info: n_embd_head_v_swa     = {}\n", h.n_embd_head_v_swa);
            crate::llama_log_info!("print_info: n_rot_swa             = {}\n", h.n_rot_swa);
        }
        crate::llama_log_info!("print_info: n_ctx_orig_yarn       = {}\n", h.n_ctx_orig_yarn);
        crate::llama_log_info!("print_info: rope_yarn_log_mul     = {:.4}\n", h.rope_yarn_log_mul);
        crate::llama_log_info!(
            "print_info: rope_finetuned        = {}\n",
            if h.rope_finetuned { "yes" } else { "unknown" }
        );
        if arch == LlmArch::GRANITE
            && h.deepstack_mapping_arr.iter().any(|&e| e >= 0)
        {
            crate::llama_log_info!(
                "print_info: deepstack_mapping_arr = {}\n",
                print_f(h, h.n_layer_all, |il| h.deepstack_mapping_arr[il as usize] as i64)
            );
        }
        // MRoPE sections (llama-model.cpp:2046-2049)
        {
            let s = &h.rope_sections;
            if s[0] != 0 || s[1] != 0 || s[2] != 0 || s[3] != 0 {
                crate::llama_log_info!(
                    "print_info: mrope sections        = [{}, {}, {}, {}]\n",
                    s[0],
                    s[1],
                    s[2],
                    s[3]
                );
            }
        }
        if !model.classifier_labels.is_empty() {
            crate::llama_log_info!("print_info: n_cls_out             = {}\n", h.n_cls_out);
            for (i, label) in model.classifier_labels.iter().enumerate() {
                crate::llama_log_info!("print_info: cls_label[{:2}]         = {}\n", i, label);
            }
        }

        if matches!(
            arch,
            MAMBA
                | MAMBA2
                | JAMBA
                | FALCON_H1
                | PLAMO2
                | GRANITE_HYBRID
                | QWEN3NEXT
                | QWEN35
                | QWEN35MOE
                | NEMOTRON_H
                | NEMOTRON_H_MOE
        ) {
            crate::llama_log_info!("print_info: ssm_d_conv            = {}\n", h.ssm_d_conv);
            crate::llama_log_info!("print_info: ssm_d_inner           = {}\n", h.ssm_d_inner);
            crate::llama_log_info!("print_info: ssm_d_state           = {}\n", h.ssm_d_state);
            crate::llama_log_info!("print_info: ssm_dt_rank           = {}\n", h.ssm_dt_rank);
            crate::llama_log_info!("print_info: ssm_n_group           = {}\n", h.ssm_n_group);
            crate::llama_log_info!("print_info: ssm_dt_b_c_rms        = {}\n", h.ssm_dt_b_c_rms as i32);
        }

        crate::llama_log_info!("print_info: model type            = {}\n", llm_type_name(ty));
        if model.n_elements >= 1e12 as u64 {
            crate::llama_log_info!(
                "print_info: model params          = {:.2} T\n",
                model.n_elements as f64 * 1e-12
            );
        } else if model.n_elements >= 1e9 as u64 {
            crate::llama_log_info!(
                "print_info: model params          = {:.2} B\n",
                model.n_elements as f64 * 1e-9
            );
        } else if model.n_elements >= 1e6 as u64 {
            crate::llama_log_info!(
                "print_info: model params          = {:.2} M\n",
                model.n_elements as f64 * 1e-6
            );
        } else {
            crate::llama_log_info!(
                "print_info: model params          = {:.2} K\n",
                model.n_elements as f64 * 1e-3
            );
        }

        // general kv (llama-model.cpp:2096)
        crate::llama_log_info!("print_info: general.name          = {}\n", model.name);

        // per-arch blocks (llama-model.cpp:2098-2171)
        if arch == DEEPSEEK {
            crate::llama_log_info!("print_info: n_layer_dense_lead    = {}\n", h.n_layer_dense_lead as i32);
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_expert_shared       = {}\n", h.n_expert_shared as i32);
            crate::llama_log_info!(
                "print_info: expert_weights_scale  = {:.1}\n",
                h.expert_weights_scale
            );
        }
        if matches!(
            arch,
            DEEPSEEK2 | DEEPSEEK2OCR | DEEPSEEK32 | GLM_DSA | DOTS3NOTE | MISTRAL4 | HY_V4
        ) {
            crate::llama_log_info!("print_info: n_layer_dense_lead    = {}\n", h.n_layer_dense_lead as i32);
            crate::llama_log_info!("print_info: n_lora_q              = {}\n", h.n_lora_q as i32);
            crate::llama_log_info!("print_info: n_lora_kv             = {}\n", h.n_lora_kv as i32);
            crate::llama_log_info!("print_info: n_embd_head_k_mla     = {}\n", h.n_embd_head_k_mla() as i32);
            crate::llama_log_info!("print_info: n_embd_head_v_mla     = {}\n", h.n_embd_head_v_mla() as i32);
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_expert_shared       = {}\n", h.n_expert_shared as i32);
            crate::llama_log_info!(
                "print_info: expert_weights_scale  = {:.1}\n",
                h.expert_weights_scale
            );
            crate::llama_log_info!(
                "print_info: expert_weights_norm   = {}\n",
                h.expert_weights_norm as i32
            );
            crate::llama_log_info!(
                "print_info: expert_gating_func    = {}\n",
                expert_gating_func_name(h.expert_gating_func)
            );
        }
        if arch == QWEN2MOE {
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_ff_shexp            = {}\n", h.n_ff_shexp as i32);
        }
        if matches!(arch, MELLUM | COHERE2MOE | QWEN3MOE | OPENAI_MOE | QWEN3VLMOE | RND1) {
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
        }
        if matches!(
            arch,
            MINICPM | GRANITE | GRANITE_MOE | GRANITE_HYBRID | GRANITE_SWITCH | NEMOTRON_H_MOE
        ) {
            crate::llama_log_info!("print_info: f_embedding_scale     = {:.6}\n", h.f_embedding_scale);
            crate::llama_log_info!("print_info: f_residual_scale      = {:.6}\n", h.f_residual_scale);
            crate::llama_log_info!("print_info: f_attention_scale     = {:.6}\n", h.f_attention_scale);
            crate::llama_log_info!("print_info: n_ff_shexp            = {}\n", h.n_ff_shexp as i32);
        }
        if arch == BAILINGMOE {
            crate::llama_log_info!("print_info: n_layer_dense_lead    = {}\n", h.n_layer_dense_lead as i32);
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_expert_shared       = {}\n", h.n_expert_shared as i32);
            crate::llama_log_info!(
                "print_info: expert_weights_scale  = {:.1}\n",
                h.expert_weights_scale
            );
            crate::llama_log_info!(
                "print_info: expert_weights_norm   = {}\n",
                h.expert_weights_norm as i32
            );
        }
        if matches!(arch, BAILINGMOE2 | BAILINGMOE3) {
            crate::llama_log_info!("print_info: n_layer_dense_lead    = {}\n", h.n_layer_dense_lead as i32);
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_ff_shexp            = {}\n", h.n_ff_shexp as i32);
            crate::llama_log_info!("print_info: n_expert_shared       = {}\n", h.n_expert_shared as i32);
            crate::llama_log_info!(
                "print_info: expert_weights_scale  = {:.1}\n",
                h.expert_weights_scale
            );
            crate::llama_log_info!(
                "print_info: expert_weights_norm   = {}\n",
                h.expert_weights_norm as i32
            );
            crate::llama_log_info!(
                "print_info: expert_gating_func    = {}\n",
                expert_gating_func_name(h.expert_gating_func)
            );
            crate::llama_log_info!("print_info: n_layer_nextn         = {}\n", h.n_layer_nextn as i32);
        }
        if matches!(arch, SMALLTHINKER | LFM2MOE) {
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!(
                "print_info: expert_gating_func    = {}\n",
                expert_gating_func_name(h.expert_gating_func)
            );
        }
        if arch == GROVEMOE {
            crate::llama_log_info!("print_info: n_ff_exp              = {}\n", h.n_ff_exp(0) as i32);
            crate::llama_log_info!("print_info: n_ff_chexp            = {}\n", h.n_ff_chexp as i32);
            crate::llama_log_info!("print_info: n_group_experts       = {}\n", h.n_group_experts as i32);
            crate::llama_log_info!(
                "print_info: expert_group_scale    = {:.2}\n",
                h.expert_group_scale
            );
        }
    }

    vocab_print_info(vocab);
}

/// `llama_vocab::print_info` (llama-vocab.cpp:3896-3930; called from
/// `llama_model::print_info`, llama-model.cpp:2174).
pub fn vocab_print_info(vocab: &Vocab) {
    let type_name = match vocab.ty {
        VocabType::None => "no vocab",
        VocabType::Spm => "SPM",
        VocabType::Bpe => "BPE",
        VocabType::Wpm => "WPM",
        VocabType::Ugm => "UGM",
        VocabType::Rwkv => "RWKV",
        VocabType::Plamo2 => "PLaMo2",
        VocabType::Test => "TEST",
    };
    let tok = |id: i32| vocab.id_to_token[id as usize].text.clone();
    crate::llama_log_info!("print_info: vocab type            = {}\n", type_name);
    crate::llama_log_info!("print_info: n_vocab               = {}\n", vocab.id_to_token.len());
    crate::llama_log_info!("print_info: n_merges              = {}\n", vocab.get_bpe_merges().len());

    // special tokens
    if vocab.special_bos_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: BOS token             = {} '{}'\n",
            vocab.special_bos_id,
            tok(vocab.special_bos_id)
        );
    }
    if vocab.special_eos_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: EOS token             = {} '{}'\n",
            vocab.special_eos_id,
            tok(vocab.special_eos_id)
        );
    }
    if vocab.special_eot_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: EOT token             = {} '{}'\n",
            vocab.special_eot_id,
            tok(vocab.special_eot_id)
        );
    }
    if vocab.special_eom_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: EOM token             = {} '{}'\n",
            vocab.special_eom_id,
            tok(vocab.special_eom_id)
        );
    }
    if vocab.special_unk_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: UNK token             = {} '{}'\n",
            vocab.special_unk_id,
            tok(vocab.special_unk_id)
        );
    }
    if vocab.special_sep_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: SEP token             = {} '{}'\n",
            vocab.special_sep_id,
            tok(vocab.special_sep_id)
        );
    }
    if vocab.special_pad_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: PAD token             = {} '{}'\n",
            vocab.special_pad_id,
            tok(vocab.special_pad_id)
        );
    }
    if vocab.special_mask_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: MASK token            = {} '{}'\n",
            vocab.special_mask_id,
            tok(vocab.special_mask_id)
        );
    }
    if vocab.linefeed_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: LF token              = {} '{}'\n",
            vocab.linefeed_id,
            tok(vocab.linefeed_id)
        );
    }
    if vocab.special_fim_pre_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM PRE token         = {} '{}'\n",
            vocab.special_fim_pre_id,
            tok(vocab.special_fim_pre_id)
        );
    }
    if vocab.special_fim_suf_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM SUF token         = {} '{}'\n",
            vocab.special_fim_suf_id,
            tok(vocab.special_fim_suf_id)
        );
    }
    if vocab.special_fim_mid_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM MID token         = {} '{}'\n",
            vocab.special_fim_mid_id,
            tok(vocab.special_fim_mid_id)
        );
    }
    if vocab.special_fim_pad_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM PAD token         = {} '{}'\n",
            vocab.special_fim_pad_id,
            tok(vocab.special_fim_pad_id)
        );
    }
    if vocab.special_fim_rep_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM REP token         = {} '{}'\n",
            vocab.special_fim_rep_id,
            tok(vocab.special_fim_rep_id)
        );
    }
    if vocab.special_fim_sep_id != crate::vocab::TOKEN_NULL {
        crate::llama_log_info!(
            "print_info: FIM SEP token         = {} '{}'\n",
            vocab.special_fim_sep_id,
            tok(vocab.special_fim_sep_id)
        );
    }
    for &id in &vocab.special_eog_ids {
        crate::llama_log_info!("print_info: EOG token             = {} '{}'\n", id, tok(id));
    }
    crate::llama_log_info!("print_info: max token length      = {}\n", vocab.max_token_len);
}

// ---------------------------------------------------------------------------
// desc / system info
// ---------------------------------------------------------------------------

/// `llama_model::type_name` (llama-model.cpp:1896-1898).
pub fn model_type_name(model: &LlamaModel, n_vocab: u32) -> String {
    llm_type_name(llm_type_of(model.arch, &model.hparams, n_vocab)).to_string()
}

/// `llama_model::desc` (llama-model.cpp:1415 `desc_str = arch_name() + " " +
/// type_name() + " " + ml.ftype_name()`; :1900-1902).
pub fn model_desc(model: &LlamaModel, n_vocab: u32) -> String {
    format!(
        "{} {} {}",
        model.arch.name(),
        model_type_name(model, n_vocab),
        ftype_name(model.ftype)
    )
}

/// `llama_print_system_info` (llama.cpp:598-617): the backend registry's
/// feature dump. The port has a single CPU backend; the feature list mirrors
/// the flags the reference's CPU backend reports for this machine (the
/// reference's exact string for the x86-64 build is pinned by the parity
/// banner test).
pub fn print_system_info() -> String {
    let mut s = String::new();
    s += "CPU : ";
    let mut first = true;
    let mut feat = |name: &str, on: bool| {
        if on {
            if !first {
                s += " | ";
            }
            first = false;
            s += name;
            s += " = 1";
        }
    };
    #[cfg(target_arch = "x86_64")]
    {
        feat("SSE3", std::arch::is_x86_feature_detected!("sse3"));
        feat("SSSE3", std::arch::is_x86_feature_detected!("ssse3"));
        feat("AVX", std::arch::is_x86_feature_detected!("avx"));
        feat("AVX2", std::arch::is_x86_feature_detected!("avx2"));
        feat("F16C", std::arch::is_x86_feature_detected!("f16c"));
        feat("FMA", std::arch::is_x86_feature_detected!("fma"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        feat("NEON", true);
    }
    s += " | ";
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C `%.1e` semantics: 1e-6 -> "1.0e-06" (the reference banner's
    /// f_norm_rms_eps shape), 0.0 -> "0.0e+00".
    #[test]
    fn cfmt_e_matches_c() {
        assert_eq!(cfmt_e(1e-6, 1), "1.0e-06");
        assert_eq!(cfmt_e(0.0, 1), "0.0e+00");
        assert_eq!(cfmt_e(1e32, 1), "1.0e+32");
        assert_eq!(cfmt_e(-2.5e8, 2), "-2.50e+08");
        assert_eq!(cfmt_e(5e-5, 1), "5.0e-05");
    }

    /// C `%g` semantics: trailing zeros stripped, plain integers without
    /// decimal point, 1.0 -> "1".
    #[test]
    fn cfmt_g_matches_c() {
        assert_eq!(cfmt_g(1.0), "1");
        assert_eq!(cfmt_g(0.5), "0.5");
        assert_eq!(cfmt_g(0.0001), "0.0001");
        assert_eq!(cfmt_g(100000.0), "100000");
        assert_eq!(cfmt_g(1000000.0), "1e+06");
        assert_eq!(cfmt_g(0.00001), "1e-05");
    }

    /// The full name table round-trips (llama-model.cpp:868-1005 spot checks).
    #[test]
    fn type_names() {
        assert_eq!(llm_type_name(LlmType::B0_5), "0.5B");
        assert_eq!(llm_type_name(LlmType::B2_8T_A50), "2.8T.A50B");
        assert_eq!(llm_type_name(LlmType::B17B_16E), "17Bx16E (Scout)");
        assert_eq!(llm_type_name(LlmType::UNKNOWN), "?B");
        assert_eq!(llm_type_name(LlmType::B124B_A5_1), "124B.A5.1B");
    }

    /// qwen2.5-0.5b: n_layer 24, n_embd 896 -> 0.5B (models/qwen2.cpp:7).
    #[test]
    fn qwen2_detection() {
        let mut h = LlamaHparams::new();
        h.n_layer_all = 24; // n_layer() = 24 (nextn 0)
        h.n_embd = 896;
        h.n_head_arr = vec![14; 24];
        h.n_head_kv_arr = vec![2; 24];
        h.n_ff_arr = vec![4864; 24];
        // models/qwen2.cpp:7: n_embd == 1024 -> 0.5B; 896 (the 0.5B file) -> 1B
        assert_eq!(llm_type_of(LlmArch::QWEN2, &h, 151936), LlmType::B1);
        h.n_embd = 1024;
        assert_eq!(llm_type_of(LlmArch::QWEN2, &h, 151936), LlmType::B0_5);
    }

    /// ftype_name: Q4_K_M raw value 15 (guessed bit never set by files) — the
    /// guessed-bit branch prepends the C prefix (llama-model-loader.cpp:75).
    #[test]
    fn ftype_names() {
        assert_eq!(ftype_name(15), "Q4_K - Medium");
        assert_eq!(ftype_name(15 | LLAMA_FTYPE_GUESSED), "(guessed) Q4_K - Medium");
        assert_eq!(ftype_name(0), "all F32");
                assert_eq!(ftype_name(-1), "(guessed) unknown, may not work"); // -1 carries the GUESSED bit
    }

    #[test]
    fn rope_and_gating_names() {
        assert_eq!(rope_scaling_type_name(LlamaRopeScalingType::YARN), "yarn");
        assert_eq!(rope_scaling_type_name(LlamaRopeScalingType::NONE), "none");
        assert_eq!(expert_gating_func_name(1), "softmax");
        assert_eq!(expert_gating_func_name(2), "sigmoid");
        assert_eq!(expert_gating_func_name(4), "sqrtsoftplus");
        assert_eq!(expert_gating_func_name(99), "unknown");
        assert_eq!(flash_attn_type_name(0), "auto");
        assert_eq!(file_version_name(3), "GGUF V3 (latest)");
    }
}
