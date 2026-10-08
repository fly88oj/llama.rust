//! `llama_model_desc` for the markdown `model` column plus
//! `llama_model_size` / `llama_model_n_params`.
//!
//! * `desc_str = arch_name() + " " + type_name() + " " + ftype_name()`
//!   (llama-model.cpp:1415) — `type_name` is `llm_type_name(model->type)`,
//!   whose `type` each `src/models/<arch>.cpp` derives from
//!   n_layer/n_embd/n_ff/n_vocab (`llm_type_name` itself: llama-model.cpp:868-1005);
//! * `ftype_name` = `llama_ftype_name(ftype)` (llama-model-loader.cpp:33-76),
//!   `ftype` guessed from the most frequent tensor type and then overridden by
//!   `general.file_type` when the file carries it (llama-model-loader.cpp:731-797);
//! * `size()` = sum of `ggml_nbytes` and `n_elements()` = sum of
//!   `ggml_nelements` over every tensor of the file
//!   (llama-model-loader.cpp:700-711).

use ggml::gguf::Gguf;
use llama::arch::LlmArch;
use llama::hparams::LlamaHparams;
use llama::quant::Ftype;

/// `llama_ftype_name` (llama-model-loader.cpp:33-76): the `LLAMA_FTYPE_PREFIX`
/// ("mostly ") is stripped when the value did *not* come from
/// `general.file_type` (llama-model-loader.cpp:75), i.e. every real quantized
/// file prints as `Q4_K - Medium` and a file without the metadata key prints as
/// `mostly Q4_K - Medium`.
pub fn ftype_name(ftype: i32) -> String {
    let guessed = ftype & Ftype::Guessed as i32 != 0;
    let name = match ftype & !(Ftype::Guessed as i32) {
        0 => "all F32",
        1 => "F16",
        32 => "BF16",
        40 => "Q1_0",
        41 => "Q2_0",
        2 => "Q4_0",
        3 => "Q4_1",
        8 => "Q5_0",
        9 => "Q5_1",
        7 => "Q8_0",
        38 => "MXFP4 MoE",
        39 => "NVFP4",
        10 => "Q2_K - Medium",
        21 => "Q2_K - Small",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        36 => "TQ1_0 - 1.69 bpw ternary",
        37 => "TQ2_0 - 2.06 bpw ternary",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        24 => "IQ1_S - 1.5625 bpw",
        31 => "IQ1_M - 1.75 bpw",
        25 => "IQ4_NL - 4.5 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        26 => "IQ3_S - 3.4375 bpw",
        27 => "IQ3_S mix - 3.66 bpw",
        _ => "unknown, may not work",
    };
    // the C's table carries the "mostly " prefix and strips it (`name +
    // guessed_prefix_len`) for a non-guessed value
    if guessed {
        format!("mostly {name}")
    } else {
        name.to_string()
    }
}

/// The ftype `llama_model_loader` ends up with (llama-model-loader.cpp:731-797):
/// the type that appears most often among the file's tensors (iterating
/// `weights_map`, i.e. tensor-name order, with a strict `<` so a tie keeps the
/// earlier type), marked GUESSED, then overridden by `general.file_type`.
pub fn guess_ftype(gguf: &Gguf) -> i32 {
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    let mut type_max = ggml::types::GgmlType::F32;
    let mut n_type_max = 0u32;
    // weights_map is a std::map keyed by tensor name -> name order
    let mut tensors: Vec<&ggml::gguf::TensorInfo> = gguf.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for t in tensors {
        let c = counts.entry(t.ty as u32).or_insert(0);
        *c += 1;
        if n_type_max < *c {
            n_type_max = *c;
            type_max = t.ty;
        }
    }
    // llama-model-loader.cpp:754-789
    use ggml::types::GgmlType as T;
    let ftype: i32 = match type_max {
        T::F32 => Ftype::AllF32 as i32,
        T::F16 => Ftype::MostlyF16 as i32,
        T::Bf16 => Ftype::MostlyBF16 as i32,
        T::Q1_0 => Ftype::MostlyQ1_0 as i32,
        T::Q2_0 => Ftype::MostlyQ2_0 as i32,
        T::Q4_0 => Ftype::MostlyQ4_0 as i32,
        T::Q4_1 => Ftype::MostlyQ4_1 as i32,
        T::Q5_0 => Ftype::MostlyQ5_0 as i32,
        T::Q5_1 => Ftype::MostlyQ5_1 as i32,
        T::Q8_0 => Ftype::MostlyQ8_0 as i32,
        T::Q2K => Ftype::MostlyQ2_K as i32,
        T::Q3K => Ftype::MostlyQ3_K_M as i32,
        T::Q4K => Ftype::MostlyQ4_K_M as i32,
        T::Q5K => Ftype::MostlyQ5_K_M as i32,
        T::Q6K => Ftype::MostlyQ6_K as i32,
        T::Tq1_0 => Ftype::MostlyTQ1_0 as i32,
        T::Tq2_0 => Ftype::MostlyTQ2_0 as i32,
        T::Iq2Xxs => Ftype::MostlyIQ2_XXS as i32,
        T::Iq2Xs => Ftype::MostlyIQ2_XS as i32,
        T::Iq2S => Ftype::MostlyIQ2_S as i32,
        T::Iq3Xxs => Ftype::MostlyIQ3_XXS as i32,
        T::Iq1S => Ftype::MostlyIQ1_S as i32,
        T::Iq1M => Ftype::MostlyIQ1_M as i32,
        T::Iq4Nl => Ftype::MostlyIQ4_NL as i32,
        T::Iq4Xs => Ftype::MostlyIQ4_XS as i32,
        T::Iq3S => Ftype::MostlyIQ3_S as i32,
        T::Nvfp4 => Ftype::MostlyNVFP4 as i32,
        _ => Ftype::AllF32 as i32, // with LLAMA_LOG_WARN in the C
    };
    let guessed = ftype | Ftype::Guessed as i32;
    match gguf.get_u32("general.file_type") {
        Some(v) => v as i32,
        None => guessed,
    }
}

/// `model->desc()` for the archs this port can run: `llm_arch_name(arch)`
/// (crates/llama/src/arch.rs, `LlmArch::name`) + `type_name` + `ftype_name`.
pub fn model_type(arch: LlmArch, hp: &LlamaHparams, n_vocab: u32, gguf: &Gguf) -> String {
    format!("{} {} {}", arch.name(), type_name(arch, hp, n_vocab), ftype_name(guess_ftype(gguf)))
}

/// One `switch (hparams.n_layer())` per `src/models/<arch>.cpp` `load_arch_hparams`
/// (the `type = LLM_TYPE_*` assignments), mapped through `llm_type_name`
/// (llama-model.cpp:868-1005); `default: LLM_TYPE_UNKNOWN` -> `"?B"`.
pub fn type_name(arch: LlmArch, hp: &LlamaHparams, n_vocab: u32) -> &'static str {
    let nl = hp.n_layer();
    let n_embd = hp.n_embd;
    let n_head = |il: usize| hp.n_head(il);
    match arch {
        // src/models/qwen2.cpp:7-16
        LlmArch::QWEN2 => match nl {
            24 => if n_embd == 1024 { "0.5B" } else { "1B" },
            28 => if n_embd == 1536 { "1.5B" } else { "7B" },
            32 => "7B",
            36 => "3B",
            40 => if n_head(0) == 20 { "4B" } else { "13B" },
            48 => "14B",
            64 => "32B",
            80 => "70B",
            _ => "?B",
        },
        // src/models/qwen3.cpp:7-12
        LlmArch::QWEN3 => match nl {
            28 => if n_embd == 1024 { "0.6B" } else { "1.7B" },
            36 => if n_embd == 2560 { "4B" } else { "8B" },
            40 => "14B",
            64 => "32B",
            _ => "?B",
        },
        // src/models/qwen35.cpp:26-30
        LlmArch::QWEN35 => match nl {
            24 => if n_embd == 1024 { "0.8B" } else { "2B" },
            32 => if n_embd == 2560 { "4B" } else { "9B" },
            64 => "27B",
            _ => "?B",
        },
        // src/models/llama.cpp:9-31
        LlmArch::LLAMA => {
            if hp.n_expert == 8 {
                match nl {
                    32 => "8x7B",
                    56 => "8x22B",
                    _ => "?B",
                }
            } else {
                match nl {
                    16 | 22 => "1B",
                    26 | 28 => "3B",
                    30 => "256M",
                    32 if n_vocab == 49152 => "3B",
                    32 if n_vocab < 40000 => "7B",
                    32 => "8B",
                    36 => "8B",
                    40 => "13B",
                    48 => "34B",
                    60 => "30B",
                    80 => if n_head(0) == hp.n_head_kv(0) { "65B" } else { "70B" },
                    _ => "?B",
                }
            }
        }
        // src/models/phi3.cpp:7-11
        LlmArch::PHI3 => match nl {
            24 => "1B",
            32 => "3B",
            40 => "14B",
            _ => "?B",
        },
        // src/models/gemma2.cpp:18-22
        LlmArch::GEMMA2 => match nl {
            26 => "2B",
            42 => "9B",
            46 => "27B",
            _ => "?B",
        },
        // src/models/gemma3.cpp:19-26
        LlmArch::GEMMA3 => match nl {
            18 => "270M",
            26 => "1B",
            32 => "8B",
            34 => "4B",
            48 => "12B",
            62 => "27B",
            _ => "?B",
        },
        // src/models/gemma4.cpp:28-33
        LlmArch::GEMMA4 => match nl {
            30 => "26B.A4B",
            35 => "E2B",
            42 => "E4B",
            60 => "31B",
            _ => "?B",
        },
        // src/models/granite.cpp:39-53
        LlmArch::GRANITE => match nl {
            32 => "3B",
            40 => match n_embd {
                2048 => "2B",
                2560 => "3B",
                4096 => "8B",
                _ => "?B",
            },
            64 => "30B",
            _ => "?B",
        },
        // src/models/granite-hybrid.cpp:28-35
        LlmArch::GRANITE_HYBRID => match n_embd {
            768 => "350M",
            1536 => if hp.n_ff(0) == 512 { "7B.A1B" } else { "1B" },
            2048 | 2560 => "3B",
            4096 => "32B.A9B",
            _ => "?B",
        },
        // src/models/lfm2.cpp:17-24 (`n_ff()` on the first layer here)
        LlmArch::LFM2MOE => match hp.n_ff(0) {
            2560 => "230M",
            4608 => "350M",
            6912 => "700M",
            8192 => "1.2B",
            10752 => "2.6B",
            _ => "?B",
        },
        // src/models/openai-moe.cpp:16-19
        LlmArch::OPENAI_MOE => match nl {
            24 => "20B",
            36 => "120B",
            _ => "?B",
        },
        _ => "?B",
    }
}

/// `llama_model_size` / `llama_model_n_params` (llama-model-loader.cpp:700-711):
/// the sums over the file's tensors.
pub fn size_and_params(gguf: &Gguf) -> (u64, u64) {
    let mut n_bytes = 0u64;
    let mut n_elements = 0u64;
    for t in &gguf.tensors {
        n_bytes += t.size_bytes();
        n_elements += t.n_elements().max(0) as u64;
    }
    (n_bytes, n_elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftype_names_match_the_reference_table() {
        // the values the reference prints for the local models
        assert_eq!(ftype_name(15), "Q4_K - Medium"); // qwen2.5 q4_k_m metadata
        assert_eq!(ftype_name(1), "F16");
        assert_eq!(ftype_name(26), "IQ3_S - 3.4375 bpw");
        assert_eq!(ftype_name(999), "unknown, may not work");
        // LLAMA_FTYPE_GUESSED (1024) keeps the C's "mostly " prefix
        assert_eq!(ftype_name(15 | 1024), "mostly Q4_K - Medium");
    }
}