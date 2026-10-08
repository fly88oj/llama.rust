//! ggml — Rust 1:1 port of the llama.cpp ggml tensor library (CPU path).
//!
//! Reference: llama.cpp pinned revision `bd4f514db1`
//! (`/home/jeffrey/llm/llama.cpp-pinned`) — the same tree the parity
//! artifacts in `parity/` were generated from. Mapping:
//!   ggml/src/ggml-common.h  -> blocks.rs
//!   ggml/src/ggml.c(+cpp)   -> types.rs, tensor.rs, graph.rs, ops.rs
//!   ggml/src/gguf.cpp       -> gguf.rs, gguf_write.rs
//!   ggml/src/ggml-quants.c  -> quants.rs, quants_k.rs, quants_impl.rs
//!   ggml/src/ggml-cpu/*     -> compute.rs, vec_dot.rs, simd_x86.rs,
//!                              repack.rs, tinyblas.rs, ssm.rs, flash_attn.rs,
//!                              gdn.rs, wkv.rs (see PARITY.md / FILE_MAP.md)
//!   ggml/src/ggml-backend.cpp / -reg.cpp / ggml-alloc.c /
//!   ggml-cpu/ggml-cpu.cpp (backend iface) ->
//!                            backend.rs, backend_sched.rs, sysffi.rs

pub mod backend;
pub mod backend_emit;
pub mod backend_sched;
pub mod blocks;
pub mod compute;
pub mod flash_attn;
pub mod gdn;
pub mod gguf;
pub mod gguf_write;
pub mod graph;
pub mod ops;
pub mod quants;
pub mod quants_k;
pub mod quants_impl;
pub mod repack;
pub mod simd_x86;
pub mod ssm;
pub mod sysffi;
pub mod tensor;
pub mod tiled;
pub mod tinyblas;
pub mod types;
pub mod vec_dot;
pub mod wkv;

pub use gguf::{Gguf, GgufError, GgufType, TensorInfo, Value};
pub use graph::Graph;
pub use tensor::{Context, GgmlOp, Storage, TensorId, TensorMeta};
pub use types::{GgmlType, MAX_DIMS, MAX_NAME, MAX_OP_PARAMS, MAX_SRC};

#[cfg(test)]
mod tests {
    use crate::*;

    /// Real-file smoke test: parse the small local Qwen2.5 GGUF and sanity-check
    /// its metadata against known values for this model (hidden size 896,
    /// 24 layers, Qwen2 arch, Q4_K_M tensor mix).
    #[test]
    fn parse_real_qwen_gguf() {
        let path = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
        let Ok(gguf) = Gguf::open(path) else {
            eprintln!("skipping: {path} not present");
            return;
        };
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.get_str("general.architecture"), Some("qwen2"));
        assert_eq!(gguf.get_u32("qwen2.block_count"), Some(24));
        assert_eq!(gguf.get_u32("qwen2.embedding_length"), Some(896));
        assert_eq!(gguf.get_u32("qwen2.context_length"), Some(32768));
        // every declared tensor must be fully inside the file
        let file_len = std::fs::metadata(path).unwrap().len();
        for ti in &gguf.tensors {
            let end = gguf.data_offset + ti.offset + ti.size_bytes();
            assert!(end <= file_len, "{} overruns file", ti.name);
        }
        // Q4_K_M mix (cross-checked against reference gguf-py reader for this file:
        // output=Q8_0, token_embd=Q5_0, ffn_down=Q6_K)
        assert_eq!(gguf.find_tensor("token_embd.weight").unwrap().ty, GgmlType::Q5_0);
        assert_eq!(gguf.find_tensor("blk.0.ffn_down.weight").unwrap().ty, GgmlType::Q6K);
        // data section starts where the reference reader puts the first tensor
        assert_eq!(gguf.data_offset, 5947744);
        assert_eq!(
            gguf.find_tensor("token_embd.weight").unwrap().offset,
            150590816 - 5947744
        );
        // tensor data views must be non-empty
        assert!(gguf.tensor_data("token_embd.weight").unwrap().len() > 1_000_000);
        // total data section close to file size
        let last = gguf.tensors.iter().map(|t| t.offset + t.size_bytes()).max().unwrap();
        assert!(last + gguf.data_offset <= file_len);
        let vocab: Vec<&str> = gguf
            .find_key("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array().map(|(_, a)| a))
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(vocab.len(), 151936);
        assert_eq!(vocab[0], "!");
    }
}
