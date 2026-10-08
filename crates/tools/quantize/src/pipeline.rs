//! pipeline.rs — port of `llama_model_quantize_impl` (llama-quant.cpp:913-1359):
//! read the input GGUF, decide a target type per tensor, write the output GGUF.
//!
//! Deliberate deviations from the reference (all reported, none affecting the
//! bytes written):
//! * no model instantiation — arch/hparams come from `llama::meta::load_hparams`
//!   and the quantizer never touches the graph (llama-quant.cpp:935-952);
//! * `--prune-layers`, `--keep-split`, `--override-kv` are rejected by the CLI
//!   (see main.rs), so the state/params code for them is absent; since
//!   `--prune-layers` is what fills `remap_imatrix`'s `mapped` map
//!   (llama-quant.cpp:79-100), `remapped_imatrix_name` is the tensor name
//!   itself;
//! * single-threaded *or* rayon per row (the reference's threads split by row;
//!   identical output);
//! * `--dry-run` is supported (llama-quant.cpp:1208-1225).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ggml::gguf::{Gguf, TensorInfo, Value, GGUF_DEFAULT_ALIGNMENT};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use llama::arch::{kv_name, LlmKv};
use llama::imatrix::common_imatrix_load;
use llama::meta;
use llama::quant::{
    init_quantize_state_counters, llama_tensor_get_type, tensor_requires_imatrix, QuantizeParams,
    QuantizeState, QuantModelInfo, TensorMetadata, LLAMA_QUANT_MAX_BUF_SIZE,
};

use crate::rows::{dequantize_into, has_quantizer, quantize_rows_imatrix};

/// `GGML_QNT_VERSION` (ggml.h:219)
const GGML_QNT_VERSION: u32 = 2;

/// `LLM_KV_GENERAL_QUANTIZATION_VERSION` / `..._FILE_TYPE` (llama-quant.cpp:991-992)
const KV_QUANTIZATION_VERSION: &str = "general.quantization_version";
const KV_FILE_TYPE: &str = "general.file_type";

/// `LLM_KV_QUANTIZE_IMATRIX_*` (tools/quantize/quantize.cpp:77-80)
const KV_QUANTIZE_IMATRIX_FILE: &str = "quantize.imatrix.file";
const KV_QUANTIZE_IMATRIX_DATASET: &str = "quantize.imatrix.dataset";
const KV_QUANTIZE_IMATRIX_N_ENTRIES: &str = "quantize.imatrix.entries_count";
const KV_QUANTIZE_IMATRIX_N_CHUNKS: &str = "quantize.imatrix.chunks_count";

/// `LLM_KV_SPLIT_*` — removed from the output (llama-quant.cpp:995-997).
fn split_keys() -> [String; 3] {
    [
        LlmKv::SPLIT_NO.template().to_string(),
        LlmKv::SPLIT_COUNT.template().to_string(),
        LlmKv::SPLIT_TENSORS_COUNT.template().to_string(),
    ]
}

/// The normalized importance matrix handed to the quantizer: the port of
/// `load_imatrix` (tools/quantize/quantize.cpp:183-260), i.e. the raw sums of
/// `common_imatrix_load` divided by their per-expert token counts.
pub struct ImatrixInput {
    /// `imatrix_file`
    pub file: String,
    /// `imatrix_datasets` (`loaded.datasets`)
    pub datasets: Vec<String>,
    /// `loaded.chunk_count` — the C's `m_last_call`
    pub chunk_count: i32,
    /// `imatrix_data` — name → `ne[0] * ne[2]` normalized weights
    pub entries: HashMap<String, Vec<f32>>,
}

/// `static int load_imatrix(...)` + `prepare_imatrix` (quantize.cpp:183-301),
/// minus `--include-weights` / `--exclude-weights` (rejected by the CLI).
pub fn load_imatrix(imatrix_file: &str) -> Result<ImatrixInput, String> {
    let Some(loaded) = common_imatrix_load(imatrix_file) else {
        return Err(format!("failed to load imatrix from '{imatrix_file}'"));
    };
    if !loaded.is_legacy && !loaded.has_metadata {
        return Err(format!("missing imatrix metadata in file {imatrix_file}"));
    }

    let mut entries: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, entry) in loaded.entries.iter() {
        let mut e = vec![0f32; entry.sums.len()];

        if !loaded.is_legacy {
            // GGUF format: normalize by per-expert counts
            let ncounts = entry.counts.len() as i64;
            let ne0 = entry.sums.len() as i64 / ncounts;
            for j in 0..ncounts {
                let count = entry.counts[j as usize] as f32;
                for i in 0..ne0 {
                    e[(j * ne0 + i) as usize] = if count > 0.0 {
                        entry.sums[(j * ne0 + i) as usize] / count
                    } else {
                        1.0
                    };
                }
            }
        } else {
            // Legacy format: sums contain (raw/count)*ncall, divide by ncall
            let ncall = entry.counts.first().copied().unwrap_or(0);
            if ncall > 0 {
                for (i, s) in entry.sums.iter().enumerate() {
                    e[i] = s / ncall as f32;
                }
            } else {
                e.copy_from_slice(&entry.sums);
            }
        }
        entries.insert(name.clone(), e);
    }

    if !loaded.datasets.is_empty() {
        print!("load_imatrix: imatrix datasets=['{}'", loaded.datasets[0]);
        for d in loaded.datasets.iter().skip(1) {
            print!(", '{d}'");
        }
        println!("]");
    }
    println!(
        "load_imatrix: loaded {} importance matrix entries from {} computed on {} chunks",
        entries.len(),
        imatrix_file,
        loaded.chunk_count
    );

    Ok(ImatrixInput {
        file: imatrix_file.to_string(),
        datasets: loaded.datasets,
        chunk_count: loaded.chunk_count,
        entries,
    })
}

pub struct QuantizeResult {
    pub total_size_org: u64,
    pub total_size_new: u64,
    pub n_fallback: i32,
    pub n_tensors: usize,
}

/// `llama_model_quantize_impl` (llama-quant.cpp:913-1359).
pub fn llama_model_quantize(
    fname_inp: &str,
    fname_out: &str,
    params: &QuantizeParams,
    imatrix: Option<&ImatrixInput>,
) -> Result<QuantizeResult, String> {
    // llama-quant.cpp:935-952: load metadata + hparams (no tensors are touched)
    let gguf = Gguf::open(fname_inp).map_err(|e| format!("failed to load model from {fname_inp}: {e}"))?;
    let (arch, hparams) =
        meta::load_hparams(&gguf).map_err(|e| format!("failed to load hparams: {e}"))?;

    let n_vocab = {
        let key = kv_name(arch, LlmKv::VOCAB_SIZE);
        gguf.get_u32(&key)
            .or_else(|| {
                // llama-model-base.cpp: `get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab)`
                gguf.find_key("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|(_, items)| items.len() as u32)
            })
            .unwrap_or(0) as i32
    };
    let model = QuantModelInfo::from_hparams(arch, &hparams, n_vocab);

    // The reference loads *all* shards of a split model (llama-model-loader.cpp
    // reads split.count and opens the siblings). This port reads the single
    // file it is handed, so refuse split inputs instead of silently writing an
    // incomplete model.
    if let Some(n) = gguf.get_u32(LlmKv::SPLIT_COUNT.template()) {
        if n > 1 {
            return Err(format!(
                "input is a split model (split.count = {n}); split inputs are not supported by \
                 this port (merge the shards first, e.g. with llama-gguf-split)"
            ));
        }
    }

    // llama-quant.cpp:915-925
    let mut ftype = params.ftype;
    if params.only_copy {
        // llama-quant.cpp:954-956 — COPY keeps the input file type
        ftype = gguf
            .get_u32(KV_FILE_TYPE)
            .map(|v| ftype_from_u32(v))
            .unwrap_or(llama::quant::Ftype::AllF32);
    }
    let default_type = match ftype.default_type() {
        Some(t) => t,
        None => return Err(format!("invalid output file type {}", ftype as i32)),
    };

    // llama-quant.cpp:1019-1053: build the weight list. `ml.weights_map` is a
    // std::map with `weight_name_comparer` (llama-model-loader.h:54-65, 126),
    // so tensors are emitted sorted by (blk layer, name) — NOT in file order.
    let mut tensors: Vec<&TensorInfo> = gguf.tensors.iter().collect();
    llama::quant::sort_weights_by_name(&mut tensors, |t| t.name.as_str());
    let n_tensors = tensors.len();

    // llama-quant.cpp:1049-1056: per-tensor metadata, computed in the
    // preliminary loop and used by the main loop.
    let mut metadata: Vec<TensorMetadata> =
        tensors.iter().map(|t| TensorMetadata::new(t.name.clone())).collect();
    let mut qs = QuantizeState::new(&model, &params.tt_overrides);
    init_quantize_state_counters(&mut qs, &mut metadata);

    // llama-quant.cpp:958-975: with an imatrix the state is flagged and every
    // value is checked for finiteness.
    if let Some(im) = imatrix {
        println!();
        println!(
            "llama_model_quantize_impl: have importance matrix data with {} entries",
            im.entries.len()
        );
        qs.has_imatrix = true;
        for (name, data) in im.entries.iter() {
            for f in data {
                if !f.is_finite() {
                    return Err(format!("imatrix contains non-finite value {f} (tensor {name})"));
                }
            }
        }
    }

    // ---- preliminary loop (llama-quant.cpp:1077-1112) ----
    for (i, t) in tensors.iter().enumerate() {
        metadata[i].allows_quantization =
            llama::quant::tensor_allows_quantization(params, arch, &t.name, &t.ne);

        metadata[i].target_type = if metadata[i].allows_quantization {
            llama_tensor_get_type(
                &mut qs,
                params,
                &t.name,
                &t.ne,
                t.ty,
                default_type,
                &metadata[i],
            )?
        } else {
            t.ty
        };

        metadata[i].requires_imatrix = tensor_requires_imatrix(&t.name, metadata[i].target_type, ftype);

        // llama-quant.cpp:1097-1109: with an imatrix the tensor name is looked
        // up (identity here — the `mapped` map comes from `--prune-layers`,
        // llama-quant.cpp:79-100); without one, a tensor that needs it is a hard
        // error.
        if imatrix.is_some() {
            metadata[i].remapped_imatrix_name = t.name.clone();
        } else if metadata[i].allows_quantization && metadata[i].requires_imatrix {
            llama::quant::log_error(
                "\n============================================================================\n\
                 \x20ERROR: this quantization requires an importance matrix!\n",
            );
            llama::quant::log_error(&format!(
                "        - offending tensor: {}\n        - target type: {}\n\
                 ============================================================================\n\n",
                metadata[i].name,
                metadata[i].target_type.name()
            ));
            return Err("this quantization requires an imatrix!".to_string());
        }

        // this port has no (revision-exact) quantizer for these → fail early,
        // before writing anything (see rows.rs::has_quantizer)
        if metadata[i].target_type != t.ty && !has_quantizer(metadata[i].target_type) {
            return Err(format!(
                "tensor {}: target type {} cannot be written by this port \
                 (no revision-exact quantizer)",
                t.name,
                metadata[i].target_type.name()
            ));
        }
        if metadata[i].target_type != t.ty {
            // llama-quant.cpp:1272-1274
            if t.ty.is_quantized() && !params.allow_requantize {
                return Err(format!(
                    "requantizing from type {} is disabled",
                    t.ty.name()
                ));
            }
            // llama-quant.cpp:220-228 (dequantizability of the source)
            if t.ty.is_quantized() {
                llama::quant::check_dequantizable(t.ty)?;
            } else if !matches!(t.ty, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16) {
                return Err(format!(
                    "cannot dequantize/convert tensor type {}",
                    t.ty.name()
                ));
            }
        }
    }

    // ---- output gguf context (llama-quant.cpp:979-1014) ----
    // gguf_set_kv copies every input KV (each via gguf_set_val_*, which
    // *removes and re-appends* the key — gguf.cpp:1265-1269), then
    // quantization_version and file_type are set, moving them to the end.
    let mut w = GgufWriter::new(GGUF_DEFAULT_ALIGNMENT);
    let remove = split_keys();
    for (k, v) in &gguf.kv {
        if k == KV_QUANTIZATION_VERSION || k == KV_FILE_TYPE || remove.contains(k) {
            continue;
        }
        w.set_kv(k, v.clone());
    }
    w.set_kv(KV_QUANTIZATION_VERSION, Value::U32(GGML_QNT_VERSION));
    w.set_kv(KV_FILE_TYPE, Value::U32(ftype as u32));

    // llama-quant.cpp:999-1014: the `--imatrix` bookkeeping keys land after
    // file_type, as u32 for the int overrides (LLAMA_KV_OVERRIDE_TYPE_INT) and
    // strings for the paths (quantize.cpp:500-541)
    if let Some(im) = imatrix {
        w.set_kv(KV_QUANTIZE_IMATRIX_FILE, Value::String(im.file.clone()));
        if let Some(d) = im.datasets.first() {
            w.set_kv(KV_QUANTIZE_IMATRIX_DATASET, Value::String(d.clone()));
        }
        w.set_kv(KV_QUANTIZE_IMATRIX_N_ENTRIES, Value::U32(im.entries.len() as u32));
        if im.chunk_count > 0 {
            w.set_kv(KV_QUANTIZE_IMATRIX_N_CHUNKS, Value::U32(im.chunk_count as u32));
        }
    }

    // gguf_add_tensor (llama-quant.cpp:1085). Offsets are computed from the
    // final types, which is what repeated gguf_set_tensor_type calls converge
    // to (gguf.cpp:1410-1435).
    for (t, meta) in tensors.iter().zip(&metadata) {
        w.add_tensor(&t.name, meta.target_type, t.ne);
    }

    let mut total_size_org: u64 = 0;
    let mut total_size_new: u64 = 0;

    if params.dry_run {
        // llama-quant.cpp:1208-1225 — size only, no output file
        for (i, (t, meta)) in tensors.iter().zip(&metadata).enumerate() {
            let tensor_size = t.size_bytes();
            let new_size: u64 = if t.ty != meta.target_type {
                // ggml_nrows(tensor) * ggml_row_size(new_type, ne[0])
                let nrows = (t.ne[1] * t.ne[2] * t.ne[3]) as u64;
                nrows * meta.target_type.row_size(t.ne[0] as usize) as u64
            } else {
                tensor_size
            };
            println!(
                "[{:4}/{:4}] {:<36} - type = {:>6} -> {:>6}, size = {:8.2} MiB -> {:8.2} MiB",
                i + 1,
                n_tensors,
                t.name,
                t.ty.name(),
                meta.target_type.name(),
                tensor_size as f64 / 1024.0 / 1024.0,
                new_size as f64 / 1024.0 / 1024.0,
            );
            total_size_org += tensor_size;
            total_size_new += new_size;
        }
        return Ok(QuantizeResult {
            total_size_org,
            total_size_new,
            n_fallback: qs.n_fallback,
            n_tensors,
        });
    }

    // ---- write (llama-quant.cpp:1133-1340) ----
    let f = File::create(fname_out).map_err(|e| format!("failed to open {fname_out}: {e}"))?;
    let mut fout = BufWriter::with_capacity(1 << 20, f);

    let max_buf_size = if params.max_buf_size != 0 {
        params.max_buf_size
    } else {
        LLAMA_QUANT_MAX_BUF_SIZE
    };

    // Metadata is known in full before the data section, so it can be written
    // first (the reference writes zeros then seeks back — same bytes).
    write_tensor_data(
        &mut fout,
        &gguf,
        &tensors,
        &metadata,
        &mut w,
        max_buf_size,
        imatrix,
        &mut total_size_org,
        &mut total_size_new,
    )?;

    fout.flush().map_err(|e| format!("write error: {e}"))?;
    drop(fout);

    Ok(QuantizeResult {
        total_size_org,
        total_size_new,
        n_fallback: qs.n_fallback,
        n_tensors,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_tensor_data(
    fout: &mut impl Write,
    gguf: &Gguf,
    tensors: &[&TensorInfo],
    metadata: &[TensorMetadata],
    w: &mut GgufWriter,
    max_buf_size: usize,
    imatrix: Option<&ImatrixInput>,
    total_size_org: &mut u64,
    total_size_new: &mut u64,
) -> Result<(), String> {
    let align = w.alignment;

    // Meta is complete now (all tensors added with their final types).
    w.write_meta(fout).map_err(|e| format!("write error: {e}"))?;

    for (idx, (t, meta)) in tensors.iter().zip(metadata).enumerate() {
        let tensor_size = t.size_bytes();
        let src = gguf
            .tensor_data(&t.name)
            .ok_or_else(|| format!("tensor {} missing from input data", t.name))?;

        let quantize = t.ty != meta.target_type;
        print!(
            "[{:4}/{:4}] {:<36} - type = {:>6}, ",
            idx + 1,
            tensors.len(),
            t.name,
            t.ty.name()
        );

        let new_size: u64 = if !quantize {
            // llama-quant.cpp:1228-1239 — copy in slabs of whole rows
            print!("size = {:8.3} MiB\n", tensor_size as f64 / 1024.0 / 1024.0);
            let row_size = t.ty.row_size(t.ne[0] as usize);
            let slab_size = std::cmp::max(row_size, (max_buf_size / row_size) * row_size);
            let mut offs = 0usize;
            while offs < src.len() {
                let n = std::cmp::min(slab_size, src.len() - offs);
                fout.write_all(&src[offs..offs + n])
                    .map_err(|e| format!("write error: {e}"))?;
                offs += n;
            }
            tensor_size
        } else {
            // llama-quant.cpp:1241-1266: pick this tensor's importance matrix
            // (identity name mapping) and check its size against the tensor
            let mut tensor_imatrix: Option<&[f32]> = None;
            if let Some(im) = imatrix {
                match im.entries.get(&meta.remapped_imatrix_name) {
                    None => {
                        println!();
                        println!(
                            "====== llama_model_quantize_impl: did not find weights for {}",
                            t.name
                        );
                    }
                    Some(weights) => {
                        let want = (t.ne[0] * t.ne[2]) as usize;
                        if weights.len() == want {
                            tensor_imatrix = Some(weights);
                        } else {
                            println!();
                            println!(
                                "====== llama_model_quantize_impl: imatrix size {} is different from tensor size {} for {}",
                                weights.len(),
                                want,
                                t.name
                            );
                            // tok_embd should be ignored in this case, since it
                            // always causes this warning (llama-quant.cpp:1262)
                            if !llama::quant::tensor_name_match_token_embd(&t.name) {
                                return Err(format!(
                                    "imatrix size {} is different from tensor size {} for {}",
                                    weights.len(),
                                    want,
                                    t.name
                                ));
                            }
                        }
                    }
                }
            }
            if tensor_imatrix.is_none() && meta.requires_imatrix {
                llama::quant::log_error(
                    "\n\n============================================================\n",
                );
                llama::quant::log_error(&format!(
                    "Missing importance matrix for tensor {} in a very low-bit quantization\n",
                    t.name
                ));
                llama::quant::log_error("The result will be garbage, so bailing out\n");
                llama::quant::log_error("============================================================\n\n");
                return Err(format!(
                    "Missing importance matrix for tensor {} in a very low-bit quantization",
                    t.name
                ));
            }

            print!("converting to {} .. ", meta.target_type.name());
            let n_per_row = t.ne[0];
            let nrows_per_expert = t.ne[1];
            let nrows_total = t.ne[1] * t.ne[2];

            let row_size_src = t.ty.row_size(n_per_row as usize);
            let row_size_dst = meta.target_type.row_size(n_per_row as usize);
            // llama-quant.cpp:1287
            let bytes_per_row = row_size_src
                + row_size_dst
                + if t.ty == GgmlType::F32 {
                    0
                } else {
                    n_per_row as usize * std::mem::size_of::<f32>()
                };
            let nrows_slab =
                std::cmp::max(1, std::cmp::min(nrows_total as usize, max_buf_size / bytes_per_row));

            let mut f32_buf: Vec<f32> = Vec::new();
            let mut work: Vec<u8> = Vec::new();
            let mut new_size = 0u64;

            let mut ir = 0i64;
            while ir < nrows_total {
                let nrows_cur = std::cmp::min(nrows_slab as i64, nrows_total - ir) as usize;
                let nelements_cur = nrows_cur * n_per_row as usize;

                let src_slab =
                    &src[ir as usize * row_size_src..(ir as usize + nrows_cur) * row_size_src];

                let f32_data: &[f32] = if t.ty == GgmlType::F32 {
                    bytemuck::cast_slice(src_slab)
                } else {
                    if f32_buf.len() < nelements_cur {
                        f32_buf.resize(nelements_cur, 0.0);
                    }
                    dequantize_into(t.ty, src_slab, &mut f32_buf[..nelements_cur])?;
                    &f32_buf[..nelements_cur]
                };

                if work.len() < nrows_cur * row_size_dst {
                    work.resize(nrows_cur * row_size_dst, 0);
                }

                let n_written = quantize_rows_imatrix(
                    meta.target_type,
                    f32_data,
                    nrows_cur as i64,
                    n_per_row,
                    &mut work[..nrows_cur * row_size_dst],
                    ir,
                    nrows_per_expert,
                    tensor_imatrix,
                )?;
                fout.write_all(&work[..n_written])
                    .map_err(|e| format!("write error: {e}"))?;
                new_size += n_written as u64;

                ir += nrows_cur as i64;
            }

            print!(
                "size = {:8.2} MiB -> {:8.2} MiB\n",
                tensor_size as f64 / 1024.0 / 1024.0,
                new_size as f64 / 1024.0 / 1024.0
            );
            new_size
        };

        *total_size_org += tensor_size;
        *total_size_new += new_size;

        // llama-quant.cpp:1334 — pad the tensor payload to the output alignment
        let pad = new_size.div_ceil(align) * align - new_size;
        fout.write_all(&vec![0u8; pad as usize])
            .map_err(|e| format!("write error: {e}"))?;
    }

    Ok(())
}

fn ftype_from_u32(v: u32) -> llama::quant::Ftype {
    // llama-model-loader.cpp: `if (!ftype) ftype = LLAMA_FTYPE_ALL_F32;`
    match v {
        0 => llama::quant::Ftype::AllF32,
        1 => llama::quant::Ftype::MostlyF16,
        2 => llama::quant::Ftype::MostlyQ4_0,
        3 => llama::quant::Ftype::MostlyQ4_1,
        7 => llama::quant::Ftype::MostlyQ8_0,
        8 => llama::quant::Ftype::MostlyQ5_0,
        9 => llama::quant::Ftype::MostlyQ5_1,
        10 => llama::quant::Ftype::MostlyQ2_K,
        11 => llama::quant::Ftype::MostlyQ3_K_S,
        12 => llama::quant::Ftype::MostlyQ3_K_M,
        13 => llama::quant::Ftype::MostlyQ3_K_L,
        14 => llama::quant::Ftype::MostlyQ4_K_S,
        15 => llama::quant::Ftype::MostlyQ4_K_M,
        16 => llama::quant::Ftype::MostlyQ5_K_S,
        17 => llama::quant::Ftype::MostlyQ5_K_M,
        18 => llama::quant::Ftype::MostlyQ6_K,
        19 => llama::quant::Ftype::MostlyIQ2_XXS,
        20 => llama::quant::Ftype::MostlyIQ2_XS,
        21 => llama::quant::Ftype::MostlyQ2_K_S,
        22 => llama::quant::Ftype::MostlyIQ3_XS,
        23 => llama::quant::Ftype::MostlyIQ3_XXS,
        24 => llama::quant::Ftype::MostlyIQ1_S,
        25 => llama::quant::Ftype::MostlyIQ4_NL,
        26 => llama::quant::Ftype::MostlyIQ3_S,
        27 => llama::quant::Ftype::MostlyIQ3_M,
        28 => llama::quant::Ftype::MostlyIQ2_S,
        29 => llama::quant::Ftype::MostlyIQ2_M,
        30 => llama::quant::Ftype::MostlyIQ4_XS,
        31 => llama::quant::Ftype::MostlyIQ1_M,
        32 => llama::quant::Ftype::MostlyBF16,
        36 => llama::quant::Ftype::MostlyTQ1_0,
        37 => llama::quant::Ftype::MostlyTQ2_0,
        38 => llama::quant::Ftype::MostlyMXFP4_MOE,
        39 => llama::quant::Ftype::MostlyNVFP4,
        40 => llama::quant::Ftype::MostlyQ1_0,
        41 => llama::quant::Ftype::MostlyQ2_0,
        _ => llama::quant::Ftype::Guessed,
    }
}

/// Path helper used by the CLI to reject in == out (quantize.cpp:620-625).
pub fn is_same_file(a: &str, b: &str) -> bool {
    let (pa, pb) = (Path::new(a), Path::new(b));
    match (pa.canonicalize(), pb.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}