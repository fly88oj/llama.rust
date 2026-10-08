//! imatrix.rs — 1:1 port of llama.cpp `tools/imatrix/imatrix.cpp` (bd4f514db1)
//! plus the shared imatrix *file* layer of `common/imatrix-loader.cpp`:
//!
//! * `IMatrixCollector` — the activation collector driven from the eval
//!   callback (`common_imatrix_eval_callback` in older revisions; in this
//!   pinned one the collector itself lives entirely in imatrix.cpp:234-424 and
//!   is registered as `params.cb_eval`, imatrix.cpp:1154).
//! * `save_imatrix` / `save_imatrix_legacy` — the GGUF and `.dat` writers
//!   (imatrix.cpp:426-640).
//! * `load_imatrix` (imatrix.cpp:642-711) on top of `common_imatrix_load`
//!   (imatrix-loader.cpp:10-173), used by both this tool's `--in-file` and
//!   `llama-quantize --imatrix`.
//! * `--show-statistics` (imatrix.cpp:96-223, 968-1075).
//!
//! The C collector is a file-static (`static IMatrixCollector g_collector`,
//! imatrix.cpp:713) reached through the plain C function pointer
//! `ik_collect_imatrix` (imatrix.cpp:715-717). The port keeps the same global
//! (a `Mutex<Option<IMatrixCollector>>`) and registers
//! [`eval_callback`] in `ggml::compute` — see that module's docs.

use std::collections::HashMap;
use std::io::Write as _;

use ggml::compute::EvalNode;
use ggml::gguf::{Value, GGUF_DEFAULT_ALIGNMENT};
use ggml::gguf_write::GgufWriter;
use ggml::tensor::GgmlOp;
use ggml::types::GgmlType;

/// `LLM_KV_IMATRIX_*` (common/imatrix-loader.h:8-13).
pub const KV_IMATRIX_DATASETS: &str = "imatrix.datasets";
pub const KV_IMATRIX_CHUNK_COUNT: &str = "imatrix.chunk_count";
pub const KV_IMATRIX_CHUNK_SIZE: &str = "imatrix.chunk_size";
pub const KV_IMATRIX_STATS_SCHEMA: &str = "imatrix.stats_schema";
pub const KV_IMATRIX_N_LAYER_NEXTN: &str = "imatrix.n_layer_nextn";

// ---------------------------------------------------------------------------
// params (the subset of common_params the collector reads: common.h:716-725)
// ---------------------------------------------------------------------------

/// `common_params` imatrix fields + the context/batch geometry the collector
/// derives `chunk_size` from (imatrix.cpp:241).
#[derive(Debug, Clone)]
pub struct ImatrixParams {
    /// `params.n_ctx` *after* the tool's n_seq rewrite (imatrix.cpp:1110-1118)
    pub n_ctx: i32,
    /// `params.n_parallel` == `n_seq` (imatrix.cpp:1114)
    pub n_parallel: i32,
    /// `--output-frequency` (common.h:717)
    pub n_out_freq: i32,
    /// `--save-frequency` (common.h:718)
    pub n_save_freq: i32,
    /// `--process-output` (common.h:722)
    pub process_output: bool,
    /// `-o` (imatrix.cpp:1082)
    pub out_file: String,
    /// `params.prompt_file` — the dataset *path*, written into the GGUF
    /// (imatrix.cpp:516, 600)
    pub prompt_file: String,
    /// `--output-format dat` (common.h:720: `imat_dat > 0`)
    pub imat_dat: i8,
    /// `params.load_mtp` (common.h:589, upstream a7b94df2c) — `--nextn` /
    /// `--model-draft`: collect data for the MTP/NextN layers. The port's
    /// loader always loads nextn tensors (see model.rs docs), so this flag
    /// only gates the *collector* filter (`m_params.load_mtp && is_nextn`,
    /// imatrix.cpp:577)
    pub load_mtp: bool,
    /// the collector's `m_chunk_size` (imatrix.cpp:79-82) — the chunk size
    /// of the loaded imatrix files, overriding `n_ctx/n_parallel` once set
    /// (`e_chunk_size()`, imatrix.cpp:83-85). Folded into the params struct
    /// because the port's save/collect take `params` where the C reads its
    /// collector fields; 0 = not loaded
    pub chunk_size_loaded: i32,
    /// the collector's `m_n_layer_nextn` (imatrix.cpp:84) — written into the
    /// GGUF (`imatrix.n_layer_nextn`) and read by the statistics pass
    pub n_layer_nextn: i32,
}

impl Default for ImatrixParams {
    fn default() -> Self {
        ImatrixParams {
            n_ctx: 512,
            n_parallel: 1,
            n_out_freq: 10,
            n_save_freq: 0,
            process_output: false,
            out_file: "imatrix.gguf".to_string(),
            prompt_file: String::new(),
            imat_dat: 0,
            load_mtp: false,
            chunk_size_loaded: 0,
            n_layer_nextn: 0,
        }
    }
}

/// `e_chunk_size()` (imatrix.cpp:83-85).
pub fn e_chunk_size(params: &ImatrixParams) -> i32 {
    if params.chunk_size_loaded > 0 {
        params.chunk_size_loaded
    } else {
        params.n_ctx / params.n_parallel
    }
}

// ---------------------------------------------------------------------------
// collector state (imatrix.cpp:38-75)
// ---------------------------------------------------------------------------

/// `struct Stats` (imatrix.cpp:38-41).
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub activations: Vec<f32>,
    pub values: Vec<f32>,
    pub counts: Vec<i64>,
}

/// `struct tensor_statistics` (imatrix.cpp:44-66, upstream a7b94df2c) — the
/// rewritten `--show-statistics` payload.
#[derive(Debug, Clone)]
pub struct TensorStatistics {
    pub tensor: String,
    pub legacy: bool,
    pub sum: f64,
    pub mean: f32,
    pub elements: i64,
    pub std_deviation: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub gain: f32,
    pub entropy: f32,
    pub l2_dist: f32,
    pub cossim: f32,
    pub pearson: f32,
    pub covariance: f32,
    pub cov_sum: f64,
    pub var_c_sum: f64,
    pub var_p_sum: f64,
    pub dot_prod: f64,
    pub norm1_sq: f64,
    pub norm2_sq: f64,
    pub l2_dist_sq: f64,
    pub sum_prev: f64,
    pub elements_prev: i64,
    pub n_features: i64,
}

impl Default for TensorStatistics {
    fn default() -> Self {
        TensorStatistics {
            tensor: String::new(),
            legacy: true,
            sum: 0.0,
            mean: 0.0,
            elements: 0,
            std_deviation: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            gain: f32::NAN,
            entropy: 0.0,
            l2_dist: f32::NAN,
            cossim: f32::NAN,
            pearson: f32::NAN,
            covariance: f32::NAN,
            cov_sum: 0.0,
            var_c_sum: 0.0,
            var_p_sum: 0.0,
            dot_prod: 0.0,
            norm1_sq: 0.0,
            norm2_sq: 0.0,
            l2_dist_sq: 0.0,
            sum_prev: 0.0,
            elements_prev: 0,
            n_features: 0,
        }
    }
}

/// `process_tensor_name` (imatrix.cpp:96-145): split a tensor name into its
/// `blk.<layer>` number ("-" when absent) and its role. The role is every
/// dot-separated part before `weight`, minus the `blk.<n>` prefix — note the
/// upstream `j += name.size() > 4 ? 1 : 2` quirk: for a 4-part
/// `blk.N.role.weight` the `+2` plus the loop increment skips the role part
/// too, leaving the tensor empty and falling back to the full input.
pub fn process_tensor_name(input: &str) -> (String, String) {
    let name: Vec<&str> = input.split('.').collect();
    let mut layer = String::new();
    for i in 0..name.len() {
        if name[i] == "blk" && i + 1 < name.len() {
            layer = name[i + 1].to_string();
            break;
        }
    }
    let mut tensor = String::new();
    'outer: for i in 0..name.len() {
        if name[i] == "weight" && i > 0 {
            let mut j = 0usize;
            while j < name.len() {
                if name[j] == "blk" {
                    // `j += name.size() > 4 ? 1 : 2` then the loop's `++j`
                    j += if name.len() > 4 { 1 } else { 2 };
                    j += 1; // the for's ++j of the C loop
                    continue;
                }
                if j == i {
                    break 'outer;
                }
                if !tensor.is_empty() {
                    tensor += ".";
                }
                tensor += name[j];
                j += 1;
            }
            break;
        }
    }
    if tensor.is_empty() {
        tensor = input.to_string();
    }
    if layer.is_empty() {
        layer = "-".to_string();
    }
    (layer, tensor)
}

/// `compute_tensor_averages` (imatrix.cpp:147-176): row-wise averages of the
/// energy (`values`) or the activations; NaN rows where the count is zero.
pub fn compute_tensor_averages(tstats: &Stats, use_activations: bool) -> Vec<f32> {
    if tstats.counts.is_empty() {
        return Vec::new();
    }
    let n_mat = tstats.counts.len();
    let len = if use_activations {
        tstats.activations.len()
    } else {
        tstats.values.len()
    };
    if len == 0 || n_mat == 0 || len % n_mat != 0 {
        return Vec::new();
    }

    let row = len / n_mat;
    let mut vec = vec![f32::NAN; len];

    let mut has_valid = false;
    for m in 0..n_mat {
        let c = tstats.counts[m] as f32;
        if c <= 0.0 {
            continue;
        }
        has_valid = true;
        let scale = 1.0 / c;
        let off = m * row;
        let src = if use_activations {
            &tstats.activations[off..off + row]
        } else {
            &tstats.values[off..off + row]
        };
        for (j, v) in vec[off..off + row].iter_mut().enumerate() {
            *v = src[j] * scale;
        }
    }

    if !has_valid {
        return Vec::new();
    }
    vec
}

/// `compute_vector_statistics` (imatrix.cpp:178-279): mean / std deviation /
/// skewness / kurtosis / entropy of one tensor's activations. Returns false
/// where the C logs an error and returns early.
pub fn compute_vector_statistics(
    tstats: &mut Vec<TensorStatistics>,
    name: &str,
    e: &Stats,
) -> bool {
    let legacy = e.activations.is_empty();
    let n_mat = e.counts.len();
    let len = if legacy { e.values.len() } else { e.activations.len() };

    if n_mat == 0 || len == 0 || len % n_mat != 0 {
        eprintln!("compute_vector_statistics: data size mismatch or empty for tensor {name}");
        return false;
    }
    if !legacy && e.values.len() != len {
        eprintln!("compute_vector_statistics: activations/values size mismatch for {name}");
        return false;
    }

    let row_size = len / n_mat;
    let mut sum = 0.0f64;
    let mut mean = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_cu_diff = 0.0f64;
    let mut sum_qd_diff = 0.0f64;
    let mut sum_energy = 0.0f64;
    let mut valid_n: usize = 0;

    // Mean (Welford-style, exactly the C's `mean += delta / valid_n`)
    for i in 0..n_mat {
        let c = e.counts[i] as f32;
        if c <= 0.0 {
            continue;
        }
        let inv_c = 1.0 / c as f64;
        let off = i * row_size;
        for j in 0..row_size {
            let v_act = if legacy { 0.0 } else { e.activations[off + j] as f64 * inv_c };
            let v_val = e.values[off + j] as f64 * inv_c;
            let v = if legacy { v_val } else { v_act }; // activation average for non-legacy
            if !v.is_finite() || !v_val.is_finite() {
                continue;
            }
            sum += v_val;
            valid_n += 1;
            let delta = v - mean;
            mean += delta / valid_n as f64;

            if v_val > 0.0 {
                sum_energy += v_val;
            }
        }
    }

    if valid_n == 0 {
        return false;
    }

    // Std Dev, Skew, Kurtosis, Entropy
    let mut entropy = 0.0f64;
    let inv_sum_energy = if sum_energy > 0.0 { 1.0 / sum_energy } else { 0.0 };
    let log2_inv = 1.0 / 2.0f64.ln();

    for i in 0..n_mat {
        let c = e.counts[i] as f32;
        if c <= 0.0 {
            continue;
        }
        let inv_c = 1.0 / c as f64;
        let off = i * row_size;
        for j in 0..row_size {
            let v_act = if legacy { 0.0 } else { e.activations[off + j] as f64 * inv_c };
            let v_val = e.values[off + j] as f64 * inv_c;
            let v = if legacy { v_val } else { v_act };
            if !v.is_finite() || !v_val.is_finite() {
                continue;
            }
            let diff = v - mean;

            sum_sq_diff += diff * diff;
            sum_cu_diff += diff * diff * diff;
            sum_qd_diff += diff * diff * diff * diff;

            // Entropy (Distribution of Energy)
            if inv_sum_energy > 0.0 {
                let v_energy = e.values[off + j] as f64 * inv_c;
                let p = v_energy.max(0.0) * inv_sum_energy;
                if p > 1e-10 {
                    entropy -= p * p.ln() * log2_inv;
                }
            }
        }
    }

    let variance = if valid_n > 1 { sum_sq_diff / valid_n as f64 } else { 0.0 };
    let std_deviation = variance.max(0.0).sqrt() as f32;
    let mut skewness = 0.0f32;
    let mut kurtosis = 0.0f32;
    if std_deviation > 1e-10 {
        let m2 = sum_sq_diff / valid_n as f64;
        skewness = (sum_cu_diff / valid_n as f64 / (m2 * m2.sqrt())) as f32;
        kurtosis = (sum_qd_diff / valid_n as f64 / (m2 * m2) - 3.0) as f32;
    }

    let mut ts = TensorStatistics {
        tensor: name.to_string(),
        legacy,
        sum,
        mean: mean as f32,
        elements: valid_n as i64,
        std_deviation,
        skewness,
        kurtosis,
        entropy: entropy as f32,
        ..Default::default()
    };
    ts.gain = f32::NAN;
    ts.l2_dist = f32::NAN;
    ts.cossim = f32::NAN;
    ts.pearson = f32::NAN;
    ts.covariance = f32::NAN;
    tstats.push(ts);

    true
}

/// `nextn_layer_start` (imatrix.cpp:281-299): the layer index the NextN
/// blocks start at — `max_blk + 1 - n_layer_nextn` when the count is known,
/// else the first layer whose tensor is a `nextn.` one. `i32::MAX` when
/// none.
pub fn nextn_layer_start(tstats: &[TensorStatistics], n_layer_nextn: i32) -> i32 {
    let mut max_blk: i32 = -1;
    let mut first: i32 = i32::MAX;

    for ts in tstats {
        let (layer_str, name) = process_tensor_name(&ts.tensor);
        let Ok(blk) = layer_str.parse::<i32>() else {
            continue;
        };
        max_blk = max_blk.max(blk);
        if name.starts_with("nextn.") {
            first = first.min(blk);
        }
    }

    if n_layer_nextn > 0 && max_blk >= 0 {
        return (max_blk + 1 - n_layer_nextn).max(0);
    }
    first
}

/// `layer_label` (imatrix.cpp:301-307).
fn layer_label(blk: i32, nextn_start: i32) -> String {
    if blk < 0 || blk == i32::MAX {
        return "-".to_string();
    }
    if blk >= nextn_start {
        return format!("mtp{}", blk - nextn_start);
    }
    blk.to_string()
}

/// `compute_tensor_statistics` (imatrix.cpp:309-455): per-tensor comparison
/// against the same tensor of the preceding layer — cossim / l2 / pearson /
/// covariance / gain.
pub fn compute_tensor_statistics(
    tstats: &mut [TensorStatistics],
    mstats: &HashMap<String, Stats>,
    nextn_start: i32,
) {
    // owned keys: the map outlives the mutable borrows of `tstats` below
    let mut tensor_map: HashMap<String, usize> = HashMap::with_capacity(tstats.len());
    for (i, ts) in tstats.iter().enumerate() {
        tensor_map.insert(ts.tensor.clone(), i);
    }

    for i in 0..tstats.len() {
        let (layer_str, _) = process_tensor_name(&tstats[i].tensor);
        let Ok(blk) = layer_str.parse::<i32>() else {
            continue;
        };
        if blk <= 0 {
            continue;
        }
        if blk == nextn_start {
            continue;
        }
        let blk_first = if blk > nextn_start { nextn_start } else { 0 };
        let Some(blk_start_pos) = tstats[i].tensor.find(&format!("blk.{layer_str}")) else {
            continue;
        };

        // walk down to the first preceding layer that has this tensor
        let curr_name = tstats[i].tensor.clone();
        let curr_legacy = tstats[i].legacy;
        let mut it: Option<usize> = None;
        let mut prev = blk - 1;
        while prev >= blk_first {
            let tname = format!(
                "{}{}{}",
                &curr_name[..blk_start_pos],
                &format!("blk.{prev}"),
                &curr_name[blk_start_pos + layer_str.len() + 4..]
            );
            if let Some(&idx) = tensor_map.get(&tname) {
                it = Some(idx);
                break;
            }
            prev -= 1;
        }

        let Some(prev_idx) = it else {
            eprintln!(
                "compute_tensor_statistics: no preceding-layer tensor for '{}'",
                curr_name
            );
            continue;
        };

        let prev_name = tstats[prev_idx].tensor.clone();
        let prev_legacy = tstats[prev_idx].legacy;
        let prev_sum = tstats[prev_idx].sum;
        let prev_elements = tstats[prev_idx].elements;
        let Some(curr_e) = mstats.get(&curr_name) else {
            continue;
        };
        let Some(prev_e) = mstats.get(&prev_name) else {
            continue;
        };

        // one side may not have no activation sums so compare both on the energy
        let use_activations = !curr_legacy && !prev_legacy;
        let curr_avg = compute_tensor_averages(curr_e, use_activations);
        let prev_avg = compute_tensor_averages(prev_e, use_activations);

        if curr_avg.is_empty() || curr_avg.len() != prev_avg.len() {
            continue;
        }

        let n = curr_avg.len();
        let mut valid_n: usize = 0;
        let mut sum_c = 0.0f64;
        let mut sum_p = 0.0f64;

        // Sums for Means
        for k in 0..n {
            if curr_avg[k].is_finite() && prev_avg[k].is_finite() {
                sum_c += curr_avg[k] as f64;
                sum_p += prev_avg[k] as f64;
                valid_n += 1;
            }
        }
        if valid_n == 0 {
            continue;
        }
        let mean_c = sum_c / valid_n as f64;
        let mean_p = sum_p / valid_n as f64;

        let mut cov_sum = 0.0f64;
        let mut var_c_sum = 0.0f64;
        let mut var_p_sum = 0.0f64;
        let mut dot_prod = 0.0f64;
        let mut norm1_sq = 0.0f64;
        let mut norm2_sq = 0.0f64;
        let mut l2_dist_sq = 0.0f64;

        // Metrics
        for k in 0..n {
            let c_val = curr_avg[k] as f64;
            let p_val = prev_avg[k] as f64;
            if !c_val.is_finite() || !p_val.is_finite() {
                continue;
            }

            // Cosine Similarity & L2 Distance
            dot_prod += c_val * p_val;
            norm1_sq += c_val * c_val;
            norm2_sq += p_val * p_val;
            let diff = c_val - p_val;
            l2_dist_sq += diff * diff;

            // Pearson (Centered stats)
            let dc = c_val - mean_c;
            let dp = p_val - mean_p;
            cov_sum += dc * dp;
            var_c_sum += dc * dc;
            var_p_sum += dp * dp;
        }

        let ts = &mut tstats[i];
        ts.n_features = valid_n as i64;
        // (prev values were snapshotted above — `prev_ts` is still immutable
        // in the C, the port reads through the copies)
        ts.dot_prod = dot_prod;
        ts.norm1_sq = norm1_sq;
        ts.norm2_sq = norm2_sq;
        ts.cov_sum = cov_sum;
        ts.var_c_sum = var_c_sum;
        ts.var_p_sum = var_p_sum;
        ts.l2_dist_sq = l2_dist_sq;
        ts.l2_dist = l2_dist_sq.sqrt() as f32;

        if valid_n > 1 {
            ts.covariance = (cov_sum / valid_n as f64) as f32;
        }

        if norm1_sq > 0.0 && norm2_sq > 0.0 {
            ts.cossim = (dot_prod / (norm1_sq.sqrt() * norm2_sq.sqrt())) as f32;
            ts.cossim = ts.cossim.clamp(-1.0, 1.0);
        } else {
            ts.cossim = if norm1_sq == 0.0 && norm2_sq == 0.0 {
                f32::NAN
            } else {
                0.0
            };
        }

        if var_c_sum > 0.0 && var_p_sum > 0.0 {
            ts.pearson = (cov_sum / (var_c_sum.sqrt() * var_p_sum.sqrt())) as f32;
            ts.pearson = ts.pearson.clamp(-1.0, 1.0);
        } else {
            ts.pearson = if var_c_sum == 0.0 && var_p_sum == 0.0 {
                f32::NAN
            } else {
                0.0
            };
        }

        if prev_sum > 1e-10 {
            ts.gain = ((ts.sum / ts.elements as f64).sqrt()
                / (prev_sum / prev_elements as f64).sqrt()) as f32;
            ts.sum_prev = prev_sum;
            ts.elements_prev = prev_elements;
        } else {
            ts.gain = if ts.sum <= 1e-10 { 1.0 } else { f32::NAN };
        }
    }
}

/// the five per-layer aggregates `compute_layer_statistics` fills
/// (imatrix.cpp:457-578)
#[derive(Default)]
pub struct LayerStatistics {
    pub layer_cossim: std::collections::BTreeMap<i32, f32>,
    pub layer_l2_dist: std::collections::BTreeMap<i32, f32>,
    pub layer_pearson: std::collections::BTreeMap<i32, f32>,
    pub layer_covariance: std::collections::BTreeMap<i32, f32>,
    pub layer_gain: std::collections::BTreeMap<i32, f32>,
}

/// `compute_layer_statistics` (imatrix.cpp:457-578).
pub fn compute_layer_statistics(tstats: &[TensorStatistics]) -> LayerStatistics {
    #[derive(Default, Clone, Copy)]
    struct LayerAggregation {
        sum_dot_prod: f64,
        sum_norm1_sq: f64,
        sum_norm2_sq: f64,
        sum_l2_dist_sq: f64,
        sum_cov: f64,
        sum_var_c: f64,
        sum_var_p: f64,
        sum_energy_curr: f64,
        sum_energy_prev: f64,
        sum_elements_curr: i64,
        sum_elements_prev: i64,
        sum_n_features: i64,
        n_tensors: i32,
    }

    let mut laggr: std::collections::BTreeMap<i32, LayerAggregation> =
        std::collections::BTreeMap::new();

    for ts in tstats {
        let (layer_str, _) = process_tensor_name(&ts.tensor);
        let blk: i32 = match layer_str.parse() {
            Ok(b) => b,
            Err(_) => {
                if layer_str == "-" {
                    -1
                } else {
                    -1
                }
            }
        };
        if blk <= 0 {
            continue;
        }

        if ts.norm1_sq == 0.0 && ts.norm2_sq == 0.0 && ts.l2_dist_sq == 0.0 {
            continue;
        }
        let entry = laggr.entry(blk).or_default();
        entry.sum_dot_prod += ts.dot_prod;
        entry.sum_norm1_sq += ts.norm1_sq;
        entry.sum_norm2_sq += ts.norm2_sq;
        entry.sum_l2_dist_sq += ts.l2_dist_sq;
        entry.sum_cov += ts.cov_sum;
        entry.sum_var_c += ts.var_c_sum;
        entry.sum_var_p += ts.var_p_sum;
        entry.n_tensors += 1;
        if ts.n_features > 0 {
            entry.sum_n_features += ts.n_features;
        }

        // skip tensors with no match in a previous layer
        if ts.elements_prev > 0 {
            entry.sum_energy_curr += ts.sum;
            entry.sum_energy_prev += ts.sum_prev;
            entry.sum_elements_curr += ts.elements;
            entry.sum_elements_prev += ts.elements_prev;
        }
    }

    let mut out = LayerStatistics::default();
    for (&layer, agg) in laggr.iter() {
        if agg.n_tensors == 0 {
            continue;
        }

        let mut cossim = 0.0f32;
        if agg.sum_norm1_sq > 0.0 && agg.sum_norm2_sq > 0.0 {
            cossim =
                (agg.sum_dot_prod / (agg.sum_norm1_sq.sqrt() * agg.sum_norm2_sq.sqrt())) as f32;
            cossim = cossim.clamp(-1.0, 1.0);
        } else if agg.sum_norm1_sq == 0.0 && agg.sum_norm2_sq == 0.0 {
            cossim = f32::NAN;
        }

        let mut gain = f32::NAN;
        if agg.sum_elements_curr > 0 && agg.sum_elements_prev > 0 && agg.sum_energy_prev > 0.0 {
            let rms_curr = (agg.sum_energy_curr / agg.sum_elements_curr as f64).sqrt();
            let rms_prev = (agg.sum_energy_prev / agg.sum_elements_prev as f64).sqrt();
            gain = (rms_curr / rms_prev) as f32;
        }

        out.layer_cossim.insert(layer, cossim);
        out.layer_l2_dist.insert(layer, agg.sum_l2_dist_sq.sqrt() as f32);
        out.layer_gain.insert(layer, gain);

        if agg.sum_n_features > 0 {
            out.layer_covariance
                .insert(layer, (agg.sum_cov / agg.sum_n_features as f64) as f32);
        } else {
            out.layer_covariance.insert(layer, f32::NAN);
        }

        if agg.sum_var_c > 0.0 && agg.sum_var_p > 0.0 {
            let pearson =
                (agg.sum_cov / (agg.sum_var_c.sqrt() * agg.sum_var_p.sqrt())) as f32;
            out.layer_pearson.insert(layer, pearson.clamp(-1.0, 1.0));
        } else if agg.sum_var_c == 0.0 && agg.sum_var_p == 0.0 {
            out.layer_pearson.insert(layer, f32::NAN);
        } else {
            out.layer_pearson.insert(layer, 0.0);
        }
    }
    out
}

/// `class IMatrixCollector` (imatrix.cpp:58-75).
pub struct IMatrixCollector {
    /// `std::unordered_map<std::string, Stats> m_stats`
    pub m_stats: HashMap<String, Stats>,
    pub m_params: ImatrixParams,
    /// `m_datasets` — datasets merged in by `load_imatrix` (imatrix.cpp:697)
    pub m_datasets: Vec<String>,
    /// `m_last_chunk` (imatrix.cpp:72)
    pub m_last_chunk: i32,
    /// `m_src1_data` — the CPU copy of an off-host src1 (imatrix.cpp:73)
    m_src1_data: Vec<u8>,
    /// `m_ids` — `ggml_mul_mat_id` expert ids (imatrix.cpp:74)
    m_ids: Vec<u8>,
}

impl Default for IMatrixCollector {
    fn default() -> Self {
        IMatrixCollector {
            m_stats: HashMap::new(),
            m_params: ImatrixParams::default(),
            m_datasets: Vec::new(),
            m_last_chunk: 0,
            m_src1_data: Vec::new(),
            m_ids: Vec::new(),
        }
    }
}

impl IMatrixCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// `void set_params(common_params params)` (imatrix.cpp:61).
    pub fn set_params(&mut self, params: ImatrixParams) {
        self.m_params = params;
    }

    pub fn get_mstats(&self) -> &HashMap<String, Stats> {
        &self.m_stats
    }
}

/// `filter_tensor_name` (imatrix.cpp:79-94): strip any `#` prefix/suffix —
/// `CUDA0#blk.0.attn_k.weight#0` -> `blk.0.attn_k.weight`.
pub fn filter_tensor_name(name: &str) -> String {
    match name.find('#') {
        Some(p) => {
            let rest = &name[p + 1..];
            match rest.find('#') {
                Some(q) => rest[..q].to_string(),
                None => rest.to_string(),
            }
        }
        None => name.to_string(),
    }
}

/// `all_finite` (imatrix.cpp:225-232).
fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// `rows_to_chunks` (imatrix.cpp:580-582, upstream a7b94df2c): round to
/// nearest instead of truncating.
fn rows_to_chunks(n_rows: i64, chunk_size: i32) -> i32 {
    ((n_rows + chunk_size as i64 / 2) / chunk_size as i64) as i32
}

// ---------------------------------------------------------------------------
// collect_imatrix (imatrix.cpp:234-424)
// ---------------------------------------------------------------------------

/// `bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask,
///                                         void * user_data)` (imatrix.cpp:234-424).
///
/// `t` is the graph node as the C scheduler would pass it: `src[0]` the weight
/// (its `name` is what lands in the imatrix), `src[1]` the activations, and for
/// `GGML_OP_MUL_MAT_ID` additionally `src[2]` = the expert ids.
///
/// Side-effect-free when `ask` is true, as in C.
pub fn collect_imatrix(
    stats: &mut HashMap<String, Stats>,
    params: &ImatrixParams,
    m_last_chunk: &mut i32,
    _m_src1_data: &mut Vec<u8>,
    m_ids: &mut Vec<u8>,
    t: &EvalNode<'_>,
    ask: bool,
    save: &mut dyn FnMut(&HashMap<String, Stats>, i32, i32),
) -> bool {
    let (Some(src0), Some(src1)) = (t.src[0].as_ref(), t.src[1].as_ref()) else {
        return false;
    };
    let wname = filter_tensor_name(src0.name);

    let chunk_size = e_chunk_size(params);

    // when ask is true, the scheduler wants to know if we are interested in
    // data from this tensor; a follow-up call with ask == false then collects
    if ask {
        if t.op == GgmlOp::MulMatId {
            return true; // collect all indirect matrix multiplications
        }
        if t.op != GgmlOp::MulMat {
            return false;
        }
        // why are small batches ignored (<16 tokens)?
        if src1.ne[1] < 16 || src1.ty != GgmlType::F32 {
            return false;
        }
        // (imatrix.cpp:573-577, upstream a7b94df2c)
        let is_output = wname == "output.weight"
            || wname == "token_embd.weight"
            || wname == "nextn.post_projection.weight";
        let is_nextn = wname == "nextn.pre_projection.weight";
        if !(wname.starts_with("blk.")
            || (params.process_output && is_output)
            || (params.load_mtp && is_nextn))
        {
            return false;
        }
        return true;
    }

    // copy the data from the GPU memory if needed — this port's storages are
    // all host-visible, so `is_host` is always true and no copy happens
    // (imatrix.cpp:256-266)
    let Some(data) = src1.data else { return false };
    let data: &[u8] = data;
    // `GGML_ASSERT(src1->nb[0] == ggml_element_size(src1))` (imatrix.cpp:266)
    debug_assert_eq!(src1.nb[0], src1.ty.type_size() as u64);

    let rd_f32 = |off: usize| -> f32 { f32::from_le_bytes(data[off..off + 4].try_into().unwrap()) };

    // this has been adapted to the new format of storing merged experts in a
    // single 3d tensor (imatrix.cpp:270-355)
    // the C's `m_src1_data`/`m_ids` copies exist because a backend tensor may
    // live off-host; this port's storages are all host-visible (see `data`
    // above), so only the small ids buffer is still copied — mirroring
    // imatrix.cpp:289-290.
    if t.op == GgmlOp::MulMatId {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        let Some(ids) = t.src[2].as_ref() else {
            return false;
        };
        let n_as = src0.ne[2];
        let n_ids = ids.ne[0];

        // GGML_ASSERT(ids->ne[1] == src1->ne[2]);
        debug_assert_eq!(ids.ne[1], src1.ne[2]);

        // the extra dimension would need to be stored somewhere to be
        // reflected in the imatrix file
        if src1.ne[3] != 1 {
            llp_err(&format!(
                "collect_imatrix: tensor has more than 3 dimensions: {wname}"
            ));
            std::process::exit(1);
        }

        m_ids.clear();
        if let Some(id_data) = ids.data {
            m_ids.extend_from_slice(id_data);
        }
        let ids_data: Vec<u8> = m_ids.clone();

        let e = stats.entry(wname.clone()).or_default();

        if e.counts.len() == 1 && n_as > 1 {
            // broadcast, when loading an old imatrix
            let c0 = e.counts[0];
            e.counts.resize(n_as as usize, c0);
        }
        if e.values.is_empty() {
            e.activations.resize((src1.ne[0] * n_as) as usize, 0.0);
            e.values.resize((src1.ne[0] * n_as) as usize, 0.0);
            e.counts.resize(n_as as usize, 0);
        } else if e.values.len() != (src1.ne[0] * n_as) as usize {
            llp_err(&format!(
                "collect_imatrix: inconsistent size for {} ({} vs {})\n",
                wname,
                e.values.len(),
                src1.ne[0] * n_as
            ));
            std::process::exit(1);
        } else if e.counts.len() != n_as as usize {
            llp_err(&format!(
                "collect_imatrix: inconsistent expert count for {} ({} vs {})\n",
                wname,
                e.counts.len(),
                n_as
            ));
            std::process::exit(1);
        }

        let ne0 = src1.ne[0];
        let n_tokens = src1.ne[2];
        // legacy imatrix tensors do not have activation sums
        let has_act = !e.activations.is_empty();

        // single pass over the routing ids
        let mut touched = vec![0u8; n_as as usize];
        for idx in 0..n_ids {
            for row in 0..n_tokens {
                let ids_off = (row as u64 * ids.nb[1] + idx as u64 * ids.nb[0]) as usize;
                let ex = i32::from_le_bytes(ids_data[ids_off..ids_off + 4].try_into().unwrap());

                debug_assert!(ex >= 0 && (ex as i64) < n_as); // sanity check

                let i11 = idx % src1.ne[1];
                let x_off = (i11 as u64 * src1.nb[1] + row as u64 * src1.nb[2]) as usize;
                let acc_off = ex as usize * (ne0 as usize);
                let act_off = acc_off;

                e.counts[ex as usize] += 1;
                touched[ex as usize] = 1;
                for j in 0..ne0 as usize {
                    let v = rd_f32(x_off + j * 4);
                    e.values[acc_off + j] += v * v;
                    if has_act {
                        e.activations[act_off + j] += v;
                    }
                }
            }
        }

        // check for non-finite values, only expert slices that were touched
        for ex in 0..n_as as usize {
            if touched[ex] != 0
                && !all_finite(&e.values[ex * ne0 as usize..(ex + 1) * ne0 as usize])
            {
                llp_err(&format!(
                    "collect_imatrix: non-finite values detected in {wname}\n"
                ));
                std::process::exit(1);
            }
        }

        let counts = e.counts.clone();
        let mut saves: Vec<(i32, i32)> = Vec::new();
        for &count in counts.iter() {
            update_chunks(m_last_chunk, count, chunk_size, params, &mut saves);
        }
        trace(format!(
            "collect_imatrix[{:5}]: {:32}, {:?}, {:5} x {:5}, {:?}",
            *m_last_chunk, wname, t.op, src1.ne[0], src1.ne[2], src1.ty
        ));
        for (n_chunk, last) in saves {
            save(stats, n_chunk, last);
        }
    } else {
        let n_mat = src0.ne[2] * src0.ne[3];
        let e = stats.entry(wname.clone()).or_default();

        // use a single count per dense tensor (necessary when merging older
        // GGUF-imatrix files with 3d tensors)
        if e.counts.len() > 1 {
            let all_equal = e.counts[1..].iter().all(|&c| c == e.counts[0]);
            if all_equal {
                e.counts.resize(1, 0);
            }
        }
        if e.values.is_empty() {
            e.activations.resize((src1.ne[0] * n_mat) as usize, 0.0);
            e.values.resize((src1.ne[0] * n_mat) as usize, 0.0);
            e.counts.resize(1, 0);
        } else if e.values.len() != (src1.ne[0] * n_mat) as usize {
            llp_err(&format!(
                "collect_imatrix: inconsistent size for {} ({} vs {})\n",
                wname,
                e.values.len(),
                src1.ne[0] * n_mat
            ));
            std::process::exit(1);
        }

        let ne0 = src1.ne[0];
        let has_act = !e.activations.is_empty();

        for i3 in 0..src1.ne[3] {
            for i2 in 0..src1.ne[2] {
                // handle 3D+ tensors, but flatten 3D+ activations when model
                // tensor is 2D
                let mat_id = (i3 % src0.ne[3]) * src0.ne[2] + (i2 % src0.ne[2]);
                let acc_off = mat_id * ne0;
                let act_off = acc_off;

                for row in 0..src1.ne[1] {
                    let x_off = (row as u64 * src1.nb[1]
                        + i2 as u64 * src1.nb[2]
                        + i3 as u64 * src1.nb[3]) as usize;
                    for j in 0..ne0 as usize {
                        let v = rd_f32(x_off + j * 4);
                        e.values[(acc_off as usize) + j] += v * v;
                        if has_act {
                            e.activations[(act_off as usize) + j] += v;
                        }
                    }
                }
            }
        }

        // check for non-finite values
        if !all_finite(&e.values) {
            llp_err(&format!(
                "collect_imatrix: non-finite values detected in {wname}\n"
            ));
            std::process::exit(1);
        }
        // only 1 count in practice, except when a tensor is used for both
        // MUL_MAT_ID and MUL_MAT
        let nrows = src1.ne[1] * src1.ne[2] * src1.ne[3];
        let mut saves: Vec<(i32, i32)> = Vec::new();
        for count in e.counts.iter_mut() {
            *count += nrows / n_mat;
            update_chunks(m_last_chunk, *count, chunk_size, params, &mut saves);
        }
        trace(format!(
            "collect_imatrix[{:5}]: {:32}, {:?}, {:5} x {:5} x {:5}, {:?}",
            *m_last_chunk, wname, t.op, src1.ne[0], src1.ne[1], src1.ne[2], src1.ty
        ));
        for (n_chunk, last) in saves {
            save(stats, n_chunk, last);
        }
    }

    true
}

/// The `n_chunk`/`n_out_freq`/`n_save_freq` block the C repeats in both
/// branches (imatrix.cpp:343-355 and 407-420). `(n_chunk_arg, last_chunk)` of
/// each pending `save_imatrix` call, in order.
fn update_chunks(
    m_last_chunk: &mut i32,
    count: i64,
    chunk_size: i32,
    params: &ImatrixParams,
    saves: &mut Vec<(i32, i32)>,
) {
    // `rows_to_chunks` (imatrix.cpp:685) — round to nearest
    let n_chunk = rows_to_chunks(count, chunk_size);
    if n_chunk > *m_last_chunk {
        let chunk_step = n_chunk - *m_last_chunk;
        *m_last_chunk = n_chunk;
        if (*m_last_chunk % params.n_out_freq) / chunk_step == 0 {
            saves.push((-1, *m_last_chunk));
        }
        if params.n_save_freq > 0 && (*m_last_chunk % params.n_save_freq) / chunk_step == 0 {
            saves.push((*m_last_chunk, *m_last_chunk));
        }
    }
}

fn llp_err(msg: &str) {
    eprint!("{msg}");
}

fn trace_enabled() -> bool {
    std::env::var("LLAMA_IMATRIX_TRACE").is_ok()
}

fn trace(msg: String) {
    if trace_enabled() {
        eprintln!("{msg}");
    }
}

/// `LOG_DBGV(2, ...)` (imatrix.cpp:310, 382) equivalent: with
/// `LLAMA_IMATRIX_TRACE=1` every callback invocation prints the node it was
/// asked about — the diagnostic that answers "why is tensor X missing from the
/// matrix?".
fn trace_node(t: &EvalNode<'_>, ask: bool) {
    if !trace_enabled() {
        return;
    }
    let src0 = t.src[0].as_ref().map(|s| s.name).unwrap_or("-");
    let src1 = t.src[1]
        .as_ref()
        .map(|s| {
            format!(
                "{}x{}x{}x{} {:?} data={}",
                s.ne[0],
                s.ne[1],
                s.ne[2],
                s.ne[3],
                s.ty,
                s.data.is_some()
            )
        })
        .unwrap_or_else(|| "-".to_string());
    eprintln!(
        "collect_imatrix[ask={ask}]: op={:?} src0={:?} src1={src1}",
        t.op, src0
    );
}

// ---------------------------------------------------------------------------
// save_imatrix / save_imatrix_legacy (imatrix.cpp:426-640)
// ---------------------------------------------------------------------------

/// The `n_zeros` scan + warnings shared by both writers (imatrix.cpp:437-475 /
/// 554-580). Returns the sorted list of entries to store.
fn prepare_store(stats: &HashMap<String, Stats>, legacy: bool) -> Vec<String> {
    let mut to_store: Vec<String> = Vec::new();
    let mut is_first = true;
    for (name, s) in stats.iter() {
        let n_all = s.counts.len();
        if legacy && n_all == 0 {
            continue;
        }
        let n_zeros = s.counts.iter().filter(|&&c| c == 0).count();

        if n_zeros != 0 && is_first {
            println!();
            is_first = false;
        }

        if legacy {
            if n_zeros == n_all {
                eprintln!(
                    "save_imatrix_legacy: entry '{:>40}' has no data - skipping",
                    name
                );
                continue;
            }
            if n_zeros > 0 {
                eprintln!(
                    "save_imatrix_legacy: entry '{:>40}' has partial data ({:.2}%)",
                    name,
                    100.0 * (n_all - n_zeros) as f32 / n_all as f32
                );
            }
        } else if n_zeros > 0 {
            eprintln!(
                "save_imatrix: entry '{:>40}' has partial data ({:.2}%)",
                name,
                100.0 * (n_all - n_zeros) as f32 / n_all as f32
            );
        }

        to_store.push(name.clone());
    }

    if legacy && to_store.len() < stats.len() {
        eprintln!(
            "save_imatrix_legacy: storing only {} out of {} entries",
            to_store.len(),
            stats.len()
        );
    }

    // deterministic tensor name order
    to_store.sort();
    to_store
}

/// `void IMatrixCollector::save_imatrix(int32_t n_chunk) const`
/// (imatrix.cpp:532-640) — GGUF or `.dat` depending on `params.imat_dat`.
///
/// `last_chunk` is `m_last_chunk` (the C reads it from the collector state).
pub fn save_imatrix(
    stats: &HashMap<String, Stats>,
    params: &ImatrixParams,
    datasets: &[String],
    last_chunk: i32,
    n_chunk: i32,
) -> Result<(), String> {
    if params.imat_dat > 0 {
        return save_imatrix_legacy(stats, params, datasets, last_chunk, n_chunk);
    }

    let mut fname = params.out_file.clone();
    // only warn when `--output-format gguf` is not specified
    if params.imat_dat == 0 && !fname.ends_with(".gguf") {
        eprintln!();
        eprintln!(
            "save_imatrix: saving imatrix using GGUF format with a different suffix than .gguf"
        );
        eprintln!("save_imatrix: if you want the previous imatrix format, use --output-format dat");
    }

    if n_chunk > 0 {
        fname += ".at_";
        fname += &n_chunk.to_string();
    }

    let to_store = prepare_store(stats, false);

    let mut w = GgufWriter::new(GGUF_DEFAULT_ALIGNMENT);

    {
        let mut ds: Vec<Value> = datasets.iter().map(|d| Value::String(d.clone())).collect();
        if !params.prompt_file.is_empty() {
            ds.push(Value::String(params.prompt_file.clone()));
        }

        w.set_kv("general.type", Value::String("imatrix".to_string()));
        // Write the dataset paths
        w.set_kv(
            KV_IMATRIX_DATASETS,
            Value::Array(ggml::gguf::GgufType::String, ds),
        );
        // Write the number of chunks the matrix was computed with
        w.set_kv(KV_IMATRIX_CHUNK_COUNT, Value::U32(last_chunk as u32));
        w.set_kv(KV_IMATRIX_CHUNK_SIZE, Value::U32(e_chunk_size(params) as u32));
        // Write how many of the top layers are NextN layers, so statistics can
        // tell them apart (imatrix.cpp:965-966, upstream a7b94df2c)
        if params.n_layer_nextn > 0 {
            w.set_kv(
                KV_IMATRIX_N_LAYER_NEXTN,
                Value::U32(params.n_layer_nextn as u32),
            );
        }
        // Define the schema for the tensor statistics (for use in quantize.cpp)
        // (imatrix.cpp:967-973)
        const STATS_SCHEMA: [&str; 12] = [
            "sum_sq", "mean", "elements", "std_deviation", "skewness", "kurtosis", "gain",
            "h_norm", "l2_dist", "cossim", "pearson", "covariance",
        ];
        w.set_kv(
            KV_IMATRIX_STATS_SCHEMA,
            Value::Array(
                ggml::gguf::GgufType::String,
                STATS_SCHEMA
                    .iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
        );
    }

    // Compute per-tensor statistics (imatrix.cpp:926-941, upstream a7b94df2c)
    let mut tstats: Vec<TensorStatistics> = Vec::with_capacity(stats.len());
    for (name, st) in stats.iter() {
        compute_vector_statistics(&mut tstats, name, st);
    }
    if !tstats.is_empty() {
        let nextn_start = nextn_layer_start(&tstats, params.n_layer_nextn);
        compute_tensor_statistics(&mut tstats, stats, nextn_start);
    }

    // index by tensor name
    let tstat_index: HashMap<&str, &TensorStatistics> = tstats
        .iter()
        .map(|ts| (ts.tensor.as_str(), ts))
        .collect();

    // payloads, in the same (sorted) order the tensors are added
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for name in &to_store {
        let stat = &stats[name];
        let nval = stat.values.len() as i32;
        let nmat = stat.counts.len() as i32;
        if nval > 0 && nmat > 0 {
            w.add_tensor(
                &format!("{name}.in_sum2"),
                GgmlType::F32,
                [(nval / nmat) as i64, nmat as i64, 1, 1],
            );
            w.add_tensor(
                &format!("{name}.counts"),
                GgmlType::F32,
                [1, nmat as i64, 1, 1],
            );
            payloads.push(stat.values.iter().flat_map(|v| v.to_le_bytes()).collect());
            payloads.push(
                stat.counts
                    .iter()
                    .flat_map(|c| (*c as f32).to_le_bytes())
                    .collect(),
            );

            // the activation sums (first moment), only when collected
            // (imatrix.cpp:984-991)
            if !stat.activations.is_empty() {
                let nact = stat.activations.len() as i32;
                w.add_tensor(
                    &format!("{name}.in_sum"),
                    GgmlType::F32,
                    [(nact / nmat) as i64, nmat as i64, 1, 1],
                );
                payloads.push(
                    stat.activations
                        .iter()
                        .flat_map(|v| (*v as f32).to_le_bytes())
                        .collect(),
                );
            }
        } else {
            eprintln!("save_imatrix: no data for tensor {name}");
        }

        // Store per-tensor statistics as a small 1D tensor
        // (imatrix.cpp:994-1036) — the same order as STATS_SCHEMA
        let ts = tstat_index.get(name.as_str()).copied();
        let fnan = f32::NAN;
        let mut fields = [fnan; 12];
        if let Some(ts) = ts {
            fields[0] = ts.sum as f32; // sum_sq
            fields[1] = ts.mean; // mean
            fields[2] = ts.elements as f32; // elements
            fields[3] = ts.std_deviation;
            fields[4] = ts.skewness;
            fields[5] = ts.kurtosis;
            fields[6] = ts.gain;
            fields[7] = if ts.elements > 1 {
                100.0 * (ts.entropy / (ts.elements as f32).log2())
            } else {
                fnan
            }; // h_norm
            fields[8] = ts.l2_dist;
            fields[9] = ts.cossim;
            fields[10] = ts.pearson;
            fields[11] = ts.covariance;
        }
        w.add_tensor(&format!("{name}.stats"), GgmlType::F32, [12, 1, 1, 1]);
        payloads.push(fields.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    let f = std::fs::File::create(&fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    let mut out = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    w.write(&mut out, &refs)
        .map_err(|e| format!("write error: {e}"))?;
    out.flush().map_err(|e| format!("write error: {e}"))?;

    println!();
    print_stored_line("save_imatrix", last_chunk, &fname);

    Ok(())
}

/// `void IMatrixCollector::save_imatrix_legacy(int32_t ncall) const`
/// (imatrix.cpp:426-530).
pub fn save_imatrix_legacy(
    stats: &HashMap<String, Stats>,
    params: &ImatrixParams,
    datasets: &[String],
    last_chunk: i32,
    ncall: i32,
) -> Result<(), String> {
    let mut fname = params.out_file.clone();
    if ncall > 0 {
        fname += ".at_";
        fname += &ncall.to_string();
    }

    let to_store = prepare_store(stats, true);
    let chunk_size = e_chunk_size(params);

    let f = std::fs::File::create(&fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    let mut out = std::io::BufWriter::new(f);
    let mut wr = |b: &[u8]| out.write_all(b);

    let n_entries = to_store.len() as i32;
    wr(&n_entries.to_le_bytes()).map_err(|e| e.to_string())?;
    for name in &to_store {
        let stat = &stats[name];
        let len = name.len() as i32;
        wr(&len.to_le_bytes()).map_err(|e| e.to_string())?;
        wr(name.as_bytes()).map_err(|e| e.to_string())?;
        // ceiling division to avoid accidental zeros
        let max_count = stat.counts.iter().cloned().max().unwrap_or(0);
        let ncall = ((max_count + (chunk_size as i64 - 1)) / chunk_size as i64) as i32;
        wr(&ncall.to_le_bytes()).map_err(|e| e.to_string())?;
        let nval = stat.values.len() as i32;
        let nmat = stat.counts.len() as i32;
        wr(&nval.to_le_bytes()).map_err(|e| e.to_string())?;
        if nval > 0 && nmat > 0 {
            let mut tmp = Vec::with_capacity(nval as usize);
            for i in 0..nval as usize {
                let mut count = stat.counts[i / (nval as usize / nmat as usize)] as f32;
                let mut value = stat.values[i];
                if count == 0.0 {
                    // store 1 for partial data
                    value = 1.0;
                    count = 1.0;
                }
                tmp.push((value / count) * ncall as f32);
            }
            wr(bytemuck::cast_slice(&tmp)).map_err(|e| e.to_string())?;
        }
    }

    // Write the number of calls the matrix was computed with
    wr(&last_chunk.to_le_bytes()).map_err(|e| e.to_string())?;

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        // When there is no prompt but there were other imatrix files loaded,
        // use the last dataset
        let (dataset_file, len) = if params.prompt_file.is_empty() {
            match datasets.last() {
                Some(d) => (d.clone(), d.len()),
                None => (String::new(), 0),
            }
        } else {
            (params.prompt_file.clone(), params.prompt_file.len())
        };
        wr(&(len as i32).to_le_bytes()).map_err(|e| e.to_string())?;
        wr(dataset_file.as_bytes()).map_err(|e| e.to_string())?;
    }

    out.flush().map_err(|e| e.to_string())?;
    println!();
    print_stored_line("save_imatrix_legacy", last_chunk, &fname);
    Ok(())
}

// ---------------------------------------------------------------------------
// imatrix file loading (common/imatrix-loader.cpp:10-173, imatrix.cpp:642-711)
// ---------------------------------------------------------------------------

/// `struct common_imatrix_entry` (common/imatrix-loader.h:12-17, upstream
/// a7b94df2c: + `activations`).
#[derive(Debug, Clone, Default)]
pub struct CommonImatrixEntry {
    pub sums: Vec<f32>,
    pub activations: Vec<f32>,
    pub counts: Vec<i64>,
}

/// `struct common_imatrix` (common/imatrix-loader.h:18-29).
#[derive(Debug, Clone, Default)]
pub struct CommonImatrix {
    /// `std::map<std::string, entry>` — sorted by name
    pub entries: std::collections::BTreeMap<String, CommonImatrixEntry>,
    pub datasets: Vec<String>,
    pub chunk_count: i32,
    pub chunk_size: i32,
    pub n_layer_nextn: i32,
    pub has_metadata: bool,
    pub is_legacy: bool,
}

/// `string_remove_suffix` (common/common.cpp): strip `suffix`, return true if it
/// was present.
fn string_remove_suffix(s: &mut String, suffix: &str) -> bool {
    if s.len() >= suffix.len() && s.ends_with(suffix) {
        let n = s.len() - suffix.len();
        s.truncate(n);
        true
    } else {
        false
    }
}

/// `common_imatrix_load_legacy` (common/imatrix-loader.cpp:10-80).
fn common_imatrix_load_legacy(fname: &str) -> Option<CommonImatrix> {
    let data = std::fs::read(fname).ok()?;
    if !fname.is_empty() && data.is_empty() {
        eprintln!("common_imatrix_load_legacy: failed to open {fname}");
        return None;
    }
    let mut imatrix = CommonImatrix::default();
    let mut off = 0usize;
    let rd_i32 = |d: &[u8], o: usize| i32::from_le_bytes(d[o..o + 4].try_into().unwrap());

    if data.len() < 4 {
        return None;
    }
    let n_entries = rd_i32(&data, 0);
    off += 4;
    if n_entries < 1 {
        eprintln!("common_imatrix_load_legacy: no data in file {fname}");
        return None;
    }

    for i in 0..n_entries {
        if off + 4 > data.len() {
            return None;
        }
        let len = rd_i32(&data, off) as usize;
        off += 4;
        if off + len > data.len() {
            eprintln!(
                "common_imatrix_load_legacy: failed reading name for entry {} from {fname}",
                i + 1
            );
            return None;
        }
        let name = String::from_utf8_lossy(&data[off..off + len]).to_string();
        off += len;

        if off + 8 > data.len() {
            return None;
        }
        let ncall = rd_i32(&data, off);
        let nval = rd_i32(&data, off + 4);
        off += 8;
        if nval < 1 {
            eprintln!("common_imatrix_load_legacy: failed reading number of values for entry {i}");
            return None;
        }

        let mut e = CommonImatrixEntry::default();
        e.sums.resize(nval as usize, 0.0);
        if off + nval as usize * 4 > data.len() {
            eprintln!("common_imatrix_load_legacy: failed reading data for entry {i}");
            return None;
        }
        for (j, s) in e.sums.iter_mut().enumerate() {
            *s = f32::from_le_bytes(data[off + j * 4..off + j * 4 + 4].try_into().unwrap());
        }
        off += nval as usize * 4;

        e.counts.resize(1, 0);
        e.counts[0] = ncall as i64;
        imatrix.entries.insert(name, e);
    }

    // the trailing data (chunk count + dataset name) is optional
    if off < data.len() {
        imatrix.chunk_count = rd_i32(&data, off);
        off += 4;
        if off + 4 <= data.len() {
            let len = rd_i32(&data, off) as usize;
            off += 4;
            if len > 0 && off + len <= data.len() {
                imatrix
                    .datasets
                    .push(String::from_utf8_lossy(&data[off..off + len]).to_string());
            }
        }
    }

    imatrix.chunk_size = 0;
    imatrix.is_legacy = true;
    Some(imatrix)
}

/// `bool common_imatrix_load(const std::string & fname, common_imatrix &)`
/// (common/imatrix-loader.cpp:82-173): GGUF first, legacy `.dat` as fallback.
pub fn common_imatrix_load(fname: &str) -> Option<CommonImatrix> {
    let Ok(gguf) = ggml::gguf::Gguf::open(fname) else {
        return common_imatrix_load_legacy(fname);
    };

    let n_entries = gguf.tensors.len() as i64;
    if n_entries < 1 {
        eprintln!("common_imatrix_load: no data in file {fname}");
        return None;
    }

    let mut imatrix = CommonImatrix::default();

    if let Some(Value::Array(ggml::gguf::GgufType::String, items)) =
        gguf.find_key(KV_IMATRIX_DATASETS)
    {
        for v in items {
            if let Value::String(s) = v {
                imatrix.datasets.push(s.clone());
            }
        }
    }

    let datasets_key = gguf.find_key(KV_IMATRIX_DATASETS).is_some();
    let chunk_count_key = gguf.find_key(KV_IMATRIX_CHUNK_COUNT).is_some();
    let chunk_size_key = gguf.find_key(KV_IMATRIX_CHUNK_SIZE).is_some();
    let nextn_key = gguf.find_key(KV_IMATRIX_N_LAYER_NEXTN).is_some();
    imatrix.has_metadata = datasets_key && chunk_count_key && chunk_size_key;
    imatrix.chunk_count = gguf.get_u32(KV_IMATRIX_CHUNK_COUNT).unwrap_or(0) as i32;
    imatrix.chunk_size = gguf.get_u32(KV_IMATRIX_CHUNK_SIZE).unwrap_or(0) as i32;
    imatrix.n_layer_nextn = if nextn_key {
        gguf.get_u32(KV_IMATRIX_N_LAYER_NEXTN).unwrap_or(0) as i32
    } else {
        0
    };

    const IN_SUM_SUFFIX: &str = ".in_sum";
    const IN_SUM2_SUFFIX: &str = ".in_sum2";
    const COUNTS_SUFFIX: &str = ".counts";

    /// `struct sum_tensors` (imatrix-loader.cpp:126-131, upstream a7b94df2c)
    #[derive(Default, Clone, Copy)]
    struct SumTensors<'a> {
        in_sum: Option<&'a ggml::TensorInfo>,
        in_sum2: Option<&'a ggml::TensorInfo>,
        counts: Option<&'a ggml::TensorInfo>,
    }

    // sums_counts_for, keyed by tensor name (`std::map` order — the port's
    // BTreeMap gives the same iteration order)
    let mut sums_counts_for: std::collections::BTreeMap<String, SumTensors> =
        std::collections::BTreeMap::new();

    for t in &gguf.tensors {
        let mut name = t.name.clone();
        if name.is_empty() {
            continue;
        }
        if string_remove_suffix(&mut name, IN_SUM_SUFFIX) {
            sums_counts_for.entry(name).or_default().in_sum = Some(t);
        } else if string_remove_suffix(&mut name, IN_SUM2_SUFFIX) {
            sums_counts_for.entry(name).or_default().in_sum2 = Some(t);
        } else if string_remove_suffix(&mut name, COUNTS_SUFFIX) {
            sums_counts_for.entry(name).or_default().counts = Some(t);
        }
    }

    for (name, sc) in sums_counts_for.iter() {
        let SumTensors {
            in_sum,
            in_sum2,
            counts,
        } = *sc;
        let (Some(in_sum2), Some(counts)) = (in_sum2, counts) else {
            eprintln!("common_imatrix_load: mismatched sums and counts for {name}");
            return None;
        };
        // an `in_sum` of a different width than `in_sum2` is rejected
        // (imatrix-loader.cpp:152-155, upstream a7b94df2c)
        if let Some(in_sum) = in_sum {
            if in_sum.n_elements() != in_sum2.n_elements() {
                eprintln!("common_imatrix_load: mismatched sums and counts for {name}");
                return None;
            }
        }

        if in_sum2.ty != GgmlType::F32 || counts.ty != GgmlType::F32 {
            eprintln!("common_imatrix_load: sums and counts for {name} must be F32");
            return None;
        }

        let sums_data = gguf.tensor_data(&in_sum2.name)?;
        let counts_data = gguf.tensor_data(&counts.name)?;

        let nval = in_sum2.n_elements() as usize;
        let ncounts = counts.n_elements() as usize;

        let mut e = CommonImatrixEntry::default();
        e.sums.resize(nval, 0.0);
        for (j, s) in e.sums.iter_mut().enumerate() {
            *s = f32::from_le_bytes(sums_data[j * 4..j * 4 + 4].try_into().unwrap());
        }
        e.counts.resize(ncounts, 0);
        for (j, c) in e.counts.iter_mut().enumerate() {
            // std::lround: round half away from zero (imatrix-loader.cpp:166)
            let v = f32::from_le_bytes(counts_data[j * 4..j * 4 + 4].try_into().unwrap());
            *c = v.round() as i64;
        }

        // the activation sums (imatrix-loader.cpp:178-184, upstream a7b94df2c)
        if let Some(in_sum) = in_sum {
            if in_sum.n_elements() as usize == nval {
                let in_sum_data = gguf.tensor_data(&in_sum.name)?;
                e.activations.resize(nval, 0.0);
                for (j, a) in e.activations.iter_mut().enumerate() {
                    *a = f32::from_le_bytes(in_sum_data[j * 4..j * 4 + 4].try_into().unwrap());
                }
            }
        }
        imatrix.entries.insert(name.clone(), e);
    }

    Some(imatrix)
}

/// `bool IMatrixCollector::load_imatrix(const char * file_name)`
/// (imatrix.cpp:642-1160, upstream a7b94df2c: activations merge + chunk-size
/// and n_layer_nextn propagation). Merges `file_name` into `m_stats` (raw
/// sums/counts). `params` is the collector's live params: the loaded
/// `chunk_size`/`n_layer_nextn` are folded back into it (the C writes its
/// `m_chunk_size`/`m_n_layer_nextn` fields here).
pub fn load_imatrix_into(
    stats: &mut HashMap<String, Stats>,
    m_datasets: &mut Vec<String>,
    m_last_chunk: &mut i32,
    params: &mut ImatrixParams,
    file_name: &str,
) -> bool {
    let Some(loaded) = common_imatrix_load(file_name) else {
        return false;
    };

    let is_legacy = loaded.is_legacy;

    // `m_n_layer_nextn` adoption (imatrix.cpp:1072-1080)
    if loaded.n_layer_nextn > 0 {
        if params.n_layer_nextn == 0 {
            params.n_layer_nextn = loaded.n_layer_nextn;
        } else if params.n_layer_nextn != loaded.n_layer_nextn {
            eprintln!(
                "load_imatrix: NextN layer count mismatch in {file_name}: {} != {}, using {}",
                loaded.n_layer_nextn, params.n_layer_nextn, params.n_layer_nextn
            );
        }
    }

    // `m_chunk_size` adoption (imatrix.cpp:1082-1090)
    if !is_legacy && loaded.chunk_size > 0 {
        if params.chunk_size_loaded == 0 {
            params.chunk_size_loaded = loaded.chunk_size;
        } else if params.chunk_size_loaded != loaded.chunk_size {
            eprintln!(
                "load_imatrix: chunk size mismatch in {file_name}: {} != {}, using {}",
                loaded.chunk_size, params.chunk_size_loaded, params.chunk_size_loaded
            );
        }
    }

    let chunk_size = e_chunk_size(params);

    for (name, entry) in loaded.entries.iter() {
        let e = stats.entry(name.clone()).or_default();

        if is_legacy {
            // Legacy format: sums contain (raw_sum/raw_count)*ncall, counts
            // contain {ncall}. Reconstruct raw form by multiplying by chunk_size
            if e.values.is_empty() {
                e.values.resize(entry.sums.len(), 0.0);
                e.counts.resize(1, 0);
            }
            for j in 0..entry.sums.len() {
                e.values[j] += entry.sums[j] * chunk_size as f32;
            }
            for j in 0..e.counts.len() {
                e.counts[j] += entry.counts[0] * chunk_size as i64;
            }
            e.activations.clear();
        } else {
            // GGUF format: raw sums and counts, accumulate directly
            let nval = entry.sums.len();
            let ncounts = entry.counts.len();
            let nact = entry.activations.len();
            let first_contribution = e.values.is_empty();

            if e.values.is_empty() {
                e.values.resize(nval, 0.0);
            } else if nval != e.values.len() {
                eprintln!(
                    "load_imatrix: mismatched sums size for {name}: {nval} != {}",
                    e.values.len()
                );
                return false;
            }

            if e.counts.is_empty() {
                e.counts.resize(ncounts, 0);
            } else if e.counts.len() == 1 && ncounts > 1 {
                let c0 = e.counts[0];
                e.counts.resize(ncounts, c0);
            } else if ncounts != e.counts.len() {
                eprintln!(
                    "load_imatrix: mismatched counts size for {name}: {ncounts} != {}",
                    e.counts.len()
                );
                return false;
            }

            for j in 0..nval {
                e.values[j] += entry.sums[j];
            }
            for j in 0..ncounts {
                e.counts[j] += entry.counts[j];
            }

            // the activation sums merge (imatrix.cpp:1131-1157): a side with
            // none drops them for the tensor
            if nact > 0 && (first_contribution || !e.activations.is_empty()) {
                if nact != nval {
                    eprintln!(
                        "load_imatrix: mismatched activations size for {name}: {nact} != {nval}"
                    );
                    return false;
                }
                if e.activations.is_empty() {
                    e.activations.resize(nact, 0.0);
                } else if nact != e.activations.len() {
                    eprintln!(
                        "load_imatrix: mismatched activations size for {name}: {nact} != {}",
                        e.activations.len()
                    );
                    return false;
                }
                for j in 0..nact {
                    e.activations[j] += entry.activations[j];
                }
            } else {
                e.activations.clear();
            }
        }
    }

    m_datasets.extend(loaded.datasets.iter().cloned());

    // Calculate the last chunk count (`rows_to_chunks`, imatrix.cpp:1158)
    let mut max_count = 0i64;
    for s in stats.values() {
        for &count in s.counts.iter() {
            if count > max_count {
                max_count = count;
            }
        }
    }
    *m_last_chunk = rows_to_chunks(max_count, chunk_size);

    true
}

// ---------------------------------------------------------------------------
// show_statistics (imatrix.cpp:968-1075)
// ---------------------------------------------------------------------------


/// printf's `%.<p>f` with the C "nan" spelling, pre-padded to `w`
fn fmt_p4(v: f32, w: usize, p: usize) -> String {
    if v.is_nan() {
        return pad_bytes(&fmt_nan(v), w);
    }
    pad_bytes(&format!("{v:.p$}", p = p), w)
}

/// printf's `%f` spelling of NaN ("nan"/"-nan") — Rust's `{}` prints "NaN"
fn fmt_nan(v: f32) -> String {
    if v.is_nan() {
        if v.is_sign_negative() { "-nan".to_string() } else { "nan".to_string() }
    } else {
        v.to_string()
    }
}

/// pad `s` to `w` BYTES like printf's `%*s` (UTF-8 aware strings pad by
/// bytes in C)
fn pad_bytes(s: &str, w: usize) -> String {
    let len = s.len();
    if len >= w {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(w - len))
    }
}

/// `static bool show_statistics(const common_params & params)`
/// (imatrix.cpp:1601-1816, upstream a7b94df2c — the rewritten tables) —
/// prints the per-tensor and per-layer statistics tables.
pub fn show_statistics(in_file: &str, collector: &IMatrixCollector) -> bool {
    let fnan = f32::NAN;

    // Load and process data
    let mut ts: Vec<TensorStatistics> = Vec::with_capacity(collector.get_mstats().len());
    for (name, st) in collector.get_mstats() {
        compute_vector_statistics(&mut ts, name, st);
    }

    if ts.is_empty() {
        return false;
    }

    let n_legacy = ts.iter().filter(|t| t.legacy).count();
    if n_legacy > 0 && n_legacy < ts.len() {
        eprintln!(
            "show_statistics: {n_legacy} of {} tensors have no activation data, using the legacy layout",
            ts.len()
        );
    }

    let legacy = n_legacy > 0;
    let nextn_start = nextn_layer_start(&ts, collector.m_params.n_layer_nextn);
    compute_tensor_statistics(&mut ts, collector.get_mstats(), nextn_start);

    // Sorting logic (Layer index -> Tensor Name) — `tensor_comparer`
    // (imatrix.cpp:1639-1668)
    ts.sort_by(|a, b| {
        let (lay_a, name_a) = process_tensor_name(&a.tensor);
        let (lay_b, name_b) = process_tensor_name(&b.tensor);

        // Handle non-numeric layers (e.g., "output")
        let blk_a: i32 = match lay_a.parse() {
            Ok(v) => v,
            Err(_) => {
                if a.tensor.contains("output") {
                    i32::MAX
                } else {
                    i32::MAX - 1
                }
            }
        };
        let blk_b: i32 = match lay_b.parse() {
            Ok(v) => v,
            Err(_) => {
                if b.tensor.contains("output") {
                    i32::MAX
                } else {
                    i32::MAX - 1
                }
            }
        };

        if blk_a != blk_b {
            blk_a.cmp(&blk_b)
        } else {
            name_a.cmp(&name_b)
        }
    });

    #[derive(Default, Clone, Copy)]
    struct LayerStats {
        layer_sum: f32,
        n: i64,
    }
    let mut ls: std::collections::BTreeMap<i32, LayerStats> = std::collections::BTreeMap::new();

    // Shorten names for table formatting (`label_fmt`, imatrix.cpp:1673-1677)
    let label_fmt = |s: &str, w: usize| -> String {
        if s.len() <= w {
            s.to_string()
        } else {
            format!("..{}", &s[s.len() - (w - 2)..])
        }
    };

    const W_LAY: usize = 6;
    const W_NAM: usize = 40; // Should be wide enough for most tensor names
    let sep = " | ";

    println!(
        "\nComputing tensor statistics for {in_file} ({} tensors)",
        ts.len()
    );

    // the multibyte `∑ E[A²]` header pads by BYTES in printf (%17s) — the
    // port pads the raw string the same way
    let sum_hdr = pad_bytes("\u{2211} E[A\u{b2}]", 17);
    if legacy {
        println!(
            "\n{:>width_lay$}{sep}{:<width_nam$}{sep}{:>10}{:>10}{:>12}{:>12}{:>9}{sep}{sum_hdr}{:>8}{sep}{:>10}{:>10}",
            "Layer", "Tensor",
            "Mean", "StdDev", "Skew", "Kurt", "H Norm",
            "Gain",
            "PCC", "Cov",
            width_lay = W_LAY, width_nam = W_NAM, sep = sep, sum_hdr = sum_hdr
        );
        println!("{}", "-".repeat(153));
    } else {
        println!(
            "\n{:>width_lay$}{sep}{:<width_nam$}{sep}{:>10}{:>10}{:>12}{:>12}{:>9}{sep}{sum_hdr}{:>8}{sep}{:>12}{:>10}{:>10}",
            "Layer", "Tensor",
            "Mean", "StdDev", "Skew", "Kurt", "H Norm",
            "Gain",
            "L2 Dist", "PCC", "Cov",
            width_lay = W_LAY, width_nam = W_NAM, sep = sep, sum_hdr = sum_hdr
        );
        println!("{}", "-".repeat(165));
    }

    // Tensor Statistics
    for tstat in &ts {
        let (layer, _) = process_tensor_name(&tstat.tensor);
        let h_norm = if tstat.elements > 1 {
            100.0 * (tstat.entropy / (tstat.elements as f32).log2())
        } else {
            fnan
        };

        let blk: i32 = match layer.parse() {
            Ok(v) => v,
            Err(_) => {
                if tstat.tensor.contains("output") {
                    i32::MAX
                } else {
                    -1
                }
            }
        };

        let layer = layer_label(blk, nextn_start);
        // printf renders NaN as "nan"; Rust's Display says "NaN" — the three
        // columns that can be NaN are pre-formatted
        let gain_s = fmt_p4(tstat.gain, 8, 2);
        let pcc_s = fmt_p4(tstat.pearson, 10, 4);
        let cov_s = fmt_p4(tstat.covariance, 10, 4);
        let l2_s = fmt_p4(tstat.l2_dist, 12, 4);
        if legacy {
            println!(
                "{:>width_lay$}{sep}{:<width_nam$}{sep}{mean:>10.4}{sd:>10.4}{sk:>12.4}{ku:>12.4}{hn:>8.2}%{sep}{sum:>14.4}{gain_s:>8}{sep}{pcc_s:>10}{cov_s:>10}",
                layer,
                label_fmt(&tstat.tensor, W_NAM),
                mean = tstat.mean,
                sd = tstat.std_deviation,
                sk = tstat.skewness,
                ku = tstat.kurtosis,
                hn = h_norm,
                sum = tstat.sum,
                width_lay = W_LAY, width_nam = W_NAM, sep = sep
            );
        } else {
            println!(
                "{:>width_lay$}{sep}{:<width_nam$}{sep}{mean:>10.4}{sd:>10.4}{sk:>12.4}{ku:>12.4}{hn:>8.2}%{sep}{sum:>14.4}{gain_s:>8}{sep}{l2_s:>12}{pcc_s:>10}{cov_s:>10}",
                layer,
                label_fmt(&tstat.tensor, W_NAM),
                mean = tstat.mean,
                sd = tstat.std_deviation,
                sk = tstat.skewness,
                ku = tstat.kurtosis,
                hn = h_norm,
                sum = tstat.sum,
                width_lay = W_LAY, width_nam = W_NAM, sep = sep
            );
        }

        // Aggregate Layer Stats — the C's float += double promotes per add;
        // the port matches that rounding
        let l = ls.entry(blk).or_default();
        l.layer_sum = (l.layer_sum as f64 + tstat.sum) as f32;
        l.n += tstat.elements;
    }

    // Layer Statistics
    let layer_stats = compute_layer_statistics(&ts);

    let mut layers: usize = 0;
    let mut trunk_layers: usize = 0;
    let mut min = i32::MAX;
    let mut max: i32 = -1;

    for (&layer, stats) in ls.iter() {
        if (0..9999).contains(&layer) && stats.n > 0 {
            layers += 1;
            if layer < nextn_start {
                trunk_layers += 1;
                min = layer.min(min);
                max = layer.max(max);
            }
        }
    }

    if trunk_layers > 0 {
        let expected = (max - min + 1) as usize;
        if trunk_layers != expected {
            eprintln!(
                "\nshow_statistics: layer sequence gap detected (found {trunk_layers} layers in range {min}-{max}, expected {expected}); layer statistics will not be shown"
            );
            return false;
        }
    }

    println!(
        "\nComputing layer statistics for {in_file} ({layers} layers)\n"
    );

    if legacy {
        println!(
            "{:>width_lay$}{sep}{sum_hdr}{:>8}{sep}{:>9}{:>9}{:>12}",
            "Layer", "Gain",
            "CosSim", "PCC", "Cov",
            width_lay = W_LAY, sep = sep, sum_hdr = sum_hdr
        );
        println!("{}", "-".repeat(64));
    } else {
        println!(
            "{:>width_lay$}{sep}{sum_hdr}{:>8}{sep}{:>12}{:>9}{:>9}{:>12}",
            "Layer", "Gain",
            "L2 Dist", "CosSim", "PCC", "Cov",
            width_lay = W_LAY, sep = sep, sum_hdr = sum_hdr
        );
        println!("{}", "-".repeat(76));
    }

    let get_layer_stat =
        |map: &std::collections::BTreeMap<i32, f32>, layer: i32| -> f32 {
            map.get(&layer).copied().unwrap_or(fnan)
        };

    for (&layer, stats) in ls.iter() {
        if layer < 0 || stats.n == 0 {
            continue;
        }

        let skip = layer == 0 || layer == i32::MAX;
        let lgn = if skip { fnan } else { get_layer_stat(&layer_stats.layer_gain, layer) };
        let ll2 = if skip { fnan } else { get_layer_stat(&layer_stats.layer_l2_dist, layer) };
        let lcs = if skip { fnan } else { get_layer_stat(&layer_stats.layer_cossim, layer) };
        let lpc = if skip { fnan } else { get_layer_stat(&layer_stats.layer_pearson, layer) };
        let lcv = if skip { fnan } else { get_layer_stat(&layer_stats.layer_covariance, layer) };
        let lyr = layer_label(layer, nextn_start);

        let lgn_s = fmt_p4(lgn, 8, 2);
        let ll2_s = fmt_p4(ll2, 12, 4);
        let lcs_s = fmt_p4(lcs, 9, 4);
        let lpc_s = fmt_p4(lpc, 9, 4);
        let lcv_s = fmt_p4(lcv, 12, 4);
        if legacy {
            println!(
                "{:>width_lay$}{sep}{sum:>14.4}{lgn_s:>8}{sep}{lcs_s:>9}{lpc_s:>9}{lcv_s:>12}",
                lyr,
                sum = stats.layer_sum,
                width_lay = W_LAY, sep = sep
            );
        } else {
            println!(
                "{:>width_lay$}{sep}{sum:>14.4}{lgn_s:>8}{sep}{ll2_s:>12}{lcs_s:>9}{lpc_s:>9}{lcv_s:>12}",
                lyr,
                sum = stats.layer_sum,
                width_lay = W_LAY, sep = sep
            );
        }
    }

    println!();
    true
}

// ---------------------------------------------------------------------------
// the global collector (`static IMatrixCollector g_collector`,
// `ik_collect_imatrix`, imatrix.cpp:713-717)
// ---------------------------------------------------------------------------

static COLLECTOR: std::sync::Mutex<Option<IMatrixCollector>> = std::sync::Mutex::new(None);

/// Run `f` on the global collector (created on first use).
pub fn with_collector<R>(f: impl FnOnce(&mut IMatrixCollector) -> R) -> R {
    let mut guard = COLLECTOR.lock().unwrap_or_else(|e| e.into_inner());
    let c = guard.get_or_insert_with(IMatrixCollector::new);
    f(c)
}

/// Verbosity of the collector's debug output (`LOG_DBGV(1, ...)` lines) —
/// `common_log_set_verbosity_thold`, only the two lines the collector emits
/// (imatrix.cpp:529, 636).
pub fn print_stored_line(func: &str, last_chunk: i32, fname: &str) {
    if log_verbose() {
        eprintln!("{func}: stored collected data after {last_chunk} chunks in {fname}");
    }
}

fn log_verbose() -> bool {
    VERBOSITY.load(std::sync::atomic::Ordering::Relaxed) >= 1
        || std::env::var("LLAMA_LOG_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
}

/// `common_log_set_verbosity_thold` stand-in — the tool's `--verbose` count.
static VERBOSITY: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_verbosity(v: u8) {
    VERBOSITY.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// `ik_collect_imatrix` (imatrix.cpp:715-717) — the `params.cb_eval` the tool
/// registers. Locks the global collector (the C `m_mutex`, imatrix.cpp:254)
/// and drives [`collect_imatrix`], including the periodic saves.
pub fn eval_callback(t: &EvalNode<'_>, ask: bool) -> bool {
    trace_node(t, ask);
    with_collector(|c| {
        let IMatrixCollector {
            m_stats,
            m_params,
            m_last_chunk,
            m_src1_data,
            m_ids,
            m_datasets,
            ..
        } = c;
        let datasets = m_datasets.clone();
        let mut save = |stats: &HashMap<String, Stats>, n_chunk: i32, last_chunk: i32| {
            if let Err(e) = save_imatrix(stats, m_params, &datasets, last_chunk, n_chunk) {
                eprintln!("save_imatrix: {e}");
            }
        };
        collect_imatrix(
            m_stats,
            m_params,
            m_last_chunk,
            m_src1_data,
            m_ids,
            t,
            ask,
            &mut save,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `filter_tensor_name` (imatrix.cpp:79-94).
    #[test]
    fn filter_tensor_name_strips_backend_markers() {
        assert_eq!(
            filter_tensor_name("CUDA0#blk.0.attn_k.weight#0"),
            "blk.0.attn_k.weight"
        );
        assert_eq!(
            filter_tensor_name("CUDA0#blk.0.attn_k.weight"),
            "blk.0.attn_k.weight"
        );
        assert_eq!(
            filter_tensor_name("blk.0.attn_k.weight"),
            "blk.0.attn_k.weight"
        );
    }

    /// `process_tensor_name` (imatrix.cpp:96-123).
    #[test]
    fn process_tensor_name_splits_layer_and_role() {
        // (imatrix.cpp:96-145, upstream a7b94df2c): the `j += size > 4 ? 1 : 2`
        // quirk skips the role part of a 4-part `blk.N.role.weight`, so the
        // tensor falls back to the full input; longer names keep the dotted
        // role path
        assert_eq!(
            process_tensor_name("blk.12.ffn_down.weight"),
            ("12".to_string(), "blk.12.ffn_down.weight".to_string())
        );
        assert_eq!(
            process_tensor_name("output.weight"),
            ("-".to_string(), "output".to_string())
        );
        // 5 parts: nextn.blk.28.post_projection.weight keeps its prefix+role
        assert_eq!(
            process_tensor_name("nextn.blk.28.post_projection.weight"),
            ("28".to_string(), "nextn.post_projection".to_string())
        );
        // no `weight` suffix: the whole input is the role
        assert_eq!(
            process_tensor_name("token_embd"),
            ("-".to_string(), "token_embd".to_string())
        );
    }

    /// The accumulator rule: `values += x*x` per row, `counts += nrows` per
    /// MUL_MAT (imatrix.cpp:384-408). A synthetic node with 2 rows of 4 F32
    /// activations must give sum(x²) and count 2.
    #[test]
    fn dense_accumulate_matches_reference_rule() {
        use ggml::compute::EvalSrc;

        let rows: [[f32; 4]; 2] = [[1.0, 2.0, -3.0, 0.5], [0.0, 1.0, 1.0, -1.0]];
        let mut bytes: Vec<u8> = Vec::new();
        for r in &rows {
            for v in r {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let weight = EvalSrc {
            name: "blk.0.ffn_down.weight",
            ty: GgmlType::Q4K,
            ne: [4, 1, 1, 1],
            nb: [18, 18, 18, 18],
            data: None,
        };
        let act = EvalSrc {
            name: "",
            ty: GgmlType::F32,
            ne: [4, 2, 1, 1],
            nb: [4, 16, 32, 32],
            data: Some(&bytes),
        };
        let mut src = [None; ggml::MAX_SRC];
        src[0] = Some(weight);
        src[1] = Some(act);
        let node = EvalNode {
            op: GgmlOp::MulMat,
            name: "",
            ty: GgmlType::F32,
            ne: [0; 4],
            nb: [0; 4],
            op_params: [0; ggml::types::MAX_OP_PARAMS / 4],
            data: None,
            src,
        };

        let mut stats = HashMap::new();
        let mut last_chunk = 0;
        let params = ImatrixParams {
            n_ctx: 512,
            n_parallel: 1,
            ..Default::default()
        };
        let mut save = |_s: &HashMap<String, Stats>, _n: i32, _l: i32| {};
        let mut src1_data = Vec::new();
        let mut ids = Vec::new();
        assert!(collect_imatrix(
            &mut stats,
            &params,
            &mut last_chunk,
            &mut src1_data,
            &mut ids,
            &node,
            false,
            &mut save
        ));
        let e = &stats["blk.0.ffn_down.weight"];
        assert_eq!(e.counts, vec![2]);
        // per column: 1+0, 4+1, 9+1, 0.25+1
        assert_eq!(e.values, vec![1.0, 5.0, 10.0, 1.25]);
        // a 2-token activation batch is below the 16-token gate for ask
        let ask = collect_imatrix(
            &mut stats,
            &params,
            &mut last_chunk,
            &mut src1_data,
            &mut ids,
            &node,
            true,
            &mut save,
        );
        assert!(
            !ask,
            "src1->ne[1] < 16 is rejected by the ask branch (imatrix.cpp:249)"
        );
    }

    /// The ask gate (imatrix.cpp:245-252): only `blk.*` MUL_MATs of F32
    /// activations with >= 16 tokens, plus every MUL_MAT_ID.
    #[test]
    fn ask_gate() {
        use ggml::compute::EvalSrc;
        let mk = |op: GgmlOp, name: &'static str, n_tokens: i64, ty: GgmlType| {
            let w = EvalSrc {
                name,
                ty: GgmlType::Q4K,
                ne: [32, 1, 1, 1],
                nb: [0; 4],
                data: None,
            };
            let a = EvalSrc {
                name: "",
                ty,
                ne: [32, n_tokens, 1, 1],
                nb: [4, 128, 0, 0],
                data: None,
            };
            let mut src = [None; ggml::MAX_SRC];
            src[0] = Some(w);
            src[1] = Some(a);
            EvalNode {
                op,
                name: "",
                ty: GgmlType::F32,
                ne: [0; 4],
                nb: [0; 4],
                op_params: [0; ggml::types::MAX_OP_PARAMS / 4],
                data: None,
                src,
            }
        };
        let mut stats = HashMap::new();
        let mut last_chunk = 0;
        let mut save = |_s: &HashMap<String, Stats>, _n: i32, _l: i32| {};
        let mut src1_data = Vec::new();
        let mut ids = Vec::new();
        let p = ImatrixParams {
            n_ctx: 512,
            n_parallel: 1,
            ..Default::default()
        };
        let mut ask = |n: &EvalNode| {
            collect_imatrix(
                &mut stats,
                &p,
                &mut last_chunk,
                &mut src1_data,
                &mut ids,
                n,
                true,
                &mut save,
            )
        };
        assert!(ask(&mk(
            GgmlOp::MulMat,
            "blk.3.attn_k.weight",
            16,
            GgmlType::F32
        )));
        assert!(!ask(&mk(
            GgmlOp::MulMat,
            "blk.3.attn_k.weight",
            8,
            GgmlType::F32
        )));
        assert!(!ask(&mk(
            GgmlOp::MulMat,
            "output.weight",
            16,
            GgmlType::F32
        )));
        assert!(!ask(&mk(
            GgmlOp::MulMat,
            "token_embd.weight",
            16,
            GgmlType::F32
        )));
        assert!(!ask(&mk(
            GgmlOp::MulMat,
            "blk.3.attn_k.weight",
            16,
            GgmlType::F16
        )));
        assert!(ask(&mk(
            GgmlOp::MulMatId,
            "blk.3.ffn_gate_exps.weight",
            8,
            GgmlType::F32
        )));
        assert!(!ask(&mk(
            GgmlOp::GetRows,
            "token_embd.weight",
            16,
            GgmlType::F32
        )));
    }
}
