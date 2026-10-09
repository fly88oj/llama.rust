//! LoRA adapters — port of src/llama-adapter.cpp/.h @ bd4f514db1.
//!
//! Ported: the adapter GGUF loader (`llama_adapter_lora_init_impl`,
//! llama-adapter.cpp:151-423) with its metadata contract (`general.type` ==
//! "adapter", `general.architecture` == the base model arch, `adapter.type` ==
//! "lora", `adapter.lora.alpha`, `adapter.alora.invocation_tokens`), the
//! `.lora_a`/`.lora_b` tensor-name pairing, the base-model shape validation,
//! `llama_adapter_lora_weight::get_scale` (llama-adapter.h:53-57), the
//! `llama_adapter_meta_*` accessors and the graph-side application
//! (`llm_graph_context::build_lora_mm` / `build_lora_mm_id`,
//! llama-graph.cpp:1514-1581, and
//! `build_inp_embd`'s lora branch, llama-graph.cpp:2389-2405).
//!
//! `llama_context::set_adapters_lora` (llama-context.cpp:1332-1348) keeps the
//! active set per context. The port cannot thread a `loras` field through
//! `DecodeContext` (context.rs is owned elsewhere), so the active set lives in
//! a thread-local written by [`set_adapters_lora`] and read by the builders via
//! [`lora_mm`]/[`lora_embd`] — same lifetime rules as the C (one set per
//! context, replaced wholesale, zero scales dropped), one set per thread
//! instead of one per `llama_context`.
//!
//! Not ported (see PARITY.md):
//!   * control vectors (`llama_adapter_cvec`, llama-adapter.cpp:16-136 +
//!     `llm_graph_context::build_cvec` llama-graph.cpp:1508-1512 +
//!     `llama_context::set_adapter_cvec` llama-context.cpp:1376-1389). Their
//!     loader is `common_control_vector_load` (common/common.cpp:2002-2100),
//!     i.e. outside llama-adapter.cpp; the `_norm.weight` adapter tensors that
//!     feed the norm path are recognized but skipped (llama-adapter.cpp:290-293).
//!   * the CPU extra-buft fallback (llama-adapter.cpp:299-353): the port has a
//!     single CPU storage class (Context arena / mmap), so there is no
//!     repacking buffer type to fall back from.
//!   * `llama_context::sched_need_reserve` bookkeeping (llama-context.cpp:1347,
//!     `get_n_nodes` llama-adapter.h:85-87): the port has no graph scheduler and
//!     rebuilds the graph per step.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use ggml::{Context, GgmlPrec, GgmlType, Gguf, GgufType, TensorId, Value};

use crate::arch::{kv_name, LlmArch, LlmKv};

/// `adapter.type` values of this revision: llama-adapter.cpp:216-219 accepts
/// only "lora" (control vectors are not loaded through this file at all).
pub const ADAPTER_TYPE_LORA: &str = "lora";

// ---------------------------------------------------------------------------
// llama_adapter_lora_weight (llama-adapter.h:48-61)
// ---------------------------------------------------------------------------

/// One `<name>.lora_a` / `<name>.lora_b` pair, as created in the adapter's
/// Context (llama-adapter.h:48-51: `ggml_tensor * a, * b`).
#[derive(Clone, Debug)]
pub struct LoraWeight {
    pub a: TensorId,
    pub b: TensorId,
    /// The adapter-file tensor names (C keeps them in `w->a->name` /
    /// `w->b->name`, llama-adapter.cpp:376-377). The port's Context is
    /// truncatable (`reset_graph_to`), so [`set_adapters_lora`] uses them to
    /// prove the pair is still live.
    pub a_name: String,
    pub b_name: String,
}

impl LoraWeight {
    pub fn new(a: TensorId, b: TensorId, a_name: String, b_name: String) -> Self {
        LoraWeight {
            a,
            b,
            a_name,
            b_name,
        }
    }

    /// `get_scale` (llama-adapter.h:53-57): the rank is `b->ne[0]`, and an
    /// alpha of 0 (missing `adapter.lora.alpha`, llama-adapter.cpp:221) leaves
    /// the user scale untouched.
    pub fn get_scale(&self, ctx: &Context, alpha: f32, adapter_scale: f32) -> f32 {
        let rank = ctx.ne(self.b)[0] as f32;
        if alpha != 0.0 {
            adapter_scale * alpha / rank
        } else {
            adapter_scale
        }
    }
}

// ---------------------------------------------------------------------------
// llama_adapter_lora (llama-adapter.h:63-88)
// ---------------------------------------------------------------------------

/// `llama_adapter_lora` (llama-adapter.h:63-88). The C also holds the owning
/// model (to deregister itself in `llama_adapter_lora_free`, :499-510) and the
/// per-buft ggml contexts/buffers; the port's tensors live in the caller's
/// Context and are shared through `Rc`, so neither is needed.
pub struct AdapterLora {
    /// `general.architecture` of the adapter file — must equal the base model
    /// arch (llama-adapter.cpp:210-214).
    pub arch: LlmArch,
    /// `adapter.lora.alpha` (llama-adapter.cpp:221), 0 when absent.
    pub alpha: f32,
    /// map tensor name (suffix stripped) to lora_a_b (llama-adapter.h:67)
    pub ab_map: HashMap<String, LoraWeight>,
    /// gguf metadata (llama-adapter.h:75): non-array keys only, in file order;
    /// the C uses an unordered_map and drops duplicates (`emplace`,
    /// llama-adapter.cpp:185), which this ordered pair list reproduces.
    pub gguf_kv: Vec<(String, String)>,
    /// activated lora (aLoRA) (llama-adapter.h:78)
    pub alora_invocation_tokens: Vec<i32>,
    /// `Context::mark()` right after the adapter tensors were created — the
    /// port's replacement for the C's heap ownership of the pairs
    /// (`llama_adapter_lora_free`, llama-adapter.cpp:499-510). [`set_adapters_lora`]
    /// refuses a set whose mark is past the context's current top, which means
    /// the pairs were dropped by `reset_graph_to`.
    pub load_mark: usize,
}

impl std::fmt::Debug for AdapterLora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterLora")
            .field("arch", &self.arch.name())
            .field("alpha", &self.alpha)
            .field("n_tensors", &(self.ab_map.len() * 2))
            .finish()
    }
}

impl AdapterLora {
    /// `llama_adapter_lora::get_weight` (llama-adapter.cpp:140-149): the C keys
    /// on `w->name`, so the caller passes the *base model* tensor's name.
    pub fn get_weight(&self, name: &str) -> Option<&LoraWeight> {
        self.ab_map.get(name)
    }

    /// `get_n_nodes` (llama-adapter.h:85-87): a, b, scale, add, 2 x mul_mat.
    pub fn get_n_nodes(&self) -> u32 {
        self.ab_map.len() as u32 * 6
    }

    /// `llama_adapter_meta_val_str` (llama-adapter.cpp:460-469).
    pub fn meta_val_str(&self, key: &str) -> Option<&str> {
        self.gguf_kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// `llama_adapter_meta_count` (llama-adapter.cpp:471-473).
    pub fn meta_count(&self) -> usize {
        self.gguf_kv.len()
    }

    /// `llama_adapter_meta_key_by_index` (llama-adapter.cpp:475-485) — file
    /// order here, hash order in the C (unspecified either way).
    pub fn meta_key_by_index(&self, i: usize) -> Option<&str> {
        self.gguf_kv.get(i).map(|(k, _)| k.as_str())
    }

    /// `llama_adapter_meta_val_str_by_index` (llama-adapter.cpp:487-497).
    pub fn meta_val_str_by_index(&self, i: usize) -> Option<&str> {
        self.gguf_kv.get(i).map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------------------
// metadata formatting (llama-impl.cpp:119-171, used by the loader's dump)
// ---------------------------------------------------------------------------

/// `gguf_data_to_str` (llama-impl.cpp:119-134): `std::to_string` formatting.
fn data_to_str(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        // std::to_string(float) == "%f" (6 decimals, trailing zeros kept)
        Value::F32(x) => format!("{x:.6}"),
        Value::F64(x) => format!("{x:.6}"),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::String(s) => s.clone(),
        // nested arrays never reach here (kv_to_str prints "???" for them)
        Value::Array(ty, _) => format!("unknown type {}", *ty as u32),
    }
}

/// `gguf_kv_to_str` (llama-impl.cpp:136-171): arrays render as
/// `[a, b, c]` (strings quoted + `\`/`"` escaped), scalars via
/// [`data_to_str`].
fn kv_to_str(v: &Value) -> String {
    match v {
        Value::Array(arr_ty, items) => {
            let mut out = String::from("[");
            for (j, item) in items.iter().enumerate() {
                if *arr_ty == GgufType::String {
                    let s = item
                        .as_str()
                        .unwrap_or("")
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"");
                    out.push('"');
                    out.push_str(&s);
                    out.push('"');
                } else if *arr_ty == GgufType::Array {
                    out.push_str("???");
                } else {
                    out.push_str(&data_to_str(item));
                }
                if j + 1 < items.len() {
                    out.push_str(", ");
                }
            }
            out.push(']');
            out
        }
        _ => data_to_str(v),
    }
}

/// `gguf_type_name` of a KV value plus the C's array decoration
/// (llama-adapter.cpp:177-180): `array[type,n]`.
fn type_name(v: &Value) -> String {
    match v {
        Value::Array(arr_ty, items) => {
            format!("array[{},{}]", arr_ty.name(), items.len())
        }
        other => other.type_().name().to_string(),
    }
}

fn get_kv_str(gguf: &Gguf, key: &str) -> String {
    gguf.get_str(key).unwrap_or("").to_string()
}

fn get_kv_f32(gguf: &Gguf, key: &str) -> f32 {
    gguf.get_f32(key).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// llama_adapter_lora_init_impl (llama-adapter.cpp:151-423)
// ---------------------------------------------------------------------------

/// `llama_adapter_lora_init_impl` (llama-adapter.cpp:151-423) over an already
/// parsed adapter GGUF. `model_dims` is the C's `model.get_tensor(name)->ne`
/// (llama-adapter.cpp:333): the base-model tensor dims, or `None` when the
/// model has no such tensor.
///
/// Errors are the C's exception texts verbatim (`llama_adapter_lora_init_from_file_ptr`
/// wraps them with "failed to apply lora adapter: %s", :452).
pub fn adapter_lora_init_impl(
    ctx: &mut Context,
    arch: LlmArch,
    model_dims: &dyn Fn(&str) -> Option<[i64; 4]>,
    gguf: &Gguf,
) -> Result<AdapterLora, String> {
    let mut adapter = AdapterLora {
        arch,
        alpha: 0.0,
        ab_map: HashMap::new(),
        gguf_kv: Vec::new(),
        alora_invocation_tokens: Vec::new(),
        load_mark: 0,
    };

    // check metadata (llama-adapter.cpp:168-242)
    {
        eprintln!("llama_adapter_lora_init_impl: Dumping metadata keys/values.");

        for (i, (name, value)) in gguf.kv.iter().enumerate() {
            let tname = type_name(value);
            let svalue = kv_to_str(value);

            // non-array keys go into the adapter's metadata map (:184-186)
            if !matches!(value, Value::Array(..)) && !adapter.gguf_kv.iter().any(|(k, _)| k == name)
            {
                adapter.gguf_kv.push((name.clone(), svalue.clone()));
            }

            // (:188-192) values longer than 40 chars are truncated to 37 + "..."
            const MAX_VALUE_LEN: usize = 40;
            let mut print_value = if svalue.len() > MAX_VALUE_LEN {
                format!("{}...", &svalue[..MAX_VALUE_LEN - 3])
            } else {
                svalue
            };
            print_value = print_value.replace('\n', "\\n");

            // (%42s / %-16s of the C printf)
            eprintln!(
                "llama_adapter_lora_init_impl: - kv {i:3}: {name:>42} {tname:<16} = {print_value}"
            );
        }

        let general_type = get_kv_str(gguf, &kv_name(LlmArch::UNKNOWN, LlmKv::GENERAL_TYPE));
        if general_type != "adapter" {
            return Err(format!(
                "expect general.type to be 'adapter', but got: {general_type}"
            ));
        }

        let general_arch_str = get_kv_str(
            gguf,
            &kv_name(LlmArch::UNKNOWN, LlmKv::GENERAL_ARCHITECTURE),
        );
        // `llm_arch_from_string` maps an unknown name to LLM_ARCH_UNKNOWN
        let general_arch = LlmArch::from_name(&general_arch_str).unwrap_or(LlmArch::UNKNOWN);
        if general_arch != arch {
            return Err("model arch and LoRA arch mismatch".to_string());
        }

        let adapter_type = get_kv_str(gguf, &kv_name(LlmArch::UNKNOWN, LlmKv::ADAPTER_TYPE));
        if adapter_type != ADAPTER_TYPE_LORA {
            return Err(format!(
                "expect adapter.type to be 'lora', but got: {adapter_type}"
            ));
        }

        adapter.alpha = get_kv_f32(gguf, &kv_name(LlmArch::UNKNOWN, LlmKv::ADAPTER_LORA_ALPHA));

        // parse alora invocation sequence vector (:224-241)
        let key = kv_name(LlmArch::UNKNOWN, LlmKv::ADAPTER_ALORA_INVOCATION_TOKENS);
        if let Some(v) = gguf.find_key(&key) {
            let Value::Array(arr_ty, items) = v else {
                return Err(format!("invalid gguf type for {key}"));
            };
            if *arr_ty != GgufType::Uint32 {
                return Err(format!("invalid gguf element type for {key}"));
            }
            // the C is a blind memcpy of gguf_get_arr_data into vector<llama_token>
            adapter.alora_invocation_tokens = items
                .iter()
                .map(|it| it.as_u32().unwrap_or(0) as i32)
                .collect();
        }
    }

    // bundle lora_a and lora_b into pairs (:269-297); the C's pairing map is a
    // std::map (sorted), which BTreeMap reproduces — the resulting tensor
    // creation/validation order is that order.
    let mut ab_map: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    let str_endswith = |s: &str, suffix: &str| s.len() >= suffix.len() && s.ends_with(suffix);

    for ti in &gguf.tensors {
        let name = ti.name.as_str();
        if str_endswith(name, ".lora_a") {
            let key = name.replace(".lora_a", "");
            match ab_map.get_mut(&key) {
                None => {
                    ab_map.insert(key, (Some(ti.name.clone()), None));
                }
                Some(pair) => pair.0 = Some(ti.name.clone()),
            }
        } else if str_endswith(name, ".lora_b") {
            let key = name.replace(".lora_b", "");
            match ab_map.get_mut(&key) {
                None => {
                    ab_map.insert(key, (None, Some(ti.name.clone())));
                }
                Some(pair) => pair.1 = Some(ti.name.clone()),
            }
        } else if str_endswith(name, "_norm.weight") {
            // TODO: add support for norm vector (:290-293)
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            return Err(format!("LoRA tensor '{name}' has unexpected suffix"));
        }
    }

    // add tensors (:322-379) — the extra-buft fallback of :299-353 has no
    // counterpart: the port's storage is always CPU (arena or mmap).
    for (name, (a_file, b_file)) in &ab_map {
        let (Some(a_file), Some(b_file)) = (a_file, b_file) else {
            return Err(format!(
                "LoRA tensor pair for '{name}' is missing one component"
            ));
        };

        let ti_a = gguf
            .find_tensor(a_file)
            .expect("pairing kept the file names");
        let ti_b = gguf
            .find_tensor(b_file)
            .expect("pairing kept the file names");
        let is_token_embd = str_endswith(name, "token_embd.weight");

        // device buft and device ctx (model_tensor is the base-model tensor, :333)
        let Some(model_ne) = model_dims(name) else {
            return Err(format!(
                "LoRA tensor '{name}' does not exist in base model (hint: maybe wrong base model?)"
            ));
        };

        // validate tensor shape (:358-371)
        if is_token_embd {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if model_ne[0] != ti_b.ne[1] || model_ne[1] != ti_a.ne[1] {
                return Err(format!(
                    "tensor '{name}' has incorrect shape (hint: maybe wrong base model?)"
                ));
            }
        } else {
            if model_ne[0] != ti_a.ne[0] || model_ne[1] != ti_b.ne[1] {
                return Err(format!(
                    "tensor '{name}' has incorrect shape (hint: maybe wrong base model?)"
                ));
            }
            if ti_a.ne[1] != ti_b.ne[0] {
                return Err(
                    "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)"
                        .to_string(),
                );
            }
        }

        // ggml_dup_tensor + ggml_set_name + the payload copy of :374-417: the
        // adapter file's type and shape are kept as-is (no conversion).
        let mut new_tensor = |file: &str, ti: &ggml::TensorInfo| -> Result<TensorId, String> {
            let id = ctx.new_tensor(ti.ty, ti.ne);
            ctx.arena_resize_tensor(id);
            ctx.set_name(id, file);
            let bytes = gguf.tensor_data(file).ok_or_else(|| {
                format!("LoRA tensor '{file}' data is not within the file bounds, file is corrupted or incomplete")
            })?;
            let dst = ctx.data_bytes_mut(id).expect("arena tensor");
            dst.copy_from_slice(bytes);
            Ok(id)
        };
        let tensor_a = new_tensor(a_file, ti_a)?;
        let tensor_b = new_tensor(b_file, ti_b)?;

        adapter.ab_map.insert(
            name.clone(),
            LoraWeight::new(tensor_a, tensor_b, a_file.clone(), b_file.clone()),
        );
    }

    eprintln!(
        "llama_adapter_lora_init_impl: loaded {} tensors from lora file",
        adapter.ab_map.len() * 2
    );

    adapter.load_mark = ctx.mark();

    Ok(adapter)
}

/// `llama_adapter_lora_init` (llama-adapter.cpp:425-438) +
/// `llama_adapter_lora_init_from_file_ptr` (:440-458): opens the file, loads
/// the adapter, logs the C's wrapper line on failure and returns the error text
/// (the C returns nullptr after deleting the half-built adapter).
pub fn load_adapter_lora(
    ctx: &mut Context,
    arch: LlmArch,
    model_dims: &dyn Fn(&str) -> Option<[i64; 4]>,
    path: &str,
) -> Result<Rc<AdapterLora>, String> {
    eprintln!("llama_adapter_lora_init: loading lora adapter from '{path}' ...");

    let result = (|| -> Result<AdapterLora, String> {
        let gguf =
            Gguf::open(path).map_err(|_| "failed to load lora adapter from file".to_string())?;
        adapter_lora_init_impl(ctx, arch, model_dims, &gguf)
    })();

    match result {
        Ok(adapter) => Ok(Rc::new(adapter)),
        Err(e) => {
            eprintln!("llama_adapter_lora_init_from_file_ptr: failed to apply lora adapter: {e}");
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// llama_context::set_adapters_lora (llama-context.cpp:1332-1374)
// ---------------------------------------------------------------------------

/// `llama_adapter_loras` (llama-adapter.h:90): adapter → scale, in
/// `llama_set_adapters_lora` order. The C's container is an unordered_map, so
/// its iteration order is unspecified; a Vec keeps the caller's order, which is
/// what the single-adapter case (the only one with defined output) needs.
pub type AdaptersLora = Vec<(Rc<AdapterLora>, f32)>;

thread_local! {
    /// `llama_context::loras` (llama-context.h:288) — see the module docs.
    static ACTIVE_LORAS: RefCell<AdaptersLora> = const { RefCell::new(Vec::new()) };
}

/// `llama_context::set_adapters_lora` (llama-context.cpp:1332-1348; the public
/// entry point is `llama_set_adapters_lora`, :4069-4079): zero scales are never
/// inserted. The C additionally resets the scheduler's reserve
/// (`sched_need_reserve`), which the port has no counterpart for.
///
/// Port extra: the C's tensors are heap objects that live until
/// `llama_adapter_lora_free`; the port's live in a truncatable Context, so the
/// pairs must still be there (`reset_graph_to` drops everything created after
/// the watermark — load the adapter *before* constructing the DecodeContext).
pub fn set_adapters_lora(ctx: &Context, adapters: &[(Rc<AdapterLora>, f32)]) -> Result<(), String> {
    for (adapter, _) in adapters {
        if ctx.mark() < adapter.load_mark {
            return Err(format!(
                "lora tensors of a '{}' adapter are no longer alive in the context \
                 (load the adapter before the DecodeContext: reset_graph_to drops \
                 everything created after its watermark)",
                adapter.arch.name()
            ));
        }
        // in range (checked above), so the name lookup cannot panic; a name
        // mismatch means the ids were reused by later tensors
        for (name, w) in &adapter.ab_map {
            if ctx.name(w.a) != w.a_name {
                return Err(format!(
                    "lora tensor for '{name}' is no longer alive in the context"
                ));
            }
        }
    }

    ACTIVE_LORAS.with(|l| {
        let mut active = l.borrow_mut();
        active.clear();
        for (adapter, scale) in adapters {
            if *scale == 0.0 {
                continue;
            }
            // `loras->insert({adapters[i], scales[i]})` (:1343): an already
            // present adapter (same pointer in C, same Rc here) keeps its first
            // scale. common_init_from_params loads a *new* object per
            // `--lora`/`--lora-scaled` entry, so repeated flags still apply
            // twice — with two independent tensors.
            if active.iter().any(|(a, _)| Rc::ptr_eq(a, adapter)) {
                continue;
            }
            active.push((adapter.clone(), *scale));
        }
    });

    Ok(())
}

/// `llama_clear_adapter_lora` equivalent: `set_adapters_lora` with no adapters
/// (the C reaches the same state through `llama_set_adapters_lora(ctx, {…}, 0, {…})`,
/// llama-context.cpp:4069).
pub fn clear_adapter_lora() {
    set_adapters_lora(&Context::new(), &[]).expect("empty set is always valid");
}

/// `llama_context::adapters_lora_are_same` (llama-context.cpp:1350-1374) —
/// only ever a scheduler-reserve optimization in the C; kept for the port's
/// own tests and for the C parity of `set_adapters_lora`'s early-out.
pub fn adapters_lora_are_same(adapters: &[(Rc<AdapterLora>, f32)]) -> bool {
    ACTIVE_LORAS.with(|l| {
        let active = l.borrow();
        let mut n_non_zero = 0usize;
        for (adapter, scale) in adapters {
            if *scale == 0.0 {
                continue;
            }
            n_non_zero += 1;
            let Some((_, existing)) = active.iter().find(|(a, _)| Rc::ptr_eq(a, adapter)) else {
                return false;
            };
            if *existing != *scale {
                return false;
            }
        }
        n_non_zero == active.len()
    })
}

/// Borrow the active set for the duration of `f`
/// (llama-graph.cpp:1524 `for (const auto & lora : *loras)`).
pub fn with_active_loras<R>(f: impl FnOnce(&AdaptersLora) -> R) -> R {
    ACTIVE_LORAS.with(|l| f(&l.borrow()))
}

fn has_active_loras() -> bool {
    ACTIVE_LORAS.with(|l| !l.borrow().is_empty())
}

// ---------------------------------------------------------------------------
// graph side
// ---------------------------------------------------------------------------

/// `llm_graph_context::build_lora_mm` (llama-graph.cpp:1514-1543) without the
/// per-tensor scale (`w_s`), which the port's builders do not carry.
pub fn lora_mm(ctx: &mut Context, w: TensorId, cur: TensorId) -> TensorId {
    lora_mm_s(ctx, w, cur, None)
}

/// `llm_graph_context::build_lora_mm` (llama-graph.cpp:1514-1549): plain
/// `w @ cur`, then for every active adapter that has a pair for `w`'s name the
/// scaled `b @ (a @ cur)` added on top. `w_s` is the C's per-tensor scale
/// (`ggml_mul(res, w_s)`, :1520-1522; NVFP4/repack weights) — unused in the
/// port but kept in the signature so the call sites read like the C.
///
/// e9f824d8c: the C applies the model's `prec_policy` right after the
/// mul_mat (`prec_policy->apply(res)`, :1526-1530 — `ggml_prec_set_src`
/// writes the src1-precision op-param the GPU W4A4 kernels read; a CPU build
/// has no such kernel, so the port keeps the policy in
/// [`crate::model::PrecPolicy`] for the saver round-trip and skips the
/// op-param here; same for `build_lora_mm_id`, :1562-1566).
pub fn lora_mm_s(ctx: &mut Context, w: TensorId, cur: TensorId, w_s: Option<TensorId>) -> TensorId {
    let mut res = ctx.mul_mat(w, cur);

    // a7b94df2c (llama-graph.cpp:1560-1562): NVFP4 weights take a BF16
    // accumulator hint — `if (w->type == GGML_TYPE_NVFP4)
    // ggml_prec_set_acc(res, GGML_PREC_BF16)`. Batch 42b wired the port's
    // `Context::prec_set_acc`: the flag lands in op_params[0] bit-for-bit
    // like the reference (a GPU W4A4 kernel hint; the CPU backend has no
    // consumer for the slot, so numerics are unchanged).
    if ctx.ty(w) == GgmlType::Nvfp4 {
        ctx.prec_set_acc(res, GgmlPrec::Bf16);
    }

    if let Some(s) = w_s {
        res = ctx.mul(res, s);
    }

    if !has_active_loras() {
        return res;
    }

    // C keys the lookup on `w->name` (llama-adapter.cpp:141)
    let name = ctx.name(w).to_string();

    with_active_loras(|loras| {
        for (adapter, adapter_scale) in loras.iter() {
            let Some(lw) = adapter.get_weight(&name) else {
                continue;
            };

            let scale = lw.get_scale(ctx, adapter.alpha, *adapter_scale);

            // ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur)) :1533-1536
            let a_cur = ctx.mul_mat(lw.a, cur);
            let mut ab_cur = ctx.mul_mat(lw.b, a_cur);

            ab_cur = ctx.scale(ab_cur, scale);
            res = ctx.add(res, ab_cur);
        }
    });

    res
}

/// `llm_graph_context::build_lora_mm_id` (llama-graph.cpp:1545-1581): the
/// `ggml_mul_mat_id` twin of [`lora_mm_s`] used by the MoE experts. `w_s` (the
/// per-expert scale, :1552-1559) has no port counterpart and is not passed.
///
/// The C inlines the scale here instead of calling `get_scale` (:1566-1568);
/// the formula is the same one [`LoraWeight::get_scale`] implements.
pub fn lora_mm_id(ctx: &mut Context, w: TensorId, cur: TensorId, ids: TensorId) -> TensorId {
    let mut res = ctx.mul_mat_id(w, cur, ids);

    // a7b94df2c (llama-graph.cpp:1604-1606): the mul_mat_id twin of the
    // NVFP4 BF16 accumulator hint above.
    if ctx.ty(w) == GgmlType::Nvfp4 {
        ctx.prec_set_acc(res, GgmlPrec::Bf16);
    }

    if !has_active_loras() {
        return res;
    }

    let name = ctx.name(w).to_string();

    with_active_loras(|loras| {
        for (adapter, adapter_scale) in loras.iter() {
            let Some(lw) = adapter.get_weight(&name) else {
                continue;
            };

            let rank = ctx.ne(lw.b)[0] as f32;
            let alpha = adapter.alpha;
            let scale = if alpha != 0.0 {
                *adapter_scale * alpha / rank
            } else {
                *adapter_scale
            };

            // ggml_mul_mat_id(ctx0, lw->b, ggml_mul_mat_id(ctx0, lw->a, cur, ids), ids)
            let a_cur = ctx.mul_mat_id(lw.a, cur, ids);
            let mut ab_cur = ctx.mul_mat_id(lw.b, a_cur, ids);

            ab_cur = ctx.scale(ab_cur, scale);
            res = ctx.add(res, ab_cur);
        }
    });

    res
}

/// `build_inp_embd`'s lora branch (llama-graph.cpp:2389-2405): the token
/// embeddings are *rows* of the adapter's `lora_a`, so the delta is
/// `scale * (lora_b @ get_rows(lora_a, tokens))` with `lora_b`
/// non-transposed (the flipped layout the loader validates at
/// llama-adapter.cpp:359-363).
pub fn lora_embd(
    ctx: &mut Context,
    tok_embd: TensorId,
    cur: TensorId,
    tokens: TensorId,
) -> TensorId {
    if !has_active_loras() {
        return cur;
    }

    let name = ctx.name(tok_embd).to_string();
    let mut cur = cur;

    with_active_loras(|loras| {
        for (adapter, adapter_scale) in loras.iter() {
            let Some(lw) = adapter.get_weight(&name) else {
                continue;
            };

            let scale = lw.get_scale(ctx, adapter.alpha, *adapter_scale);

            // ggml_scale(ctx0, ggml_mul_mat(lw->b, ggml_get_rows(ctx0, lw->a, tokens)), scale)
            let a_rows = ctx.get_rows(lw.a, tokens);
            let ab = ctx.mul_mat(lw.b, a_rows);
            let delta = ctx.scale(ab, scale);

            cur = ctx.add(cur, delta);
        }
    });

    cur
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ggml::gguf_write::GgufWriter;
    use ggml::types::GgmlType;

    /// Write a tiny adapter GGUF to a temp path and return it. Tensors are F32
    /// `[n0, n1]` with deterministic contents; `extra_kv` appends metadata.
    fn write_adapter(
        tag: &str,
        arch: &str,
        adapter_type: &str,
        alpha: Option<f32>,
        tensors: &[(&str, [i64; 4])],
        extra_kv: &[(&str, Value)],
    ) -> String {
        let path =
            std::env::temp_dir().join(format!("lora-test-{tag}-{}.gguf", std::process::id()));
        let mut w = GgufWriter::new(32);
        w.set_kv("general.type", Value::String("adapter".into()));
        w.set_kv("general.architecture", Value::String(arch.into()));
        w.set_kv("adapter.type", Value::String(adapter_type.into()));
        if let Some(a) = alpha {
            w.set_kv("adapter.lora.alpha", Value::F32(a));
        }
        for (k, v) in extra_kv {
            w.set_kv(k, v.clone());
        }
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        for (name, ne) in tensors {
            w.add_tensor(name, GgmlType::F32, *ne);
            let n: i64 = ne.iter().product();
            payloads.push(
                (0..n)
                    .map(|i| (i as f32 * 0.25) - 1.0)
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            );
        }
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        let mut buf = Vec::new();
        w.write(&mut buf, &refs).unwrap();
        std::fs::write(&path, &buf).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// `get_scale` (llama-adapter.h:53-57): alpha != 0 → `scale * alpha / rank`,
    /// alpha == 0 → `scale`, rank = `b->ne[0]`.
    #[test]
    fn lora_weight_get_scale() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 8, 4); // [n_embd, rank]
        let b = ctx.new_tensor_2d(GgmlType::F32, 4, 8); // [rank, n_embd]
        let w = LoraWeight::new(a, b, "x.lora_a".into(), "x.lora_b".into());
        assert_eq!(w.get_scale(&ctx, 4.0, 1.0), 4.0 * 1.0 / 4.0);
        assert_eq!(w.get_scale(&ctx, 4.0, 0.5), 0.5 * 4.0 / 4.0);
        // alpha == 0 keeps the adapter scale (the C's `alpha ? … : …`)
        assert_eq!(w.get_scale(&ctx, 0.0, 2.5), 2.5);
        assert_eq!(w.get_scale(&ctx, 0.0, 0.0), 0.0);
    }

    /// Metadata contract (llama-adapter.cpp:204-241): good file loads, the
    /// `general.type` / `adapter.type` / arch checks throw the C's messages,
    /// and `adapter.lora.alpha` lands in `alpha`.
    #[test]
    fn adapter_metadata_contract() {
        let dims = |name: &str| -> Option<[i64; 4]> {
            match name {
                "blk.0.attn_q.weight" => Some([8, 8, 1, 1]),
                _ => None,
            }
        };

        let path = write_adapter(
            "meta-ok",
            "qwen2",
            "lora",
            Some(16.0),
            &[
                ("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.0.attn_q.weight.lora_b", [4, 8, 1, 1]),
            ],
            &[],
        );
        let mut ctx = Context::new();
        let a =
            adapter_lora_init_impl(&mut ctx, LlmArch::QWEN2, &dims, &Gguf::open(&path).unwrap())
                .expect("valid adapter");
        assert_eq!(a.alpha, 16.0);
        assert_eq!(a.ab_map.len(), 1);
        assert!(a.get_weight("blk.0.attn_q.weight").is_some());
        assert!(a.get_weight("blk.0.attn_k.weight").is_none());
        assert_eq!(a.get_n_nodes(), 6);
        // the pair carries the file tensor names and the file payload
        let w = a.get_weight("blk.0.attn_q.weight").unwrap();
        assert_eq!(ctx.name(w.a), "blk.0.attn_q.weight.lora_a");
        assert_eq!(ctx.name(w.b), "blk.0.attn_q.weight.lora_b");
        assert_eq!(*ctx.ne(w.a), [8, 4, 1, 1]);
        let av: &[f32] = bytemuck::cast_slice(ctx.data_bytes(w.a).unwrap());
        assert_eq!(av[..4], [-1.0, -0.75, -0.5, -0.25]);
        // metadata accessors (:460-497); general.* is a non-array key
        assert_eq!(a.meta_val_str("general.architecture"), Some("qwen2"));
        assert_eq!(a.meta_val_str("adapter.lora.alpha"), Some("16.000000"));
        assert_eq!(a.meta_val_str("nope"), None);
        assert_eq!(a.meta_count(), 4);
        assert_eq!(a.meta_key_by_index(0), Some("general.type"));
        assert_eq!(a.meta_val_str_by_index(0), Some("adapter"));
        assert_eq!(a.meta_key_by_index(a.meta_count()), None);
        std::fs::remove_file(&path).ok();

        // wrong general.type (:205-208)
        let path = write_adapter("meta-gt", "qwen2", "lora", Some(1.0), &[], &[]);
        let mut g = Gguf::open(&path).unwrap();
        g.kv[0].1 = Value::String("model".into());
        let e = adapter_lora_init_impl(&mut Context::new(), LlmArch::QWEN2, &dims, &g).unwrap_err();
        assert_eq!(e, "expect general.type to be 'adapter', but got: model");
        std::fs::remove_file(&path).ok();

        // arch mismatch (:210-214)
        let path = write_adapter("meta-arch", "llama", "lora", Some(1.0), &[], &[]);
        let e = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap_err();
        assert_eq!(e, "model arch and LoRA arch mismatch");
        std::fs::remove_file(&path).ok();

        // wrong adapter.type (:216-219): control vectors are not accepted here
        let path = write_adapter("meta-at", "qwen2", "control_vector", Some(1.0), &[], &[]);
        let e = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            e,
            "expect adapter.type to be 'lora', but got: control_vector"
        );
        std::fs::remove_file(&path).ok();

        // missing alpha → 0.0 (:199-202 get_kv_f32 default)
        let path = write_adapter(
            "meta-noalpha",
            "qwen2",
            "lora",
            None,
            &[
                ("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.0.attn_q.weight.lora_b", [4, 8, 1, 1]),
            ],
            &[],
        );
        let a = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap();
        assert_eq!(a.alpha, 0.0);
        std::fs::remove_file(&path).ok();
    }

    /// Shape validation + suffix handling (llama-adapter.cpp:274-297, :358-371):
    /// every C error text is reproduced, `_norm.weight` is skipped, and an
    /// unknown suffix aborts.
    #[test]
    fn adapter_shape_and_suffix_checks() {
        let dims = |name: &str| -> Option<[i64; 4]> {
            match name {
                "blk.0.attn_q.weight" => Some([8, 8, 1, 1]),
                "token_embd.weight" => Some([8, 100, 1, 1]),
                _ => None,
            }
        };
        let load = |tensors: &[(&str, [i64; 4])]| {
            let path = write_adapter("shape", "qwen2", "lora", Some(1.0), tensors, &[]);
            let r = adapter_lora_init_impl(
                &mut Context::new(),
                LlmArch::QWEN2,
                &dims,
                &Gguf::open(&path).unwrap(),
            )
            .map(|_| ());
            std::fs::remove_file(&path).ok();
            r
        };

        // unknown suffix (:294-296)
        assert_eq!(
            load(&[("blk.0.attn_q.weight.lora_c", [8, 4, 1, 1])]).unwrap_err(),
            "LoRA tensor 'blk.0.attn_q.weight.lora_c' has unexpected suffix"
        );
        // `_norm.weight` is skipped, the (empty) adapter still loads
        load(&[("blk.0.attn_norm.weight_norm.weight", [8, 1, 1, 1])]).unwrap();
        // missing component (:328-330)
        assert_eq!(
            load(&[("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1])]).unwrap_err(),
            "LoRA tensor pair for 'blk.0.attn_q.weight' is missing one component"
        );
        // not in the base model (:333-336)
        assert_eq!(
            load(&[
                ("blk.9.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.9.attn_q.weight.lora_b", [4, 8, 1, 1]),
            ])
            .unwrap_err(),
            "LoRA tensor 'blk.9.attn_q.weight' does not exist in base model \
             (hint: maybe wrong base model?)"
        );
        // wrong shape (:364-367): model ne[1] = 8 but lora_b ne[1] = 4
        assert_eq!(
            load(&[
                ("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.0.attn_q.weight.lora_b", [4, 4, 1, 1]),
            ])
            .unwrap_err(),
            "tensor 'blk.0.attn_q.weight' has incorrect shape (hint: maybe wrong base model?)"
        );
        // rank mismatch → "not transposed" (:368-370)
        assert_eq!(
            load(&[
                ("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.0.attn_q.weight.lora_b", [8, 8, 1, 1]),
            ])
            .unwrap_err(),
            "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)"
        );
        // token_embd uses the flipped layout (:359-363): b ne[1] == n_embd,
        // a ne[1] == n_vocab, and the rank is a/b ne[0] (A and B are flipped
        // because build_inp_embd gathers *rows* of a)
        load(&[
            ("token_embd.weight.lora_a", [4, 100, 1, 1]),
            ("token_embd.weight.lora_b", [4, 8, 1, 1]),
        ])
        .unwrap();
        assert_eq!(
            load(&[
                ("token_embd.weight.lora_a", [8, 4, 1, 1]),
                ("token_embd.weight.lora_b", [4, 8, 1, 1]),
            ])
            .unwrap_err(),
            "tensor 'token_embd.weight' has incorrect shape (hint: maybe wrong base model?)"
        );
    }

    /// aLoRA metadata (:224-241): the array must be uint32; the values land in
    /// `alora_invocation_tokens` as i32.
    #[test]
    fn adapter_alora_invocation_tokens() {
        let dims = |_: &str| None;
        let key = "adapter.alora.invocation_tokens";
        let path = write_adapter(
            "alora",
            "qwen2",
            "lora",
            Some(1.0),
            &[],
            &[(
                key,
                Value::Array(
                    GgufType::Uint32,
                    vec![Value::U32(3), Value::U32(7), Value::U32(11)],
                ),
            )],
        );
        let a = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap();
        assert_eq!(a.alora_invocation_tokens, vec![3, 7, 11]);
        // arrays never enter gguf_kv (:184-186)
        assert!(a.meta_val_str(key).is_none());
        std::fs::remove_file(&path).ok();

        // wrong element type (:231-233)
        let path = write_adapter(
            "alora-bad",
            "qwen2",
            "lora",
            Some(1.0),
            &[],
            &[(key, Value::Array(GgufType::Int32, vec![Value::I32(3)]))],
        );
        let e = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap_err();
        assert_eq!(e, format!("invalid gguf element type for {key}"));
        std::fs::remove_file(&path).ok();

        // wrong value type (:227-229)
        let path = write_adapter(
            "alora-scalar",
            "qwen2",
            "lora",
            Some(1.0),
            &[],
            &[(key, Value::U32(3))],
        );
        let e = adapter_lora_init_impl(
            &mut Context::new(),
            LlmArch::QWEN2,
            &dims,
            &Gguf::open(&path).unwrap(),
        )
        .unwrap_err();
        assert_eq!(e, format!("invalid gguf type for {key}"));
        std::fs::remove_file(&path).ok();
    }

    /// The active-set plumbing (llama-context.cpp:1332-1374): zero scales are
    /// dropped, a cleared set applies nothing, and the "same" check compares
    /// scales like the C.
    #[test]
    fn set_adapters_lora_scale_plumbing() {
        let mut ctx = Context::new();
        let path = write_adapter(
            "active",
            "qwen2",
            "lora",
            Some(8.0),
            &[
                ("blk.0.attn_q.weight.lora_a", [8, 4, 1, 1]),
                ("blk.0.attn_q.weight.lora_b", [4, 8, 1, 1]),
            ],
            &[],
        );
        let dims = |name: &str| -> Option<[i64; 4]> {
            match name {
                "blk.0.attn_q.weight" => Some([8, 8, 1, 1]),
                _ => None,
            }
        };
        let a = load_adapter_lora(&mut ctx, LlmArch::QWEN2, &dims, &path).unwrap();

        // scale 0 → the adapter is not inserted at all (:1342-1344)
        set_adapters_lora(&ctx, &[(a.clone(), 0.0)]).unwrap();
        with_active_loras(|l| assert!(l.is_empty()));

        set_adapters_lora(&ctx, &[(a.clone(), 1.5)]).unwrap();
        with_active_loras(|l| {
            assert_eq!(l.len(), 1);
            assert_eq!(l[0].1, 1.5);
        });
        assert!(adapters_lora_are_same(&[(a.clone(), 1.5)]));
        assert!(!adapters_lora_are_same(&[(a.clone(), 1.0)]));
        assert!(adapters_lora_are_same(&[
            (a.clone(), 0.0),
            (a.clone(), 1.5)
        ]));
        assert!(!adapters_lora_are_same(&[]));

        // the C's insert semantics (:1343): the same adapter twice keeps the
        // first scale, a second load of the same file is a second adapter
        set_adapters_lora(&ctx, &[(a.clone(), 1.5), (a.clone(), 2.5)]).unwrap();
        with_active_loras(|l| {
            assert_eq!(l.len(), 1);
            assert_eq!(l[0].1, 1.5);
        });
        let a2 = load_adapter_lora(&mut ctx, LlmArch::QWEN2, &dims, &path).unwrap();
        set_adapters_lora(&ctx, &[(a.clone(), 1.5), (a2.clone(), 2.5)]).unwrap();
        with_active_loras(|l| assert_eq!(l.len(), 2));
        std::fs::remove_file(&path).ok();

        // a truncated context (graph watermark) is rejected instead of jumping
        // through a dangling TensorId: an empty context is below the adapter's
        // load mark
        let small = Context::new();
        let r = set_adapters_lora(&small, &[(a.clone(), 1.0)]);
        assert!(r.is_err());

        clear_adapter_lora();
        with_active_loras(|l| assert!(l.is_empty()));
    }

    /// `lora_mm` numerics vs a hand-computed f64 reference (the identity
    /// `a @ b * scale + w @ x`): a transposed or scale-blind implementation
    /// cannot pass this.
    #[test]
    fn lora_mm_matches_f64_reference() {
        let dims = |name: &str| -> Option<[i64; 4]> {
            match name {
                "blk.0.attn_q.weight" => Some([4, 4, 1, 1]),
                _ => None,
            }
        };
        // rank 2, alpha 6 → scale = adapter_scale * 6 / 2 = 3 * adapter_scale
        let mut data_a: Vec<f32> = Vec::new();
        let mut data_b: Vec<f32> = Vec::new();
        for i in 0..8 {
            data_a.push(((i as f32) * 0.31).sin());
        }
        for i in 0..8 {
            data_b.push(((i as f32) * 0.17).cos());
        }

        let path = std::env::temp_dir().join(format!("lora-mm-{}.gguf", std::process::id()));
        {
            let mut w = GgufWriter::new(32);
            w.set_kv("general.type", Value::String("adapter".into()));
            w.set_kv("general.architecture", Value::String("qwen2".into()));
            w.set_kv("adapter.type", Value::String("lora".into()));
            w.set_kv("adapter.lora.alpha", Value::F32(6.0));
            w.add_tensor("blk.0.attn_q.weight.lora_a", GgmlType::F32, [4, 2, 1, 1]);
            w.add_tensor("blk.0.attn_q.weight.lora_b", GgmlType::F32, [2, 4, 1, 1]);
            let ab: Vec<u8> = data_a.iter().flat_map(|x| x.to_le_bytes()).collect();
            let bb: Vec<u8> = data_b.iter().flat_map(|x| x.to_le_bytes()).collect();
            let mut buf = Vec::new();
            w.write(&mut buf, &[&ab, &bb]).unwrap();
            std::fs::write(&path, &buf).unwrap();
        }

        let mut ctx = Context::new();
        let adapter =
            load_adapter_lora(&mut ctx, LlmArch::QWEN2, &dims, path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        // base weight w [4,4] and input x [4,3] in the same context
        let w_id = ctx.new_tensor_2d(GgmlType::F32, 4, 4);
        let x_id = ctx.new_tensor_2d(GgmlType::F32, 4, 3);
        ctx.set_name(w_id, "blk.0.attn_q.weight");
        ctx.arena_resize_tensor(w_id);
        ctx.arena_resize_tensor(x_id);
        let wv: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.07).cos()).collect();
        let xv: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.11).sin()).collect();
        ctx.with_f32_mut(w_id, |p| p.copy_from_slice(&wv)).unwrap();
        ctx.with_f32_mut(x_id, |p| p.copy_from_slice(&xv)).unwrap();

        // base + lora at adapter scale 2.0, alpha 6, rank 2 → scale = 6.0
        set_adapters_lora(&ctx, &[(adapter, 2.0)]).unwrap();
        let out = lora_mm(&mut ctx, w_id, x_id);

        // f64 reference: out = w @ x + scale * b @ (a @ x)
        let scale = 2.0f64 * 6.0 / 2.0;
        let mut want = vec![0f64; 4 * 3];
        for row in 0..4 {
            for t in 0..3 {
                let mut acc = 0f64;
                for k in 0..4 {
                    acc += wv[row * 4 + k] as f64 * xv[t * 4 + k] as f64;
                }
                let mut delta = 0f64;
                for r in 0..2 {
                    let mut ar = 0f64;
                    for k in 0..4 {
                        ar += data_a[r * 4 + k] as f64 * xv[t * 4 + k] as f64;
                    }
                    delta += data_b[row * 2 + r] as f64 * ar;
                }
                want[row + t * 4] = acc + scale * delta;
            }
        }

        let mut g = ggml::graph::Graph::new(64);
        g.build_forward(&ctx, out);
        ggml::compute::graph_compute(&mut ctx, &mut g, 1);
        let got: Vec<f32> = bytemuck::cast_slice(ctx.data_bytes(out).unwrap()).to_vec();

        for i in 0..12 {
            assert!(
                (got[i] as f64 - want[i]).abs() < 1e-5,
                "element {i}: {} vs {}",
                got[i],
                want[i]
            );
        }

        // the lora nodes must actually be in the graph: res = MulMat(w,x) →
        // MulMat(a,x) → MulMat(b,·) → Scale → Add, i.e. the C's lora_mm + add
        // pattern (llama-graph.cpp:1533-1539)
        assert_eq!(ctx.op(out), ggml::tensor::GgmlOp::Add);
        let base = ctx.src(out)[0].unwrap();
        assert_eq!(ctx.op(base), ggml::tensor::GgmlOp::MulMat);
        let s = ctx.src(base);
        assert_eq!((s[0], s[1]), (Some(w_id), Some(x_id)));
        let scaled = ctx.src(out)[1].unwrap();
        assert_eq!(ctx.op(scaled), ggml::tensor::GgmlOp::Scale);
        let bb = ctx.src(scaled)[0].unwrap();
        assert_eq!(ctx.op(bb), ggml::tensor::GgmlOp::MulMat);
        let aa = ctx.src(bb)[1].unwrap();
        assert_eq!(ctx.op(aa), ggml::tensor::GgmlOp::MulMat);
        assert_eq!(
            ctx.name(ctx.src(aa)[0].unwrap()),
            "blk.0.attn_q.weight.lora_a"
        );
        assert_eq!(
            ctx.name(ctx.src(bb)[0].unwrap()),
            "blk.0.attn_q.weight.lora_b"
        );

        // cleared → exactly `w @ x`, no extra nodes (bit-for-bit base model)
        clear_adapter_lora();
        let out2 = lora_mm(&mut ctx, w_id, x_id);
        assert_eq!(ctx.op(out2), ggml::tensor::GgmlOp::MulMat);
        let mut g = ggml::graph::Graph::new(4);
        g.build_forward(&ctx, out2);
        ggml::compute::graph_compute(&mut ctx, &mut g, 1);
        let base: Vec<f32> = bytemuck::cast_slice(ctx.data_bytes(out2).unwrap()).to_vec();
        for i in 0..12 {
            let mut acc = 0f64;
            let row = i % 4;
            let t = i / 4;
            for k in 0..4 {
                acc += wv[row * 4 + k] as f64 * xv[t * 4 + k] as f64;
            }
            assert!((base[i] as f64 - acc).abs() < 1e-5);
        }
    }

    /// a7b94df2c (llama-graph.cpp:1560-1562 / 1604-1606): NVFP4 weights flag
    /// the mul_mat result with the BF16 accumulator hint
    /// (`ggml_prec_set_acc(res, GGML_PREC_BF16)` → op_params[0] == 15, the
    /// exact bytes parity/prec_acc_ref.txt pins on the reference .so); any
    /// other weight type leaves the slot untouched.
    #[test]
    fn lora_mm_nvfp4_prec_acc() {
        let mut ctx = Context::new();
        // NVFP4 w [64,4] (block size 64), cur [64,2] F32
        let w4 = ctx.new_tensor_2d(GgmlType::Nvfp4, 64, 4);
        let cur = ctx.new_tensor_2d(GgmlType::F32, 64, 2);
        let out = lora_mm(&mut ctx, w4, cur);
        assert_eq!(ctx.op(out), ggml::tensor::GgmlOp::MulMat);
        assert_eq!(ctx.op_params(out)[0], 15); // GGML_PREC_BF16

        // F32 weight: no hint
        let wf = ctx.new_tensor_2d(GgmlType::F32, 64, 4);
        let out2 = lora_mm(&mut ctx, wf, cur);
        assert_eq!(ctx.op_params(out2)[0], 0);

        // mul_mat_id twin: experts [64,4,2] NVFP4, cur3 [64,2,2], ids [2,2]
        let we4 = ctx.new_tensor_3d(GgmlType::Nvfp4, 64, 4, 2);
        let cur3 = ctx.new_tensor_3d(GgmlType::F32, 64, 2, 2);
        let ids = ctx.new_tensor_2d(GgmlType::I32, 2, 2);
        let out3 = lora_mm_id(&mut ctx, we4, cur3, ids);
        assert_eq!(ctx.op(out3), ggml::tensor::GgmlOp::MulMatId);
        assert_eq!(ctx.op_params(out3)[0], 15); // GGML_PREC_BF16

        // F32 experts: no hint
        let wef = ctx.new_tensor_3d(GgmlType::F32, 64, 4, 2);
        let out4 = lora_mm_id(&mut ctx, wef, cur3, ids);
        assert_eq!(ctx.op_params(out4)[0], 0);
    }
}
