//! GGUF container reader — port of ggml/src/gguf.cpp (`gguf_init_from_file`).
//!
//! Format (v3): magic "GGUF", u32 version, u64 tensor_count, u64 kv_count,
//! kv pairs, tensor infos, then (aligned) tensor data. Little-endian throughout.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::types::GgmlType;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
pub const GGUF_KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

/// `LLM_KV_SPLIT_*` (llama-arch.cpp:338-340) — the gguf-split convention.
pub const LLM_KV_SPLIT_NO: &str = "split.no";
pub const LLM_KV_SPLIT_COUNT: &str = "split.count";
pub const LLM_KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// `llama_split_path` (llama.cpp:544-563) — `{prefix}-{no+1:05}-{count:05}.gguf`.
pub fn split_path(prefix: &str, split_no: i32, split_count: i32) -> String {
    // SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf"
    format!("{prefix}-{:05}-of-{:05}.gguf", split_no + 1, split_count)
}

/// `llama_split_prefix` (llama.cpp:569-593): the path minus the
/// `-{no+1:05}-of-{count:05}.gguf` postfix, `None` when it does not match.
pub fn split_prefix(path: &str, split_no: i32, split_count: i32) -> Option<String> {
    let postfix = format!("-{:05}-of-{:05}.gguf", split_no + 1, split_count);
    if path.len() <= postfix.len() {
        return None;
    }
    let size_prefix = path.len() - postfix.len();
    if path.ends_with(&postfix) {
        Some(path[..size_prefix].to_string())
    } else {
        None
    }
}

/// `enum gguf_type` — wire discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgufType::*;
        Some(match v {
            0 => Uint8,
            1 => Int8,
            2 => Uint16,
            3 => Int16,
            4 => Uint32,
            5 => Int32,
            6 => Float32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => Uint64,
            11 => Int64,
            12 => Float64,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        use GgufType::*;
        match self {
            Uint8 => "uint8",
            Int8 => "int8",
            Uint16 => "uint16",
            Int16 => "int16",
            Uint32 => "uint32",
            Int32 => "int32",
            Float32 => "float32",
            Bool => "bool",
            String => "string",
            Array => "array",
            Uint64 => "uint64",
            Int64 => "int64",
            Float64 => "float64",
        }
    }
}

/// Scalar payload; arrays hold their element values.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufType, Vec<Value>),
}

impl Value {
    pub fn type_(&self) -> GgufType {
        match self {
            Value::U8(_) => GgufType::Uint8,
            Value::I8(_) => GgufType::Int8,
            Value::U16(_) => GgufType::Uint16,
            Value::I16(_) => GgufType::Int16,
            Value::U32(_) => GgufType::Uint32,
            Value::I32(_) => GgufType::Int32,
            Value::F32(_) => GgufType::Float32,
            Value::Bool(_) => GgufType::Bool,
            Value::String(_) => GgufType::String,
            Value::U64(_) => GgufType::Uint64,
            Value::I64(_) => GgufType::Int64,
            Value::F64(_) => GgufType::Float64,
            Value::Array(t, _) => GgufType::Array, // element type kept inside
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(x) => Some(*x),
            Value::F64(x) => Some(*x as f32),
            Value::U8(x) => Some(*x as f32),
            Value::I8(x) => Some(*x as f32),
            Value::U16(x) => Some(*x as f32),
            Value::I16(x) => Some(*x as f32),
            Value::U32(x) => Some(*x as f32),
            Value::I32(x) => Some(*x as f32),
            Value::U64(x) => Some(*x as f32),
            Value::I64(x) => Some(*x as f32),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(x) => Some(*x),
            Value::U8(x) => Some(*x as u32),
            Value::U16(x) => Some(*x as u32),
            Value::I8(x) => Some(*x as u32),
            Value::I16(x) => Some(*x as u32),
            Value::I32(x) => u32::try_from(*x).ok(),
            Value::U64(x) => u32::try_from(*x).ok(),
            Value::I64(x) => u32::try_from(*x).ok(),
            _ => None,
        }
    }
    /// `gguf_get_val_u16` — exact-type read (GGUFMeta::GKV type check).
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Value::U16(x) => Some(*x),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<(&GgufType, &Vec<Value>)> {
        match self {
            Value::Array(t, v) => Some((t, v)),
            _ => None,
        }
    }
    /// f32 slice for numeric arrays (ints widened, per llama.cpp implicit conversions in loaders).
    pub fn as_f32_slice(&self) -> Option<Vec<f32>> {
        let (t, v) = self.as_array()?;
        match t {
            GgufType::Float32 => Some(v.iter().filter_map(|x| x.as_f32()).collect()),
            _ => {
                let mut out = Vec::with_capacity(v.len());
                for x in v {
                    out.push(x.as_f32()?);
                }
                Some(out)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0] = fastest-varying (row length)
    pub ne: [i64; 4],
    pub ty: GgmlType,
    /// byte offset of this tensor inside the data section
    pub offset: u64,
    /// byte strides (nb), computed like gguf.cpp does
    pub nb: [u64; 4],
    /// index of the file (split) this tensor's storage lives in — 0 for a
    /// single-file GGUF; set by [`Gguf::open`]'s split pass, mirroring
    /// `weights_map` over `files[]` of llama-model-loader.cpp:583-651
    pub part: u16,
}

impl TensorInfo {
    pub fn n_elements(&self) -> i64 {
        self.ne.iter().product()
    }
    pub fn size_bytes(&self) -> u64 {
        self.ty.type_size() as u64 * (self.n_elements() as u64 / self.ty.blck_size() as u64)
    }
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    /// key → value, insertion order preserved (the *first split's* kv for a
    /// multi-part open — gguf-split writes all metadata into part 1 only)
    pub kv: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// absolute file offset where the data section starts (part 0's)
    pub data_offset: u64,
    /// one entry per source file; `tensors[i].part` indexes this
    parts: Vec<GgufPart>,
}

/// The mmap + data-section offset of one source file of the container.
#[derive(Debug)]
struct GgufPart {
    data_offset: u64,
    data: Arc<Mmap>,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated(&'static str),
    BadType(u32),
    BadArrayType(u32),
    DuplicateKey(String),
    DuplicateTensor(String),
    NameTooLong(String),
    BadShape(String),
    KeyNotFound(String),
    BadCast { key: String, want: &'static str, got: &'static str },
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {e}"),
            GgufError::BadMagic => write!(f, "not a GGUF file (bad magic)"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v} (max {GGUF_VERSION})"),
            GgufError::Truncated(what) => write!(f, "truncated file while reading {what}"),
            GgufError::BadType(v) => write!(f, "invalid gguf metadata type {v}"),
            GgufError::BadArrayType(v) => write!(f, "invalid gguf array element type {v}"),
            GgufError::DuplicateKey(k) => write!(f, "duplicate key '{k}'"),
            GgufError::DuplicateTensor(n) => write!(f, "duplicate tensor name '{n}'"),
            GgufError::NameTooLong(n) => write!(f, "tensor name too long: '{n}'"),
            GgufError::BadShape(n) => write!(f, "tensor '{n}' has invalid shape/type"),
            GgufError::KeyNotFound(k) => write!(f, "key not found: '{k}'"),
            GgufError::BadCast { key, want, got } => write!(f, "key '{key}' is {got}, expected {want}"),
        }
    }
}
impl std::error::Error for GgufError {}
impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.buf.len() {
            return Err(GgufError::Truncated(what));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self, what: &'static str) -> Result<u8, GgufError> {
        Ok(self.take(1, what)?[0])
    }
    fn i8(&mut self, what: &'static str) -> Result<i8, GgufError> {
        Ok(self.take(1, what)?[0] as i8)
    }
    fn u16(&mut self, what: &'static str) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn i16(&mut self, what: &'static str) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn i32(&mut self, what: &'static str) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn f32(&mut self, what: &'static str) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn i64(&mut self, what: &'static str) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn f64(&mut self, what: &'static str) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn string(&mut self, what: &'static str) -> Result<String, GgufError> {
        let n = self.u64(what)? as usize;
        if n > self.buf.len() - self.pos {
            return Err(GgufError::Truncated(what));
        }
        let bytes = self.take(n, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::Truncated("invalid utf-8 in string"))
    }
    fn value(&mut self, ty: GgufType, what: &'static str) -> Result<Value, GgufError> {
        Ok(match ty {
            GgufType::Uint8 => Value::U8(self.u8(what)?),
            GgufType::Int8 => Value::I8(self.i8(what)?),
            GgufType::Uint16 => Value::U16(self.u16(what)?),
            GgufType::Int16 => Value::I16(self.i16(what)?),
            GgufType::Uint32 => Value::U32(self.u32(what)?),
            GgufType::Int32 => Value::I32(self.i32(what)?),
            GgufType::Float32 => Value::F32(self.f32(what)?),
            GgufType::Bool => Value::Bool(self.u8(what)? != 0),
            GgufType::String => Value::String(self.string(what)?),
            GgufType::Uint64 => Value::U64(self.u64(what)?),
            GgufType::Int64 => Value::I64(self.i64(what)?),
            GgufType::Float64 => Value::F64(self.f64(what)?),
            GgufType::Array => {
                let et_raw = self.u32(what)?;
                let et = GgufType::from_u32(et_raw).ok_or(GgufError::BadArrayType(et_raw))?;
                let n = self.u64(what)? as usize;
                if matches!(et, GgufType::Array) {
                    return Err(GgufError::BadArrayType(et_raw));
                }
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(et, what)?);
                }
                Value::Array(et, items)
            }
        })
    }
}

impl Gguf {
    /// Open a GGUF, transparently following the `split.*` convention when the
    /// file declares itself part 1 of a multi-part model — the port-side
    /// equivalent of the split pass in the `llama_model_loader` constructor
    /// (llama-model-loader.cpp:596-669): part 0's kv becomes the metadata,
    /// every part's tensors are appended in order (each pointing into its own
    /// mmap), duplicate tensor names and a mismatching
    /// `split.tensors.count` are errors.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let first = Self::open_single(path)?;

        // uint16_t n_split = 0; get_key(LLM_KV_SPLIT_COUNT, n_split, false)
        // (llama-model-loader.cpp:596-597 — absent key is not an error)
        let Some(n_split) = first.find_key(LLM_KV_SPLIT_COUNT) else {
            return Ok(first);
        };
        let Some(n_split) = n_split.as_u16() else {
            return Err(GgufError::BadCast {
                key: LLM_KV_SPLIT_COUNT.into(),
                want: "uint16",
                got: n_split.type_().name(),
            });
        };
        if n_split <= 1 {
            return Ok(first);
        }

        // get_key(LLM_KV_SPLIT_NO, idx) — required, and it must be 0
        // (llama-model-loader.cpp:600-606)
        let idx = first.find_key(LLM_KV_SPLIT_NO).and_then(|v| v.as_u16()).ok_or(
            GgufError::KeyNotFound(LLM_KV_SPLIT_NO.into()),
        )?;
        if idx != 0 {
            return Err(GgufError::BadShape(format!(
                "illegal split file idx: {idx} (file: {}), model must be loaded with the first split",
                path.display()
            )));
        }

        // llama_get_list_splits (llama-model-loader.cpp:82-105)
        let prefix = split_prefix(&path.to_string_lossy(), 0, n_split as i32).ok_or(
            GgufError::BadShape(format!("invalid split file name: {}", path.display())),
        )?;

        let mut parts: Vec<GgufPart> = Vec::with_capacity(n_split as usize);
        let Gguf { version, kv, tensors: first_tensors, alignment, data_offset, parts: first_parts } =
            first;
        parts.push(first_parts.into_iter().next().unwrap());
        let mut g = Gguf {
            version,
            kv,
            tensors: first_tensors,
            alignment,
            data_offset,
            parts,
        };

        // load other splits (llama-model-loader.cpp:622-651)
        for idx in 1..n_split {
            let fname_split = split_path(&prefix, idx as i32, n_split as i32);
            let part = Self::open_single(&fname_split).map_err(|_| {
                GgufError::BadShape(format!(
                    "llama_model_loader: failed to load GGUF split from {fname_split}"
                ))
            })?;
            let idx_gguf = part.find_key(LLM_KV_SPLIT_NO).and_then(|v| v.as_u16()).ok_or(
                GgufError::BadShape(format!("missing key {LLM_KV_SPLIT_NO} in GGUF split {fname_split}")),
            )?;
            if idx_gguf != idx {
                return Err(GgufError::BadShape(format!(
                    "invalid split file idx: {idx_gguf} (file: {fname_split}), expected {idx}"
                )));
            }
            let Gguf { tensors: part_tensors, parts: part_parts, .. } = part;
            let part_no = g.parts.len() as u16;
            g.parts.extend(part_parts);
            for mut t in part_tensors {
                if g.tensors.iter().any(|x| x.name == t.name) {
                    return Err(GgufError::DuplicateTensor(t.name));
                }
                t.part = part_no;
                g.tensors.push(t);
            }
        }

        // get_key(LLM_KV_SPLIT_TENSORS_COUNT, n_tensors) + sanity check
        // (llama-model-loader.cpp:663-670)
        let n_tensors = g
            .find_key(LLM_KV_SPLIT_TENSORS_COUNT)
            .and_then(|v| v.as_u32())
            .ok_or(GgufError::KeyNotFound(LLM_KV_SPLIT_TENSORS_COUNT.into()))?;
        if n_tensors as usize != g.tensors.len() {
            return Err(GgufError::BadShape(format!(
                "corrupted model: {} tensors expected but {} found",
                n_tensors,
                g.tensors.len()
            )));
        }

        Ok(g)
    }

    /// Open exactly one GGUF file, ignoring any `split.*` metadata (the plain
    /// `gguf_init_from_file` — what tools/gguf-split operate on).
    pub fn open_single(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        // ponytail: SAFETY — model files are read-only for us; concurrent writers are out of scope
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_bytes(Arc::new(mmap))
    }

    pub fn from_bytes(data: Arc<Mmap>) -> Result<Self, GgufError> {
        let mut c = Cursor { buf: &data, pos: 0 };
        let magic = c.take(4, "magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = c.u32("version")?;
        if version > GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = c.u64("tensor count")?;
        let n_kv = c.u64("kv count")?;

        let mut kv: Vec<(String, Value)> = Vec::with_capacity(n_kv.min(1 << 16) as usize);
        for i in 0..n_kv {
            let key = c.string(&format_kv_what(i))?;
            if key.is_empty() {
                return Err(GgufError::Truncated("empty kv key"));
            }
            if kv.iter().any(|(k, _)| *k == key) {
                return Err(GgufError::DuplicateKey(key));
            }
            let type_raw = c.u32("kv type")?;
            let value = c.value(
                GgufType::from_u32(type_raw).ok_or(GgufError::BadType(type_raw))?,
                "kv value",
            )?;
            kv.push((key, value));
        }

        let alignment = match kv.iter().find(|(k, _)| k == GGUF_KEY_GENERAL_ALIGNMENT) {
            Some((_, v)) => {
                let a = v.as_u32().ok_or_else(|| GgufError::BadCast {
                    key: GGUF_KEY_GENERAL_ALIGNMENT.into(),
                    want: "uint32",
                    got: v.type_().name(),
                })?;
                if a == 0 || (a & (a - 1)) != 0 {
                    return Err(GgufError::BadShape(format!("alignment {a} not a power of 2")));
                }
                a as u64
            }
            None => GGUF_DEFAULT_ALIGNMENT,
        };

        let mut tensors: Vec<TensorInfo> = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
        for i in 0..n_tensors {
            let name = c.string(&format_tensor_what(i))?;
            if name.len() >= crate::types::MAX_NAME {
                return Err(GgufError::NameTooLong(name));
            }
            if tensors.iter().any(|t| t.name == name) {
                return Err(GgufError::DuplicateTensor(name));
            }
            let n_dims = c.u32("tensor n_dims")?;
            if n_dims as usize > crate::types::MAX_DIMS {
                return Err(GgufError::BadShape(name));
            }
            let mut ne = [1i64; 4];
            for (j, dim) in ne.iter_mut().enumerate() {
                if j < n_dims as usize {
                    *dim = c.i64("tensor ne")?;
                    if *dim < 0 {
                        return Err(GgufError::BadShape(name));
                    }
                }
            }
            let ty = GgmlType::from_u32(c.u32("tensor type")?).ok_or_else(|| GgufError::BadShape(name.clone()))?;
            let blck = ty.blck_size() as i64;
            if blck == 0 || ne[0] % blck != 0 {
                return Err(GgufError::BadShape(name));
            }
            let type_size = ty.type_size() as u64;
            let nb0 = type_size;
            let nb1 = nb0 * (ne[0] / blck) as u64;
            let nb2 = nb1 * ne[1] as u64;
            let nb3 = nb2 * ne[2] as u64;
            let offset = c.u64("tensor offset")?;
            tensors.push(TensorInfo { name, ne, ty, offset, nb: [nb0, nb1, nb2, nb3], part: 0 });
        }

        // data section is padded to alignment (only when tensors exist)
        let data_offset = if n_tensors > 0 {
            let unaligned = c.pos as u64;
            unaligned.div_ceil(alignment) * alignment
        } else {
            c.pos as u64
        };

        Ok(Gguf {
            version,
            kv,
            tensors,
            alignment,
            data_offset,
            parts: vec![GgufPart { data_offset, data }],
        })
    }

    pub fn find_key(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.find_key(key).and_then(|v| v.as_str())
    }
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.find_key(key).and_then(|v| v.as_u32())
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.find_key(key).and_then(|v| v.as_f32())
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.find_key(key).and_then(|v| v.as_bool())
    }
    pub fn get_f32_slice(&self, key: &str) -> Option<Vec<f32>> {
        self.find_key(key).and_then(|v| v.as_f32_slice())
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The mmap and *absolute* byte offset of a tensor's storage — the
    /// multi-file generalization of `gguf_get_data_offset +
    /// gguf_get_tensor_offset` that `weights_map`/`files[]` implement for
    /// splits (llama-model-loader.cpp:583-651): the tensor lives in the part
    /// that declares it, at that part's data offset plus the tensor offset.
    pub fn tensor_storage(&self, name: &str) -> Option<(Arc<Mmap>, usize)> {
        let ti = self.find_tensor(name)?;
        let (data, base) = self.part_storage(ti.part);
        Some((data, (base + ti.offset) as usize))
    }

    /// The mmap and data-section offset of one source file of the container
    /// (`part` = `TensorInfo::part`; always valid for tensors of this Gguf).
    pub fn part_storage(&self, part: u16) -> (Arc<Mmap>, u64) {
        let p = &self.parts[part as usize];
        (p.data.clone(), p.data_offset)
    }

    /// Byte slice of a tensor's storage (zero-copy view into its part's mmap).
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let ti = self.find_tensor(name)?;
        let p = self.parts.get(ti.part as usize)?;
        let start = (p.data_offset + ti.offset) as usize;
        let end = start + ti.size_bytes() as usize;
        p.data.get(start..end)
    }

    pub fn data_len(&self) -> usize {
        self.parts[0].data.len().saturating_sub(self.data_offset as usize)
    }
}

fn format_kv_what(i: u64) -> &'static str {
    // &'static str for the small hot errors; index is embedded via Truncated(what) generically
    let _ = i;
    "kv pair"
}
fn format_tensor_what(i: u64) -> &'static str {
    let _ = i;
    "tensor info"
}
