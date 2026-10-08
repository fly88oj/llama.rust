//! GGUF writer — port of gguf.cpp's write path (`gguf_add_tensor`,
//! `gguf_set_tensor_type`, `gguf_write_out` + the buf/file writers).
//!
//! Tensor offsets: first = 0, each next = prev.offset + PAD(prev_nbytes,
//! alignment) (gguf.cpp:1425-1428); the meta section is padded to alignment
//! before the data section (gguf.cpp:1642-1643) and every tensor payload is
//! padded after it (gguf.cpp:1621).

use std::io::Write;

use crate::gguf::{Value, GGUF_MAGIC, GGUF_VERSION};
use crate::types::GgmlType;

#[derive(Debug, Clone)]
pub struct WriterTensor {
    pub name: String,
    pub ty: GgmlType,
    pub ne: [i64; 4],
    /// byte offset relative to the data section start
    pub offset: u64,
}

impl WriterTensor {
    pub fn n_elements(&self) -> i64 {
        self.ne.iter().product()
    }
    pub fn nbytes(&self) -> u64 {
        self.ty.type_size() as u64 * (self.n_elements() as u64 / self.ty.blck_size() as u64)
    }
    fn n_dims(&self) -> u32 {
        self.ne.iter().rposition(|&d| d != 1).map(|i| i + 1).unwrap_or(1) as u32
    }
}

pub struct GgufWriter {
    version: u32,
    pub kv: Vec<(String, Value)>,
    pub tensors: Vec<WriterTensor>,
    pub alignment: u64,
}

impl GgufWriter {
    pub fn new(alignment: u64) -> Self {
        assert!(alignment > 0 && alignment.is_power_of_two());
        GgufWriter { version: GGUF_VERSION, kv: Vec::new(), tensors: Vec::new(), alignment }
    }

    /// `gguf_set_val_*`: override an existing key (in place) or append.
    pub fn set_kv(&mut self, key: &str, value: Value) {
        assert!(!key.is_empty());
        if let Some(slot) = self.kv.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.kv.push((key.to_string(), value));
        }
    }

    /// `gguf_add_tensor` (gguf.cpp:1395-1408): duplicate names abort; offsets
    /// computed from the running aligned sizes.
    pub fn add_tensor(&mut self, name: &str, ty: GgmlType, ne: [i64; 4]) -> usize {
        assert!(self.tensors.iter().all(|t| t.name != name), "duplicate tensor name: {name}");
        assert_eq!(ne[0] % ty.blck_size() as i64, 0, "row not divisible by block size");
        let offset = match self.tensors.last() {
            None => 0,
            Some(p) => p.offset + p.nbytes().div_ceil(self.alignment) * self.alignment,
        };
        self.tensors.push(WriterTensor { name: name.to_string(), ty, ne, offset });
        self.tensors.len() - 1
    }

    /// `gguf_set_tensor_type` (gguf.cpp:1410-1435): changes the type and
    /// recomputes the offsets of every following tensor.
    pub fn set_tensor_type(&mut self, idx: usize, ty: GgmlType) {
        let t = &mut self.tensors[idx];
        assert_eq!(t.ne[0] % ty.blck_size() as i64, 0, "row not divisible by block size");
        t.ty = ty;
        for i in idx + 1..self.tensors.len() {
            let prev = &self.tensors[i - 1];
            self.tensors[i].offset =
                prev.offset + prev.nbytes().div_ceil(self.alignment) * self.alignment;
        }
    }

    /// Size of the padded meta section (header + kv + tensor infos).
    pub fn meta_size(&self) -> u64 {
        let mut n = 4 + 4 + 8 + 8;
        for (k, v) in &self.kv {
            n += 8 + k.len() as u64 + 4 + value_size(v);
        }
        for t in &self.tensors {
            n += 8 + t.name.len() as u64 + 4 + 8 * t.n_dims() as u64 + 4 + 8;
        }
        n.div_ceil(self.alignment) * self.alignment
    }

    /// Header + kv + tensor infos, padded to alignment (gguf_write_out prefix).
    pub fn write_meta(&self, w: &mut impl Write) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::with_capacity(self.meta_size() as usize);
        buf.extend_from_slice(GGUF_MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&value_type_id(v).to_le_bytes());
            write_value(&mut buf, v);
        }
        for t in &self.tensors {
            buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
            buf.extend_from_slice(t.name.as_bytes());
            buf.extend_from_slice(&t.n_dims().to_le_bytes());
            for d in &t.ne[..t.n_dims() as usize] {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(t.ty as u32).to_le_bytes());
            buf.extend_from_slice(&t.offset.to_le_bytes());
        }
        let pad = self.meta_size() as usize - buf.len();
        buf.extend(std::iter::repeat_n(0u8, pad));
        w.write_all(&buf)
    }

    /// Full write: meta, then each tensor's payload followed by alignment
    /// padding (gguf_write_out + write_tensor_data).
    pub fn write(&self, w: &mut impl Write, data: &[&[u8]]) -> std::io::Result<()> {
        assert_eq!(data.len(), self.tensors.len());
        self.write_meta(w)?;
        for (t, bytes) in self.tensors.iter().zip(data) {
            assert_eq!(bytes.len() as u64, t.nbytes(), "tensor {} payload size", t.name);
            w.write_all(bytes)?;
            let pad = t.nbytes().div_ceil(self.alignment) * self.alignment - t.nbytes();
            w.write_all(&vec![0u8; pad as usize])?;
        }
        Ok(())
    }
}

fn value_type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_, _) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn value_size(v: &Value) -> u64 {
    match v {
        Value::U8(_) | Value::I8(_) | Value::Bool(_) => 1,
        Value::U16(_) | Value::I16(_) => 2,
        Value::U32(_) | Value::I32(_) | Value::F32(_) => 4,
        Value::U64(_) | Value::I64(_) | Value::F64(_) => 8,
        Value::String(s) => 8 + s.len() as u64,
        Value::Array(_, items) => 4 + 8 + items.iter().map(value_size).sum::<u64>(),
    }
}

fn write_value(buf: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => buf.push(*x),
        Value::I8(x) => buf.push(*x as u8),
        Value::Bool(x) => buf.push(*x as u8),
        Value::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::String(s) => {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Array(t, items) => {
            buf.extend_from_slice(&(*t as u32).to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value(buf, it);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: write a small file, read it back with our reader, compare
    /// every field. Byte-parity against the reference writer is in
    /// `gguf_write_bit_exact_vs_reference` (parity/gguf_write_ref.bin).
    #[test]
    fn write_read_roundtrip() {
        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("test".into()));
        w.set_kv("test.block_count", Value::U32(2));
        w.set_kv("test.arr", Value::Array(crate::gguf::GgufType::Int32, vec![Value::I32(-1), Value::I32(7)]));
        let payload: Vec<Vec<u8>> = vec![
            (0..64u8).collect(),           // F32 [16]  (64 B)
            (0..18u8).collect::<Vec<_>>(), // Q4_0 [32] (18 B)
        ];
        w.add_tensor("a.weight", GgmlType::F32, [16, 1, 1, 1]);
        w.add_tensor("b.weight", GgmlType::Q4_0, [32, 1, 1, 1]);
        assert_eq!(w.tensors[1].offset, 64); // 64 B is already 32-aligned
        let mut out = Vec::new();
        let refs: Vec<&[u8]> = payload.iter().map(|v| v.as_slice()).collect();
        w.write(&mut out, &refs).unwrap();

        let path = std::env::temp_dir().join("gguf_write_roundtrip.gguf");
        std::fs::write(&path, &out).unwrap();
        let g = crate::gguf::Gguf::open(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.get_str("general.architecture"), Some("test"));
        assert_eq!(g.get_u32("test.block_count"), Some(2));
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.find_tensor("a.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.find_tensor("b.weight").unwrap().ty, GgmlType::Q4_0);
        assert_eq!(g.find_tensor("b.weight").unwrap().offset, 64);
        assert_eq!(g.tensor_data("a.weight").unwrap(), payload[0].as_slice());
        assert_eq!(g.tensor_data("b.weight").unwrap(), payload[1].as_slice());
        assert_eq!(g.data_offset, w.meta_size());
        std::fs::remove_file(&path).ok();
    }

    /// Byte-identical to the reference writer (gguf.cpp @ bd4f514db1) for the
    /// same logical content — parity/gguf_write_ref.bin, generated by
    /// parity/ref_gguf_write.c.
    #[test]
    fn gguf_write_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/gguf_write_ref.bin");
        let Ok(reference) = std::fs::read(path) else {
            eprintln!("skipping: build parity/ref_gguf_write.c first");
            return;
        };
        // LCG data identical to the C harness
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        };
        let a: Vec<u8> = (0..64).map(|_| next()).collect();
        let b: Vec<u8> = (0..18).map(|_| next()).collect();

        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("test".into()));
        w.set_kv("test.block_count", Value::U32(2));
        w.set_kv("test.f", Value::F32(1.5));
        w.set_kv("test.arr", Value::Array(crate::gguf::GgufType::Int32, vec![Value::I32(-1), Value::I32(7), Value::I32(42)]));
        w.set_kv("test.strs", Value::Array(
            crate::gguf::GgufType::String,
            vec![Value::String("alpha".into()), Value::String("beta".into())],
        ));
        w.add_tensor("a.weight", GgmlType::F32, [16, 1, 1, 1]);
        w.add_tensor("b.weight", GgmlType::Q4_0, [32, 1, 1, 1]);
        let mut mine = Vec::new();
        w.write(&mut mine, &[a.as_slice(), b.as_slice()]).unwrap();

        assert_eq!(mine.len(), reference.len(), "file size differs");
        let first_diff = mine
            .iter()
            .zip(&reference)
            .position(|(x, y)| x != y);
        assert_eq!(first_diff, None, "byte {first_diff:?} differs");
    }
}