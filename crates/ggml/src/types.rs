//! ggml type system — port of the `ggml_type` enum (ggml.h) and the
//! `type_traits` table (ggml.c). Discriminants are GGUF wire values; do not reorder.

use crate::blocks::*;

pub const MAX_DIMS: usize = 4;
pub const MAX_NAME: usize = 64;
pub const MAX_OP_PARAMS: usize = 64;
/// ggml.h:224 `#define GGML_MAX_SRC 10` — raised from 4 by agent T for
/// GGML_OP_SSM_SCAN (7 sources: s, x, dt, A, B, C, ids).
pub const MAX_SRC: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5: removed (Q4_2/Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    // 31..33: removed repacked variants
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

pub const GGML_TYPE_COUNT: u32 = 43;

/// Metadata for one type, mirroring one row of the C `type_traits` table.
#[derive(Clone, Copy)]
pub struct TypeTraits {
    pub blck_size: usize,
    pub type_size: usize, // bytes per block
    pub is_quantized: bool,
    pub name: &'static str,
}

macro_rules! traits_row {
    ($bs:expr, $sz:expr, $q:expr, $name:literal) => {
        TypeTraits { blck_size: $bs, type_size: $sz, is_quantized: $q, name: $name }
    };
}

fn traits(t: GgmlType) -> TypeTraits {
    use GgmlType::*;
    match t {
        I8 => traits_row!(1, size_of::<i8>(), false, "I8"),
        I16 => traits_row!(1, size_of::<i16>(), false, "I16"),
        I32 => traits_row!(1, size_of::<i32>(), false, "I32"),
        I64 => traits_row!(1, size_of::<i64>(), false, "I64"),
        F64 => traits_row!(1, size_of::<f64>(), false, "F64"),
        F32 => traits_row!(1, size_of::<f32>(), false, "F32"),
        F16 => traits_row!(1, size_of::<half::f16>(), false, "F16"),
        Bf16 => traits_row!(1, size_of::<half::bf16>(), false, "BF16"),

        Q1_0 => traits_row!(QK1_0, size_of::<BlockQ1_0>(), true, "Q1_0"),
        Q2_0 => traits_row!(QK2_0, size_of::<BlockQ2_0>(), true, "Q2_0"),
        Q4_0 => traits_row!(QK4_0, size_of::<BlockQ4_0>(), true, "Q4_0"),
        Q4_1 => traits_row!(QK4_1, size_of::<BlockQ4_1>(), true, "Q4_1"),
        Q5_0 => traits_row!(QK5_0, size_of::<BlockQ5_0>(), true, "Q5_0"),
        Q5_1 => traits_row!(QK5_1, size_of::<BlockQ5_1>(), true, "Q5_1"),
        Q8_0 => traits_row!(QK8_0, size_of::<BlockQ8_0>(), true, "Q8_0"),
        Q8_1 => traits_row!(QK8_1, size_of::<BlockQ8_1>(), true, "Q8_1"),
        Mxfp4 => traits_row!(QK_MXFP4, size_of::<BlockMxfp4>(), true, "MXFP4"),
        Nvfp4 => traits_row!(QK_NVFP4, size_of::<BlockNvfp4>(), true, "NVFP4"),
        Tq1_0 => traits_row!(QK_K, size_of::<BlockTq1_0>(), true, "TQ1_0"),
        Tq2_0 => traits_row!(QK_K, size_of::<BlockTq2_0>(), true, "TQ2_0"),

        Q2K => traits_row!(QK_K, size_of::<BlockQ2K>(), true, "Q2_K"),
        Q3K => traits_row!(QK_K, size_of::<BlockQ3K>(), true, "Q3_K"),
        Q4K => traits_row!(QK_K, size_of::<BlockQ4K>(), true, "Q4_K"),
        Q5K => traits_row!(QK_K, size_of::<BlockQ5K>(), true, "Q5_K"),
        Q6K => traits_row!(QK_K, size_of::<BlockQ6K>(), true, "Q6_K"),
        Q8K => traits_row!(QK_K, size_of::<BlockQ8K>(), true, "Q8_K"),

        Iq2Xxs => traits_row!(QK_K, size_of::<BlockIq2Xxs>(), true, "IQ2_XXS"),
        Iq2Xs => traits_row!(QK_K, size_of::<BlockIq2Xs>(), true, "IQ2_XS"),
        Iq2S => traits_row!(QK_K, size_of::<BlockIq2S>(), true, "IQ2_S"),
        Iq3Xxs => traits_row!(QK_K, size_of::<BlockIq3Xxs>(), true, "IQ3_XXS"),
        Iq3S => traits_row!(QK_K, size_of::<BlockIq3S>(), true, "IQ3_S"),
        Iq1S => traits_row!(QK_K, size_of::<BlockIq1S>(), true, "IQ1_S"),
        Iq1M => traits_row!(QK_K, size_of::<BlockIq1M>(), true, "IQ1_M"),
        Iq4Nl => traits_row!(QK4_NL, size_of::<BlockIq4Nl>(), true, "IQ4_NL"),
        Iq4Xs => traits_row!(QK_K, size_of::<BlockIq4Xs>(), true, "IQ4_XS"),
    }
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => Bf16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            40 => Nvfp4,
            41 => Q1_0,
            42 => Q2_0,
            _ => return None,
        })
    }

    pub fn blck_size(self) -> usize {
        traits(self).blck_size
    }
    pub fn type_size(self) -> usize {
        traits(self).type_size
    }
    pub fn is_quantized(self) -> bool {
        traits(self).is_quantized
    }
    pub fn name(self) -> &'static str {
        traits(self).name
    }
    /// `ggml_row_size`
    pub fn row_size(self, ne: usize) -> usize {
        assert_eq!(ne % self.blck_size(), 0, "row not divisible by blck_size");
        self.type_size() * ne / self.blck_size()
    }
    pub fn sizef(self) -> f64 {
        traits(self).type_size as f64 / traits(self).blck_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte sizes cross-checked against ggml.c's table (bpw = sizef).
    #[test]
    fn type_table_matches_reference() {
        let cases: &[(GgmlType, usize, usize, f64)] = &[
            (GgmlType::F32, 1, 4, 4.0),
            (GgmlType::F16, 1, 2, 2.0),
            (GgmlType::Bf16, 1, 2, 2.0),
            (GgmlType::Q4_0, 32, 18, 0.5625),
            (GgmlType::Q4_1, 32, 20, 0.625),
            (GgmlType::Q5_0, 32, 22, 0.6875),
            (GgmlType::Q5_1, 32, 24, 0.75),
            (GgmlType::Q8_0, 32, 34, 1.0625),
            (GgmlType::Q2K, 256, 84, 0.328125),
            (GgmlType::Q3K, 256, 110, 0.4296875),
            (GgmlType::Q4K, 256, 144, 0.5625),
            (GgmlType::Q5K, 256, 176, 0.6875),
            (GgmlType::Q6K, 256, 210, 0.8203125),
            (GgmlType::Q8K, 256, 292, 292.0 / 256.0),
            (GgmlType::Iq2Xxs, 256, 66, 0.2578125),
            (GgmlType::Iq2Xs, 256, 74, 0.2890625),
            (GgmlType::Iq2S, 256, 82, 0.3203125),
            (GgmlType::Iq3Xxs, 256, 98, 0.3828125),
            (GgmlType::Iq3S, 256, 110, 0.4296875),
            (GgmlType::Iq1S, 256, 50, 0.1953125),
            (GgmlType::Iq1M, 256, 56, 0.21875),
            (GgmlType::Iq4Nl, 32, 18, 0.5625),
            (GgmlType::Iq4Xs, 256, 136, 0.53125),
            (GgmlType::Tq1_0, 256, 54, 0.2109375),
            (GgmlType::Tq2_0, 256, 66, 0.2578125),
            (GgmlType::Q1_0, 128, 18, 0.140625),
            (GgmlType::Q2_0, 64, 18, 0.28125),
        ];
        for &(t, bs, ts, bpw) in cases {
            assert_eq!(t.blck_size(), bs, "{t:?} blck_size");
            assert_eq!(t.type_size(), ts, "{t:?} type_size");
            assert!((t.sizef() - bpw).abs() < 1e-12, "{t:?} bpw");
            assert!(t.is_quantized() == (t != GgmlType::F32 && t != GgmlType::F16 && t != GgmlType::Bf16 && !matches!(t, GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 | GgmlType::F64)));
        }
        assert_eq!(GgmlType::Q4K.row_size(512), 144 * 2);
    }
}
