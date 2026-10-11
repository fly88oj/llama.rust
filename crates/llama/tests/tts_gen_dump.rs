//! tts_gen_dump.rs (worktree probe) — Rust-side mirror of
//! `parity/ref_gen_audio_dump.cpp --nodes`: the same `DECDMP1` node stream
//! for the TTS gen pipelines (pockettts_gen genwav-feats cold call), so
//! `parity/decode_dump_cmp.py --no-names` can bisect the first divergent
//! graph node against the NEW-reference node stream.
//!
//! Loads the EXISTING fixtures of /tmp/tts-gen-synth (written by
//! tests/tts_gen_e2e.rs) — no fixture regeneration; runs the same
//! GEN_CODE + GEN_WAV sequence the e2e test runs (its leftover buffers are
//! proven bit-identical through the state segments), with the eval callback
//! active ONLY around the main GEN_WAV call.
//!
//! Run (release):
//!   cargo test --release -p llama --test tts_gen_dump -- --ignored --nocapture
//! env:
//!   TTS_DUMP_ARCH=pockettts|qwen3tts   (default pockettts)

use std::io::Write as _;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::types::GgmlType;
use llama::clip::{
    clip_init_from_file, ClipContextParams, ClipFlashAttn, ClipGenProcess, GenInp,
};

const FIX: &str = "/tmp/tts-gen-synth";
const SEED: u32 = 42;

const ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static ACTIVE: AtomicU32 = AtomicU32::new(0);

fn op_desc(op: ggml::GgmlOp) -> &'static str {
    use ggml::GgmlOp::*;
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        Mul => "MUL",
        Div => "DIV",
        Sub => "SUB",
        Norm => "NORM",
        SquaredMulMat => "MUL_MAT_SQ",
        MulMat => "MUL_MAT",
        Scale => "SCALE",
        Cpy => "CPY",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        DiagMaskInf => "DIAG_MASK_INF",
        SoftMax => "SOFT_MAX",
        RoPE => "ROPE",
        RoPEBack => "ROPE_BACK",
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => "SILU",
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => "GLU",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        Clamp => "CLAMP",
        Gdn => "GATED_DELTA_NET",
        Im2col => "IM2COL",
        Upscale => "UPSCALE",
        Fill => "FILL",
        LightningIndexer => "LIGHTNING_INDEXER",
        TopK => "TOP_K",
        Sqrt => "SQRT",
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPreGated => "dsv4_hc_pre_gated(x, gate, scale)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        Pad => "PAD",
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_D",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn type_desc(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Bf16 => "bf16",
        GgmlType::I64 => "i64",
        GgmlType::I32 => "i32",
        GgmlType::I16 => "i16",
        GgmlType::I8 => "i8",
        _ => "other",
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len().min(255);
    buf.push(len as u8);
    buf.extend_from_slice(&s.as_bytes()[..len]);
}

fn dump_cb(node: &EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true;
    }
    let mut guard = DUMP.get().unwrap().lock().unwrap();
    let Some(st) = guard.as_mut() else {
        return true;
    };
    let n: i64 = node.ne.iter().product();
    st.nodes += 1;
    put_str(&mut st.out, op_desc(node.op));
    put_str(&mut st.out, node.name);
    put_str(&mut st.out, type_desc(node.ty));
    st.out
        .extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
    st.out.extend_from_slice(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= ELEM_CAP {
        return true;
    }
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
        return true;
    }
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v: f32 = if node.ty == GgmlType::F32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            let h = half::f16::from_le_bytes([data[off], data[off + 1]]);
            h.to_f32()
        };
        st.out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

fn read_f32_blob(path: &str) -> Vec<f32> {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    d.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32_blob(path: &str) -> Vec<i32> {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    d.chunks(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_dump(path: &str, nodes: u32, body: Vec<u8>) {
    let mut f = std::fs::File::create(path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&0i32.to_le_bytes()).unwrap();
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    println!("tts_gen_dump: {nodes} nodes -> {path}");
}

/// node-stream dump of one GEN_WAV call (pockettts feats cold / qwen3tts
/// codes cold), with the e2e test's GEN_CODE call first (leftover buffers)
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the tts gen parity"]
fn tts_gen_node_dump() {
    let arch = std::env::var("TTS_DUMP_ARCH").unwrap_or_else(|_| "pockettts".to_string());
    let params = ClipContextParams {
        flash_attn_type: ClipFlashAttn::Disabled,
        ..Default::default()
    };

    DUMP.get_or_init(|| Mutex::new(None));

    if arch == "pockettts" {
        let mut cx =
            clip_init_from_file(&format!("{FIX}/mmproj-pockettts-gen.gguf"), &params)
                .expect("load gen mmproj");
        // the e2e sequence: GEN_CODE first (seed 42, temp 0.7)
        let h = read_f32_blob(&format!("{FIX}/pt-hstate.bin"));
        let inp = GenInp {
            process: ClipGenProcess::GenCode,
            code0: 0,
            embd: Some(&h),
            top_k: 50,
            top_p: 1.0,
            seed: SEED,
            temp: 0.7,
            codes: None,
            feats: None,
            state: None,
        };
        let _ = cx.gen_audio_process(&inp).expect("gen_code warm");

        let feats = read_f32_blob(&format!("{FIX}/pt-feats1.bin"));
        let inp = GenInp {
            process: ClipGenProcess::GenWav,
            code0: 0,
            embd: None,
            top_k: 50,
            top_p: 1.0,
            seed: SEED,
            temp: 0.7,
            codes: None,
            feats: Some(&feats),
            state: None,
        };
        {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            *guard = Some(DumpState { out: Vec::new(), nodes: 0 });
        }
        set_eval_callback(Some(dump_cb));
        ACTIVE.store(1, Ordering::SeqCst);
        let out = cx.gen_audio_process(&inp).expect("gen_wav cold");
        ACTIVE.store(0, Ordering::SeqCst);
        set_eval_callback(None);
        let (nodes, body) = {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            let st = guard.take().unwrap();
            (st.nodes, st.out)
        };
        println!(
            "pockettts genwav1: {} audio samples, {} state bytes",
            out.audio.len(),
            out.state.len()
        );
        write_dump("/tmp/r4o4/pt1-port-nodes.bin", nodes, body);
    } else {
        let mut cx =
            clip_init_from_file(&format!("{FIX}/mmproj-qwen3tts-gen.gguf"), &params)
                .expect("load gen mmproj");
        let h = read_f32_blob(&format!("{FIX}/q3t-hstate.bin"));
        let inp = GenInp {
            process: ClipGenProcess::GenCode,
            code0: 5,
            embd: Some(&h),
            top_k: 50,
            top_p: 1.0,
            seed: SEED,
            temp: 0.9,
            codes: None,
            feats: None,
            state: None,
        };
        let _ = cx.gen_audio_process(&inp).expect("gen_code warm");

        let codes = read_i32_blob(&format!("{FIX}/q3t-codes1.bin"));
        let inp = GenInp {
            process: ClipGenProcess::GenWav,
            code0: 0,
            embd: None,
            top_k: 50,
            top_p: 1.0,
            seed: SEED,
            temp: 0.0,
            codes: Some(&codes),
            feats: None,
            state: None,
        };
        {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            *guard = Some(DumpState { out: Vec::new(), nodes: 0 });
        }
        set_eval_callback(Some(dump_cb));
        ACTIVE.store(1, Ordering::SeqCst);
        let out = cx.gen_audio_process(&inp).expect("gen_wav cold");
        ACTIVE.store(0, Ordering::SeqCst);
        set_eval_callback(None);
        let (nodes, body) = {
            let mut guard = DUMP.get().unwrap().lock().unwrap();
            let st = guard.take().unwrap();
            (st.nodes, st.out)
        };
        println!(
            "qwen3tts genwav1: {} audio samples, {} state bytes",
            out.audio.len(),
            out.state.len()
        );
        write_dump("/tmp/r4o4/q3t1-port-nodes.bin", nodes, body);
    }
}
