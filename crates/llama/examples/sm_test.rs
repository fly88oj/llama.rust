// reproduce: port soft_max over a [33,33,4] input row from the parakeet dump
fn main() {
    let data = std::fs::read("/tmp/nodes/parakeet_sm_input.bin").unwrap();
    let vals: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // row 0 = first 33 values
    let row = &vals[..33];
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    let mut out = vec![0f32; 33];
    for i in 0..33 {
        let v = (row[i] - max).exp();
        out[i] = v;
        sum += v as f64;
    }
    println!(
        "scalar softmax row0[:4]: {:?}",
        &(out[..4].iter().map(|v| v / sum as f32).collect::<Vec<_>>())
    );
}
