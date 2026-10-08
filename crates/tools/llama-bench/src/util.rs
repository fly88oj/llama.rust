//! Small helpers for llama-bench: wall clock, the C `std::rand()` stream, the
//! `/proc`-derived host strings and the C++ stream formatting quirks the
//! reference output depends on.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// llama-bench.cpp:31-34 `get_time_ns` — `high_resolution_clock` epoch
/// nanoseconds. Only differences of these values are ever used (the C takes
/// `get_time_ns() - t_start`), so a monotonic clock is equivalent.
pub fn get_time_ns() -> u64 {
    // process-start anchor; Instant has no epoch, the C value would overflow
    // nothing anyway (it is monotonic in practice too)
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// `llama-bench.cpp:1565` — `strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t))`,
/// i.e. RFC 3339 in UTC with second resolution.
pub fn utc_time_rfc3339() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    // civil-from-days (Howard Hinnant's algorithm), UTC only — avoids a chrono dep
    let days = secs.div_euclid(86_400);
    let mut rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = {
        let h = rem / 3600;
        rem %= 3600;
        (h, rem / 60, rem % 60)
    };
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// llama-bench.cpp:1967-1969 `avg` — integer arithmetic on the sample vector
/// (`sum / size`, truncating for the u64 overload), 0 for an empty vector.
pub fn avg_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let sum = v.iter().fold(0u64, |a, x| a.wrapping_add(*x));
    sum / v.len() as u64
}

/// llama-bench.cpp:1971-1980 `stdev` (sample standard deviation, n-1):
/// `sqrt(sq_sum/(n-1) - mean*mean*n/(n-1))` in the vector's own type. For the
/// u64 instantiation the C computes that expression entirely in u64 with the
/// truncations that implies (`avg` is integer, the subtraction wraps); the port
/// reproduces the formula in f64 — see `stdev_ns_u64` for the faithful
/// integer-only path that the `stddev_ns` field prints.
pub fn stdev_f64(v: &[f64]) -> f64 {
    if v.len() <= 1 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sq_sum: f64 = v.iter().map(|x| x * x).sum();
    let n = v.len() as f64;
    (sq_sum / (n - 1.0) - mean * mean * n / (n - 1.0)).sqrt()
}

/// `stdev<uint64_t>` of llama-bench.cpp:1971-1980, evaluated exactly like the
/// C `T` template does (integer division, wraparound subtraction) so the
/// printed `stddev_ns` matches the reference's value for the same samples.
pub fn stdev_ns_u64(v: &[u64]) -> u64 {
    if v.len() <= 1 {
        return 0;
    }
    let mean = avg_u64(v);
    let sq_sum = v.iter().fold(0u64, |a, x| a.wrapping_add(x.wrapping_mul(*x)));
    let n = v.len() as u64;
    let a = sq_sum / (n - 1);
    let b = mean.wrapping_mul(mean).wrapping_mul(n) / (n - 1);
    let d = a.wrapping_sub(b) as f64;
    if d <= 0.0 {
        // the C casts the (often wrapped) u64 to double and calls sqrt; a
        // negative value yields NaN there, 0 here (never printed in practice)
        return 0;
    }
    d.sqrt() as u64
}

/// llama-bench.cpp:1619-1627 `get_ts` — `1e9 * n_tokens / t` per sample with
/// the *double* arithmetic of the C (`std::transform` into `std::vector<double>`).
pub fn get_ts(samples_ns: &[u64], n_tokens: i32) -> Vec<f64> {
    samples_ns.iter().map(|t| 1e9 * n_tokens as f64 / *t as f64).collect()
}

/// llama-bench.cpp:1967 `avg<double>` — plain mean, 0.0 when empty.
pub fn avg(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// `std::to_string(double)` = `printf("%f")`: fixed, 6 fractional digits, so
/// `avg_ts` / `stddev_ts` keep their trailing zeros exactly like the reference.
pub fn fmt_to_string_f64(v: f64) -> String {
    format!("{v:.6}")
}

/// `std::ostringstream << double` with the default precision (6 *significant*
/// digits) = `printf("%g")` — used by `join(t.get_ts(), ", ")` for `samples_ts`.
/// C's %g rounds to P significant digits first and picks the style from the
/// *rounded* exponent; `{:.5e}` gives that rounding for P = 6.
pub fn fmt_g6(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let e = format!("{v:.5e}"); // rust: "6.43191e2" / "1.23457e-5"
    let (mant, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let mant = strip_trailing_zeros_fixed(mant);
    if ex < -4 || ex >= 6 {
        format!("{mant}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
    } else {
        let prec = (6 - 1 - ex).max(0) as usize;
        strip_trailing_zeros_fixed(&format!("{v:.prec$}"))
    }
}

fn strip_trailing_zeros_fixed(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    let t = t.trim_end_matches('.');
    t.to_string()
}

/// glibc `std::rand()` (TYPE_3 additive feedback, `srandom(1)` state) as used
/// by llama-bench's `test_prompt`/`test_gen` (llama-bench.cpp:2136/2160). The C
/// uses naked `std::rand()`; reproducing glibc's generator keeps the two tools'
/// pseudo-random token streams identical (same default seed 1).
pub struct GlibcRand {
    r: [u32; 344],
    idx: usize,
}

impl Default for GlibcRand {
    fn default() -> Self {
        Self::new()
    }
}

impl GlibcRand {
    /// `srand(1)` / `srandom(1)` — the state every C program starts with.
    pub fn new() -> Self {
        let mut r = [0u32; 344];
        r[0] = 1;
        for i in 1..31 {
            // r[i] = (16807 * r[i-1]) % 2147483647 with the C's signed trick
            let prev = r[i - 1] as i64;
            let hi = prev / 127_773;
            let lo = prev % 127_773;
            let word = 16_807 * lo - 2_836 * hi;
            r[i] = if word < 0 { (word + 2_147_483_647) as u32 } else { word as u32 };
        }
        for i in 31..34 {
            r[i] = r[i - 31];
        }
        // f = 3, r[34..344] = r[i-31] + r[i-3]
        for i in 34..344 {
            r[i] = r[i - 31].wrapping_add(r[i - 3]);
        }
        GlibcRand { r, idx: 344 }
    }

    /// glibc `random()`: `(r[i-31] + r[i-3]) >> 1` after advancing.
    pub fn next_u31(&mut self) -> u32 {
        let i = self.idx;
        let v = self.r[(i + 344 - 31) % 344].wrapping_add(self.r[(i + 344 - 3) % 344]);
        self.r[i % 344] = v;
        self.idx = i + 1;
        v >> 1
    }

    /// `std::rand() % n` — RAND_MAX is 2^31-1, so `% n` on the 31-bit value.
    pub fn below(&mut self, n: i32) -> i32 {
        (self.next_u31() % n as u32) as i32
    }
}

/// `ggml_backend_cpu_device_description` on Linux (ggml-cpu.cpp:288-318): the
/// `/proc/cpuinfo` "model name" with surrounding whitespace trimmed, "CPU"
/// when absent. `get_cpu_info` (llama-bench.cpp:120-131) joins exactly the
/// CPU + ACCEL devices with ", " — the port has one CPU device.
pub fn cpu_info() -> String {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return "CPU".to_string();
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some((_, v)) = rest.split_once(':') {
                let v = v.trim();
                if !v.is_empty() {
                    return v.to_string();
                }
            }
        }
    }
    "CPU".to_string()
}

/// llama-bench.cpp:220-227 with `common_cpu_get_num_math` (common/common.cpp:
/// 200-228) on x86_64 Linux: `_SC_NPROCESSORS_ONLN` cores, physical cores when
/// the CPU is not hybrid. `common_cpu_get_num_physical_cores` (common/common.cpp
/// :78-99) counts the distinct `/sys/.../topology/thread_siblings` sets.
pub fn num_physical_cores() -> i32 {
    let mut siblings = std::collections::BTreeSet::new();
    for cpu in 0..u32::MAX {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings");
        let Ok(line) = std::fs::read_to_string(&path) else {
            break;
        };
        let line = line.trim();
        if !line.is_empty() {
            siblings.insert(line.to_string());
        }
    }
    if !siblings.is_empty() {
        return siblings.len() as i32;
    }
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0) as i32;
    if n > 0 {
        n
    } else {
        4
    }
}

/// `is_hybrid_cpu` (common/common.cpp, x86 feature detection) — only Intel
/// hybrid parts take the `cpu_count_math_cpus` branch; the port reports the
/// physical-core count unconditionally (see PARITY.md).
pub fn cpu_get_num_math() -> i32 {
    num_physical_cores()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_and_stdev_match_c_formula() {
        // hand-computed: samples 100, 200, 300 -> mean 200,
        // sq_sum = 140000, stdev = sqrt(140000/2 - 200*200*3/2) = sqrt(10000) = 100
        let v = [100u64, 200, 300];
        assert_eq!(avg_u64(&v), 200);
        assert_eq!(stdev_ns_u64(&v), 100);

        // single sample / empty vector -> 0 (the C's v.size() <= 1 branch)
        assert_eq!(stdev_ns_u64(&[42]), 0);
        assert_eq!(stdev_ns_u64(&[]), 0);
        assert_eq!(avg_u64(&[]), 0);

        // get_ts: 1e9 * n_tokens / t
        let ts = get_ts(&[1_000_000_000, 2_000_000_000], 64);
        assert_eq!(ts, vec![64.0, 32.0]);
        // the f64 overload is the plain mean
        assert!((avg(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
        // 5 samples: mean 5, sumsq 30+... hand-computed population-free formula
        let w = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let got = stdev_f64(&w);
        let want = (2.5f64).sqrt(); // sumsq=55, 55/4 - 9*5/4 = 13.75-11.25 = 2.5
        assert!((got - want).abs() < 1e-12);
    }

    #[test]
    fn glibc_rand_matches_the_c_stream() {
        // glibc `rand()` / `random()` from `srand(1)` (checked against the
        // running libc through ctypes: `srand(1); [rand() for _ in range(5)]`)
        let mut r = GlibcRand::new();
        let first: Vec<u32> = (0..5).map(|_| r.next_u31()).collect();
        assert_eq!(first, vec![1_804_289_383, 846_930_886, 1_681_692_777, 1_714_636_915, 1_957_747_793]);
        // `rand() % n` — the second token stream below is the third value of a
        // fresh seed-1 stream modulo a qwen2.5-sized vocabulary
        let mut r = GlibcRand::new();
        assert_eq!(r.below(151_643), 1_804_289_383 % 151_643);
        let mut r = GlibcRand::new();
        let _ = (r.below(151_643), r.below(151_643));
        assert_eq!(r.below(151_643), 1_681_692_777 % 151_643);
    }

    #[test]
    fn cpp_stream_double_formatting() {
        // std::to_string(double) = %f (6 decimals, trailing zeros kept)
        assert_eq!(fmt_to_string_f64(646.6241482), "646.624148");
        assert_eq!(fmt_to_string_f64(4.5), "4.500000");
        assert_eq!(fmt_to_string_f64(0.0), "0.000000");
        // ostringstream << double = %g (6 significant digits); expectations
        // are `'%g' % v` from the same libc the reference links
        assert_eq!(fmt_g6(643.1913), "643.191");
        assert_eq!(fmt_g6(650.0571), "650.057");
        assert_eq!(fmt_g6(646.6241482), "646.624");
        assert_eq!(fmt_g6(96.5), "96.5");
        assert_eq!(fmt_g6(1.0), "1");
        assert_eq!(fmt_g6(100.0), "100");
        assert_eq!(fmt_g6(1_234_567.0), "1.23457e+06");
        assert_eq!(fmt_g6(999_999.9), "1e+06");
        assert_eq!(fmt_g6(0.0001234567), "0.000123457");
        assert_eq!(fmt_g6(0.0), "0");
        assert_eq!(fmt_g6(1_804.0), "1804");
    }

    #[test]
    fn rfc3339_format_is_utc() {
        let s = utc_time_rfc3339();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.ends_with('Z'));
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
        // 2026-09-24T22:56:39Z style: digits everywhere else
        for (i, c) in s.char_indices() {
            if i == 4 || i == 7 || i == 10 || i == 13 || i == 16 || i == 19 {
                continue;
            }
            assert!(c.is_ascii_digit(), "{s}");
        }
    }
}