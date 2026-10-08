//! llama-tokenize (rust) — port of `tools/tokenize/tokenize.cpp` (pinned
//! bd4f514db1), CLI-level re-verification of the vocab layer.
//!
//! Reference behaviour reproduced here:
//!   * `-f` is read verbatim (a trailing newline is a token), escapes are
//!     applied locally; `-p` arrives escape-processed (common_params_parse);
//!     `--stdin` wins over both and is escape-processed after reading.
//!   * `-e/--escape` defaults to true (`string_process_escapes`: \n \r \t \' \"
//!     \\ and \xHH, anything else keeps the backslash).
//!   * the prompt is tokenized with `add_bos = vocab.add_bos && !--no-bos` and
//!     `parse_special` (default true, `--no-parse-special` disables).
//!   * pieces come from `common_token_to_piece(ctx, id)` = `token_to_piece(id,
//!     special = true)`; raw bytes are written through unchanged.
//!   * output shapes: `%6d -> '<piece>'` per token, or `[1, 2, 3]` with
//!     `--ids`; `Total number of tokens: N` with `--show-count`.
//!
//! Difference vs the C tool: the model is loaded vocab-only at the reader
//! level (`Vocab::load`, exactly the `llama_model_params.vocab_only = true`
//! data set) and no llama_context is created — tokenization output is
//! identical, but `--version`/timings/perf prints of the C tool are absent.

use std::io::{Read as _, Write as _};

use ggml::Gguf;
use llama::vocab::Vocab;

/// `string_process_escapes` (common.cpp:645).
fn process_escapes(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let n = input.len();
    let mut i = 0;
    while i < n {
        if input[i] == b'\\' && i + 1 < n {
            i += 1;
            match input[i] {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'x' if i + 2 < n => {
                    // `\x12`: exactly two hex digits (strtol with errp == x+2)
                    let h = std::str::from_utf8(&input[i + 1..i + 3])
                        .ok()
                        .and_then(|s| u8::from_str_radix(s, 16).ok());
                    match h {
                        Some(v) => {
                            i += 2;
                            out.push(v);
                        }
                        None => {
                            out.push(b'\\');
                            out.push(input[i]);
                        }
                    }
                }
                c => {
                    out.push(b'\\');
                    out.push(c);
                }
            }
        } else {
            out.push(input[i]);
        }
        i += 1;
    }
    out
}

fn usage(prog: &str) {
    println!("usage: {prog} -m model.gguf -p \"Hello world\"");
    println!("       {prog} -m model.gguf -f prompt.txt --ids");
    println!("       cat prompt.txt | {prog} -m model.gguf --stdin --show-count");
    println!();
    println!("  -m, --model FILE       model to load (vocabulary only)");
    println!("  -p, --prompt TEXT      prompt to tokenize");
    println!("  -f, --file FILE        file to read the prompt from (verbatim)");
    println!("  --stdin                read the prompt from stdin (takes precedence)");
    println!("  --ids                  print only the token ids as [1, 2, 3]");
    println!("  --show-count           print the total number of tokens");
    println!("  --no-bos               do not add BOS even if the model normally does");
    println!("  --no-parse-special     do not parse special tokens (chat, tool, ...)");
    println!("  -e, --escape / --no-escape   process \\n \\r \\t \\' \\\" \\\\ \\xHH (default: on)");
    println!("  -h, --help             this help");
}

struct Args {
    model: Option<String>,
    prompt: Option<String>,
    file: Option<String>,
    stdin: bool,
    ids: bool,
    show_count: bool,
    no_bos: bool,
    parse_special: bool,
    escape: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: None,
            prompt: None,
            file: None,
            stdin: false,
            ids: false,
            show_count: false,
            no_bos: false,
            // "-no-parse-special" help text: parse special tokens by default
            // (arg.cpp:1402, the tokenize tool's own default)
            parse_special: true,
            escape: true,
        }
    }
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let mut val = |name: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("error: {name} requires an argument"))
        };
        match arg {
            "-h" | "--help" | "--usage" => {
                usage(&argv[0]);
                std::process::exit(0);
            }
            "-m" | "--model" => a.model = Some(val(arg)?),
            "-p" | "--prompt" => a.prompt = Some(val(arg)?),
            "-f" | "--file" => a.file = Some(val(arg)?),
            "--stdin" => a.stdin = true,
            "--ids" => a.ids = true,
            "--show-count" => a.show_count = true,
            "--no-bos" => a.no_bos = true,
            "--no-parse-special" => a.parse_special = false,
            "-e" | "--escape" => a.escape = true,
            "--no-escape" => a.escape = false,
            s if s.starts_with("--model=") => a.model = Some(s["--model=".len()..].to_string()),
            s if s.starts_with("--prompt=") => a.prompt = Some(s["--prompt=".len()..].to_string()),
            s if s.starts_with("--file=") => a.file = Some(s["--file=".len()..].to_string()),
            s => return Err(format!("error: unknown option '{s}'")),
        }
        i += 1;
    }
    Ok(a)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // tokenize.cpp:112 — must have some prompt. The reference tests
    // `params.prompt.empty()`, so an explicitly empty `-p ""` counts as missing.
    if !args.stdin && args.file.is_none() && args.prompt.as_deref().unwrap_or("").is_empty() {
        eprintln!("error: must specify one of: --stdin, --file or --prompt");
        std::process::exit(1);
    }

    let Some(model_path) = args.model.as_deref() else {
        eprintln!("error: no model specified (-m/--model)");
        std::process::exit(1);
    };

    // read the prompt *before* loading the model where the C tool does
    // (-f/-p are read by the arg parser; --stdin takes precedence over both,
    // tokenize.cpp:163-178)
    let bytes: Vec<u8> = if args.stdin {
        let mut buf = Vec::new();
        if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
            eprintln!("error: could not read the entire standard input: {e}");
            std::process::exit(1);
        }
        buf
    } else if let Some(f) = &args.file {
        // read the file verbatim (common's -f handler strips a trailing
        // newline; the tokenizer must not)
        match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: could not open file '{f}' for reading: {e}");
                std::process::exit(1);
            }
        }
    } else {
        args.prompt.clone().unwrap_or_default().into_bytes()
    };

    let bytes = if args.escape { process_escapes(&bytes) } else { bytes };

    let gguf = match Gguf::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: could not load model from file '{model_path}': {e}");
            std::process::exit(1);
        }
    };
    let vocab = match Vocab::load(&gguf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: could not load vocabulary from '{model_path}': {e}");
            std::process::exit(1);
        }
    };

    let add_bos = vocab.get_add_bos() && !args.no_bos;
    // C `llama_tokenize` takes a char* + length: arbitrary bytes reach the
    // tokenizer and are matched through the byte-fallback vocabulary. Going
    // through `&str` here would reject them.
    let tokens = vocab.tokenize_bytes(&bytes, add_bos, args.parse_special);

    let mut out: Vec<u8> = Vec::new();
    if args.ids {
        out.push(b'[');
        for (i, t) in tokens.iter().enumerate() {
            if i > 0 {
                out.extend_from_slice(b", ");
            }
            out.extend_from_slice(t.to_string().as_bytes());
        }
        out.extend_from_slice(b"]\n");
    } else {
        for t in &tokens {
            // the prediction cache holds `token_to_piece(id, special = true)`
            let piece = vocab.token_to_piece_bytes(*t);
            let _ = write!(out, "{:6} -> '", t);
            out.extend_from_slice(piece);
            out.extend_from_slice(b"'\n");
        }
    }
    if args.show_count {
        let _ = writeln!(out, "Total number of tokens: {}", tokens.len());
    }

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(&out);
    let _ = lock.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_match_string_process_escapes() {
        assert_eq!(process_escapes(b"a\\nb"), b"a\nb");
        assert_eq!(process_escapes(b"\\r\\t\\'\\\"\\\\"), b"\r\t'\"\\");
        assert_eq!(process_escapes(b"\\x41"), b"A");
        assert_eq!(process_escapes(b"\\\\x41"), b"\\x41");
        // unknown escape keeps the backslash
        assert_eq!(process_escapes(b"\\q"), b"\\q");
        // trailing backslash is literal
        assert_eq!(process_escapes(b"a\\"), b"a\\");
        // incomplete \x
        assert_eq!(process_escapes(b"\\x4"), b"\\x4");
        // non-hex digits
        assert_eq!(process_escapes(b"\\xzz"), b"\\xzz");
    }
}