#!/usr/bin/env python3
"""gen_fixtures.py — regenerate tokenizer parity fixtures from the reference
llama.cpp build (llama-tokenize).

Each fixture file contains records of the form:

    # <human-readable sentence>
    <hex of the UTF-8 sentence bytes>
    <space-separated token ids from the reference llama-tokenize>
    RT|NORT          (whether detokenize(tokenize(s)) must equal s)

Reference invocation mirrors the tool defaults: add_special == add_bos_token
of the model, parse_special == true, no escape processing (--no-escape).
"""

import os
import subprocess
import sys

TOKENIZE = "/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-tokenize"
MODELS = "/home/jeffrey/llm/llama.cpp-pinned/models"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

QWEN2_SENTENCES = [
    ("Hello world", True),
    (" Hello world", True),
    ("The capital of France is", True),
    ("中文English混合text", True),
    ("你好，世界！", True),
    ("这是一个测试句子，包含中文标点。", True),
    ("emoji test 🦙🚀🎉 done", True),
    ("numbers 1234567890 and 007", True),
    ("w048 7tuijk dsdfhu", True),
    ("tabs\tand\tspaces", True),
    ("line\nbreak", True),
    ("multiple   spaces   here", True),
    ("don't can't won't I'm we've they're he'll she'd it's", True),
    ("Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?", True),
    ("Ünïcödé àccented tëxt", True),
    ("日本語のテキストです", True),
    ("한국어 텍스트", True),
    ("   leading spaces", True),
    ("trailing spaces   ", True),
    ("Mixed 中English, numbers 42, and symbols #tag!", True),
    ("3.14159265358979323846", True),
    ("1,234,567.89", True),
    ("CamelCaseWords and snake_case_words and kebab-case-words", True),
    ("<|endoftext|>", False),
    ("text with <|endoftext|> in the middle", False),
    ("ΑΒΓΔ ΕΖΗΘ greek", True),
    ("Привет мир русский текст", True),
    ("a", True),
    ("0", True),
    ("llama.cpp is a C++ inference engine", True),
]

LLAMA_BPE_SENTENCES = [
    ("Hello world", True),
    ("The quick brown fox jumps over the lazy dog", True),
    ("The capital of France is", True),
    ("中文测试", True),
    ("Hello 🦙 emoji", True),
    ("123456789", True),
    ("don't stop believing", True),
    ("MiXeD CaSe WoRdS", True),
    ("tabs\tinside", True),
    ("new\nline", True),
    ("  double  spaces  ", True),
    ("Ünïcödé", True),
    ("I am 42 years old", True),
    ("नमस्ते दुनिया", True),
]

LLAMA_SPM_SENTENCES = [
    ("Hello world", True),
    (" Hello world", True),
    ("The capital of France is", True),
    ("中文测试", True),
    ("你好世界", True),
    ("emoji 🦙 here", True),
    ("1234567890", True),
    ("tabs\tand new\nlines", True),
    ("Ünïcödé àccënts", True),
    ("don't won't", True),
    ("Special: !@#$%", True),
    ("日本語テキスト", True),
    ("Привет мир", True),
]


# NOTE: WPM round-trip keeps a leading space in llama.cpp detokenize
# (add_space_prefix == false means no lstrip on the first piece), so all
# bert-bge cases are ids-only.
BERT_BGE_SENTENCES = [
    ("hello world", False),
    ("the quick brown fox", False),
    ("hello, world", False),
    ("Numbers 123 and 456", False),   # normalizer lowercases
    ("caf\u00e9 na\u00efve", False),  # accents stripped
    ("don't stop", False),
    ("[CLS] style specials [SEP]", False),
    ("mixed \u4e2d\u6587 bert", False),
]

GPT2_SENTENCES = [
    ("Hello world", True),
    ("The capital of France is", True),
    ("hello \U0001f999 emoji", True),
    ("w048 7tuijk dsdfhu", True),
    ("   leading spaces", True),
    ("don't can't won't", True),
    ("1234567890", True),
    ("\u4e2d\u6587\u6d4b\u8bd5", True),
]

DEEPSEEK_LLM_SENTENCES = [
    ("Hello world", True),
    ("\u4e2d\u6587\u6d4b\u8bd5", True),
    ("def f():\n    return 1", False),
    ("123456789", True),
    ("\u03b1\u03b2\u03b3 symbols", True),
    ("\tindented code", True),
    ("don't \u00e9\u00e8", True),
    ("line1\nline2", True),
]

QWEN35_SENTENCES = [
    ("Hello world", True),
    ("\u4f60\u597d\u4e16\u754c", True),
    ("don't won't I'm", True),
    ("1 22 333 4444", True),
    ("caf\u00e9 na\u00efve combining", True),
    ("<|endoftext|>", False),
    ("e\u0301 combining marks", True),
    ("Mixed \u4e2dEnglish 42 #tag", True),
]

FALCON_SENTENCES = [
    ("Hello world", True),
    ("code {a+b}*2", True),
    ("x**2 + y_1 = z", True),
    ("123456", True),
    ("don't stop", True),
    ("\u4e2d\u6587\u5b57\u7b26", True),
    ("tabs\tand spaces", True),
    ("The capital of France is", True),
]

COMMAND_R_SENTENCES = [
    ("Hello world", True),
    ("Bonjour le monde", True),
    ("def main():\n    pass", False),
    ("42.195 km marathon", True),
    (" \u65e5\u672c\u8a9e\u30c6\u30ad\u30b9\u30c8", True),
    ("emoji \U0001f680 rocket", True),
    ("<|END_OF_TURN_TOKEN|>", False),
    ("don't can't", True),
]

PHI3_SENTENCES = [
    ("Hello world", True),
    (" Phi-3 test sentence", True),
    ("\u4f60\u597d\u4e16\u754c", True),
    ("1234 numbers", True),
    ("don't won't", True),
    ("tab\there", True),
    ("\u00dcn\u00efc\u00f6d\u00e9", True),
    ("<|endoftext|>", False),
]

GEMMA4_SENTENCES = [
    ("Hello world", True),
    ("\u4f60\u597d\u4e16\u754c", True),
    ("<start_of_turn>", False),
    ("numbers 12345", True),
    ("emoji \U0001f999 llama", True),
    ("line\nbreak", True),
    ("don't stop", True),
    ("\u00dcn\u00efc\u00f6d\u00e9", True),
]

QWEN25_REAL_SENTENCES = [
    ("The capital of France is", True),
    ("Hello world", True),
    ("你好，世界！", True),
]


def ref_tokenize(model: str, text: str) -> list:
    proc = subprocess.run(
        [TOKENIZE, "-m", model, "--stdin", "--ids", "--no-escape"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    out = proc.stdout.decode().strip()
    assert out.startswith("[") and out.endswith("]"), out
    return [int(x) for x in out[1:-1].split(",")]


def write_fixture(path: str, model: str, sentences):
    lines = [f"# model: {model}"]
    for text, rt in sentences:
        ids = ref_tokenize(model, text)
        lines.append(f"# {text!r}")
        lines.append(text.encode("utf-8").hex())
        lines.append(" ".join(str(i) for i in ids))
        lines.append("RT" if rt else "NORT")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    n_rt = sum(1 for _, rt in sentences if rt)
    print(f"{os.path.basename(path)}: {len(sentences)} sentences ({n_rt} round-trip)")


def main():
    write_fixture(os.path.join(OUT_DIR, "qwen2.txt"), f"{MODELS}/ggml-vocab-qwen2.gguf", QWEN2_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "llama-bpe.txt"), f"{MODELS}/ggml-vocab-llama-bpe.gguf", LLAMA_BPE_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "llama-spm.txt"), f"{MODELS}/ggml-vocab-llama-spm.gguf", LLAMA_SPM_SENTENCES)
    write_fixture(
        os.path.join(OUT_DIR, "qwen25-real.txt"),
        "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        QWEN25_REAL_SENTENCES,
    )
    write_fixture(os.path.join(OUT_DIR, "bert-bge.txt"), f"{MODELS}/ggml-vocab-bert-bge.gguf", BERT_BGE_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "gpt-2.txt"), f"{MODELS}/ggml-vocab-gpt-2.gguf", GPT2_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "deepseek-llm.txt"), f"{MODELS}/ggml-vocab-deepseek-llm.gguf", DEEPSEEK_LLM_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "qwen35.txt"), f"{MODELS}/ggml-vocab-qwen35.gguf", QWEN35_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "falcon.txt"), f"{MODELS}/ggml-vocab-falcon.gguf", FALCON_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "command-r.txt"), f"{MODELS}/ggml-vocab-command-r.gguf", COMMAND_R_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "phi-3.txt"), f"{MODELS}/ggml-vocab-phi-3.gguf", PHI3_SENTENCES)
    write_fixture(os.path.join(OUT_DIR, "gemma-4.txt"), f"{MODELS}/ggml-vocab-gemma-4.gguf", GEMMA4_SENTENCES)


if __name__ == "__main__":
    main()
