#!/usr/bin/env python3
"""vocab_pre_gen.py — one-shot generator for the BPE pre-tokenizer regex table.

Parses the `llm_tokenizer_bpe` constructor switch in llama-vocab.cpp and emits
a Rust `match` returning (regex exprs, byte_encode) per pre-type. Output is
pasted into vocab.rs verbatim (run once; kept for provenance).

Usage: python3 vocab_pre_gen.py <llama-vocab.cpp> <output.rust.txt>
"""

import re
import sys


def strip_comments(text: str) -> str:
    out = []
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                out.append(text[i : i + 2])
                i += 2
                continue
            if c == '"':
                in_str = False
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # line comment: skip to newline
            j = text.find("\n", i)
            i = len(text) if j == -1 else j
            continue
        if c == "/" and i + 1 < len(text) and text[i + 1] == "*":
            j = text.find("*/", i)
            i = len(text) if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def camel(name: str) -> str:
    # LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM -> DeepseekLlm
    parts = name.split("_")
    out = "".join(p.capitalize() if p not in ("LLM", "BPE", "GPT", "MOE", "WPM") else p.title() for p in parts)
    fixes = {
        "DeepseekLlm": "DeepseekLlm",
        "Deepseek3Llm": "Deepseek3Llm",
        "DeepseekCoder": "DeepseekCoder",
        "Gpt2": "Gpt2",
        "Gpt3Finnish": "Gpt3Finnish",
        "Gpt4O": "Gpt4O",
        "Chatglm3": "Chatglm3",
        "Chatglm4": "Chatglm4",
        "Stablelm2": "Stablelm2",
        "Smollm": "Smollm",
        "Codeshell": "Codeshell",
        "CommandR": "CommandR",
        "Starcoder": "Starcoder",
        "Refact": "Refact",
        "Olmo": "Olmo",
        "Dbrx": "Dbrx",
        "Smaug": "Smaug",
        "Poro": "Poro",
        "Jais": "Jais",
        "Jais2": "Jais2",
        "Tekken": "Tekken",
        "Exaone": "Exaone",
        "ExaoneMoe": "ExaoneMoe",
        "Chameleon": "Chameleon",
        "Minerva": "Minerva",
        "Superbpe": "Superbpe",
        "Trillion": "Trillion",
        "Bailingmoe": "Bailingmoe",
        "Llama3": "Llama3",
        "Llama4": "Llama4",
        "Pixtral": "Pixtral",
        "SeedCoder": "SeedCoder",
        "Hunyuan": "Hunyuan",
        "HunyuanDense": "HunyuanDense",
        "KimiK2": "KimiK2",
        "Grok2": "Grok2",
        "GraniteDocling": "GraniteDocling",
        "MinimaxM2": "MinimaxM2",
        "Afmoe": "Afmoe",
        "SolarOpen": "SolarOpen",
        "Youtu": "Youtu",
        "Qwen2": "Qwen2",
        "Qwen35": "Qwen35",
        "TinyAya": "TinyAya",
        "JoyaiLlm": "JoyaiLlm",
        "Gemma4": "Gemma4",
        "SarvamMoe": "SarvamMoe",
        "Minicpm5": "Minicpm5",
        "Whitespace": "Whitespace",
        "GraniteEmbMulti": "GraniteEmbMulti",
        "Mellum2": "Mellum2",
        "Laguna": "Laguna",
        "HyV4": "HyV4",
        "Spark2_5": "Spark2_5",
        "Ufakzeka": "Ufakzeka",
        "Mpt": "Mpt",
        "Falcon": "Falcon",
        "Default": "Default",
    }
    return fixes.get(out, out)


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    start = src.find("llm_tokenizer_bpe(const llama_vocab & vocab) {")
    end = src.find("struct llm_tokenizer_bpe_session")
    region = strip_comments(src[start:end])

    # parse: groups of case labels -> (exprs, byte_encode_false)
    case_re = re.compile(r"case LLAMA_VOCAB_PRE_TYPE_(\w+):")
    expr_re = re.compile(r"regex_exprs\s*=\s*\{(.*?)\};", re.S)

    # split region at 'case ' boundaries; a label whose body carries no
    # regex_exprs assignment is a C++ fallthrough — it joins the next group
    pieces = case_re.split(region)
    # pieces: [prelude, label1, body1, label2, body2, ...]
    groups = []  # ([labels], [exprs], byte_encode_false)
    pending_labels = []
    for i in range(1, len(pieces) - 1, 2):
        label = pieces[i]
        body = pieces[i + 1]
        pending_labels.append(label)
        m = expr_re.search(body)
        if not m:
            continue  # fallthrough into the next case
        exprs = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        no_byte_encode = "byte_encode = false" in body
        groups.append((pending_labels, exprs, no_byte_encode))
        pending_labels = []

    # trailing `default:` arm (also consumes any pending fallthrough labels)
    last_case_pos = region.rfind("case LLAMA_VOCAB_PRE_TYPE_")
    tail = region[last_case_pos:]
    # take text after the final case group's closing "};"
    tail = tail[tail.find("};") + 2 :] if "};" in tail else tail
    m = re.search(r"default:\s*regex_exprs\s*=\s*\{(.*?)\};", tail, re.S)
    if m is None:
        m = expr_re.search(tail)
    assert m, "default arm not found"
    default_exprs = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))

    lines = []
    lines.append("/// Pre-tokenizer regex table — port of the `llm_tokenizer_bpe` constructor")
    lines.append("/// switch in llama-vocab.cpp (generated by vocab_pre_gen.py; do not hand-edit).")
    lines.append("fn bpe_pre_regexes(pre: PreType) -> (Vec<String>, bool) {")
    lines.append("    let exprs: &[&str] = match pre {")
    no_enc = []
    for labels, exprs, nbe in groups:
        pats = " | ".join(f"PreType::{camel(l)}" for l in labels)
        if nbe:
            no_enc.extend(labels)
        lines.append(f"        {pats} => &[")
        for e in exprs:
            lines.append(f'            "{e}",')
        lines.append("        ],")
    lines.append("        // default regex for BPE tokenization pre-processing")
    lines.append("        _ => &[")
    for e in default_exprs:
        lines.append(f'            "{e}",')
    lines.append("        ],")
    lines.append("    };")
    if no_enc:
        pats = " | ".join(f"PreType::{camel(l)}" for l in no_enc)
        lines.append(f"    // byte_encode = false (SPM-style BPE on raw UTF-8)")
        lines.append(f"    let byte_encode = !matches!(pre, {pats});")
    else:
        lines.append("    let byte_encode = true;")
    lines.append("    (exprs.iter().map(|s| s.to_string()).collect(), byte_encode)")
    lines.append("}")
    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(groups)} groups, {sum(len(g[0]) for g in groups)} pre-types, no_byte_encode={no_enc}")


if __name__ == "__main__":
    main()
