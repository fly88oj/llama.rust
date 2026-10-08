//! chat.rs verification harness (differential run against /tmp/chatref).
//! TEMPORARY: reads the fixture produced by the extracted C++ reference
//! (llm_chat_apply_template) and reports mismatches. Replaced by embedded
//! static expectations once the differential run is clean.

use llama::chat::{self, ChatMessage, ChatTemplate, Role};

fn cases() -> Vec<(String, Vec<(Role, &'static str)>)> {
    vec![
        ("A".into(), vec![(Role::User, "Hello")]),
        (
            "B".into(),
            vec![(Role::System, "You are helpful."), (Role::User, "Hi")],
        ),
        (
            "C".into(),
            vec![
                (Role::User, "  padded  "),
                (Role::Assistant, "a1"),
                (Role::Assistant, "a2"),
                (Role::User, "again"),
            ],
        ),
        (
            "D".into(),
            vec![
                (Role::System, "s1"),
                (Role::System, "s2"),
                (Role::User, "u"),
                (Role::Assistant, "a"),
            ],
        ),
        (
            "E".into(),
            vec![
                (Role::User, "u1"),
                (Role::Assistant, "a1"),
                (Role::User, "u2"),
            ],
        ),
        (
            "F".into(),
            vec![
                (Role::Tool, "t1"),
                (Role::Function, "f1"),
                (Role::AssistantToolCall, "atc"),
            ],
        ),
        ("G".into(), vec![]),
        ("H".into(), vec![(Role::System, "sys only")]),
        (
            "I".into(),
            vec![
                (Role::User, "  \t x \n "),
                (Role::Assistant, "  trimmed?  "),
            ],
        ),
    ]
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[test]
fn differential_all_templates() {
    let path = std::env::var("CHAT_FIXTURE").unwrap_or_else(|_| "/tmp/chatref/fixture.tsv".into());
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: cannot read {path}: {e}");
            return;
        }
    };
    let all_cases = cases();
    let mut mismatches = 0usize;
    let mut total = 0usize;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json_lite::Value = serde_json_lite::parse(line);
        let arr = v.as_array();
        let tmpl_name = arr[0].as_str().to_string();
        let case_id = arr[1].as_str().to_string();
        let add_ass = arr[2].as_str() == "1";
        let expected = arr[3].as_str().to_string();
        // enum order name -> ChatTemplate
        let tmpl = match template_by_cpp_name(&tmpl_name) {
            Some(t) => t,
            None => {
                eprintln!("no rust mapping for {tmpl_name}");
                mismatches += 1;
                continue;
            }
        };
        let (_, msgs) = all_cases.iter().find(|(id, _)| *id == case_id).unwrap();
        let messages: Vec<ChatMessage> =
            msgs.iter().map(|(r, c)| ChatMessage::new(*r, *c)).collect();
        let got = chat::apply(tmpl, &messages, add_ass).unwrap();
        total += 1;
        if escape(&got) != expected {
            mismatches += 1;
            println!("MISMATCH {tmpl_name} case={case_id} add_ass={add_ass}");
            println!("  exp: {expected}");
            println!("  got: {}", escape(&got));
        }
    }
    println!("checked {total} cases, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "{mismatches}/{total} mismatches");
}

#[test]
fn differential_detect() {
    let path = std::env::var("CHAT_DETECT_FIXTURE")
        .unwrap_or_else(|_| "/tmp/chatref/detect_fixture.tsv".into());
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: cannot read {path}: {e}");
            return;
        }
    };
    let mut mismatches = 0usize;
    let mut total = 0usize;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v = serde_json_lite::parse(line);
        let arr = v.as_array();
        let tmpl = arr[0].as_str().to_string();
        let cpp_idx: usize = arr[1].as_str().parse().unwrap();
        let got = cpp_index(chat::detect(&tmpl));
        total += 1;
        if got != cpp_idx {
            mismatches += 1;
            println!("DETECT MISMATCH cpp={cpp_idx} rust={got} tmpl={tmpl:?}");
        }
    }
    println!("detect: checked {total} templates, {mismatches} mismatches");
    assert_eq!(mismatches, 0);
}

/// C++ enum order `llm_chat_template` (src/llama-chat.h:7-64) — the value
/// produced by the extracted reference `llm_chat_detect_template`.
fn cpp_index(t: ChatTemplate) -> usize {
    use ChatTemplate::*;
    let all = [
        Chatml,
        Llama2,
        Llama2Sys,
        Llama2SysBos,
        Llama2SysStrip,
        MistralV1,
        MistralV3,
        MistralV3Tekken,
        MistralV7,
        MistralV7Tekken,
        Phi3,
        Phi4,
        Falcon3,
        Zephyr,
        Monarch,
        Gemma,
        Orion,
        Openchat,
        Vicuna,
        VicunaOrca,
        Deepseek,
        Deepseek2,
        Deepseek3,
        DeepseekOcr,
        CommandR,
        Llama3,
        Chatglm3,
        Chatglm4,
        Glmedge,
        Minicpm,
        Exaone3,
        Exaone4,
        ExaoneMoe,
        RwkvWorld,
        Granite3X,
        Granite40,
        Granite41,
        Gigachat,
        Megrez,
        Yandex,
        Bailing,
        BailingThink,
        Bailing2,
        Llama4,
        Smolvlm,
        Dots1,
        HunyuanMoe,
        OpenaiMoe,
        HunyuanDense,
        HunyuanVl,
        KimiK2,
        SeedOss,
        Grok2,
        PanguEmbed,
        SolarOpen,
        Unknown,
    ];
    all.iter()
        .position(|x| *x == t)
        .expect("template in enum order")
}

/// C++ enum order (llama-chat.h:7-64) -> ChatTemplate, for fixtures addressed
/// by template.
fn template_by_cpp_name(n: &str) -> Option<ChatTemplate> {
    use ChatTemplate::*;
    Some(match n {
        "LLM_CHAT_TEMPLATE_CHATML" => Chatml,
        "LLM_CHAT_TEMPLATE_LLAMA_2" => Llama2,
        "LLM_CHAT_TEMPLATE_LLAMA_2_SYS" => Llama2Sys,
        "LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS" => Llama2SysBos,
        "LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP" => Llama2SysStrip,
        "LLM_CHAT_TEMPLATE_MISTRAL_V1" => MistralV1,
        "LLM_CHAT_TEMPLATE_MISTRAL_V3" => MistralV3,
        "LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN" => MistralV3Tekken,
        "LLM_CHAT_TEMPLATE_MISTRAL_V7" => MistralV7,
        "LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN" => MistralV7Tekken,
        "LLM_CHAT_TEMPLATE_PHI_3" => Phi3,
        "LLM_CHAT_TEMPLATE_PHI_4" => Phi4,
        "LLM_CHAT_TEMPLATE_FALCON_3" => Falcon3,
        "LLM_CHAT_TEMPLATE_ZEPHYR" => Zephyr,
        "LLM_CHAT_TEMPLATE_MONARCH" => Monarch,
        "LLM_CHAT_TEMPLATE_GEMMA" => Gemma,
        "LLM_CHAT_TEMPLATE_ORION" => Orion,
        "LLM_CHAT_TEMPLATE_OPENCHAT" => Openchat,
        "LLM_CHAT_TEMPLATE_VICUNA" => Vicuna,
        "LLM_CHAT_TEMPLATE_VICUNA_ORCA" => VicunaOrca,
        "LLM_CHAT_TEMPLATE_DEEPSEEK" => Deepseek,
        "LLM_CHAT_TEMPLATE_DEEPSEEK_2" => Deepseek2,
        "LLM_CHAT_TEMPLATE_DEEPSEEK_3" => Deepseek3,
        "LLM_CHAT_TEMPLATE_DEEPSEEK_OCR" => DeepseekOcr,
        "LLM_CHAT_TEMPLATE_COMMAND_R" => CommandR,
        "LLM_CHAT_TEMPLATE_LLAMA_3" => Llama3,
        "LLM_CHAT_TEMPLATE_CHATGLM_3" => Chatglm3,
        "LLM_CHAT_TEMPLATE_CHATGLM_4" => Chatglm4,
        "LLM_CHAT_TEMPLATE_GLMEDGE" => Glmedge,
        "LLM_CHAT_TEMPLATE_MINICPM" => Minicpm,
        "LLM_CHAT_TEMPLATE_EXAONE_3" => Exaone3,
        "LLM_CHAT_TEMPLATE_EXAONE_4" => Exaone4,
        "LLM_CHAT_TEMPLATE_EXAONE_MOE" => ExaoneMoe,
        "LLM_CHAT_TEMPLATE_RWKV_WORLD" => RwkvWorld,
        "LLM_CHAT_TEMPLATE_GRANITE_3_X" => Granite3X,
        "LLM_CHAT_TEMPLATE_GRANITE_4_0" => Granite40,
        "LLM_CHAT_TEMPLATE_GRANITE_4_1" => Granite41,
        "LLM_CHAT_TEMPLATE_GIGACHAT" => Gigachat,
        "LLM_CHAT_TEMPLATE_MEGREZ" => Megrez,
        "LLM_CHAT_TEMPLATE_YANDEX" => Yandex,
        "LLM_CHAT_TEMPLATE_BAILING" => Bailing,
        "LLM_CHAT_TEMPLATE_BAILING_THINK" => BailingThink,
        "LLM_CHAT_TEMPLATE_BAILING2" => Bailing2,
        "LLM_CHAT_TEMPLATE_LLAMA4" => Llama4,
        "LLM_CHAT_TEMPLATE_SMOLVLM" => Smolvlm,
        "LLM_CHAT_TEMPLATE_DOTS1" => Dots1,
        "LLM_CHAT_TEMPLATE_HUNYUAN_MOE" => HunyuanMoe,
        "LLM_CHAT_TEMPLATE_OPENAI_MOE" => OpenaiMoe,
        "LLM_CHAT_TEMPLATE_HUNYUAN_DENSE" => HunyuanDense,
        "LLM_CHAT_TEMPLATE_HUNYUAN_VL" => HunyuanVl,
        "LLM_CHAT_TEMPLATE_KIMI_K2" => KimiK2,
        "LLM_CHAT_TEMPLATE_SEED_OSS" => SeedOss,
        "LLM_CHAT_TEMPLATE_GROK_2" => Grok2,
        "LLM_CHAT_TEMPLATE_PANGU_EMBED" => PanguEmbed,
        "LLM_CHAT_TEMPLATE_SOLAR_OPEN" => SolarOpen,
        _ => return None,
    })
}

/// Minimal JSON reader for the fixture rows (4-element string arrays).
mod serde_json_lite {
    pub enum Value {
        Str(String),
        Arr(Vec<Value>),
    }

    impl Value {
        pub fn as_array(&self) -> &Vec<Value> {
            match self {
                Value::Arr(a) => a,
                _ => panic!("not an array"),
            }
        }
        pub fn as_str(&self) -> &str {
            match self {
                Value::Str(s) => s,
                _ => panic!("not a string"),
            }
        }
    }

    pub fn parse(s: &str) -> Value {
        let mut chars = s.chars().peekable();
        parse_value(&mut chars)
    }

    fn parse_value(it: &mut std::iter::Peekable<std::str::Chars>) -> Value {
        let c = *it.peek().unwrap();
        match c {
            '[' => {
                it.next();
                let mut v = Vec::new();
                loop {
                    match it.peek() {
                        Some(']') => {
                            it.next();
                            break;
                        }
                        Some(',') => {
                            it.next();
                        }
                        _ => v.push(parse_value(it)),
                    }
                }
                Value::Arr(v)
            }
            '"' => {
                it.next();
                let mut s = String::new();
                loop {
                    let c = it.next().unwrap();
                    match c {
                        '"' => break,
                        '\\' => {
                            let e = it.next().unwrap();
                            match e {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                'r' => s.push('\r'),
                                'u' => {
                                    let hex: String = (0..4).map(|_| it.next().unwrap()).collect();
                                    let cp = u32::from_str_radix(&hex, 16).unwrap();
                                    s.push(char::from_u32(cp).unwrap_or('?'));
                                }
                                other => s.push(other),
                            }
                        }
                        other => s.push(other),
                    }
                }
                Value::Str(s)
            }
            c if c.is_ascii_digit() || c == '-' => {
                let mut s = String::new();
                while let Some(&c) = it.peek() {
                    if c.is_ascii_digit() || c == '-' || c == '.' || c == 'e' || c == 'E' {
                        s.push(c);
                        it.next();
                    } else {
                        break;
                    }
                }
                Value::Str(s)
            }
            _ => panic!("unexpected char {c:?}"),
        }
    }
}
