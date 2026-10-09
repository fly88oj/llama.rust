//! chat.rs — chat template layer, 1:1 port of llama.cpp `src/llama-chat.cpp` +
//! `src/llama-chat.h` (llm_chat_template enum / LLM_CHAT_TEMPLATES name map /
//! llm_chat_detect_template sniffing / llm_chat_apply_template builtin
//! formatters / llama_chat_builtin_templates), baseline bd4f514db1.
//!
//! The pinned tree's `common/chat.cpp` renders the *generic* path with a real
//! Jinja engine (`common/jinja/*`, ~6.4k lines). That engine is ported here as
//! the [`mini_jinja`] module (below) against the constructs the chat-template
//! corpus uses — value model (value.cpp), lexer (lexer.cpp), parser
//! (parser.cpp) and runtime (runtime.cpp) semantics; the capability analysis
//! (`caps`, common/jinja/caps.cpp) runs on top of it. Everything outside the
//! ported surface stays a loud parse/eval error.
//!
//! What *is* ported 1:1 is everything the non-jinja path of
//! `common/chat.cpp:1402` uses:
//! `llama_chat_apply_template(src, ...)` → `llm_chat_detect_template` +
//! `llm_chat_apply_template`, i.e. exactly what `llama-cli --chat-template
//! <name-or-gguf-template>` does without `--jinja`.
//!
//! C++ → Rust reference map (all line numbers = pinned tree):
//!   * enum `llm_chat_template`      — src/llama-chat.h:7-64
//!   * `LLM_CHAT_TEMPLATES` name map — src/llama-chat.cpp:28-83 (std::map ⇒
//!     byte-wise lexicographic order, kept for `builtin_templates`)
//!   * `llm_chat_template_from_str`  — src/llama-chat.cpp:85-87
//!   * `llm_chat_detect_template`    — src/llama-chat.cpp:89-240
//!   * `trim` (C isspace)            — src/llama-chat.cpp:16-26
//!   * `llm_chat_apply_template`     — src/llama-chat.cpp:244-946
//!   * `llama_chat_builtin_templates`— src/llama-chat.cpp:950-957
//!   * `llama_chat_apply_template` wrapper (detect+apply) — src/llama.cpp:511-536
//!   * mini-jinja                    — common/jinja/{lexer,parser,runtime,value}.cpp
//!
//! Deviations (documented, no C++ counterpart):
//!   * `apply` returns `Result<String, String>` instead of the C++ `int32_t`
//!     return (`-1` on unsupported template ⇒ `Err`; the formatted length is
//!     the Rust `String` length so the size is not returned separately).
//!   * [`eot_prefix`] is a derived helper (no such C++ function): it returns
//!     the end-of-turn marker each builtin formatter itself emits, for
//!     generation-stop detection in the cli chat loop. Where the builtin
//!     formatter emits no marker the model family's documented EOS special
//!     token is used (marked `derived`).
//!   * `apply_str`'s `strftime_now` renders in UTC with a pinnable clock
//!     (C++ value.cpp:370-381 uses `std::localtime` — timezone-dependent, so
//!     not reproducible; tests pin the epoch via `ChatTemplateCtx::now`).
//!
//! `Role` covers every role string the C++ formatter branches on
//! ("system"/"user"/"assistant"/"tool"/"function"/"assistant_tool_call");
//! pass-through templates format `role.as_str()` exactly like the C++ copies
//! `message->role`.

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Role / ChatMessage (llama.h:462-465 llama_chat_message; role literals as
// used across llama-chat.cpp)
// ---------------------------------------------------------------------------

/// Chat role. Variants are every role string `llm_chat_apply_template` branches
/// on (llama-chat.cpp: e.g. "tool" exaone-4/kimi-k2/pangu, "function" pangu,
/// "assistant_tool_call" granite).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Function,
    AssistantToolCall,
}

impl Role {
    /// Role string, exactly the literals used by the C++ code paths.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Function => "function",
            Role::AssistantToolCall => "assistant_tool_call",
        }
    }

    /// Parse a role string (inverse of [`Role::as_str`]); unknown ⇒ `None`.
    pub fn from_str(s: &str) -> Option<Role> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            "function" => Some(Role::Function),
            "assistant_tool_call" => Some(Role::AssistantToolCall),
            _ => None,
        }
    }
}

/// One chat message (`llama_chat_message`: `{ const char * role; const char *
/// content; }`, llama.h:462-465).
#[derive(Clone, PartialEq, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        ChatMessage {
            role,
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ChatTemplate — llm_chat_template (llama-chat.h:7-64)
// ---------------------------------------------------------------------------

/// Builtin chat templates, 1:1 with `enum llm_chat_template`
/// (src/llama-chat.h:7-64). `Unknown` = `LLM_CHAT_TEMPLATE_UNKNOWN`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ChatTemplate {
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
}

/// Name ⇄ template table — port of `LLM_CHAT_TEMPLATES`
/// (llama-chat.cpp:28-83). `std::map<std::string, ...>` iterates in byte-wise
/// lexicographic key order, which `llama_chat_builtin_templates` exposes; this
/// table is kept sorted the same way (asserted in tests). C++ line of each
/// entry given in the comment.
static BUILTIN_TEMPLATES: [(&str, ChatTemplate); 54] = [
    ("bailing", ChatTemplate::Bailing),            // llama-chat.cpp:69
    ("bailing-think", ChatTemplate::BailingThink), // :70
    ("bailing2", ChatTemplate::Bailing2),          // :71
    ("chatglm3", ChatTemplate::Chatglm3),          // :55
    ("chatglm4", ChatTemplate::Chatglm4),          // :56
    ("chatml", ChatTemplate::Chatml),              // :29
    ("command-r", ChatTemplate::CommandR),         // :53
    ("deepseek", ChatTemplate::Deepseek),          // :49
    ("deepseek-ocr", ChatTemplate::DeepseekOcr),   // :52
    ("deepseek2", ChatTemplate::Deepseek2),        // :50
    ("deepseek3", ChatTemplate::Deepseek3),        // :51
    ("exaone-moe", ChatTemplate::ExaoneMoe),       // :61
    ("exaone3", ChatTemplate::Exaone3),            // :59
    ("exaone4", ChatTemplate::Exaone4),            // :60
    ("falcon3", ChatTemplate::Falcon3),            // :41
    ("gemma", ChatTemplate::Gemma),                // :44
    ("gigachat", ChatTemplate::Gigachat),          // :66
    ("glmedge", ChatTemplate::Glmedge),            // :57
    ("gpt-oss", ChatTemplate::OpenaiMoe),          // :75
    ("granite", ChatTemplate::Granite3X),          // :63
    ("granite-4.0", ChatTemplate::Granite40),      // :64
    ("granite-4.1", ChatTemplate::Granite41),      // :65
    ("grok-2", ChatTemplate::Grok2),               // :80
    ("hunyuan-dense", ChatTemplate::HunyuanDense), // :76
    ("hunyuan-moe", ChatTemplate::HunyuanMoe),     // :74
    ("hunyuan-vl", ChatTemplate::HunyuanVl),       // :77
    ("kimi-k2", ChatTemplate::KimiK2),             // :78
    ("llama2", ChatTemplate::Llama2),              // :30
    ("llama2-sys", ChatTemplate::Llama2Sys),       // :31
    ("llama2-sys-bos", ChatTemplate::Llama2SysBos), // :32
    ("llama2-sys-strip", ChatTemplate::Llama2SysStrip), // :33
    ("llama3", ChatTemplate::Llama3),              // :54
    ("llama4", ChatTemplate::Llama4),              // :72
    ("megrez", ChatTemplate::Megrez),              // :67
    ("minicpm", ChatTemplate::Minicpm),            // :58
    ("mistral-v1", ChatTemplate::MistralV1),       // :34
    ("mistral-v3", ChatTemplate::MistralV3),       // :35
    ("mistral-v3-tekken", ChatTemplate::MistralV3Tekken), // :36
    ("mistral-v7", ChatTemplate::MistralV7),       // :37
    ("mistral-v7-tekken", ChatTemplate::MistralV7Tekken), // :38
    ("monarch", ChatTemplate::Monarch),            // :43
    ("openchat", ChatTemplate::Openchat),          // :46
    ("orion", ChatTemplate::Orion),                // :45
    ("pangu-embedded", ChatTemplate::PanguEmbed),  // :81
    ("phi3", ChatTemplate::Phi3),                  // :39
    ("phi4", ChatTemplate::Phi4),                  // :40
    ("rwkv-world", ChatTemplate::RwkvWorld),       // :62
    ("seed_oss", ChatTemplate::SeedOss),           // :79
    ("smolvlm", ChatTemplate::Smolvlm),            // :73
    ("solar-open", ChatTemplate::SolarOpen),       // :82
    ("vicuna", ChatTemplate::Vicuna),              // :47
    ("vicuna-orca", ChatTemplate::VicunaOrca),     // :48
    ("yandex", ChatTemplate::Yandex),              // :68
    ("zephyr", ChatTemplate::Zephyr),              // :42
];

impl ChatTemplate {
    /// Reverse lookup: canonical `--chat-template` name for this template.
    /// `None` for [`ChatTemplate::Dots1`] (detectable but has no name in the
    /// C++ map) and [`ChatTemplate::Unknown`].
    pub fn name(self) -> Option<&'static str> {
        BUILTIN_TEMPLATES
            .iter()
            .find(|(_, t)| *t == self)
            .map(|(n, _)| *n)
    }

    /// `llm_chat_template_from_str` (llama-chat.cpp:85-87; C++ throws
    /// `std::out_of_range` — here `Option`).
    pub fn from_name(name: &str) -> Option<ChatTemplate> {
        BUILTIN_TEMPLATES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
    }
}

/// `llama_chat_builtin_templates` (llama-chat.cpp:950-957): all builtin
/// template names in `std::map` (lexicographic) order. Sortedness is asserted
/// by tests.
pub fn builtin_templates() -> Vec<&'static str> {
    BUILTIN_TEMPLATES.iter().map(|(n, _)| *n).collect()
}

// ---------------------------------------------------------------------------
// trim — llama-chat.cpp:16-26 (C isspace on unsigned char)
// ---------------------------------------------------------------------------

#[inline]
fn c_isspace(c: u8) -> bool {
    // C isspace: ' ' \t \n \v(0x0B) \f(0x0C) \r
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// `c_isspace` exposed for chat_tools.rs — its trims (chat-auto-parser-helpers
/// / chat-peg-parser) share the exact C semantics.
pub fn c_isspace_pub(c: u8) -> bool {
    c_isspace(c)
}

/// [`trim`] exposed for chat_tools.rs.
pub fn trim_pub(s: &str) -> &str {
    trim(s)
}

/// Trim whitespace from both ends (ASCII, C `isspace` semantics) —
/// `trim()` llama-chat.cpp:16-26.
fn trim(s: &str) -> &str {
    let b = s.as_bytes();
    let mut start = 0usize;
    let mut end = b.len();
    while start < end && c_isspace(b[start]) {
        start += 1;
    }
    while end > start && c_isspace(b[end - 1]) {
        end -= 1;
    }
    &s[start..end]
}

// ---------------------------------------------------------------------------
// detect — llm_chat_detect_template (llama-chat.cpp:89-240)
// ---------------------------------------------------------------------------

/// Sniff a chat template from a template string: first an exact builtin-name
/// lookup, then substring heuristics. 1:1 port of `llm_chat_detect_template`
/// (llama-chat.cpp:89-240) including branch order.
pub fn detect(tmpl: &str) -> ChatTemplate {
    // llama-chat.cpp:90-94: exact name wins
    if let Some(t) = ChatTemplate::from_name(tmpl) {
        return t;
    }

    let contains = |haystack: &str| tmpl.contains(haystack);

    if contains("<|im_start|>") {
        // llama-chat.cpp:99-104
        if contains("<|im_sep|>") {
            ChatTemplate::Phi4
        } else if contains("<end_of_utterance>") {
            // SmolVLM uses <|im_start|> as BOS, but it is NOT chatml
            ChatTemplate::Smolvlm
        } else {
            ChatTemplate::Chatml
        }
    } else if tmpl.starts_with("mistral") || contains("[INST]") {
        // llama-chat.cpp:105-139
        if contains("[SYSTEM_PROMPT]") {
            ChatTemplate::MistralV7
        } else if contains("' [INST] ' + system_message")
            // official 'v1'; 'v3'/'v3-tekken' are caught by [AVAILABLE_TOOLS]
            || contains("[AVAILABLE_TOOLS]")
        {
            if contains(" [INST]") {
                ChatTemplate::MistralV1
            } else if contains("\"[INST]\"") {
                ChatTemplate::MistralV3Tekken
            } else {
                ChatTemplate::MistralV3
            }
        } else {
            // llama2 template and its variants
            let support_system_message = contains("<<SYS>>");
            let add_bos_inside_history = contains("bos_token + '[INST]");
            let strip_message = contains("content.strip()");
            if strip_message {
                ChatTemplate::Llama2SysStrip
            } else if add_bos_inside_history {
                ChatTemplate::Llama2SysBos
            } else if support_system_message {
                ChatTemplate::Llama2Sys
            } else {
                ChatTemplate::Llama2
            }
        }
    } else if contains("<|assistant|>") && contains("<|end|>") {
        // llama-chat.cpp:140-141
        ChatTemplate::Phi3
    } else if contains("[gMASK]<sop>") {
        // :142-143
        ChatTemplate::Chatglm4
    } else if contains("<|assistant|>") && contains("<|user|>") {
        // :144-148
        if contains("<|tool_declare|>") {
            ChatTemplate::ExaoneMoe
        } else if contains("</s>") {
            ChatTemplate::Falcon3
        } else {
            ChatTemplate::Glmedge
        }
    } else if contains("<|{{ item['role'] }}|>") && contains("<|begin_of_image|>") {
        // :149-150
        ChatTemplate::Glmedge
    } else if contains("<|user|>") && contains("<|endoftext|>") {
        // :151-152
        ChatTemplate::Zephyr
    } else if contains("bos_token + message['role']") {
        // :153-154
        ChatTemplate::Monarch
    } else if contains("<start_of_turn>") {
        // :155-156
        ChatTemplate::Gemma
    } else if contains("'\\n\\nAssistant: ' + eos_token") {
        // :157-159 OrionStarAI/Orion-14B-Chat (literal backslash-n in the
        // jinja source, exactly as in the C++ literal)
        ChatTemplate::Orion
    } else if contains("GPT4 Correct ") {
        // :160-162 openchat/openchat-3.5-0106
        ChatTemplate::Openchat
    } else if contains("USER: ") && contains("ASSISTANT: ") {
        // :163-168 eachadea/vicuna-13b-1.1 (and Orca variant)
        if contains("SYSTEM: ") {
            ChatTemplate::VicunaOrca
        } else {
            ChatTemplate::Vicuna
        }
    } else if contains("### Instruction:") && contains("<|EOT|>") {
        // :169-171 deepseek-ai/deepseek-coder-33b-instruct
        ChatTemplate::Deepseek
    } else if contains("<|START_OF_TURN_TOKEN|>") && contains("<|USER_TOKEN|>") {
        // :172-174 CohereForAI/c4ai-command-r-plus
        ChatTemplate::CommandR
    } else if contains("<|start_header_id|>") && contains("<|end_header_id|>") {
        // :175-176
        ChatTemplate::Llama3
    } else if contains("[gMASK]sop") {
        // :177-179 chatglm3-6b
        ChatTemplate::Chatglm3
    } else if contains("<用户>") {
        // :180-182 MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        ChatTemplate::Minicpm
    } else if contains("'Assistant: ' + message['content'] + eos_token") {
        // :183-184
        ChatTemplate::Deepseek2
    } else if contains("<｜Assistant｜>") && contains("<｜User｜>") && contains("<｜end▁of▁sentence｜>") {
        // :185-186
        ChatTemplate::Deepseek3
    } else if contains("[|system|]") && contains("[|assistant|]") && contains("[|endofturn|]") {
        // :187-193
        if contains("[|tool|]") {
            ChatTemplate::Exaone4
        } else {
            // EXAONE-3.0-7.8B-Instruct
            ChatTemplate::Exaone3
        }
    } else if contains("rwkv-world") || contains("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}") {
        // :194-195
        ChatTemplate::RwkvWorld
    } else if contains("<|start_of_role|>") {
        // :196-203
        if contains("<tool_call>") || contains("<tools>") {
            if contains("g4_default_system_message") {
                ChatTemplate::Granite40
            } else {
                ChatTemplate::Granite41
            }
        } else {
            ChatTemplate::Granite3X
        }
    } else if contains(
        "message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]",
    ) {
        // :204-205
        ChatTemplate::Gigachat
    } else if contains("<|role_start|>") {
        // :206-207
        ChatTemplate::Megrez
    } else if contains(" Ассистент:") {
        // :208-209
        ChatTemplate::Yandex
    } else if contains("<role>ASSISTANT</role>") && contains("'HUMAN'") {
        // :210-211
        ChatTemplate::Bailing
    } else if contains("<role>ASSISTANT</role>") && contains("\"HUMAN\"") && contains("<think>") {
        // :212-213
        ChatTemplate::BailingThink
    } else if contains("<role>ASSISTANT</role>")
        && contains("<role>HUMAN</role>")
        && contains("<|role_end|>")
    {
        // :214-215
        ChatTemplate::Bailing2
    } else if contains("<|header_start|>") && contains("<|header_end|>") {
        // :216-217
        ChatTemplate::Llama4
    } else if contains("<|endofuserprompt|>") {
        // :218-219
        ChatTemplate::Dots1
    } else if contains("<|extra_0|>") && contains("<|extra_4|>") {
        // :220-221
        ChatTemplate::HunyuanMoe
    } else if contains("<|start|>") && contains("<|channel|>") {
        // :222-223
        ChatTemplate::OpenaiMoe
    } else if contains("<｜hy_Assistant｜>") && contains("<｜hy_begin▁of▁sentence｜>") {
        // :224-225
        ChatTemplate::HunyuanVl
    } else if contains("<｜hy_Assistant｜>") && contains("<｜hy_place▁holder▁no▁3｜>") {
        // :226-227
        ChatTemplate::HunyuanDense
    } else if contains("<|im_assistant|>assistant<|im_middle|>") {
        // :228-229
        ChatTemplate::KimiK2
    } else if contains("<seed:bos>") {
        // :230-231
        ChatTemplate::SeedOss
    } else if contains("'Assistant: '  + message['content'] + '<|separator|>") {
        // :232-233 (note: two spaces after 'Assistant: ')
        ChatTemplate::Grok2
    } else if contains("[unused9]系统：[unused10]") {
        // :234-235
        ChatTemplate::PanguEmbed
    } else if contains("<|begin|>") && contains("<|end|>") && contains("<|content|>") {
        // :236-237
        ChatTemplate::SolarOpen
    } else {
        // :239
        ChatTemplate::Unknown
    }
}

// ---------------------------------------------------------------------------
// apply — llm_chat_apply_template (llama-chat.cpp:244-946)
// ---------------------------------------------------------------------------

/// Format `messages` with a builtin template — 1:1 port of
/// `llm_chat_apply_template` (llama-chat.cpp:244-946). C++ returns
/// `-1` for unsupported templates; here that is `Err` (only
/// [`ChatTemplate::Unknown`] reaches the final `else` at llama-chat.cpp:940).
///
/// Per-branch C++ line references are given inline below.
pub fn apply(
    template: ChatTemplate,
    messages: &[ChatMessage],
    add_ass: bool,
) -> Result<String, String> {
    let mut ss = String::new();
    match template {
        ChatTemplate::Chatml => {
            // llama-chat.cpp:250-257
            for m in messages {
                let _ = write!(
                    ss,
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    m.role.as_str(),
                    m.content
                );
            }
            if add_ass {
                ss.push_str("<|im_start|>assistant\n");
            }
        }
        ChatTemplate::MistralV7 | ChatTemplate::MistralV7Tekken => {
            // llama-chat.cpp:258-273
            let trailing_space = if template == ChatTemplate::MistralV7 {
                " "
            } else {
                ""
            };
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(
                            ss,
                            "[SYSTEM_PROMPT]{}{}[/SYSTEM_PROMPT]",
                            trailing_space, m.content
                        );
                    }
                    Role::User => {
                        let _ = write!(ss, "[INST]{}{}[/INST]", trailing_space, m.content);
                    }
                    _ => {
                        let _ = write!(ss, "{}{}</s>", trailing_space, m.content);
                    }
                }
            }
        }
        ChatTemplate::MistralV1 | ChatTemplate::MistralV3 | ChatTemplate::MistralV3Tekken => {
            // llama-chat.cpp:274-298
            let leading_space = if template == ChatTemplate::MistralV1 {
                " "
            } else {
                ""
            };
            let trailing_space = if template == ChatTemplate::MistralV3Tekken {
                ""
            } else {
                " "
            };
            let trim_assistant_message = template == ChatTemplate::MistralV3;
            let mut is_inside_turn = false;
            for m in messages {
                if !is_inside_turn {
                    let _ = write!(ss, "{}[INST]{}", leading_space, trailing_space);
                    is_inside_turn = true;
                }
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "{}\n\n", m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "{}{}[/INST]", m.content, leading_space);
                    }
                    _ => {
                        let content = if trim_assistant_message {
                            trim(&m.content)
                        } else {
                            &m.content
                        };
                        let _ = write!(ss, "{}{}</s>", trailing_space, content);
                        is_inside_turn = false;
                    }
                }
            }
        }
        ChatTemplate::Llama2
        | ChatTemplate::Llama2Sys
        | ChatTemplate::Llama2SysBos
        | ChatTemplate::Llama2SysStrip => {
            // llama-chat.cpp:299-335
            let support_system_message = template != ChatTemplate::Llama2;
            let add_bos_inside_history = template == ChatTemplate::Llama2SysBos;
            let strip_message = template == ChatTemplate::Llama2SysStrip;
            let mut is_inside_turn = true; // skip BOS at the beginning
            ss.push_str("[INST] ");
            for m in messages {
                let content = if strip_message {
                    trim(&m.content)
                } else {
                    &m.content
                };
                if !is_inside_turn {
                    is_inside_turn = true;
                    ss.push_str(if add_bos_inside_history {
                        "<s>[INST] "
                    } else {
                        "[INST] "
                    });
                }
                match m.role {
                    Role::System => {
                        if support_system_message {
                            let _ = write!(ss, "<<SYS>>\n{}\n<</SYS>>\n\n", content);
                        } else {
                            // model without system support: keep it in the
                            // first message, but without <<SYS>>
                            let _ = write!(ss, "{}\n", content);
                        }
                    }
                    Role::User => {
                        let _ = write!(ss, "{} [/INST]", content);
                    }
                    _ => {
                        let _ = write!(ss, "{}</s>", content);
                        is_inside_turn = false;
                    }
                }
            }
        }
        ChatTemplate::Phi3 => {
            // llama-chat.cpp:336-344
            for m in messages {
                let _ = write!(ss, "<|{}|>\n{}<|end|>\n", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>\n");
            }
        }
        ChatTemplate::Phi4 => {
            // llama-chat.cpp:345-352 (chatml-like with <|im_sep|>)
            for m in messages {
                let _ = write!(
                    ss,
                    "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                    m.role.as_str(),
                    m.content
                );
            }
            if add_ass {
                ss.push_str("<|im_start|>assistant<|im_sep|>");
            }
        }
        ChatTemplate::Falcon3 => {
            // llama-chat.cpp:353-361
            for m in messages {
                let _ = write!(ss, "<|{}|>\n{}\n", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>\n");
            }
        }
        ChatTemplate::Zephyr => {
            // llama-chat.cpp:362-369
            for m in messages {
                let _ = write!(ss, "<|{}|>\n{}<|endoftext|>\n", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>\n");
            }
        }
        ChatTemplate::Monarch => {
            // llama-chat.cpp:370-378 mlabonne/AlphaMonarch-7B (<s> inside history)
            for (i, m) in messages.iter().enumerate() {
                let bos = if i == 0 { "" } else { "<s>" }; // skip BOS for first message
                let _ = write!(ss, "{}{}\n{}</s>\n", bos, m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<s>assistant\n");
            }
        }
        ChatTemplate::Gemma => {
            // llama-chat.cpp:379-400 google/gemma-7b-it
            let mut system_prompt = String::new();
            for m in messages {
                if m.role == Role::System {
                    // no system message in gemma: merge with the user prompt
                    system_prompt.push_str(trim(&m.content));
                    continue;
                }
                // in gemma, "assistant" is "model"
                let role = if m.role == Role::Assistant {
                    "model"
                } else {
                    m.role.as_str()
                };
                let _ = write!(ss, "<start_of_turn>{}\n", role);
                if !system_prompt.is_empty() && role != "model" {
                    let _ = write!(ss, "{}\n\n", system_prompt);
                    system_prompt.clear();
                }
                let _ = write!(ss, "{}<end_of_turn>\n", trim(&m.content));
            }
            if add_ass {
                ss.push_str("<start_of_turn>model\n");
            }
        }
        ChatTemplate::Orion => {
            // llama-chat.cpp:401-420 OrionStarAI/Orion-14B-Chat
            let mut system_prompt = String::new();
            for m in messages {
                match m.role {
                    Role::System => {
                        // no system support: merge with user prompt
                        system_prompt.push_str(&m.content);
                    }
                    Role::User => {
                        ss.push_str("Human: ");
                        if !system_prompt.is_empty() {
                            let _ = write!(ss, "{}\n\n", system_prompt);
                            system_prompt.clear();
                        }
                        let _ = write!(ss, "{}\n\nAssistant: </s>", m.content);
                    }
                    _ => {
                        let _ = write!(ss, "{}</s>", m.content);
                    }
                }
            }
        }
        ChatTemplate::Openchat => {
            // llama-chat.cpp:421-434 openchat/openchat-3.5-0106
            for m in messages {
                if m.role == Role::System {
                    let _ = write!(ss, "{}<|end_of_turn|>", m.content);
                } else {
                    // role[0] = toupper(role[0]) — roles are ASCII here
                    let role = m.role.as_str();
                    let role = format!("{}{}", role[..1].to_ascii_uppercase(), &role[1..]);
                    let _ = write!(ss, "GPT4 Correct {}: {}<|end_of_turn|>", role, m.content);
                }
            }
            if add_ass {
                ss.push_str("GPT4 Correct Assistant:");
            }
        }
        ChatTemplate::Vicuna | ChatTemplate::VicunaOrca => {
            // llama-chat.cpp:435-454 eachadea/vicuna-13b-1.1 (and Orca variant)
            for m in messages {
                match m.role {
                    Role::System => {
                        if template == ChatTemplate::VicunaOrca {
                            // Orca-Vicuna variant uses a system prefix
                            let _ = write!(ss, "SYSTEM: {}\n", m.content);
                        } else {
                            let _ = write!(ss, "{}\n\n", m.content);
                        }
                    }
                    Role::User => {
                        let _ = write!(ss, "USER: {}\n", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "ASSISTANT: {}</s>\n", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("ASSISTANT:");
            }
        }
        ChatTemplate::Deepseek => {
            // llama-chat.cpp:455-469 deepseek-ai/deepseek-coder-33b-instruct
            for m in messages {
                match m.role {
                    Role::System => {
                        ss.push_str(&m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "### Instruction:\n{}\n", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "### Response:\n{}\n<|EOT|>\n", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("### Response:\n");
            }
        }
        ChatTemplate::CommandR => {
            // llama-chat.cpp:470-484 CohereForAI/c4ai-command-r-plus
            for m in messages {
                let open = match m.role {
                    Role::System => "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
                    Role::User => "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
                    Role::Assistant => "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                    _ => continue,
                };
                let _ = write!(ss, "{}{}<|END_OF_TURN_TOKEN|>", open, trim(&m.content));
            }
            if add_ass {
                ss.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
            }
        }
        ChatTemplate::Llama3 => {
            // llama-chat.cpp:485-493 Llama 3
            for m in messages {
                let _ = write!(
                    ss,
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    m.role.as_str(),
                    trim(&m.content)
                );
            }
            if add_ass {
                ss.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }
        ChatTemplate::Chatglm3 => {
            // llama-chat.cpp:494-503 chatglm3-6b
            ss.push_str("[gMASK]sop");
            for m in messages {
                let _ = write!(ss, "<|{}|>\n {}", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>");
            }
        }
        ChatTemplate::Chatglm4 => {
            // llama-chat.cpp:504-512
            ss.push_str("[gMASK]<sop>");
            for m in messages {
                let _ = write!(ss, "<|{}|>\n{}", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>\n");
            }
        }
        ChatTemplate::Glmedge => {
            // llama-chat.cpp:513-520
            for m in messages {
                let _ = write!(ss, "<|{}|>\n{}", m.role.as_str(), m.content);
            }
            if add_ass {
                ss.push_str("<|assistant|>");
            }
        }
        ChatTemplate::Minicpm => {
            // llama-chat.cpp:521-532 MiniCPM-3B-OpenHermes-2.5-v2-GGUF
            for m in messages {
                if m.role == Role::User {
                    let _ = write!(ss, "<用户>{}<AI>", trim(&m.content));
                } else {
                    ss.push_str(trim(&m.content));
                }
            }
        }
        ChatTemplate::Deepseek2 => {
            // llama-chat.cpp:533-547 DeepSeek-V2
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "{}\n\n", m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "User: {}\n\n", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "Assistant: {}<｜end▁of▁sentence｜>", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("Assistant:");
            }
        }
        ChatTemplate::Deepseek3 => {
            // llama-chat.cpp:548-562 DeepSeek-V3
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "{}\n\n", m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "<｜User｜>{}", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "<｜Assistant｜>{}<｜end▁of▁sentence｜>", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("<｜Assistant｜>");
            }
        }
        ChatTemplate::DeepseekOcr => {
            // llama-chat.cpp:563-567 (no template)
            for m in messages {
                ss.push_str(&m.content);
            }
        }
        ChatTemplate::Exaone3 => {
            // llama-chat.cpp:568-583 EXAONE-3.0-7.8B-Instruct
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "[|system|]{}[|endofturn|]\n", trim(&m.content));
                    }
                    Role::User => {
                        let _ = write!(ss, "[|user|]{}\n", trim(&m.content));
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "[|assistant|]{}[|endofturn|]\n", trim(&m.content));
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("[|assistant|]");
            }
        }
        ChatTemplate::Exaone4 => {
            // llama-chat.cpp:584-599
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "[|system|]{}[|endofturn|]\n", trim(&m.content));
                    }
                    Role::User => {
                        let _ = write!(ss, "[|user|]{}\n", trim(&m.content));
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "[|assistant|]{}[|endofturn|]\n", trim(&m.content));
                    }
                    Role::Tool => {
                        let _ = write!(ss, "[|tool|]{}[|endofturn|]\n", trim(&m.content));
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("[|assistant|]");
            }
        }
        ChatTemplate::ExaoneMoe => {
            // llama-chat.cpp:600-615
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "<|system|>\n{}<|endofturn|>\n", trim(&m.content));
                    }
                    Role::User => {
                        let _ = write!(ss, "<|user|>\n{}<|endofturn|>\n", trim(&m.content));
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "<|assistant|>\n{}<|endofturn|>\n", trim(&m.content));
                    }
                    Role::Tool => {
                        let _ = write!(ss, "<|tool|>\n{}<|endofturn|>\n", trim(&m.content));
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("<|assistant|>\n");
            }
        }
        ChatTemplate::RwkvWorld => {
            // llama-chat.cpp:616-630 — requires "\n\n" as EOT token
            let n = messages.len();
            for (i, m) in messages.iter().enumerate() {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "System: {}\n\n", trim(&m.content));
                    }
                    Role::User => {
                        let _ = write!(ss, "User: {}\n\n", trim(&m.content));
                        if i == n - 1 {
                            ss.push_str("Assistant:");
                        }
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "Assistant: {}\n\n", trim(&m.content));
                    }
                    _ => {}
                }
            }
        }
        ChatTemplate::Granite3X => {
            // llama-chat.cpp:631-643 IBM Granite 3.x
            for m in messages {
                let _ = write!(ss, "<|start_of_role|>{}<|end_of_role|>", m.role.as_str());
                if m.role == Role::AssistantToolCall {
                    ss.push_str("<|tool_call|>");
                }
                let _ = write!(ss, "{}<|end_of_text|>\n", m.content);
            }
            if add_ass {
                ss.push_str("<|start_of_role|>assistant<|end_of_role|>");
            }
        }
        ChatTemplate::Granite40 | ChatTemplate::Granite41 => {
            // llama-chat.cpp:644-671 IBM Granite 4.0 / 4.1
            for m in messages {
                if m.role == Role::AssistantToolCall {
                    ss.push_str("<|start_of_role|>assistant<|end_of_role|><|tool_call|>");
                } else {
                    let _ = write!(ss, "<|start_of_role|>{}<|end_of_role|>", m.role.as_str());
                }
                let _ = write!(ss, "{}<|end_of_text|>\n", m.content);
            }
            if add_ass {
                ss.push_str("<|start_of_role|>assistant<|end_of_role|>");
            }
        }
        ChatTemplate::Gigachat => {
            // llama-chat.cpp:672-697
            let has_system = matches!(messages.first(), Some(m) if m.role == Role::System);

            if has_system {
                let _ = write!(ss, "<s>{}<|message_sep|>", messages[0].content);
            } else {
                ss.push_str("<s>");
            }

            for m in messages.iter().skip(if has_system { 1 } else { 0 }) {
                match m.role {
                    Role::User => {
                        let _ = write!(
                            ss,
                            "user<|role_sep|>{}<|message_sep|>available functions<|role_sep|>[]<|message_sep|>",
                            m.content
                        );
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "assistant<|role_sep|>{}<|message_sep|>", m.content);
                    }
                    _ => {}
                }
            }

            if add_ass {
                ss.push_str("assistant<|role_sep|>");
            }
        }
        ChatTemplate::Megrez => {
            // llama-chat.cpp:698-707
            for m in messages {
                let _ = write!(
                    ss,
                    "<|role_start|>{}<|role_end|>{}<|turn_end|>",
                    m.role.as_str(),
                    m.content
                );
            }
            if add_ass {
                ss.push_str("<|role_start|>assistant<|role_end|>");
            }
        }
        ChatTemplate::Yandex => {
            // llama-chat.cpp:708-723 ("\n\n" is the EOT token)
            for m in messages {
                match m.role {
                    Role::User => {
                        let _ = write!(ss, " Пользователь: {}\n\n", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, " Ассистент: {}\n\n", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str(" Ассистент:[SEP]");
            }
        }
        ChatTemplate::Bailing | ChatTemplate::BailingThink => {
            // llama-chat.cpp:724-744 Bailing (Ling/Ring)
            for m in messages {
                // role == "user" ? "HUMAN" : toupper(role) (ASCII roles)
                let role = if m.role == Role::User {
                    "HUMAN".to_string()
                } else {
                    m.role.as_str().to_ascii_uppercase()
                };
                let _ = write!(ss, "<role>{}</role>{}", role, m.content);
            }
            if add_ass {
                ss.push_str("<role>ASSISTANT</role>");
                if template == ChatTemplate::BailingThink {
                    ss.push_str("<think>");
                }
            }
        }
        ChatTemplate::Bailing2 => {
            // llama-chat.cpp:745-767 Bailing2 (Ling 2.0)
            let has_system = matches!(messages.first(), Some(m) if m.role == Role::System);

            if !has_system {
                ss.push_str("<role>SYSTEM</role>detailed thinking off<|role_end|>");
            }

            for m in messages {
                let role = if m.role == Role::User {
                    "HUMAN".to_string()
                } else {
                    m.role.as_str().to_ascii_uppercase()
                };
                let _ = write!(ss, "<role>{}</role>{}<|role_end|>", role, m.content);
            }

            if add_ass {
                ss.push_str("<role>ASSISTANT</role>");
            }
        }
        ChatTemplate::Llama4 => {
            // llama-chat.cpp:768-776 Llama 4
            for m in messages {
                let _ = write!(
                    ss,
                    "<|header_start|>{}<|header_end|>\n\n{}<|eot|>",
                    m.role.as_str(),
                    trim(&m.content)
                );
            }
            if add_ass {
                ss.push_str("<|header_start|>assistant<|header_end|>\n\n");
            }
        }
        ChatTemplate::Smolvlm => {
            // llama-chat.cpp:777-792 SmolVLM (<|im_start|> as BOS, NOT chatml)
            ss.push_str("<|im_start|>");
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "{}\n\n", m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "User: {}<end_of_utterance>\n", m.content);
                    }
                    _ => {
                        let _ = write!(ss, "Assistant: {}<end_of_utterance>\n", m.content);
                    }
                }
            }
            if add_ass {
                ss.push_str("Assistant:");
            }
        }
        ChatTemplate::Dots1 => {
            // llama-chat.cpp:793-807 dots.llm1.inst (DOTS1)
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "<|system|>{}<|endofsystem|>", m.content);
                    }
                    Role::User => {
                        let _ = write!(ss, "<|userprompt|>{}<|endofuserprompt|>", m.content);
                    }
                    _ => {
                        let _ = write!(ss, "<|response|>{}<|endofresponse|>", m.content);
                    }
                }
            }
            if add_ass {
                ss.push_str("<|response|>");
            }
        }
        ChatTemplate::HunyuanMoe => {
            // llama-chat.cpp:808-819 tencent/Hunyuan-A13B-Instruct
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "<|startoftext|>{}<|extra_4|>", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "{}<|eos|>", m.content);
                    }
                    _ => {
                        let _ = write!(ss, "<|startoftext|>{}<|extra_0|>", m.content);
                    }
                }
            }
        }
        ChatTemplate::OpenaiMoe => {
            // llama-chat.cpp:820-829 OpenAI MoE (Harmony-based)
            for m in messages {
                let _ = write!(ss, "<|start|>{}<|message|>{}", m.role.as_str(), m.content);
                ss.push_str(if m.role == Role::Assistant {
                    "<|return|>"
                } else {
                    "<|end|>"
                });
            }
            if add_ass {
                ss.push_str("<|start|>assistant");
            }
        }
        ChatTemplate::HunyuanDense => {
            // llama-chat.cpp:830-845 tencent/Hunyuan-4B-Instruct
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == Role::System {
                    let _ = write!(ss, "{}<｜hy_place▁holder▁no▁3｜>", m.content);
                }

                match m.role {
                    Role::Assistant => {
                        let _ = write!(
                            ss,
                            "<｜hy_Assistant｜>{}<｜hy_place▁holder▁no▁2｜>",
                            m.content
                        );
                    }
                    Role::User => {
                        let _ = write!(ss, "<｜hy_User｜>{}<｜hy_Assistant｜>", m.content);
                    }
                    _ => {}
                }
            }
        }
        ChatTemplate::HunyuanVl => {
            // llama-chat.cpp:846-861 tencent/HunyuanOCR & HunyuanVL
            ss.push_str("<｜hy_begin▁of▁sentence｜>");
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == Role::System {
                    let _ = write!(ss, "{}<｜hy_place▁holder▁no▁3｜>", m.content);
                    continue;
                }

                match m.role {
                    Role::User => {
                        let _ = write!(ss, "{}<｜hy_User｜>", m.content);
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "{}<｜hy_Assistant｜>", m.content);
                    }
                    _ => {}
                }
            }
        }
        ChatTemplate::KimiK2 => {
            // llama-chat.cpp:862-880 moonshotai/Kimi-K2-Instruct
            for m in messages {
                match m.role {
                    Role::System => ss.push_str("<|im_system|>system<|im_middle|>"),
                    Role::User => ss.push_str("<|im_user|>user<|im_middle|>"),
                    Role::Assistant => ss.push_str("<|im_assistant|>assistant<|im_middle|>"),
                    Role::Tool => ss.push_str("<|im_system|>tool<|im_middle|>"),
                    _ => {}
                }
                let _ = write!(ss, "{}<|im_end|>", m.content);
            }
            if add_ass {
                ss.push_str("<|im_assistant|>assistant<|im_middle|>");
            }
        }
        ChatTemplate::SeedOss => {
            // llama-chat.cpp:881-888
            for m in messages {
                let content = if m.role == Role::Assistant {
                    trim(&m.content)
                } else {
                    &m.content
                };
                let _ = write!(ss, "<seed:bos>{}\n{}<seed:eos>", m.role.as_str(), content);
            }
            if add_ass {
                ss.push_str("<seed:bos>assistant\n");
            }
        }
        ChatTemplate::Grok2 => {
            // llama-chat.cpp:889-902
            for m in messages {
                match m.role {
                    Role::System => {
                        let _ = write!(ss, "System: {}<|separator|>\n\n", trim(&m.content));
                    }
                    Role::User => {
                        let _ = write!(ss, "Human: {}<|separator|>\n\n", trim(&m.content));
                    }
                    Role::Assistant => {
                        let _ = write!(ss, "Assistant: {}<|separator|>\n\n", m.content);
                    }
                    _ => {}
                }
            }
            if add_ass {
                ss.push_str("Assistant:");
            }
        }
        ChatTemplate::PanguEmbed => {
            // llama-chat.cpp:903-931
            // [unused9]系统：xxx[unused10] / 用户 / 助手 / 工具 / 方法
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role != Role::System {
                    ss.push_str("[unused9]系统：[unused10]");
                }

                let label = match m.role {
                    Role::System => "系统",
                    Role::User => "用户",
                    Role::Assistant => "助手",
                    Role::Tool => "工具",
                    Role::Function => "方法",
                    Role::AssistantToolCall => "",
                };
                if !label.is_empty() {
                    let _ = write!(ss, "[unused9]{}：{}[unused10]", label, m.content);
                }
            }
            if add_ass {
                ss.push_str("[unused9]助手：");
            }
        }
        ChatTemplate::SolarOpen => {
            // llama-chat.cpp:932-939
            for m in messages {
                let _ = write!(
                    ss,
                    "<|begin|>{}<|content|>{}<|end|>",
                    m.role.as_str(),
                    m.content
                );
            }
            if add_ass {
                ss.push_str("<|begin|>assistant");
            }
        }
        // llama-chat.cpp:940-942: template not supported → -1
        ChatTemplate::Unknown => {
            return Err(
                "chat template not supported (llama-chat.cpp:940-942); detect() returned Unknown"
                    .to_string(),
            );
        }
    }
    Ok(ss)
}

/// Convenience wrapper mirroring the public C entry point
/// `llama_chat_apply_template(const char * tmpl, ...)` (src/llama.cpp:511-536):
/// `tmpl == None` ⇒ `"chatml"`, then detect + apply. `Err` covers both the
/// UNKNOWN detection (`-1`, llama.cpp:527-529) and the unsupported-template
/// apply (`-1`, llama-chat.cpp:941).
pub fn apply_named(
    tmpl: Option<&str>,
    messages: &[ChatMessage],
    add_ass: bool,
) -> Result<String, String> {
    let curr_tmpl = tmpl.unwrap_or("chatml");
    let detected = detect(curr_tmpl);
    apply(detected, messages, add_ass)
}

// ---------------------------------------------------------------------------
// eot_prefix — derived helper for generation-stop detection
// ---------------------------------------------------------------------------

/// End-of-turn marker for generation-stop detection (used as an antiprompt /
/// stop prefix by the cli chat loop). No direct C++ counterpart in
/// `llama-chat.cpp`; each value is the end-of-turn marker the builtin
/// formatter itself emits (C++ line given per arm). Entries marked `derived`
/// have no explicit marker in the formatter; the model family's documented EOS
/// special token is used instead.
pub fn eot_prefix(template: ChatTemplate) -> &'static str {
    match template {
        ChatTemplate::Chatml => "<|im_end|>", // llama-chat.cpp:253
        ChatTemplate::Llama2
        | ChatTemplate::Llama2Sys
        | ChatTemplate::Llama2SysBos
        | ChatTemplate::Llama2SysStrip => "</s>", // :332
        ChatTemplate::MistralV1 | ChatTemplate::MistralV3 | ChatTemplate::MistralV3Tekken => "</s>", // :295
        ChatTemplate::MistralV7 | ChatTemplate::MistralV7Tekken => "</s>", // :271
        ChatTemplate::Phi3 => "<|end|>",                                   // :340
        ChatTemplate::Phi4 => "<|im_end|>",                                // :348
        ChatTemplate::Falcon3 => "<|end|>",                                // derived (falcon3 EOS)
        ChatTemplate::Zephyr => "<|endoftext|>",                           // :365
        ChatTemplate::Monarch => "</s>",                                   // :374
        ChatTemplate::Gemma => "<end_of_turn>",                            // :396
        ChatTemplate::Orion => "</s>",                                     // :416
        ChatTemplate::Openchat => "<|end_of_turn|>",                       // :426
        ChatTemplate::Vicuna | ChatTemplate::VicunaOrca => "</s>",         // :449
        ChatTemplate::Deepseek => "<|EOT|>",                               // :464
        ChatTemplate::Deepseek2 | ChatTemplate::Deepseek3 => "<｜end▁of▁sentence｜>", // :542/:557
        ChatTemplate::DeepseekOcr => "",                                   // no marker in :563-567
        ChatTemplate::CommandR => "<|END_OF_TURN_TOKEN|>",                 // :475
        ChatTemplate::Llama3 => "<|eot_id|>",                              // :489
        ChatTemplate::Chatglm3 => "<|endoftext|>",                         // derived (chatglm3 EOS)
        ChatTemplate::Chatglm4 => "<|endoftext|>",                         // derived (chatglm4 EOS)
        ChatTemplate::Glmedge => "<|endoftext|>",                          // derived (glm-edge EOS)
        ChatTemplate::Minicpm => "<|im_end|>",                             // derived (minicpm EOS)
        ChatTemplate::Exaone3 | ChatTemplate::Exaone4 => "[|endofturn|]",  // :574/:588
        ChatTemplate::ExaoneMoe => "<|endofturn|>",                        // :604
        ChatTemplate::RwkvWorld => "\n\n", // :617 comment ("requires \n\n as EOT")
        ChatTemplate::Granite3X | ChatTemplate::Granite40 | ChatTemplate::Granite41 => {
            "<|end_of_text|>" // :639
        }
        ChatTemplate::Gigachat => "<|message_sep|>", // :678
        ChatTemplate::Megrez => "<|turn_end|>",      // :702
        ChatTemplate::Yandex => "\n\n",              // :709 comment
        ChatTemplate::Bailing | ChatTemplate::BailingThink => "<|role_end|>", // derived (ling EOS)
        ChatTemplate::Bailing2 => "<|role_end|>",    // :762
        ChatTemplate::Llama4 => "<|eot|>",           // :772
        ChatTemplate::Smolvlm => "<end_of_utterance>", // :785
        ChatTemplate::Dots1 => "<|endofresponse|>",  // :802
        ChatTemplate::HunyuanMoe => "<|eos|>",       // :815
        ChatTemplate::OpenaiMoe => "<|return|>",     // :825
        ChatTemplate::HunyuanDense => "<｜hy_place▁holder▁no▁2｜>", // :841
        ChatTemplate::HunyuanVl => "<｜hy_User｜>",  // derived (next-turn marker after assistant)
        ChatTemplate::KimiK2 => "<|im_end|>",        // :876
        ChatTemplate::SeedOss => "<seed:eos>",       // :884
        ChatTemplate::Grok2 => "<|separator|>",      // :893
        ChatTemplate::PanguEmbed => "[unused10]",    // :918
        ChatTemplate::SolarOpen => "<|end|>",        // :935
        ChatTemplate::Unknown => "",
    }
}

// ---------------------------------------------------------------------------
// apply_str — Jinja rendering (common_chat_template_direct_apply's engine)
// ---------------------------------------------------------------------------

/// Render inputs for [`apply_str`]. Mirrors the variables the pinned jinja
/// runtime binds for chat templates (common/jinja/value.h:71-72,
/// common/chat.cpp:915-938): `messages`, `add_generation_prompt`, `bos_token`,
/// `eos_token` (`bos_tag`/`eos_tag` are bound as aliases for gemma-2-style
/// templates). Unbound template variables (`tools`, `custom_tool`,
/// `message.tool_calls`, …) evaluate to *undefined* ⇒ falsy, exactly like
/// rendering without tools in the C++ engine.
pub struct ChatTemplateCtx<'a> {
    pub messages: &'a [ChatMessage],
    pub add_generation_prompt: bool,
    pub bos_token: &'a str,
    pub eos_token: &'a str,
    /// Unix timestamp used for `strftime_now(...)` (common/jinja/value.cpp:
    /// 370-381 uses the real clock via `std::localtime`). `None` ⇒ current
    /// time. Rendering is in UTC (C++ is TZ-dependent — deviation, see module
    /// docs); tests pin this value.
    pub now: Option<i64>,
}

impl<'a> ChatTemplateCtx<'a> {
    pub fn new(
        messages: &'a [ChatMessage],
        add_generation_prompt: bool,
        bos_token: &'a str,
        eos_token: &'a str,
    ) -> Self {
        ChatTemplateCtx {
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            now: None,
        }
    }
}

/// Apply a raw Jinja chat template string (e.g. GGUF `tokenizer.chat_template`)
/// to `messages` — the port of the pinned `common/jinja` engine
/// (lexer.cpp/parser.cpp/runtime.cpp/value.cpp semantics, see [`mini_jinja`]).
///
/// * whitespace control exactly as the pinned `common/jinja/lexer.cpp:112-210`:
///   `{{- … -}}` / `{%- … -%}` trim, `lstrip_blocks` + `trim_blocks`
///   (HF-transformers compatible), one trailing template newline stripped
///   (`keep_trailing_newline=false`, lexer.cpp:54-57)
/// * statements: `if/elif/else`, `for` (+ tuple unpack, inline `if` filter,
///   `loop.*` incl. previtem/nextitem, `break`/`continue`), `set`
///   (assignment, tuple unpack, `ns.attr` member write, block `set/endset`),
///   `macro`/`endmacro` (default args, kwargs, recursion, closures),
///   `generation` markers (ignored)
/// * expressions: literals (+ list/tuple/object), `and`/`or` (value-returning),
///   `not`, comparisons, `in`/`not in`, arithmetic, `~` concat, string `*`,
///   ternary, subscripts incl. Python slices and negative indexes, method
///   calls (`.get/.items/.split/.strip/…`), filters, `is` tests, kwargs
///
/// Constructs outside the ported surface fail with `Err` (loud-fail policy).
pub fn apply_str(raw_jinja: &str, ctx: &ChatTemplateCtx) -> Result<String, String> {
    let toks = mini_jinja::lex(raw_jinja).map_err(|e| format!("mini-jinja: {e}"))?;
    let prog = mini_jinja::parse(&toks).map_err(|e| format!("mini-jinja: {e}"))?;
    mini_jinja::render(&prog, ctx).map_err(|e| format!("mini-jinja: {e}"))
}

/// Convenience: `apply_str` with empty BOS/EOS and the real clock.
pub fn apply_str_simple(
    raw_jinja: &str,
    messages: &[ChatMessage],
    add_ass: bool,
) -> Result<String, String> {
    let ctx = ChatTemplateCtx::new(messages, add_ass, "", "");
    apply_str(raw_jinja, &ctx)
}

// ---------------------------------------------------------------------------
// mini-jinja — private partial Jinja engine
// ---------------------------------------------------------------------------

pub mod mini_jinja {
    use super::ChatTemplateCtx;
    use std::fmt::Write as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ---- values -----------------------------------------------------------
    //
    // The tool-calling work (chat_tools.rs) needs the subset of minja's value
    // model that carries usage statistics: `jinja::caps` (common/jinja/caps.cpp)
    // infers template capabilities from *which* input values a template reads
    // (`stats.used`) and *how* (`stats.ops`: test_is_string / selectattr /
    // array_access, set by common/jinja/runtime.cpp:306,427,499-532,926-932).
    // Values are therefore reference-counted nodes with interior-mutable stats,
    // mirroring `jinja::value` (shared_ptr + stats_t).

    use std::cell::{Cell, RefCell};
    use std::collections::HashSet;
    use std::rc::Rc;

    /// `jinja::value_t::stats_t` (common/jinja/value.h:104-123)
    #[derive(Default, Debug)]
    pub struct Stats {
        pub used: Cell<bool>,
        pub ops: RefCell<HashSet<String>>,
    }

    impl Stats {
        pub fn mark_used(&self) {
            self.used.set(true);
        }
        pub fn add_op(&self, name: &str) {
            self.ops.borrow_mut().insert(name.to_string());
        }
        pub fn has_op(&self, name: &str) -> bool {
            self.ops.borrow().contains(name)
        }
    }

    /// `stats_t::mark_used(val, deep)` (value.cpp:1570-1584)
    pub fn mark_deep_used(v: &Val) {
        v.stats().mark_used();
        match v.kind() {
            ValKind::List(items) => {
                for item in items.borrow().iter() {
                    mark_deep_used(item);
                }
            }
            ValKind::Tuple(items) => {
                for item in items {
                    mark_deep_used(item);
                }
            }
            ValKind::Object(fields) => {
                for (_, val) in fields.borrow().iter() {
                    mark_deep_used(val);
                }
            }
            _ => {}
        }
    }

    /// `value_tuple_t` / `value_func_t` forward declarations.
    #[derive(Clone, Debug)]
    pub struct MacroDef {
        /// macro parameters; `None` = no default (bind_parameters,
        /// runtime.cpp:697-741). Defaults evaluate in the *caller's* context.
        pub params: Vec<(String, Option<Expr>)>,
        pub body: Rc<Vec<Node>>,
    }

    #[derive(Debug)]
    pub enum FuncKind {
        /// entry of `global_builtins()` (value.cpp:351-531)
        Global(&'static str),
        /// builtin bound to a receiver ("this"): methods like `.get()`,
        /// `.items()`, `.split()` dispatched on the receiver's type
        /// (value.cpp:534-1352 per-type builtin tables)
        Method { recv: Val, name: String },
        /// `{% macro %}` value (runtime.cpp:743-764)
        Macro(Rc<MacroDef>),
    }

    #[derive(Debug)]
    pub struct FuncVal {
        pub name: String,
        pub kind: FuncKind,
    }

    /// The `jinja::value_t` hierarchy (value.h:106-172). `None` (JSON null /
    /// `none`) is distinct from `Undefined` (value.h:602-636). Containers are
    /// interior-mutable: `{% set ns.attr = v %}` (runtime.cpp:672-689) and
    /// `arr.append(x)` (value.cpp:1090-1100) mutate through shared refs.
    #[derive(Clone, Debug)]
    pub enum ValKind {
        Undefined,
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        List(RefCell<Vec<Val>>),
        /// immutable array (`value_tuple_t : value_array_t`, value.h:463-480)
        Tuple(Vec<Val>),
        /// ordered object; keys are strings (chat templates only ever use
        /// string keys — minja hashes arbitrary `value` keys, value.h:484)
        Object(RefCell<Vec<(String, Val)>>),
        Func(Rc<FuncVal>),
    }

    /// `jinja::value` — Rc'd node so that usage stats are shared between every
    /// reference to the same input value (shared_ptr, value.h:21).
    #[derive(Clone, Debug)]
    pub struct Val {
        pub v: Rc<(ValKind, Stats)>,
    }

    impl Val {
        pub fn undef() -> Val {
            Val::new(ValKind::Undefined)
        }
        pub fn bool_(b: bool) -> Val {
            Val::new(ValKind::Bool(b))
        }
        pub fn int(i: i64) -> Val {
            Val::new(ValKind::Int(i))
        }
        pub fn float(f: f64) -> Val {
            Val::new(ValKind::Float(f))
        }
        pub fn str_(s: &str) -> Val {
            Val::new(ValKind::Str(s.to_string()))
        }
        pub fn new_lit_str(s: String) -> Val {
            Val::new(ValKind::Str(s))
        }
        pub fn list(items: Vec<Val>) -> Val {
            Val::new(ValKind::List(RefCell::new(items)))
        }
        pub fn tuple(items: Vec<Val>) -> Val {
            Val::new(ValKind::Tuple(items))
        }
        pub fn object(fields: Vec<(&str, Val)>) -> Val {
            Val::new(ValKind::Object(RefCell::new(
                fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            )))
        }
        /// JSON null → `value_none` (from_json, value.cpp:1358-1388)
        pub fn null_() -> Val {
            Val::new(ValKind::None)
        }
        pub fn func(name: &str, kind: FuncKind) -> Val {
            Val::new(ValKind::Func(Rc::new(FuncVal {
                name: name.to_string(),
                kind,
            })))
        }

        fn new(v: ValKind) -> Val {
            Val {
                v: Rc::new((v, Stats::default())),
            }
        }

        pub fn kind(&self) -> &ValKind {
            &self.v.0
        }
        pub fn stats(&self) -> &Stats {
            &self.v.1
        }

        pub fn is_undefined(&self) -> bool {
            matches!(self.kind(), ValKind::Undefined)
        }
        pub fn is_none(&self) -> bool {
            matches!(self.kind(), ValKind::None)
        }
        pub fn is_string(&self) -> bool {
            matches!(self.v.0, ValKind::Str(_))
        }
        /// array (mutable or tuple — `dynamic_cast<value_array_t>` accepts
        /// both, value.h:34-42)
        pub fn is_array(&self) -> bool {
            matches!(self.kind(), ValKind::List(_) | ValKind::Tuple(_))
        }
        pub fn is_object(&self) -> bool {
            matches!(self.kind(), ValKind::Object(_))
        }
        /// `is_numeric()` (value.h:152) — bool counts as numeric
        pub fn is_numeric(&self) -> bool {
            matches!(
                self.kind(),
                ValKind::Int(_) | ValKind::Float(_) | ValKind::Bool(_)
            )
        }

        /// `as_bool()` (value.h:135): empty strings/lists/objects, undefined
        /// and none are falsy; `value_func_t` does not override it → calling
        /// truthiness on a function is a type error.
        pub fn truthy(&self) -> Result<bool, String> {
            match self.kind() {
                ValKind::Undefined | ValKind::None => Ok(false),
                ValKind::Bool(b) => Ok(*b),
                ValKind::Int(i) => Ok(*i != 0),
                ValKind::Float(f) => Ok(*f != 0.0),
                ValKind::Str(s) => Ok(!s.is_empty()),
                ValKind::List(items) => Ok(!items.borrow().is_empty()),
                ValKind::Tuple(items) => Ok(!items.is_empty()),
                ValKind::Object(fields) => Ok(!fields.borrow().is_empty()),
                ValKind::Func(_) => Err("Function is not a bool value".to_string()),
            }
        }

        /// The `(val_int, val_flt)` pair used for numeric equivalence
        /// (value.h:214-249: value_int_t/value_float_t/value_bool_t
        /// constructors keep both).
        fn num_pair(&self) -> Option<(i64, f64)> {
            match self.kind() {
                ValKind::Int(i) => {
                    let f = *i as f64;
                    // int not representable in f64 saturates to ±inf (value.h:218-220)
                    let f = if f as i64 == *i {
                        f
                    } else if *i < 0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                    Some((*i, f))
                }
                ValKind::Float(f) => {
                    // value_float_t: val_int = finite ? (int64)f : 0
                    Some((if f.is_finite() { *f as i64 } else { 0 }, *f))
                }
                ValKind::Bool(b) => Some((*b as i64, *b as i64 as f64)),
                _ => None,
            }
        }

        /// `gather_string_parts_recursive` (runtime.h:717-731): output
        /// conversion appends string/int/float/bool, recurses into arrays,
        /// and skips none/undefined/objects/functions (a `TODO` in minja
        /// notes none renders as "" not "None").
        pub fn gather_out(&self, out: &mut String) {
            match self.kind() {
                ValKind::Str(s) => out.push_str(s),
                ValKind::Int(i) => {
                    let _ = write!(out, "{i}");
                }
                ValKind::Float(f) => out.push_str(&Self::float_as_string(*f)),
                ValKind::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
                ValKind::List(items) => {
                    for item in items.borrow().iter() {
                        item.gather_out(out);
                    }
                }
                ValKind::Tuple(items) => {
                    for item in items {
                        item.gather_out(out);
                    }
                }
                _ => {}
            }
        }

        pub fn to_out(&self) -> String {
            let mut s = String::new();
            self.gather_out(&mut s);
            s
        }

        /// `as_string()` (value.h:134 per-type overrides): the human string,
        /// NOT the output conversion (floats keep one decimal digit,
        /// value.h:258-263; bools are "True"/"False"; containers print their
        /// repr).
        pub fn as_string(&self) -> Result<String, String> {
            match self.kind() {
                ValKind::Str(s) => Ok(s.clone()),
                ValKind::Int(i) => Ok(i.to_string()),
                ValKind::Float(f) => Ok(Self::float_as_string(*f)),
                ValKind::Bool(b) => Ok((if *b { "True" } else { "False" }).to_string()),
                ValKind::None => Ok("None".to_string()),
                ValKind::List(items) => {
                    let parts: Vec<String> =
                        items.borrow().iter().map(Self::to_string_repr).collect();
                    Ok(format!("[{}]", parts.join(", ")))
                }
                ValKind::Tuple(items) => {
                    let parts: Vec<String> = items.iter().map(Self::to_string_repr).collect();
                    // 1-tuples print a trailing comma (value.h:409-411)
                    let comma = if items.len() == 1 { "," } else { "" };
                    Ok(format!("({}{comma})", parts.join(", ")))
                }
                ValKind::Object(fields) => {
                    let parts: Vec<String> = fields
                        .borrow()
                        .iter()
                        .map(|(k, v)| format!("{}: {}", Self::repr_str(k), Self::to_string_repr(v)))
                        .collect();
                    Ok(format!("{{{}}}", parts.join(", ")))
                }
                ValKind::Undefined => Err("Undefined is not a string value".to_string()),
                ValKind::Func(f) => Err(format!("Function {} is not a string value", f.name)),
            }
        }

        /// `value_float_t::as_string()` (value.h:258-263): `std::to_string`
        /// (6 fixed decimals) with trailing zeros removed, one kept.
        fn float_as_string(f: f64) -> String {
            let s = format!("{f:.6}");
            let s = s.trim_end_matches('0');
            if s.ends_with('.') {
                format!("{s}0")
            } else {
                s.to_string()
            }
        }

        /// `value_to_string_repr` (value.cpp:1555-1567)
        fn to_string_repr(v: &Val) -> String {
            match v.kind() {
                ValKind::Str(s) => Self::repr_str(s),
                _ => match v.as_string() {
                    Ok(s) => s,
                    // as_repr of undefined prints the type name (value.h:130)
                    Err(_) => "Undefined".to_string(),
                },
            }
        }
        fn repr_str(s: &str) -> String {
            if s.contains('\'') {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                format!("'{s}'")
            }
        }

        /// `get_item` access with stats marking (runtime.cpp:925-934):
        /// val/object/property marked used; `array_access` for integer
        /// properties, `object_access` for string ones.
        pub fn get_item(&self, key: &Val) -> Val {
            match (self.kind(), key.kind()) {
                (ValKind::Object(fields), ValKind::Str(k)) => {
                    self.stats().mark_used();
                    if let Some((_, v)) = fields.borrow().iter().find(|(fk, _)| fk == k) {
                        v.stats().mark_used();
                        self.stats().add_op("object_access");
                        return v.clone();
                    }
                    Val::undef()
                }
                (ValKind::List(items), ValKind::Int(i)) => {
                    self.stats().mark_used();
                    self.stats().add_op("array_access");
                    let mut index = *i;
                    if index < 0 {
                        index += items.borrow().len() as i64;
                    }
                    if index >= 0 && (index as usize) < items.borrow().len() {
                        let v = items.borrow()[index as usize].clone();
                        v.stats().mark_used();
                        return v;
                    }
                    Val::undef()
                }
                _ => Val::undef(),
            }
        }

        /// Shallow field access without stats (caps introspection).
        pub fn field(&self, name: &str) -> Option<Val> {
            if let ValKind::Object(fields) = self.kind() {
                fields
                    .borrow()
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
            } else {
                None
            }
        }

        pub fn as_str_val(&self) -> Option<String> {
            match self.kind() {
                ValKind::Str(s) => Some(s.clone()),
                _ => None,
            }
        }

        pub fn as_int_val(&self) -> Option<i64> {
            match self.kind() {
                ValKind::Int(i) => Some(*i),
                _ => None,
            }
        }
    }

    /// `operator==` → `equivalent` (value.h:158, 170): numerics compare by
    /// their (int, float) pair (so `true == 1` and `1 == 1.0` hold);
    /// undefined equals only undefined; containers compare element-wise and
    /// only within the same concrete type.
    pub fn jval_eq(a: &Val, b: &Val) -> bool {
        match (a.kind(), b.kind()) {
            (ValKind::Str(x), ValKind::Str(y)) => x == y,
            _ if a.is_numeric() && b.is_numeric() => {
                let (x, y) = (a.num_pair().unwrap(), b.num_pair().unwrap());
                x == y
            }
            (ValKind::Undefined, _) | (_, ValKind::Undefined) => {
                a.is_undefined() && b.is_undefined()
            }
            (ValKind::None, ValKind::None) => true,
            (ValKind::List(x), ValKind::List(y)) => {
                let (x, y) = (x.borrow(), y.borrow());
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| jval_eq(a, b))
            }
            (ValKind::Tuple(x), ValKind::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| jval_eq(a, b))
            }
            (ValKind::Object(x), ValKind::Object(y)) => {
                let (x, y) = (x.borrow(), y.borrow());
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|((ka, va), (kb, vb))| ka == kb && jval_eq(va, vb))
            }
            (ValKind::Func(x), ValKind::Func(y)) => Rc::ptr_eq(x, y),
            _ => false, // mixed types are not equal
        }
    }

    /// `value_compare` (value.cpp:1391-1452) — used by sort/min/max/dictsort
    /// and the eq/equalto tests. Unlike operator== it does NOT treat bool as
    /// numeric.
    pub fn value_compare(a: &Val, b: &Val, lt: bool) -> Result<bool, String> {
        // numerics (int/float only)
        let num = |v: &Val| matches!(v.kind(), ValKind::Int(_) | ValKind::Float(_));
        if num(a) && num(b) {
            let (x, y) = (a.num_pair().unwrap().1, b.num_pair().unwrap().1);
            return Ok(if lt { x < y } else { x > y });
        }
        // string vs string / string vs number: byte-wise as_string compare
        if (matches!(b.kind(), ValKind::Str(_)) && num(a))
            || (matches!(a.kind(), ValKind::Str(_)) && num(b))
            || (a.is_string() && b.is_string())
        {
            let (x, y) = (a.as_string()?, b.as_string()?);
            return Ok(if lt { x < y } else { x > y });
        }
        // bool vs bool: only eq/ne comparisons exist in minja (others throw
        // and are swallowed → false)
        if matches!(a.kind(), ValKind::Bool(_)) && matches!(b.kind(), ValKind::Bool(_)) {
            return Err("Unsupported comparison operator for bool type".to_string());
        }
        Ok(false)
    }

    /// `value_compare` eq flavor (a->as_float() == b->as_float() /
    /// as_string equality; bool pair compares directly).
    pub fn value_eq_cmp(a: &Val, b: &Val) -> bool {
        let num = |v: &Val| matches!(v.kind(), ValKind::Int(_) | ValKind::Float(_));
        if num(a) && num(b) {
            return a.num_pair().unwrap().1 == b.num_pair().unwrap().1;
        }
        if (matches!(b.kind(), ValKind::Str(_)) && num(a))
            || (matches!(a.kind(), ValKind::Str(_)) && num(b))
            || (a.is_string() && b.is_string())
        {
            return (a.as_string().unwrap_or_default()) == b.as_string().unwrap_or_default();
        }
        if let (ValKind::Bool(x), ValKind::Bool(y)) = (a.kind(), b.kind()) {
            return x == y;
        }
        false
    }

    /// Python-style slicing (value.cpp:72-117): clamped start/stop, `step`
    /// sign sets direction, `step == 0` yields empty.
    fn py_slice_indexes(len: i64, start: i64, stop: i64, step: i64) -> Vec<i64> {
        let direction = if step > 0 {
            1
        } else if step < 0 {
            -1
        } else {
            0
        };
        let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
        let (start_val, stop_val) = if direction >= 0 {
            (
                if start < 0 {
                    clamp(len + start, 0, len)
                } else {
                    clamp(start, 0, len)
                },
                if stop < 0 {
                    clamp(len + stop, 0, len)
                } else {
                    clamp(stop, 0, len)
                },
            )
        } else {
            (
                if start < 0 {
                    clamp(len + start, 0, len)
                } else {
                    clamp(start, 0, len - 1)
                },
                if stop < -1 {
                    clamp(len + stop, -1, len - 1)
                } else {
                    clamp(stop, -1, len - 1)
                },
            )
        };
        let mut out = Vec::new();
        if direction == 0 {
            return out;
        }
        let mut i = start_val;
        while direction * i < direction * stop_val {
            if (0..len).contains(&i) {
                out.push(i);
            }
            i += step;
        }
        out
    }

    /// `tojson` (value.cpp:235-262 + value_to_json): compact JSON with
    /// `", "` / `": "` separators by default (indent -1), JSON string
    /// escaping, non-ASCII passed through (ensure_ascii=false). Marks
    /// deep-used.
    pub fn tojson(v: &Val) -> String {
        tojson_opts(v, false)
    }

    /// `tojson` with the `ensure_ascii` kwarg: escapes non-ASCII inside
    /// string values as \uXXXX (json_ensure_ascii_preserving_format,
    /// value.cpp:175-233), preserving other format.
    pub fn tojson_opts(v: &Val, ensure_ascii: bool) -> String {
        mark_deep_used(v);
        fn escape(s: &str, ensure_ascii: bool) -> String {
            let mut out = String::with_capacity(s.len() + 2);
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{8}' => out.push_str("\\b"),
                    '\u{c}' => out.push_str("\\f"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => {
                        out.push_str(&format!("\\u{:04x}", c as u32));
                    }
                    c if ensure_ascii && (c as u32) >= 0x80 => {
                        // surrogate pair for astral codepoints
                        let cp = c as u32;
                        if cp <= 0xFFFF {
                            out.push_str(&format!("\\u{cp:04x}"));
                        } else {
                            let cp = cp - 0x10000;
                            out.push_str(&format!("\\u{:04x}", 0xD800 + ((cp >> 10) & 0x3FF)));
                            out.push_str(&format!("\\u{:04x}", 0xDC00 + (cp & 0x3FF)));
                        }
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        fn convert(v: &Val, ensure_ascii: bool) -> String {
            match v.kind() {
                // none/undefined → "null" (value.cpp:1476-1477)
                ValKind::Undefined | ValKind::None => "null".to_string(),
                ValKind::Bool(b) => {
                    if *b {
                        "true".into()
                    } else {
                        "false".into()
                    }
                }
                ValKind::Int(i) => i.to_string(),
                // C++ streams the double (6 significant digits by default);
                // the crate JSON dumper's shortest-round-trip form matches on
                // every value the vendor templates emit
                ValKind::Float(f) => crate::json_schema::dump_float_pub(*f),
                ValKind::Str(s) => escape(s, ensure_ascii),
                ValKind::List(items) => {
                    let parts: Vec<String> = items
                        .borrow()
                        .iter()
                        .map(|v| convert(v, ensure_ascii))
                        .collect();
                    format!("[{}]", parts.join(", "))
                }
                ValKind::Tuple(items) => {
                    let parts: Vec<String> =
                        items.iter().map(|v| convert(v, ensure_ascii)).collect();
                    format!("[{}]", parts.join(", "))
                }
                ValKind::Object(fields) => {
                    let parts: Vec<String> = fields
                        .borrow()
                        .iter()
                        .map(|(k, val)| {
                            format!(
                                "{}: {}",
                                escape(k, ensure_ascii),
                                convert(val, ensure_ascii)
                            )
                        })
                        .collect();
                    format!("{{{}}}", parts.join(", "))
                }
                ValKind::Func(_) => "null".to_string(),
            }
        }
        convert(v, ensure_ascii)
    }

    // ---- lexer (token-level port of common/jinja/lexer.cpp:112-227) -------

    #[derive(Clone, Debug)]
    pub enum Tok {
        Text(String),
        Output(String), // inner expression source
        Stmt(String),   // inner statement source
    }

    #[inline]
    fn is_space(c: u8) -> bool {
        matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
    }

    /// Tokenize with the whitespace semantics of the pinned lexer
    /// (`lstrip_blocks=true`, `trim_blocks=true`, `-` trim markers, one
    /// trailing template newline stripped — lexer.cpp:54-57, 112-210).
    pub fn lex(source: &str) -> Result<Vec<Tok>, String> {
        // C++ lexer normalizes line endings and strips a single trailing \n
        // (lexer.cpp:49, 54-57).
        let mut src = source.replace("\r\n", "\n").replace('\r', "\n");
        if src.ends_with('\n') {
            src.pop();
        }
        let b = src.as_bytes();
        let n = b.len();
        let mut toks: Vec<Tok> = Vec::new();

        // What kind of tag closed just before the current text run.
        // None = start of file (C++ initial state is close_statement, but the
        // `pos > 3` guard makes all trim flags false there).
        #[derive(Clone, Copy, PartialEq)]
        enum Close {
            Expr,    // }}
            Stmt,    // %}
            Comment, // #}
        }
        let mut prev_close: Option<(Close, bool /* had `-` rstrip marker */)> = None;

        let mut pos = 0usize;
        loop {
            // ---- find next tag start "{%" / "{{" / "{#" ----
            let mut tag_at = None;
            let mut i = pos;
            while i + 1 < n {
                if b[i] == b'{' && (b[i + 1] == b'{' || b[i + 1] == b'%' || b[i + 1] == b'#') {
                    tag_at = Some(i);
                    break;
                }
                i += 1;
            }
            let text_end = tag_at.unwrap_or(n);
            if pos >= n && tag_at.is_none() {
                break;
            }

            // ---- text segment [pos, text_end) — lexer.cpp:132-209 trims ----
            let mut text = &src[pos..text_end];

            // lstrip_blocks (lexer.cpp:161-179): the upcoming tag starts with
            // {% or {# (their next_pos_is({'%','#','-'}) — a '-' cannot end a
            // text scan) and only whitespace precedes it on its line.
            if let Some(t) = tag_at {
                if b[t + 1] == b'%' || b[t + 1] == b'#' {
                    let tb = text.as_bytes();
                    let mut cur = tb.len();
                    let mut end = tb.len();
                    // lexer.cpp:168-178 walks ABSOLUTE indices in the source;
                    // the `current == 1` full-trim case only fires at the very
                    // start of the file, so a relative port must not reuse it
                    // (a length-1 "\n" text would be wrongly erased).
                    while cur > 0 {
                        let c = tb[cur - 1];
                        if c == b'\n' {
                            end = cur; // trim from the start of the line (keep \n)
                            break;
                        }
                        if !is_space(c) {
                            break; // non-whitespace before newline: keep
                        }
                        cur -= 1;
                    }
                    text = &text[..end];
                }
            }

            // trim_blocks (lexer.cpp:183-188): after %} / #} (or -%} / -#})
            // remove one leading newline of the text; `-}}`/`-%}`/`-#}`
            // additionally lstrip ALL leading whitespace (is_rstrip_block,
            // lexer.cpp:190-195). After a plain `}}` nothing is trimmed.
            let (rstrip_prev, trim_newline) = match prev_close {
                Some((_, true)) => (true, true), // -%} / -#} / -}}
                Some((Close::Stmt, false)) | Some((Close::Comment, false)) => (false, true), // %} / #}
                Some((Close::Expr, _)) => (false, false),                                    // }}
                None => (false, false),
            };

            let mut text = text.to_string();
            if rstrip_prev {
                let tb = text.as_bytes();
                let mut k = 0;
                while k < tb.len() && is_space(tb[k]) {
                    k += 1;
                }
                text.drain(..k);
            } else if trim_newline && text.starts_with('\n') {
                text.remove(0);
            }

            // is_lstrip_block (lexer.cpp:197-203): current tag starts with
            // "{{-", "{%-" or "{#-" → rstrip all trailing whitespace of the
            // text.
            if let Some(t) = tag_at {
                if t + 2 < n && b[t + 2] == b'-' {
                    while text.ends_with(|c: char| is_space(c as u8)) {
                        text.pop();
                    }
                }
            }

            if !text.is_empty() {
                toks.push(Tok::Text(text));
            }

            // ---- consume the tag (lexer.cpp:212-227 for comments) ----
            let Some(t) = tag_at else { break };
            let kind = b[t + 1]; // '{' | '%' | '#'
            let close = match kind {
                b'{' => b'}', // }}
                b'%' => b'%',
                _ => b'#',
            };
            let ltrim = t + 2 < n && b[t + 2] == b'-';
            let mut j = t + 2 + usize::from(ltrim);
            let mut inner_end = None;
            if kind == b'#' {
                // comments: plain scan (C++ lexer does not honor quotes here)
                while j + 1 < n {
                    if b[j] == close && b[j + 1] == b'}' {
                        inner_end = Some(j);
                        break;
                    }
                    j += 1;
                }
            } else {
                // expressions/statements: honor quoted literals — a literal
                // may contain "{{"/"}}" (e.g. Qwen2.5's tool_call example)
                while j + 1 < n {
                    let ch = b[j];
                    if ch == b'\'' || ch == b'"' {
                        j += 1;
                        while j < n {
                            if b[j] == b'\\' {
                                j += 2;
                                continue;
                            }
                            if b[j] == ch {
                                j += 1;
                                break;
                            }
                            j += 1;
                        }
                        continue;
                    }
                    if ch == close && b[j + 1] == b'}' {
                        inner_end = Some(j);
                        break;
                    }
                    j += 1;
                }
            }
            let Some(inner_end) = inner_end else {
                return Err(match kind {
                    b'{' => "unterminated {{ … }} expression".to_string(),
                    b'%' => "unterminated {% … %} statement".to_string(),
                    _ => "unterminated {# … #} comment".to_string(),
                });
            };
            let mut inner = &src[t + 2 + usize::from(ltrim)..inner_end];
            let rtrim = inner.ends_with('-');
            if rtrim {
                inner = &inner[..inner.len() - 1];
            }
            let inner = inner.trim();

            match kind {
                b'#' => {} // comment: consumed, no token
                b'{' => toks.push(Tok::Output(inner.to_string())),
                _ => toks.push(Tok::Stmt(inner.to_string())),
            }
            prev_close = Some((
                match kind {
                    b'{' => Close::Expr,
                    b'%' => Close::Stmt,
                    _ => Close::Comment,
                },
                rtrim,
            ));
            pos = inner_end + 2;
            if pos >= n {
                break;
            }
        }
        Ok(toks)
    }

    // ---- AST (runtime.h statement/expression types) ------------------------

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum BinOp {
        // parser.cpp:112-300 — operator set of binary_expression
        Or,
        And,
        Eq,
        Ne,
        Lt,
        Gt,
        Le,
        Ge,
        In,
        NotIn,
        Add,    // '+'
        Concat, // '~'
        Sub,
        Mul,
        Div,
        Mod,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum UnOp {
        Not,
        Neg,
        Pos,
    }

    #[derive(Clone, Debug)]
    pub enum Expr {
        Lit(Val),
        Ident(String),
        Array(Vec<Expr>),
        TupleLit(Vec<Expr>),
        /// `{key: value, …}` object literal (object_literal, runtime.h:435)
        ObjectLit(Vec<(Expr, Expr)>),
        /// member access: `obj[expr]` (computed) or `obj.prop` — the static
        /// property is an `Ident` name or an integer `Lit` (member_expression,
        /// runtime.h:327-345)
        Member {
            object: Box<Expr>,
            property: Box<Expr>,
            computed: bool,
        },
        /// call with positional and keyword args (a kwarg argument is
        /// [`Expr::Kwarg`]); `callee` may be an [`Expr::Member`] (method call)
        Call {
            callee: Box<Expr>,
            args: Vec<Expr>,
        },
        Kwarg(String, Box<Expr>),
        Unary(UnOp, Box<Expr>),
        Binary(BinOp, Box<Expr>, Box<Expr>),
        /// `operand | filter(args)` — `call` records the parenthesized form
        /// (filter_expression + apply_filter's identifier/call distinction,
        /// runtime.cpp:320-374)
        Filter {
            operand: Box<Expr>,
            name: String,
            args: Vec<Expr>,
            call: bool,
        },
        /// `operand is (not) test(args)` — "value is something" translates to
        /// `test_is_something(value)` (test_expression, runtime.cpp:390-437)
        Test {
            operand: Box<Expr>,
            negate: bool,
            name: String,
            args: Vec<Expr>,
        },
        /// `a if cond else b` (ternary_expression, runtime.h:669-696)
        Ternary {
            cond: Box<Expr>,
            t: Box<Expr>,
            f: Box<Expr>,
        },
        /// `a if cond` on an iterable — the `{% for x in xs if cond %}` filter
        /// (select_expression, runtime.h:523-546)
        Select {
            lhs: Box<Expr>,
            test: Box<Expr>,
        },
        /// `[start:stop:step]` sub-script (slice_expression, runtime.h:592)
        Slice {
            start: Option<Box<Expr>>,
            stop: Option<Box<Expr>>,
            step: Option<Box<Expr>>,
        },
        /// `obj[]` — omitted subscript (blank_expression, runtime.h:320)
        Blank,
    }

    /// for-loop variable(s): identifier or tuple unpack (runtime.h:203-226)
    #[derive(Clone, Debug)]
    pub enum LoopVar {
        Ident(String),
        Tuple(Vec<String>),
    }

    #[derive(Clone, Debug)]
    pub enum Node {
        Text(String),
        Output(Expr),
        /// `{% if %}` arms — `(None, body)` is the else arm; elif chains
        /// flatten in order (parser.cpp:241-269 nests them in `alternate`,
        /// which is behaviorally the ordered arm list)
        If(Vec<(Option<Expr>, Vec<Node>)>),
        /// `{% for %}` (+ inline `if` filter via [`Expr::Select`], + `{% else %}`
        /// no-iteration block)
        For {
            loop_var: LoopVar,
            iter: Expr,
            body: Vec<Node>,
            alternate: Vec<Node>,
        },
        /// `{% set x = expr %}` or block `{% set x %}…{% endset %}` (a block
        /// set's value is the body's rendered string, runtime.cpp:644-646)
        Set {
            assignee: Expr,
            value: Option<Expr>,
            body: Vec<Node>,
        },
        /// `{% macro name(params) %}` — defines a function value in the
        /// current scope (runtime.cpp:743-764)
        Macro {
            name: String,
            def: Rc<MacroDef>,
        },
        Break,
        Continue,
        /// `{% generation %}`/`{% endgeneration %}` markers are ignored
        /// (transformers-specific; parser.cpp:207-212)
        Noop,
    }

    pub fn parse(toks: &[Tok]) -> Result<Vec<Node>, String> {
        let mut idx = 0usize;
        let (nodes, term) = parse_block(toks, &mut idx, &[])?;
        if let Some(t) = term {
            return Err(format!("unexpected '{{% {t} %}}'"));
        }
        Ok(nodes)
    }

    fn first_word(s: &str) -> (&str, &str) {
        let s = s.trim_start();
        match s.find(char::is_whitespace) {
            Some(k) => (&s[..k], s[k..].trim_start()),
            None => (s, ""),
        }
    }

    fn parse_block(
        toks: &[Tok],
        idx: &mut usize,
        terms: &[&str],
    ) -> Result<(Vec<Node>, Option<String>), String> {
        let mut nodes = Vec::new();
        while *idx < toks.len() {
            match &toks[*idx] {
                Tok::Text(s) => {
                    nodes.push(Node::Text(s.clone()));
                    *idx += 1;
                }
                Tok::Output(s) => {
                    nodes.push(Node::Output(parse_expr(s)?));
                    *idx += 1;
                }
                Tok::Stmt(s) => {
                    let (kw, rest) = first_word(s);
                    if terms.contains(&kw) {
                        *idx += 1;
                        return Ok((nodes, Some(s.clone())));
                    }
                    match kw {
                        "if" => {
                            *idx += 1;
                            nodes.push(parse_if(toks, idx, &parse_expr(rest)?, terms)?);
                        }
                        "for" => {
                            *idx += 1;
                            nodes.push(parse_for(toks, idx, rest, terms)?);
                        }
                        "set" => {
                            *idx += 1;
                            nodes.push(parse_set(toks, idx, rest, terms)?);
                        }
                        "macro" => {
                            *idx += 1;
                            nodes.push(parse_macro(toks, idx, rest)?);
                        }
                        "break" => {
                            *idx += 1;
                            nodes.push(Node::Break);
                        }
                        "continue" => {
                            *idx += 1;
                            nodes.push(Node::Continue);
                        }
                        // transformers generation markers: ignored, content
                        // renders in place (parser.cpp:207-212)
                        "generation" | "endgeneration" => {
                            *idx += 1;
                            nodes.push(Node::Noop);
                        }
                        "elif" | "else" | "endif" | "endfor" | "endset" | "endmacro" => {
                            return Err(format!("unexpected '{{% {kw} %}}'"));
                        }
                        other => {
                            return Err(format!(
                                "unsupported statement '{other}' (supported: \
                                 if/elif/else/endif, for/endfor, set/endset, \
                                 macro/endmacro, break, continue)"
                            ));
                        }
                    }
                }
            }
        }
        Ok((nodes, None))
    }

    fn parse_if(
        toks: &[Tok],
        idx: &mut usize,
        cond: &Expr,
        parent_terms: &[&str],
    ) -> Result<Node, String> {
        let mut arms: Vec<(Option<Expr>, Vec<Node>)> = Vec::new();
        let mut cur_cond: Option<Expr> = Some(cond.clone());
        let mut cur_terms: Vec<&str> = vec!["elif", "else", "endif"];
        cur_terms.extend_from_slice(parent_terms);
        loop {
            let (body, term) = parse_block(toks, idx, &cur_terms)?;
            arms.push((cur_cond.take(), body));
            let term = term.ok_or("missing {% endif %}")?;
            let (kw, rest) = first_word(&term);
            match kw {
                "elif" => cur_cond = Some(parse_expr(rest)?),
                "else" => cur_cond = None,
                "endif" => break,
                _ => return Err(format!("unexpected '{{% {kw} %}}' inside if")),
            }
        }
        // minja nests elif as an If inside `alternate`; the ordered arm list
        // is behaviorally identical
        Ok(Node::If(arms))
    }

    fn parse_for(
        toks: &[Tok],
        idx: &mut usize,
        rest: &str,
        parent_terms: &[&str],
    ) -> Result<Node, String> {
        // parser.cpp:295-325: `for <vars> in <expr>` where <vars> is an
        // identifier or a tuple of identifiers
        let mut p = P::new(rest)?;
        let loop_var = match p.parse_expression_sequence(true)? {
            Expr::Ident(name) => LoopVar::Ident(name),
            Expr::TupleLit(items) => {
                let mut names = Vec::new();
                for item in items {
                    match item {
                        Expr::Ident(name) => names.push(name),
                        _ => return Err("for-loop variables must be identifiers".to_string()),
                    }
                }
                LoopVar::Tuple(names)
            }
            _ => return Err("invalid for-loop variable(s)".to_string()),
        };
        if !p.eat_word("in") {
            return Err(format!("unsupported for syntax: 'for {rest}'"));
        }
        let iter = p.parse_expression()?;
        p.expect_end()?;
        // bare `x if cond` at the iterable position is the loop filter
        // (select_expression, parser.cpp:346-348)
        let (iter, filter) = match iter {
            Expr::Select { lhs, test } => (*lhs, Some(*test)),
            other => (other, None),
        };
        let mut terms: Vec<&str> = vec!["endfor"];
        terms.extend_from_slice(parent_terms);
        let (body, term) = parse_block(toks, idx, &{
            // for's own `{% else %}` must terminate the body — add it on top
            // of the parent terms, but NOT above an enclosing if's else
            let mut t: Vec<&str> = vec!["else"];
            t.extend_from_slice(&terms);
            t
        })?;
        let term = term.ok_or("missing {% endfor %}")?;
        let (kw, _) = first_word(&term);
        let alternate = if kw == "else" {
            let (alt, term) = parse_block(toks, idx, &terms)?;
            let term = term.ok_or("missing {% endfor %}")?;
            if first_word(&term).0 != "endfor" {
                return Err(format!(
                    "unexpected '{{% {} %}}' inside for",
                    first_word(&term).0
                ));
            }
            alt
        } else if kw != "endfor" {
            return Err(format!("unexpected '{{% {kw} %}}' inside for"));
        } else {
            Vec::new()
        };
        // re-attach the inline filter for the evaluator
        let iter = match filter {
            Some(test) => Expr::Select {
                lhs: Box::new(iter),
                test: Box::new(test),
            },
            None => iter,
        };
        Ok(Node::For {
            loop_var,
            iter,
            body,
            alternate,
        })
    }

    fn parse_set(
        toks: &[Tok],
        idx: &mut usize,
        rest: &str,
        parent_terms: &[&str],
    ) -> Result<Node, String> {
        // parser.cpp:219-239: `set` acts as both declaration and assignment;
        // without `= value` it is a block set capturing the body's output
        let mut p = P::new(rest)?;
        let assignee = p.parse_expression_sequence(false)?;
        let value = if p.eat_tok(Pt::Eq) {
            let v = p.parse_expression_sequence(false)?;
            p.expect_end()?;
            Some(v)
        } else {
            p.expect_end()?;
            None
        };
        if value.is_none() {
            // multiline set: body until {% endset %}
            let mut terms: Vec<&str> = vec!["endset"];
            terms.extend_from_slice(parent_terms);
            let (body, term) = parse_block(toks, idx, &terms)?;
            let term = term.ok_or("missing {% endset %}")?;
            if first_word(&term).0 != "endset" {
                return Err("unexpected statement inside block set".to_string());
            }
            return Ok(Node::Set {
                assignee,
                value: None,
                body,
            });
        }
        Ok(Node::Set {
            assignee,
            value,
            body: Vec::new(),
        })
    }

    fn parse_macro(toks: &[Tok], idx: &mut usize, rest: &str) -> Result<Node, String> {
        // parser.cpp:271-281
        let mut p = P::new(rest)?;
        let name = match p.parse_primary() {
            Ok(Expr::Ident(name)) => name,
            _ => return Err("macro name must be an identifier".to_string()),
        };
        if !p.eat_tok(Pt::LParen) {
            return Err("expected macro parameter list".to_string());
        }
        let mut params: Vec<(String, Option<Expr>)> = Vec::new();
        while !p.eat_tok(Pt::RParen) {
            let pname = match p.parse_primary() {
                Ok(Expr::Ident(n)) => n,
                _ => return Err("macro parameters must be identifiers".to_string()),
            };
            let default = if p.eat_tok(Pt::Eq) {
                Some(p.parse_expression()?)
            } else {
                None
            };
            params.push((pname, default));
            if !p.eat_tok(Pt::Comma) && !p.is_tok(&Pt::RParen) {
                return Err("expected ',' or ')' in macro parameters".to_string());
            }
        }
        p.expect_end()?;
        let (body, term) = parse_block(toks, idx, &["endmacro"])?;
        let term = term.ok_or("missing {% endmacro %}")?;
        if first_word(&term).0 != "endmacro" {
            return Err("unexpected statement inside macro".to_string());
        }
        Ok(Node::Macro {
            name,
            def: Rc::new(MacroDef {
                params,
                body: Rc::new(body),
            }),
        })
    }

    // ---- expression tokenizer + parser (parser.cpp) ------------------------
    //
    // minja tokenizes the whole template once and parses expressions from
    // the token stream; the port's lexer (above) yields the *inner source*
    // of `{{ … }}` / `{% … %}` tags, so each inner string is tokenized here
    // with the same token set (lexer.h:13-47, lexer.cpp:60-110) and then run
    // through a recursive-descent parser following parser.cpp:283-606.

    /// one token of an expression's inner source (lexer.h:13-47)
    #[derive(Clone, Debug, PartialEq)]
    enum Pt {
        Ident(String),
        Num(String),
        Str(String),
        Eq,
        LParen,
        RParen,
        LBrack,
        RBrack,
        LBrace,
        RBrace,
        Comma,
        Dot,
        Colon,
        Pipe,
        Add(char),   // '+' '-' '~' (additive_binary_operator)
        Mul(char),   // '*' '/' '%' (multiplicative_binary_operator)
        Cmp(String), // "<" ">" "<=" ">=" "==" "!="
    }

    /// `lexer::tokenize` restricted to expression innards
    fn tokenize_expr(src: &str) -> Result<Vec<Pt>, String> {
        let b = src.as_bytes();
        let mut toks = Vec::new();
        let mut i = 0usize;
        while i < b.len() {
            let c = b[i];
            if (c as char).is_ascii_whitespace() {
                i += 1;
                continue;
            }
            match c {
                b'\'' | b'"' => {
                    // consume_while escape handling (lexer.cpp:57-88)
                    let quote = c;
                    i += 1;
                    let mut s = String::new();
                    loop {
                        if i >= b.len() {
                            return Err("unexpected end of string literal".to_string());
                        }
                        if b[i] == b'\\' {
                            i += 1;
                            let e = *b.get(i).ok_or("dangling escape")?;
                            i += 1;
                            s.push(match e {
                                b'n' => '\n',
                                b't' => '\t',
                                b'r' => '\r',
                                b'b' => '\u{8}',
                                b'f' => '\u{c}',
                                b'v' => '\u{b}',
                                b'\\' => '\\',
                                b'\'' => '\'',
                                b'"' => '"',
                                other => {
                                    return Err(format!(
                                        "unknown escape character \\{}",
                                        other as char
                                    ))
                                }
                            });
                            continue;
                        }
                        if b[i] == quote {
                            i += 1;
                            break;
                        }
                        let ch = src[i..].chars().next().unwrap();
                        s.push(ch);
                        i += ch.len_utf8();
                    }
                    toks.push(Pt::Str(s));
                }
                b'0'..=b'9' => {
                    // consume_numeric (lexer.cpp:92-98): digits [. digits]
                    let start = i;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
                        i += 1;
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    toks.push(Pt::Num(src[start..i].to_string()));
                }
                c if c.is_ascii_alphanumeric() || c == b'_' => {
                    // is_word (lexer.h:99-101)
                    let start = i;
                    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                        i += 1;
                    }
                    toks.push(Pt::Ident(src[start..i].to_string()));
                }
                b'=' => {
                    // "==" is a comparison operator, "=" alone is assignment
                    if i + 1 < b.len() && b[i + 1] == b'=' {
                        toks.push(Pt::Cmp("==".to_string()));
                        i += 2;
                    } else {
                        toks.push(Pt::Eq);
                        i += 1;
                    }
                }
                b'(' => {
                    toks.push(Pt::LParen);
                    i += 1;
                }
                b')' => {
                    toks.push(Pt::RParen);
                    i += 1;
                }
                b'[' => {
                    toks.push(Pt::LBrack);
                    i += 1;
                }
                b']' => {
                    toks.push(Pt::RBrack);
                    i += 1;
                }
                b'{' => {
                    toks.push(Pt::LBrace);
                    i += 1;
                }
                b'}' => {
                    toks.push(Pt::RBrace);
                    i += 1;
                }
                b',' => {
                    toks.push(Pt::Comma);
                    i += 1;
                }
                b'.' => {
                    toks.push(Pt::Dot);
                    i += 1;
                }
                b':' => {
                    toks.push(Pt::Colon);
                    i += 1;
                }
                b'|' => {
                    toks.push(Pt::Pipe);
                    i += 1;
                }
                b'+' | b'-' | b'~' => {
                    toks.push(Pt::Add(c as char));
                    i += 1;
                }
                b'*' | b'/' | b'%' => {
                    toks.push(Pt::Mul(c as char));
                    i += 1;
                }
                b'<' | b'>' | b'!' => {
                    if i + 1 < b.len() && b[i + 1] == b'=' {
                        toks.push(Pt::Cmp(src[i..i + 2].to_string()));
                        i += 2;
                    } else if c == b'!' {
                        return Err("unexpected character '!'".to_string());
                    } else {
                        toks.push(Pt::Cmp(src[i..i + 1].to_string()));
                        i += 1;
                    }
                }
                other => {
                    return Err(format!("unexpected character '{}'", other as char));
                }
            }
        }
        Ok(toks)
    }

    /// recursive-descent expression parser (parser.cpp:283-606)
    struct P {
        toks: Vec<Pt>,
        i: usize,
    }

    impl P {
        fn new(src: &str) -> Result<P, String> {
            Ok(P {
                toks: tokenize_expr(src)?,
                i: 0,
            })
        }
        fn peek(&self) -> Option<&Pt> {
            self.toks.get(self.i)
        }
        fn peek_at(&self, n: usize) -> Option<&Pt> {
            self.toks.get(self.i + n)
        }
        fn is_tok(&self, t: &Pt) -> bool {
            self.peek() == Some(t)
        }
        fn eat_tok(&mut self, t: Pt) -> bool {
            if self.is_tok(&t) {
                self.i += 1;
                true
            } else {
                false
            }
        }
        fn eat_word(&mut self, w: &str) -> bool {
            if let Some(Pt::Ident(s)) = self.peek() {
                if s == w {
                    // keyword boundary is guaranteed by tokenization
                    self.i += 1;
                    return true;
                }
            }
            false
        }
        fn is_word(&self, w: &str) -> bool {
            matches!(self.peek(), Some(Pt::Ident(s)) if s == w)
        }
        fn ident(&mut self) -> Option<String> {
            if let Some(Pt::Ident(s)) = self.peek() {
                let s = s.clone();
                self.i += 1;
                Some(s)
            } else {
                None
            }
        }
        fn expect_end(&self) -> Result<(), String> {
            if self.i != self.toks.len() {
                return Err("unexpected trailing input in expression".to_string());
            }
            Ok(())
        }

        /// parser.cpp:327-330
        fn parse_expression(&mut self) -> Result<Expr, String> {
            self.parse_if_expression()
        }

        /// `a if cond else b` ternary, or `a if cond` select on iterables
        /// (parser.cpp:332-351)
        fn parse_if_expression(&mut self) -> Result<Expr, String> {
            let a = self.parse_logical_or()?;
            if self.is_word("if") {
                self.i += 1;
                let test = self.parse_logical_or()?;
                if self.is_word("else") {
                    self.i += 1;
                    let f = self.parse_if_expression()?; // chained ternaries
                    return Ok(Expr::Ternary {
                        cond: Box::new(test),
                        t: Box::new(a),
                        f: Box::new(f),
                    });
                }
                return Ok(Expr::Select {
                    lhs: Box::new(a),
                    test: Box::new(test),
                });
            }
            Ok(a)
        }

        fn parse_logical_or(&mut self) -> Result<Expr, String> {
            let mut left = self.parse_logical_and()?;
            while self.is_word("or") {
                self.i += 1;
                let right = self.parse_logical_and()?;
                left = Expr::Binary(BinOp::Or, Box::new(left), Box::new(right));
            }
            Ok(left)
        }

        fn parse_logical_and(&mut self) -> Result<Expr, String> {
            let mut left = self.parse_logical_negation()?;
            while self.is_word("and") {
                self.i += 1;
                let right = self.parse_logical_negation()?;
                left = Expr::Binary(BinOp::And, Box::new(left), Box::new(right));
            }
            Ok(left)
        }

        fn parse_logical_negation(&mut self) -> Result<Expr, String> {
            // `not` binds looser than comparison: `not a == b` = `not (a == b)`
            // (parser.cpp:373-381)
            if self.is_word("not") {
                self.i += 1;
                return Ok(Expr::Unary(
                    UnOp::Not,
                    Box::new(self.parse_logical_negation()?),
                ));
            }
            self.parse_comparison()
        }

        /// comparison + membership share precedence, left-associative
        /// (parser.cpp:383-402)
        fn parse_comparison(&mut self) -> Result<Expr, String> {
            let mut left = self.parse_additive()?;
            loop {
                let op = if self.is_word("not")
                    && matches!(self.peek_at(1), Some(Pt::Ident(s)) if s == "in")
                {
                    self.i += 2;
                    Some(BinOp::NotIn)
                } else if self.is_word("in") {
                    self.i += 1;
                    Some(BinOp::In)
                } else if let Some(Pt::Cmp(op)) = self.peek() {
                    let op = op.clone();
                    self.i += 1;
                    Some(match op.as_str() {
                        "==" => BinOp::Eq,
                        "!=" => BinOp::Ne,
                        "<" => BinOp::Lt,
                        ">" => BinOp::Gt,
                        "<=" => BinOp::Le,
                        ">=" => BinOp::Ge,
                        _ => return Err(format!("unknown comparison operator '{op}'")),
                    })
                } else {
                    None
                };
                let Some(op) = op else { break };
                let right = self.parse_additive()?;
                left = Expr::Binary(op, Box::new(left), Box::new(right));
            }
            Ok(left)
        }

        fn parse_additive(&mut self) -> Result<Expr, String> {
            let mut left = self.parse_multiplicative()?;
            while let Some(Pt::Add(op)) = self.peek() {
                let op = *op;
                self.i += 1;
                let right = self.parse_multiplicative()?;
                let bop = match op {
                    '+' => BinOp::Add,
                    '~' => BinOp::Concat,
                    '-' => BinOp::Sub,
                    _ => unreachable!(),
                };
                left = Expr::Binary(bop, Box::new(left), Box::new(right));
            }
            Ok(left)
        }

        fn parse_multiplicative(&mut self) -> Result<Expr, String> {
            let mut left = self.parse_test()?;
            while let Some(Pt::Mul(op)) = self.peek() {
                let op = *op;
                self.i += 1;
                let right = self.parse_test()?;
                left = Expr::Binary(
                    match op {
                        '*' => BinOp::Mul,
                        '/' => BinOp::Div,
                        _ => BinOp::Mod,
                    },
                    Box::new(left),
                    Box::new(right),
                );
            }
            Ok(left)
        }

        /// `is` binds tighter than `*` (parser.cpp:424-437)
        fn parse_test(&mut self) -> Result<Expr, String> {
            let mut operand = self.parse_filter()?;
            while self.is_word("is") {
                self.i += 1;
                let negate = self.eat_word("not");
                let name = match self.parse_primary() {
                    Ok(Expr::Ident(n)) => n,
                    _ => return Err("test name must be an identifier".to_string()),
                };
                let args = if self.is_tok(&Pt::LParen) {
                    self.i += 1;
                    self.parse_args_until_rparen()?
                } else if matches!(
                    self.peek(),
                    Some(Pt::Num(_)) | Some(Pt::Str(_)) | Some(Pt::LBrace) | Some(Pt::LBrack)
                ) || matches!(
                    self.peek(),
                    Some(Pt::Ident(s)) if s != "and" && s != "or" && s != "else"
                ) {
                    // non-call test statement with an argument: `x is divisibleby 3`
                    // (parser.cpp:432-439)
                    vec![self.parse_unary()?]
                } else {
                    Vec::new()
                };
                operand = Expr::Test {
                    operand: Box::new(operand),
                    negate,
                    name,
                    args,
                };
            }
            Ok(operand)
        }

        /// filters bind outside unary: `-n|abs` is `(-n)|abs`
        /// (parser.cpp:439-450)
        fn parse_filter(&mut self) -> Result<Expr, String> {
            let mut operand = self.parse_unary()?;
            while self.is_tok(&Pt::Pipe) {
                self.i += 1;
                let name = match self.parse_primary() {
                    Ok(Expr::Ident(n)) => n,
                    _ => return Err("filter name must be an identifier".to_string()),
                };
                let (args, call) = if self.is_tok(&Pt::LParen) {
                    self.i += 1;
                    (self.parse_args_until_rparen()?, true)
                } else {
                    (Vec::new(), false)
                };
                operand = Expr::Filter {
                    operand: Box::new(operand),
                    name,
                    args,
                    call,
                };
            }
            Ok(operand)
        }

        fn parse_unary(&mut self) -> Result<Expr, String> {
            if let Some(Pt::Add(op)) = self.peek() {
                let op = *op;
                match op {
                    '-' => {
                        self.i += 1;
                        return Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)));
                    }
                    '+' => {
                        self.i += 1;
                        return Ok(Expr::Unary(UnOp::Pos, Box::new(self.parse_unary()?)));
                    }
                    _ => {}
                }
            }
            self.parse_call_member()
        }

        fn parse_call_member(&mut self) -> Result<Expr, String> {
            let primary = self.parse_primary()?;
            let member = self.parse_member(primary)?;
            if self.is_tok(&Pt::LParen) {
                return self.parse_call(member);
            }
            Ok(member)
        }

        fn parse_call(&mut self, callee: Expr) -> Result<Expr, String> {
            // parser.cpp:469-476
            self.i += 1; // '('
            let args = self.parse_args_until_rparen()?;
            let expr = Expr::Call {
                callee: Box::new(callee),
                args,
            };
            let member = self.parse_member(expr)?;
            if self.is_tok(&Pt::LParen) {
                return self.parse_call(member);
            }
            Ok(member)
        }

        /// argument list after '(' consumed; `name = value` kwargs
        /// (parser.cpp:478-506). Trailing/duplicate commas are tolerated the
        /// same way minja's loop does (it only advances on a comma).
        fn parse_args_until_rparen(&mut self) -> Result<Vec<Expr>, String> {
            let mut args = Vec::new();
            while !self.eat_tok(Pt::RParen) {
                if self.peek().is_none() {
                    return Err("missing ')'".to_string());
                }
                let arg = self.parse_expression()?;
                let arg = if self.eat_tok(Pt::Eq) {
                    let name = match &arg {
                        Expr::Ident(n) => n.clone(),
                        _ => return Err("keyword argument key must be an identifier".to_string()),
                    };
                    Expr::Kwarg(name, Box::new(self.parse_expression()?))
                } else {
                    arg
                };
                args.push(arg);
                if !self.eat_tok(Pt::Comma) && !self.is_tok(&Pt::RParen) && self.peek().is_some() {
                    // minja's loop keeps parsing expressions until ')' —
                    // surface a clear error on junk instead
                    return Err("expected ',' or ')' in argument list".to_string());
                }
            }
            Ok(args)
        }

        /// `.prop` / `[expr]` / `[a:b:c]` chains (parser.cpp:508-556)
        fn parse_member(&mut self, object: Expr) -> Result<Expr, String> {
            let mut object = object;
            while self.is_tok(&Pt::Dot) || self.is_tok(&Pt::LBrack) {
                if self.eat_tok(Pt::Dot) {
                    let property = self.parse_primary()?;
                    object = Expr::Member {
                        object: Box::new(object),
                        property: Box::new(property),
                        computed: false,
                    };
                } else {
                    self.i += 1; // '['
                    let property = self.parse_member_args()?;
                    if !self.eat_tok(Pt::RBrack) {
                        return Err("missing ']'".to_string());
                    }
                    object = Expr::Member {
                        object: Box::new(object),
                        property: Box::new(property),
                        computed: true,
                    };
                }
            }
            Ok(object)
        }

        /// `['test']` / `[0]` / `[:2]` / `[1:]` / `[1:2]` / `[1:2:3]`
        /// (parser.cpp:525-556)
        fn parse_member_args(&mut self) -> Result<Expr, String> {
            let mut slices: Vec<Option<Expr>> = Vec::new();
            let mut is_slice = false;
            while !self.is_tok(&Pt::RBrack) {
                if self.peek().is_none() {
                    return Err("missing ']'".to_string());
                }
                if self.eat_tok(Pt::Colon) {
                    slices.push(None);
                    is_slice = true;
                } else {
                    slices.push(Some(self.parse_expression()?));
                    if self.eat_tok(Pt::Colon) {
                        is_slice = true;
                    }
                }
            }
            if is_slice {
                let mut it = slices.into_iter();
                let start = it.next().flatten().map(Box::new);
                let stop = it.next().flatten().map(Box::new);
                let step = it.next().flatten().map(Box::new);
                return Ok(Expr::Slice { start, stop, step });
            }
            match slices.pop() {
                None => Ok(Expr::Blank), // `obj[]`
                Some(Some(e)) => Ok(e),
                Some(None) => unreachable!(),
            }
        }

        /// parser.cpp:558-605
        fn parse_primary(&mut self) -> Result<Expr, String> {
            match self.peek().cloned() {
                Some(Pt::Num(v)) => {
                    self.i += 1;
                    if v.contains('.') {
                        let f: f64 = v.parse().map_err(|_| format!("bad float literal '{v}'"))?;
                        Ok(Expr::Lit(Val::float(f)))
                    } else {
                        let n: i64 = v
                            .parse()
                            .map_err(|_| format!("bad integer literal '{v}'"))?;
                        Ok(Expr::Lit(Val::int(n)))
                    }
                }
                Some(Pt::Str(v)) => {
                    self.i += 1;
                    // consecutive string literals concatenate (parser.cpp:568-573)
                    let mut val = v;
                    while let Some(Pt::Str(next)) = self.peek().cloned() {
                        val += &next;
                        self.i += 1;
                    }
                    Ok(Expr::Lit(Val::new_lit_str(val)))
                }
                Some(Pt::Ident(name)) => {
                    self.i += 1;
                    Ok(Expr::Ident(name))
                }
                Some(Pt::LParen) => {
                    self.i += 1;
                    let e = self.parse_expression_sequence(false)?;
                    if !self.eat_tok(Pt::RParen) {
                        return Err("missing ')'".to_string());
                    }
                    Ok(e)
                }
                Some(Pt::LBrack) => {
                    self.i += 1;
                    let mut vals = Vec::new();
                    while !self.eat_tok(Pt::RBrack) {
                        if self.peek().is_none() {
                            return Err("missing ']'".to_string());
                        }
                        vals.push(self.parse_expression()?);
                        if !self.eat_tok(Pt::Comma) && !self.is_tok(&Pt::RBrack) {
                            return Err("expected ',' or ']' in array literal".to_string());
                        }
                    }
                    Ok(Expr::Array(vals))
                }
                Some(Pt::LBrace) => {
                    self.i += 1;
                    let mut pairs = Vec::new();
                    while !self.eat_tok(Pt::RBrace) {
                        if self.peek().is_none() {
                            return Err("missing '}'".to_string());
                        }
                        let key = self.parse_expression()?;
                        if !self.eat_tok(Pt::Colon) {
                            return Err("expected ':' in object literal".to_string());
                        }
                        pairs.push((key, self.parse_expression()?));
                        if !self.eat_tok(Pt::Comma) && !self.is_tok(&Pt::RBrace) {
                            return Err("expected ',' or '}' in object literal".to_string());
                        }
                    }
                    Ok(Expr::ObjectLit(pairs))
                }
                other => Err(match other {
                    Some(t) => format!("unexpected token in expression: {t:?}"),
                    None => "unexpected end of expression".to_string(),
                }),
            }
        }

        /// comma-separated expression sequence → tuple when >1 (parser.cpp:283-293)
        fn parse_expression_sequence(&mut self, primary: bool) -> Result<Expr, String> {
            let mut exprs = vec![if primary {
                self.parse_primary()?
            } else {
                self.parse_expression()?
            }];
            let mut is_tuple = false;
            while self.eat_tok(Pt::Comma) {
                is_tuple = true;
                exprs.push(if primary {
                    self.parse_primary()?
                } else {
                    self.parse_expression()?
                });
            }
            Ok(if is_tuple {
                Expr::TupleLit(exprs)
            } else {
                exprs.pop().unwrap()
            })
        }
    }

    pub fn parse_expr(src: &str) -> Result<Expr, String> {
        let mut p = P::new(src)?;
        let e = p.parse_expression()?;
        p.expect_end()?;
        Ok(e)
    }

    // ---- evaluator ------------------------------------------------------------
    //
    // The environment is the template-context binding of
    // `common_chat_template_direct_apply_impl` (chat.cpp:905-966): a
    // [`RenderInputs`] holding `messages`/`tools`/`bos_token`/`eos_token`/
    // `enable_thinking`/`add_generation_prompt` plus arbitrary extra context,
    // mirroring `jinja::global_from_json(ctx, inp, …)` (value.cpp:1455-1464).
    //
    // Scoping (runtime.h:55-107 `context`): entering a for loop / macro
    // invocation pushes a scope that is a *flattened copy* of the caller's
    // bindings; `{% set %}` writes into the innermost scope only (a loop body
    // set persists across that loop's iterations but does not escape it —
    // which is exactly why the vendor templates thread state through
    // `namespace()` objects, whose mutation is shared).

    /// Inputs bound as template globals ([`RenderInputs`]). Mirrors the
    /// variables the pinned jinja runtime binds for chat templates
    /// (common/jinja/value.h:71-72, common/chat.cpp:914-939).
    pub struct RenderInputs {
        pub messages: Val,
        pub tools: Val,
        pub add_generation_prompt: bool,
        pub bos_token: String,
        pub eos_token: String,
        /// `enable_thinking` global; `None` leaves it *unbound* (undefined ⇒
        /// falsy) — the caps analysis (caps.cpp:42-48) binds only
        /// messages/tools/bos/eos/add_generation_prompt unless a check
        /// explicitly sets it.
        pub enable_thinking: Option<bool>,
        /// additional globals (extra_context in chat.cpp:925-936), in order;
        /// shadows the dedicated fields when names collide
        pub extra: Vec<(String, Val)>,
        /// Unix timestamp used for `strftime_now(...)`; `None` ⇒ current time
        pub now: Option<i64>,
    }

    impl Default for RenderInputs {
        fn default() -> Self {
            RenderInputs {
                messages: Val::list(Vec::new()),
                tools: Val::list(Vec::new()),
                add_generation_prompt: false,
                bos_token: String::new(),
                eos_token: String::new(),
                enable_thinking: Some(true),
                extra: Vec::new(),
                now: None,
            }
        }
    }

    impl RenderInputs {
        /// Convert from the plain-text [`ChatTemplateCtx`] (role/content
        /// messages, no tools) — the pre-tool-calling API.
        pub fn from_chat_ctx(ctx: &ChatTemplateCtx) -> RenderInputs {
            let messages = Val::list(
                ctx.messages
                    .iter()
                    .map(|m| {
                        Val::object(vec![
                            ("role", Val::str_(m.role.as_str())),
                            ("content", Val::str_(&m.content)),
                        ])
                    })
                    .collect(),
            );
            RenderInputs {
                messages,
                tools: Val::list(Vec::new()),
                add_generation_prompt: ctx.add_generation_prompt,
                bos_token: ctx.bos_token.to_string(),
                eos_token: ctx.eos_token.to_string(),
                enable_thinking: Some(true),
                extra: Vec::new(),
                now: ctx.now,
            }
        }

        /// Build a value from a [`crate::json_schema::Json`] tree
        /// (`jinja::from_json`, value.cpp:1358-1388 — null maps to `none`).
        pub fn val_from_json(j: &crate::json_schema::Json) -> Val {
            use crate::json_schema::Json as J;
            match j {
                J::Null => Val::null_(),
                J::Bool(b) => Val::bool_(*b),
                J::Int(i) => Val::int(*i),
                J::Uint(u) => Val::int(*u as i64),
                J::Double(f) => Val::float(*f),
                J::String(s) => Val::str_(s),
                J::Array(items) => {
                    Val::list(items.iter().map(RenderInputs::val_from_json).collect())
                }
                J::Object(fields) => Val::object(
                    fields
                        .iter()
                        .map(|(k, v)| (k.as_str(), RenderInputs::val_from_json(v)))
                        .collect(),
                ),
            }
        }
    }

    /// one `context` scope: insertion-ordered name→value map with
    /// replace-in-place (`value_object_t::insert`, value.h:529-548)
    #[derive(Clone, Default)]
    struct Scope {
        vars: Vec<(String, Val)>,
    }

    impl Scope {
        fn set(&mut self, name: &str, v: Val) {
            if let Some(slot) = self.vars.iter_mut().find(|(n, _)| n == name) {
                slot.1 = v;
            } else {
                self.vars.push((name.to_string(), v));
            }
        }
        fn get(&self, name: &str) -> Option<&Val> {
            self.vars.iter().find(|(n, _)| n == name).map(|(_, v)| v)
        }
    }

    /// `context(ctx)` copy constructor (runtime.h:77-86): the new scope is a
    /// flattened copy — outermost insertion order, innermost values win.
    fn flatten_scopes(scopes: &[Scope]) -> Scope {
        let mut out = Scope::default();
        for sc in scopes {
            for (k, v) in &sc.vars {
                out.set(k, v.clone());
            }
        }
        out
    }

    /// call arguments: positional and `key = value` kwargs interleaved, like
    /// minja's flat `func_args` list holding `value_kwarg_t` entries
    /// (runtime.cpp:957-969)
    #[derive(Clone)]
    enum ArgVal {
        Pos(Val),
        Kwarg(String, Val),
    }

    #[derive(Clone, Default)]
    struct Args {
        items: Vec<ArgVal>,
    }

    impl Args {
        fn count(&self) -> usize {
            self.items.len()
        }
        fn push(&mut self, a: ArgVal) {
            self.items.push(a);
        }
        /// `get_pos` (value.cpp:43-48)
        fn get_pos(&self, pos: usize) -> Result<Val, String> {
            match self.items.get(pos) {
                Some(ArgVal::Pos(v)) => Ok(v.clone()),
                Some(ArgVal::Kwarg(k, _)) => Err(format!(
                    "argument {pos} is the keyword '{k}', expected positional"
                )),
                None => Err(format!(
                    "expected at least {} arguments, got {}",
                    pos + 1,
                    self.count()
                )),
            }
        }
        /// `get_pos(pos, default)` (value.cpp:50-55)
        fn get_pos_or(&self, pos: usize, default: Val) -> Val {
            match self.items.get(pos) {
                Some(ArgVal::Pos(v)) => v.clone(),
                _ => default,
            }
        }
        /// `get_kwarg` (value.cpp:21-31)
        fn get_kwarg(&self, key: &str) -> Option<Val> {
            self.items.iter().find_map(|a| match a {
                ArgVal::Kwarg(k, v) if k == key => Some(v.clone()),
                _ => None,
            })
        }
        /// `get_kwarg_or_pos` (value.cpp:33-41): a kwarg whose *value* is
        /// undefined falls back to the positional slot.
        fn get_kwarg_or_pos(&self, key: &str, pos: usize) -> Val {
            if let Some(v) = self.get_kwarg(key) {
                if !v.is_undefined() {
                    return v;
                }
            }
            if pos < self.count() {
                if let ArgVal::Pos(v) = &self.items[pos] {
                    return v.clone();
                }
            }
            Val::undef()
        }
        fn str_pos(&self, pos: usize) -> Result<String, String> {
            let v = self.get_pos(pos)?;
            v.as_string().map_err(|e| format!("argument {pos}: {e}"))
        }
        fn int_pos(&self, pos: usize) -> Result<i64, String> {
            match self.items.get(pos) {
                Some(ArgVal::Pos(v)) if matches!(v.kind(), ValKind::Int(_)) => {
                    Ok(v.as_int_val().unwrap())
                }
                _ => Err(format!("argument {pos}: expected an integer")),
            }
        }
    }

    struct Env<'a> {
        inputs: &'a RenderInputs,
        scopes: Vec<Scope>,
        now: i64,
    }

    /// for-loop control signals (break_statement/continue_statement,
    /// runtime.h:228-254 — minja throws; the port threads an enum)
    #[derive(Clone, Copy)]
    enum Ctl {
        Break,
        Continue,
        /// internal: the for-loop finished with no iteration and its
        /// `{% else %}` block must run in the parent scope
        RunElse,
    }

    pub fn render(nodes: &[Node], ctx: &ChatTemplateCtx) -> Result<String, String> {
        let inputs = RenderInputs::from_chat_ctx(ctx);
        render_inputs(nodes, &inputs)
    }

    /// `common_chat_template_direct_apply_impl` render phase (chat.cpp:949-957):
    /// bind globals, execute the program, gather the output string.
    pub fn render_inputs(nodes: &[Node], inputs: &RenderInputs) -> Result<String, String> {
        let mut env = Env {
            inputs,
            scopes: vec![root_scope(inputs)],
            now: inputs.now.unwrap_or_else(now_epoch),
        };
        let mut out = String::new();
        match exec(nodes, &mut env, &mut out)? {
            Some(_) => Err("'break'/'continue' outside of a loop".to_string()),
            None => Ok(out),
        }
    }

    /// chat.cpp:908-948: `inp` construction order — messages, bos/eos,
    /// enable_thinking, tools (only when non-empty), extra context,
    /// add_generation_prompt (only when true; false leaves it unbound).
    fn root_scope(inputs: &RenderInputs) -> Scope {
        let mut sc = Scope::default();
        sc.set("messages", inputs.messages.clone());
        sc.set("bos_token", Val::str_(&inputs.bos_token));
        sc.set("eos_token", Val::str_(&inputs.eos_token));
        if let Some(b) = inputs.enable_thinking {
            sc.set("enable_thinking", Val::bool_(b));
        }
        if array_items_clone(&inputs.tools).is_some_and(|t| !t.is_empty()) {
            sc.set("tools", inputs.tools.clone());
        }
        for (k, v) in &inputs.extra {
            sc.set(k, v.clone());
        }
        if inputs.add_generation_prompt {
            sc.set("add_generation_prompt", Val::bool_(true));
        }
        sc
    }

    /// items of a List/Tuple (`as_array()` — tuples are arrays too); returns
    /// clones so callers can hold the value's RefCell across later mutations
    fn array_items_clone(v: &Val) -> Option<Vec<Val>> {
        match v.kind() {
            ValKind::List(items) => Some(items.borrow().clone()),
            ValKind::Tuple(items) => Some(items.clone()),
            _ => None,
        }
    }

    fn exec(nodes: &[Node], env: &mut Env, out: &mut String) -> Result<Option<Ctl>, String> {
        for node in nodes {
            match node {
                Node::Text(s) => out.push_str(s),
                Node::Output(e) => {
                    let v = eval(e, env)?;
                    v.gather_out(out);
                }
                Node::If(arms) => {
                    // if_statement (runtime.cpp:463-482)
                    for (cond, body) in arms {
                        let take = match cond {
                            None => true, // else arm
                            Some(c) => eval(c, env)?.truthy()?,
                        };
                        if take {
                            if let Some(ctl) = exec(body, env, out)? {
                                return Ok(Some(ctl));
                            }
                            break;
                        }
                    }
                }
                Node::For {
                    loop_var,
                    iter,
                    body,
                    alternate,
                } => {
                    if let Some(ctl) = exec_for(loop_var, iter, body, alternate, env, out)? {
                        return Ok(Some(ctl));
                    }
                }
                Node::Set {
                    assignee,
                    value,
                    body,
                } => {
                    // set_statement (runtime.cpp:644-695)
                    let rhs = match value {
                        Some(e) => eval(e, env)?,
                        None => {
                            // block set: the value is the body's rendered
                            // string (exec_statements + gather)
                            let mut s = String::new();
                            let ctl = exec(body, env, &mut s)?;
                            if ctl.is_some() {
                                return Ok(ctl);
                            }
                            Val::new_lit_str(s)
                        }
                    };
                    match assignee {
                        Expr::Ident(name) => {
                            if let Some(sc) = env.scopes.last_mut() {
                                sc.set(name, rhs);
                            }
                        }
                        Expr::TupleLit(names) => {
                            let arr = array_items_clone(&rhs)
                                .ok_or("Cannot unpack non-iterable type in set")?;
                            if arr.len() != names.len() {
                                return Err(if names.len() > arr.len() {
                                    "Too few items to unpack in set".to_string()
                                } else {
                                    "Too many items to unpack in set".to_string()
                                });
                            }
                            if let Some(sc) = env.scopes.last_mut() {
                                for (name, v) in names.iter().zip(arr.into_iter()) {
                                    if let Expr::Ident(name) = name {
                                        sc.set(name, v);
                                    }
                                }
                            }
                        }
                        Expr::Member {
                            object,
                            property,
                            computed: false,
                        } => {
                            // {% set ns.attr = value %} — object insert
                            let Expr::Ident(prop) = &**property else {
                                return Err("Cannot assign to member with non-identifier property"
                                    .to_string());
                            };
                            let obj = eval(object, env)?;
                            let ValKind::Object(fields) = obj.kind() else {
                                return Err("Cannot assign to member of non-object".to_string());
                            };
                            // replace in place, else append (value.h:529-548)
                            let mut f = fields.borrow_mut();
                            if let Some(slot) = f.iter_mut().find(|(k, _)| k == prop) {
                                slot.1 = rhs;
                            } else {
                                f.push((prop.clone(), rhs));
                            }
                        }
                        _ => {
                            return Err("Invalid LHS inside assignment expression".to_string());
                        }
                    }
                }
                Node::Macro { name, def } => {
                    // macro_statement (runtime.cpp:743-764): defines the
                    // function in the current scope; renders nothing
                    if let Some(sc) = env.scopes.last_mut() {
                        sc.set(name, Val::func(name, FuncKind::Macro(def.clone())));
                    }
                }
                Node::Break => return Ok(Some(Ctl::Break)),
                Node::Continue => return Ok(Some(Ctl::Continue)),
                Node::Noop => {}
            }
        }
        Ok(None)
    }

    /// for_statement (runtime.cpp:484-642): one scope for the whole loop,
    /// inline-filter pass binds the loop variable per candidate, then the
    /// kept items iterate with a fresh `loop` object.
    fn exec_for(
        loop_var: &LoopVar,
        iter: &Expr,
        body: &[Node],
        alternate: &[Node],
        env: &mut Env,
        out: &mut String,
    ) -> Result<Option<Ctl>, String> {
        // bare `x if cond` iterable = select_expression (parser.cpp:346-348)
        let (iter_expr, filter) = match iter {
            Expr::Select { lhs, test } => ((**lhs).clone(), Some((**test).clone())),
            other => (other.clone(), None),
        };

        env.scopes.push(Scope::default()); // `context scope(ctx)`
        let r = (|| -> Result<Option<Ctl>, String> {
            let iter_val = eval(&iter_expr, env)?;
            // stats: iteration marks the iterable used + array_access
            // (runtime.cpp:497-501)
            iter_val.stats().mark_used();
            iter_val.stats().add_op("array_access");

            let over_object = matches!(iter_val.kind(), ValKind::Object(_));
            let items: Vec<Val> = match iter_val.kind() {
                ValKind::Undefined => Vec::new(),
                ValKind::Object(fields) => {
                    // objects iterate (key, value) tuples (runtime.cpp:513-523)
                    iter_val.stats().mark_used();
                    iter_val.stats().add_op("object_access");
                    fields
                        .borrow()
                        .iter()
                        .map(|(k, v)| Val::tuple(vec![Val::str_(k), v.clone()]))
                        .collect()
                }
                ValKind::List(items) => {
                    iter_val.stats().mark_used();
                    iter_val.stats().add_op("array_access");
                    items.borrow().clone()
                }
                ValKind::Tuple(items) => items.clone(),
                other => {
                    return Err(format!(
                        "Expected iterable or object type in for loop: got {other:?}"
                    ))
                }
            };

            // loop-variable bindings per candidate (runtime.cpp:544-584)
            let bind = |item: &Val| -> Result<Vec<(String, Val)>, String> {
                match loop_var {
                    LoopVar::Ident(id) => {
                        if over_object {
                            // {% for key in dict %} binds the key
                            let key = array_items_clone(item)
                                .and_then(|a| a.first().cloned())
                                .unwrap_or_else(Val::undef);
                            Ok(vec![(id.clone(), key)])
                        } else {
                            Ok(vec![(id.clone(), item.clone())])
                        }
                    }
                    LoopVar::Tuple(names) => {
                        let arr = array_items_clone(item)
                            .ok_or_else(|| "Cannot unpack non-iterable type".to_string())?;
                        if arr.len() != names.len() {
                            return Err(if names.len() > arr.len() {
                                "Too few items to unpack".to_string()
                            } else {
                                "Too many items to unpack".to_string()
                            });
                        }
                        Ok(names.iter().cloned().zip(arr.into_iter()).collect())
                    }
                }
            };

            // filtering pass (runtime.cpp:536-597)
            struct Kept {
                item: Val,
                binds: Vec<(String, Val)>,
            }
            let mut kept: Vec<Kept> = Vec::new();
            for item in &items {
                let binds = bind(item)?;
                if let Some(test) = &filter {
                    // the test evaluates in a copy of the loop scope with the
                    // variable bound (runtime.cpp:586-592)
                    let mut copy = env.scopes.last().unwrap().clone();
                    for (k, v) in &binds {
                        copy.set(k, v.clone());
                    }
                    env.scopes.push(copy);
                    let t = eval(test, env);
                    env.scopes.pop();
                    if !t?.truthy()? {
                        continue;
                    }
                }
                kept.push(Kept {
                    item: item.clone(),
                    binds,
                });
            }

            // iteration pass (runtime.cpp:599-628)
            let mut no_iteration = true;
            let n = kept.len();
            for (i, k) in kept.iter().enumerate() {
                let loop_obj = Val::object(vec![
                    ("index", Val::int(i as i64 + 1)),
                    ("index0", Val::int(i as i64)),
                    ("revindex", Val::int((n - i) as i64)),
                    ("revindex0", Val::int((n - i - 1) as i64)),
                    ("first", Val::bool_(i == 0)),
                    ("last", Val::bool_(i + 1 == n)),
                    ("length", Val::int(n as i64)),
                    (
                        "previtem",
                        if i > 0 {
                            kept[i - 1].item.clone()
                        } else {
                            Val::undef()
                        },
                    ),
                    (
                        "nextitem",
                        if i + 1 < n {
                            kept[i + 1].item.clone()
                        } else {
                            Val::undef()
                        },
                    ),
                ]);
                {
                    let sc = env.scopes.last_mut().unwrap();
                    sc.set("loop", loop_obj);
                    for (name, v) in &k.binds {
                        sc.set(name, v.clone());
                    }
                }
                match exec(body, env, out)? {
                    Some(Ctl::Continue) => continue,
                    Some(Ctl::Break) => break,
                    _ => {}
                }
                no_iteration = false;
            }

            // `{% else %}` runs in the PARENT context when no iteration took
            // place (runtime.cpp:631-636); the caller pops the loop scope
            Ok(if no_iteration && !alternate.is_empty() {
                Some(Ctl::RunElse)
            } else {
                None
            })
        })();

        match r {
            Ok(Some(Ctl::RunElse)) => {
                env.scopes.pop(); // loop scope off — alternate sees the parent
                if exec(alternate, env, out)?.is_some() {
                    // break/continue inside for-else is swallowed by the for
                    return Ok(None);
                }
            }
            Ok(_) => {
                env.scopes.pop();
            }
            Err(e) => {
                env.scopes.pop();
                return Err(e);
            }
        }
        Ok(None)
    }

    // ---- expression evaluation -----------------------------------------------

    fn eval(expr: &Expr, env: &mut Env) -> Result<Val, String> {
        match expr {
            Expr::Lit(v) => Ok(v.clone()),
            Expr::Ident(name) => eval_ident(name, env),
            Expr::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(eval(item, env)?);
                }
                Ok(Val::list(out))
            }
            Expr::TupleLit(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(eval(item, env)?);
                }
                Ok(Val::tuple(out))
            }
            Expr::ObjectLit(pairs) => {
                let mut fields: Vec<(String, Val)> = Vec::new();
                for (k, v) in pairs {
                    let key = eval(k, env)?;
                    let val = eval(v, env)?;
                    let key = key.as_string().map_err(|e| format!("object key: {e}"))?;
                    // insert replaces an existing key in place (value.h:529-548)
                    if let Some(slot) = fields.iter_mut().find(|(fk, _)| *fk == key) {
                        slot.1 = val;
                    } else {
                        fields.push((key, val));
                    }
                }
                Ok(Val::new(ValKind::Object(RefCell::new(fields))))
            }
            Expr::Member {
                object,
                property,
                computed,
            } => eval_member(object, property, *computed, env),
            Expr::Call { callee, args } => {
                let mut argv = Args::default();
                for a in args {
                    match a {
                        Expr::Kwarg(k, v) => {
                            argv.push(ArgVal::Kwarg(k.clone(), eval(v, env)?));
                        }
                        other => argv.push(ArgVal::Pos(eval(other, env)?)),
                    }
                }
                let callee_val = eval(callee, env)?;
                call_value(&callee_val, argv, env)
            }
            Expr::Kwarg(..) => Err("keyword argument outside a call".to_string()),
            Expr::Unary(op, inner) => {
                let v = eval(inner, env)?;
                match op {
                    UnOp::Not => Ok(Val::bool_(!v.truthy()?)),
                    UnOp::Neg => match v.kind() {
                        ValKind::Int(i) => Ok(Val::int(i.wrapping_neg())),
                        ValKind::Float(f) => Ok(Val::float(-f)),
                        _ => Err("Unary - operator requires numeric operand".to_string()),
                    },
                    UnOp::Pos => {
                        if matches!(v.kind(), ValKind::Int(_) | ValKind::Float(_)) {
                            Ok(v)
                        } else {
                            Err("Unary + operator requires numeric operand".to_string())
                        }
                    }
                }
            }
            Expr::Binary(op, l, r) => eval_binary(*op, l, r, env),
            Expr::Filter {
                operand,
                name,
                args,
                call,
            } => apply_filter(operand, name, args, *call, env),
            Expr::Test {
                operand,
                negate,
                name,
                args,
            } => apply_test(operand, *negate, name, args, env),
            Expr::Ternary { cond, t, f } => {
                if eval(cond, env)?.truthy()? {
                    eval(t, env)
                } else {
                    eval(f, env)
                }
            }
            Expr::Select { lhs, test } => {
                // select_expression (runtime.h:533-539): `x if cond` outside a
                // for-loop yields x when the test passes, else undefined
                if eval(test, env)?.truthy()? {
                    eval(lhs, env)
                } else {
                    Ok(Val::undef())
                }
            }
            Expr::Slice { .. } => {
                Err("slice expression must be handled by member access".to_string())
            }
            Expr::Blank => Ok(Val::undef()),
        }
    }

    /// identifier::execute_impl (runtime.cpp:83-99)
    fn eval_ident(name: &str, env: &mut Env) -> Result<Val, String> {
        // scope lookup (a stored undefined shadows outer bindings —
        // get_val returns it and the caller treats it as not-found)
        for sc in env.scopes.iter().rev() {
            if let Some(v) = sc.get(name) {
                if !v.is_undefined() {
                    v.stats().mark_used();
                    return Ok(v.clone());
                }
                break;
            }
        }
        // context constants (runtime.h:67-72) — every scope contains these
        match name {
            "true" | "True" => return Ok(Val::bool_(true)),
            "false" | "False" => return Ok(Val::bool_(false)),
            "none" | "None" => return Ok(Val::null_()),
            _ => {}
        }
        // global_builtins functions (value.cpp:351-531)
        if is_global_func(name) {
            return Ok(Val::func(name, FuncKind::Global(global_func_id(name))));
        }
        Ok(Val::undef())
    }

    fn is_global_func(name: &str) -> bool {
        matches!(
            name,
            "dict" | "namespace" | "range" | "raise_exception" | "strftime_now" | "tojson"
        )
    }
    fn global_func_id(name: &str) -> &'static str {
        match name {
            "dict" => "dict",
            "namespace" => "namespace",
            "range" => "range",
            "raise_exception" => "raise_exception",
            "strftime_now" => "strftime_now",
            _ => "tojson",
        }
    }

    /// member_expression::execute_impl (runtime.cpp:806-937)
    fn eval_member(
        object_expr: &Expr,
        property: &Expr,
        computed: bool,
        env: &mut Env,
    ) -> Result<Val, String> {
        let object = eval(object_expr, env)?;

        if computed {
            if let Expr::Slice { start, stop, step } = property {
                // slices translate to obj.slice(start, stop, step)
                // (runtime.cpp:814-837)
                let arr_size = match object.kind() {
                    ValKind::List(items) => items.borrow().len() as i64,
                    ValKind::Tuple(items) => items.len() as i64,
                    ValKind::Str(s) => s.len() as i64, // byte length (string.h:56-62)
                    _ => 0,
                };
                let step_val = match step {
                    Some(e) => eval(e, env)?,
                    None => Val::int(1),
                };
                let start_val = match start {
                    Some(e) => eval(e, env)?,
                    None => Val::int(if step_val.as_int_val().unwrap_or(1) < 0 {
                        arr_size - 1
                    } else {
                        0
                    }),
                };
                let stop_val = match stop {
                    Some(e) => eval(e, env)?,
                    None => Val::int(if step_val.as_int_val().unwrap_or(1) < 0 {
                        -1
                    } else {
                        arr_size
                    }),
                };
                let mut args = Args::default();
                args.push(ArgVal::Pos(start_val));
                args.push(ArgVal::Pos(stop_val));
                args.push(ArgVal::Pos(step_val));
                return builtin_method(&object, "slice", &args);
            }
        }

        // static `.prop`: a bound builtin method wins over an object key
        // (runtime.cpp:847-865 — jinja2 semantics: `{"obj": {"items":
        // 123}}.obj.items` is the builtin, `obj['items']` is 123)
        let property = if computed {
            eval(property, env)?
        } else {
            match property {
                Expr::Ident(name) => {
                    if let Some(bound) = try_method(&object, name) {
                        return Ok(bound);
                    }
                    Val::str_(name)
                }
                Expr::Lit(v) if matches!(v.kind(), ValKind::Int(_)) => {
                    let i = v.as_int_val().unwrap();
                    if i < 0 {
                        return Err("Static member property cannot be negative".to_string());
                    }
                    v.clone()
                }
                _ => return Err("Static member property must be an identifier".to_string()),
            }
        };

        // dispatch (runtime.cpp:868-923)
        if property.is_undefined() {
            return Ok(Val::undef());
        }
        if object.is_undefined() {
            return Ok(Val::undef());
        }

        let val = if object.is_object() {
            let key = property
                .as_string()
                .map_err(|_| "object access requires a string key")?;
            let fields = match object.kind() {
                ValKind::Object(f) => f,
                _ => unreachable!(),
            };
            let found = fields
                .borrow()
                .iter()
                .find(|(fk, _)| *fk == key)
                .map(|(_, v)| v.clone());
            match found {
                Some(v) => v,
                None => match try_method(&object, &key) {
                    Some(f) => f,
                    None => Val::undef(),
                },
            }
        } else if object.is_array() || object.is_string() {
            match property.kind() {
                // int or bool index (value_bool_t IS-A value_int_t, so
                // `arr[true]` indexes 1 — runtime.cpp:891)
                ValKind::Int(_) | ValKind::Bool(_) => {
                    let mut index = property.num_pair().unwrap().0;
                    if object.is_array() {
                        let arr = array_items_clone(&object).unwrap();
                        if index < 0 {
                            index += arr.len() as i64;
                        }
                        if index >= 0 && (index as usize) < arr.len() {
                            arr[index as usize].clone()
                        } else {
                            Val::undef()
                        }
                    } else {
                        // byte-wise single char; negative indexes do NOT wrap
                        // for strings (runtime.cpp:902-907)
                        let s = object.as_str_val().unwrap();
                        if index >= 0 && (index as usize) < s.len() {
                            Val::new_lit_str(
                                String::from_utf8_lossy(&[s.as_bytes()[index as usize]])
                                    .into_owned(),
                            )
                        } else {
                            Val::undef()
                        }
                    }
                }
                ValKind::Str(key) => match try_method(&object, key) {
                    Some(f) => f,
                    None => Val::undef(),
                },
                // any other property type on an array/string receiver now
                // yields undefined instead of throwing (runtime.cpp:891-918
                // dropped the `else { throw }` arm)
                _ => Val::undef(),
            }
        } else {
            let key = property
                .as_string()
                .map_err(|_| "Cannot access property with non-string")?;
            match try_method(&object, &key) {
                Some(f) => f,
                None => Val::undef(),
            }
        };

        // stats marking (runtime.cpp:925-934)
        val.stats().mark_used();
        object.stats().mark_used();
        property.stats().mark_used();
        if matches!(object.kind(), ValKind::Object(_))
            || matches!(
                property.kind(),
                ValKind::Str(_) | ValKind::Float(_) | ValKind::List(_) | ValKind::Tuple(_) | ValKind::None
            )
        {
            object.stats().add_op("object_access");
        } else if matches!(property.kind(), ValKind::Int(_) | ValKind::Bool(_)) {
            object.stats().add_op("array_access");
        }

        Ok(val)
    }

    /// binary_expression::execute_impl (runtime.cpp:112-300)
    fn eval_binary(op: BinOp, l: &Expr, r: &Expr, env: &mut Env) -> Result<Val, String> {
        let left = eval(l, env)?;

        // and/or return the operand VALUE, not a bool (runtime.cpp:116-121)
        match op {
            BinOp::Or => {
                return Ok(if left.truthy()? { left } else { eval(r, env)? });
            }
            BinOp::And => {
                return Ok(if left.truthy()? { eval(r, env)? } else { left });
            }
            _ => {}
        }

        let right = eval(r, env)?;

        match op {
            BinOp::Eq => return Ok(Val::bool_(jval_eq(&left, &right))),
            BinOp::Ne => return Ok(Val::bool_(!jval_eq(&left, &right))),
            _ => {}
        }

        // null/undefined operand handling (runtime.cpp:133-183)
        let null_concat = |l: &Val, r: &Val| -> Option<Result<Val, String>> {
            let l_null = l.is_none() || l.is_undefined();
            let r_null = r.is_none() || r.is_undefined();
            if (l_null && r.is_string()) || (r_null && l.is_string()) {
                let ls = if l_null {
                    String::new()
                } else {
                    l.as_string().ok()?
                };
                let rs = if r_null {
                    String::new()
                } else {
                    r.as_string().ok()?
                };
                Some(Ok(Val::new_lit_str(format!("{ls}{rs}"))))
            } else {
                None
            }
        };

        if left.is_undefined() || right.is_undefined() {
            if right.is_undefined() && matches!(op, BinOp::In | BinOp::NotIn) {
                // `anything in undefined` is false (runtime.cpp:158-161)
                return Ok(Val::bool_(op == BinOp::NotIn));
            }
            if matches!(op, BinOp::Add | BinOp::Concat) {
                if let Some(res) = null_concat(&left, &right) {
                    return res;
                }
            }
            return Err(format!(
                "Cannot perform operation {op:?} on undefined values"
            ));
        }
        if left.is_none() || right.is_none() {
            if !right.is_none() && matches!(op, BinOp::In | BinOp::NotIn) {
                // `none in {…}` looks the null up like any other value
                let member = test_is_in(&left, &right)?;
                return Ok(Val::bool_(if op == BinOp::In { member } else { !member }));
            }
            if matches!(op, BinOp::Add | BinOp::Concat) {
                if let Some(res) = null_concat(&left, &right) {
                    return res;
                }
            }
            return Err("Cannot perform operation on null values".to_string());
        }

        // numeric operations (runtime.cpp:185-224)
        if left.is_numeric() && right.is_numeric() {
            let a = left.num_pair().unwrap().1;
            let b = right.num_pair().unwrap().1;
            let is_float = matches!(left.kind(), ValKind::Float(_))
                || matches!(right.kind(), ValKind::Float(_));
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    let res = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        _ => a * b,
                    };
                    return Ok(if is_float {
                        Val::float(res)
                    } else {
                        Val::int(res as i64)
                    });
                }
                BinOp::Div => return Ok(Val::float(a / b)),
                BinOp::Mod => {
                    let rem = a % b; // fmod (truncated)
                    return Ok(if is_float {
                        Val::float(rem)
                    } else {
                        Val::int(rem as i64)
                    });
                }
                BinOp::Lt => return Ok(Val::bool_(a < b)),
                BinOp::Gt => return Ok(Val::bool_(a > b)),
                BinOp::Le => return Ok(Val::bool_(a <= b)),
                BinOp::Ge => return Ok(Val::bool_(a >= b)),
                _ => {}
            }
        }

        // array concat / membership (runtime.cpp:226-248)
        if left.is_array() && right.is_array() {
            if op == BinOp::Add {
                let mut items = array_items_clone(&left).unwrap();
                items.extend(array_items_clone(&right).unwrap());
                return Ok(Val::list(items));
            }
        } else if right.is_array() {
            if matches!(op, BinOp::In | BinOp::NotIn) {
                let member = test_is_in(&left, &right)?;
                return Ok(Val::bool_(if op == BinOp::In { member } else { !member }));
            }
        }

        // string concatenation with ~ and + (runtime.cpp:250-258)
        if (left.is_string() || right.is_string()) && matches!(op, BinOp::Concat | BinOp::Add) {
            let ls = left.as_string()?;
            let rs = right.as_string()?;
            return Ok(Val::new_lit_str(format!("{ls}{rs}")));
        }

        // Python-style string repetition (runtime.cpp:260-275)
        if op == BinOp::Mul {
            let pair = match (left.kind(), right.kind()) {
                (ValKind::Str(_), ValKind::Int(_)) => {
                    Some((left.as_str_val().unwrap(), right.as_int_val().unwrap()))
                }
                (ValKind::Int(_), ValKind::Str(_)) => {
                    Some((right.as_str_val().unwrap(), left.as_int_val().unwrap()))
                }
                _ => None,
            };
            if let Some((s, repeat)) = pair {
                let mut out = String::new();
                for _ in 0..repeat.max(0) {
                    out.push_str(&s);
                }
                return Ok(Val::new_lit_str(out));
            }
        }

        // string membership (runtime.cpp:277-286)
        if left.is_string() && right.is_string() {
            if matches!(op, BinOp::In | BinOp::NotIn) {
                let member = test_is_in(&left, &right)?;
                return Ok(Val::bool_(if op == BinOp::In { member } else { !member }));
            }
        }

        // key in object (runtime.cpp:288-297)
        if right.is_object() {
            if matches!(op, BinOp::In | BinOp::NotIn) {
                let member = test_is_in(&left, &right)?;
                return Ok(Val::bool_(if op == BinOp::In { member } else { !member }));
            }
        }

        Err(format!(
            "Unknown operator {op:?} between {:?} and {:?}",
            left.kind(),
            right.kind()
        ))
    }

    /// `test_is_in` (value.cpp:481-507)
    fn test_is_in(needle: &Val, haystack: &Val) -> Result<bool, String> {
        match haystack.kind() {
            ValKind::Undefined => Ok(false),
            ValKind::List(_) | ValKind::Tuple(_) => {
                let items = array_items_clone(haystack).unwrap();
                Ok(items.iter().any(|item| jval_eq(needle, item)))
            }
            ValKind::Str(s) => {
                let needle = needle
                    .as_str_val()
                    .ok_or_else(|| "'in' test expects a string needle for a string haystack")?;
                Ok(s.contains(&needle))
            }
            ValKind::Object(fields) => match needle.kind() {
                ValKind::Str(k) => Ok(fields.borrow().iter().any(|(fk, _)| fk == k)),
                _ => Ok(false),
            },
            other => Err(format!(
                "'in' test expects iterable as second argument, got {other:?}"
            )),
        }
    }

    /// apply_filter (runtime.cpp:320-374)
    fn apply_filter(
        operand: &Expr,
        name: &str,
        args: &[Expr],
        call: bool,
        env: &mut Env,
    ) -> Result<Val, String> {
        let filter_id = match name {
            "count" => "length",
            "d" => "default",
            "e" => "escape",
            "trim" => "strip",
            other => other,
        };
        let mut input = eval(operand, env)?;
        if !call {
            // identifier form coerces non-strings for the string filters
            // (runtime.cpp:341-352)
            if !input.is_undefined()
                && !input.is_string()
                && matches!(
                    filter_id,
                    "capitalize" | "lower" | "replace" | "strip" | "title" | "upper" | "wordcount"
                )
            {
                input = Val::new_lit_str(input.as_string()?);
            }
            return builtin_method(&input, filter_id, &Args::default());
        }
        let mut argv = Args::default();
        for a in args {
            match a {
                Expr::Kwarg(k, v) => argv.push(ArgVal::Kwarg(k.clone(), eval(v, env)?)),
                other => argv.push(ArgVal::Pos(eval(other, env)?)),
            }
        }
        builtin_method(&input, filter_id, &argv)
    }

    /// test_expression::execute_impl (runtime.cpp:390-437)
    fn apply_test(
        operand: &Expr,
        negate: bool,
        name: &str,
        args: &[Expr],
        env: &mut Env,
    ) -> Result<Val, String> {
        let input = eval(operand, env)?;
        let test_name = format!("test_is_{name}");
        if !is_test(&test_name) {
            return Err(format!("Unknown test '{name}'"));
        }
        let mut argv = Args::default();
        argv.push(ArgVal::Pos(input.clone()));
        for a in args {
            match a {
                Expr::Kwarg(k, v) => argv.push(ArgVal::Kwarg(k.clone(), eval(v, env)?)),
                other => argv.push(ArgVal::Pos(eval(other, env)?)),
            }
        }
        // stats (runtime.cpp:425-428)
        input.stats().mark_used();
        input.stats().add_op(&test_name);
        let res = run_test(&test_name, &argv)?;
        if negate {
            Ok(Val::bool_(!res.truthy()?))
        } else {
            Ok(res)
        }
    }

    fn is_test(name: &str) -> bool {
        matches!(
            name,
            "test_is_boolean"
                | "test_is_callable"
                | "test_is_odd"
                | "test_is_even"
                | "test_is_false"
                | "test_is_true"
                | "test_is_divisibleby"
                | "test_is_string"
                | "test_is_integer"
                | "test_is_float"
                | "test_is_number"
                | "test_is_iterable"
                | "test_is_sequence"
                | "test_is_mapping"
                | "test_is_lower"
                | "test_is_upper"
                | "test_is_none"
                | "test_is_defined"
                | "test_is_undefined"
                | "test_is_eq"
                | "test_is_equalto"
                | "test_is_ge"
                | "test_is_gt"
                | "test_is_greaterthan"
                | "test_is_lt"
                | "test_is_lessthan"
                | "test_is_ne"
                | "test_is_in"
                | "test_is_sameas"
        )
    }

    fn run_test(name: &str, a: &Args) -> Result<Val, String> {
        let v0 = a.get_pos_or(0, Val::undef());
        Ok(match name {
            "test_is_boolean" => Val::bool_(matches!(v0.kind(), ValKind::Bool(_))),
            "test_is_callable" => Val::bool_(matches!(v0.kind(), ValKind::Func(_))),
            "test_is_odd" => Val::bool_(v0.as_int_val().map(|v| v % 2 != 0).unwrap_or(false)),
            "test_is_even" => Val::bool_(v0.as_int_val().map(|v| v % 2 == 0).unwrap_or(false)),
            "test_is_false" => Val::bool_(matches!(v0.kind(), ValKind::Bool(false))),
            "test_is_true" => Val::bool_(matches!(v0.kind(), ValKind::Bool(true))),
            "test_is_divisibleby" => {
                let d = a.get_pos(1)?;
                Val::bool_(
                    v0.as_int_val().unwrap_or(0) % d.as_int_val().ok_or("divisibleby: int")? == 0,
                )
            }
            "test_is_string" => Val::bool_(v0.is_string()),
            "test_is_integer" => Val::bool_(matches!(v0.kind(), ValKind::Int(_))),
            "test_is_float" => Val::bool_(matches!(v0.kind(), ValKind::Float(_))),
            // bool is NOT a number here (test_type_fn<value_int, value_float>)
            "test_is_number" => {
                Val::bool_(matches!(v0.kind(), ValKind::Int(_) | ValKind::Float(_)))
            }
            // iterables: object/array/tuple (via value_array), string,
            // undefined (test_type_fn<value_object, value_array, value_string,
            // value_undefined>)
            "test_is_iterable" | "test_is_sequence" => Val::bool_(matches!(
                v0.kind(),
                ValKind::Object(_)
                    | ValKind::List(_)
                    | ValKind::Tuple(_)
                    | ValKind::Str(_)
                    | ValKind::Undefined
            )),
            "test_is_mapping" => Val::bool_(v0.is_object()),
            "test_is_lower" => Val::bool_(
                v0.as_str_val()
                    .map(|s| !s.bytes().any(|c| c.is_ascii_uppercase()))
                    .unwrap_or(false),
            ),
            "test_is_upper" => Val::bool_(
                v0.as_str_val()
                    .map(|s| !s.bytes().any(|c| c.is_ascii_lowercase()))
                    .unwrap_or(false),
            ),
            "test_is_none" => Val::bool_(v0.is_none()),
            "test_is_defined" => Val::bool_(!v0.is_undefined()),
            "test_is_undefined" => Val::bool_(v0.is_undefined()),
            "test_is_eq" | "test_is_equalto" => {
                let v1 = a.get_pos(1)?;
                Val::bool_(value_eq_cmp(&v0, &v1))
            }
            "test_is_ne" => {
                let v1 = a.get_pos(1)?;
                Val::bool_(!value_eq_cmp(&v0, &v1))
            }
            "test_is_ge" => {
                let v1 = a.get_pos(1)?;
                Val::bool_(value_compare(&v0, &v1, false).unwrap_or(false))
            }
            "test_is_gt" | "test_is_greaterthan" => {
                let v1 = a.get_pos(1)?;
                Val::bool_(value_compare(&v0, &v1, false).unwrap_or(false))
            }
            "test_is_lt" | "test_is_lessthan" => {
                let v1 = a.get_pos(1)?;
                Val::bool_(value_compare(&v0, &v1, true).unwrap_or(false))
            }
            "test_is_in" => Val::bool_(test_is_in(&v0, &a.get_pos(1)?)?),
            // identity test (value.cpp:570-592): mirrors CPython — none/none
            // and bools by value; ints only within the small-int cache range
            // ([-5, 256]); everything else by shared_ptr identity (the port's
            // Rc pointer equality)
            "test_is_sameas" => {
                let b = a.get_pos(1)?;
                let mut res = false;
                if !v0.is_undefined() && !b.is_undefined() {
                    if v0.is_none() && b.is_none() {
                        res = true;
                    } else if let (ValKind::Bool(x), ValKind::Bool(y)) = (v0.kind(), b.kind()) {
                        res = x == y;
                    } else if let (ValKind::Int(x), ValKind::Int(y)) = (v0.kind(), b.kind()) {
                        res = (*x >= -5 && *x <= 256) && x == y;
                    } else {
                        res = Rc::ptr_eq(&v0.v, &b.v);
                    }
                }
                Val::bool_(res)
            }
            _ => return Err(format!("Unknown test '{name}'")),
        })
    }

    // ---- builtin functions / methods (value.cpp:351-1352) --------------------

    /// try_builtin_func with undef_on_missing (runtime.cpp:302-318): returns
    /// the bound method when the receiver's type has it; marks used + op.
    fn try_method(recv: &Val, name: &str) -> Option<Val> {
        if !type_has_builtin(recv, name) {
            return None;
        }
        recv.stats().mark_used();
        recv.stats().add_op(name);
        Some(Val::func(
            name,
            FuncKind::Method {
                recv: recv.clone(),
                name: name.to_string(),
            },
        ))
    }

    fn type_has_builtin(recv: &Val, name: &str) -> bool {
        match recv.kind() {
            ValKind::Undefined => matches!(
                name,
                "default"
                    | "capitalize"
                    | "first"
                    | "items"
                    | "join"
                    | "last"
                    | "length"
                    | "list"
                    | "lower"
                    | "map"
                    | "max"
                    | "min"
                    | "reject"
                    | "rejectattr"
                    | "replace"
                    | "reverse"
                    | "safe"
                    | "select"
                    | "selectattr"
                    | "sort"
                    | "string"
                    | "strip"
                    | "sum"
                    | "title"
                    | "truncate"
                    | "unique"
                    | "upper"
                    | "wordcount"
            ),
            ValKind::None => matches!(
                name,
                "default"
                    | "tojson"
                    | "string"
                    | "safe"
                    | "items"
                    | "map"
                    | "reject"
                    | "rejectattr"
                    | "select"
                    | "selectattr"
                    | "unique"
            ),
            ValKind::Bool(_) => {
                matches!(
                    name,
                    "default" | "int" | "float" | "safe" | "string" | "tojson"
                )
            }
            ValKind::Int(_) | ValKind::Float(_) => {
                matches!(
                    name,
                    "default" | "abs" | "int" | "float" | "safe" | "string" | "tojson"
                )
            }
            ValKind::Str(_) => matches!(
                name,
                "default"
                    | "upper"
                    | "lower"
                    | "strip"
                    | "rstrip"
                    | "lstrip"
                    | "title"
                    | "capitalize"
                    | "length"
                    | "startswith"
                    | "endswith"
                    | "split"
                    | "rsplit"
                    | "replace"
                    | "format"
                    | "int"
                    | "float"
                    | "string"
                    | "slice"
                    | "safe"
                    | "tojson"
                    | "indent"
                    | "join"
            ),
            ValKind::List(_) | ValKind::Tuple(_) => matches!(
                name,
                "default"
                    | "list"
                    | "first"
                    | "last"
                    | "length"
                    | "slice"
                    | "selectattr"
                    | "select"
                    | "rejectattr"
                    | "reject"
                    | "join"
                    | "string"
                    | "tojson"
                    | "map"
                    | "append"
                    | "pop"
                    | "sort"
                    | "reverse"
                    | "min"
                    | "max"
                    | "unique"
            ),
            ValKind::Object(_) => matches!(
                name,
                "get"
                    | "keys"
                    | "values"
                    | "items"
                    | "tojson"
                    | "string"
                    | "length"
                    | "dictsort"
                    | "join"
            ),
            ValKind::Func(_) => false,
        }
    }

    fn val_type_name(v: &Val) -> &'static str {
        match v.kind() {
            ValKind::Undefined => "Undefined",
            ValKind::None => "None",
            ValKind::Bool(_) => "Boolean",
            ValKind::Int(_) => "Integer",
            ValKind::Float(_) => "Float",
            ValKind::Str(_) => "String",
            ValKind::List(_) => "Array",
            ValKind::Tuple(_) => "Tuple",
            ValKind::Object(_) => "Object",
            ValKind::Func(_) => "Function",
        }
    }

    /// invoke a callee value with evaluated args (call_expression,
    /// runtime.cpp:939-955). For methods the receiver is argument 0.
    fn call_value(callee: &Val, args: Args, env: &mut Env) -> Result<Val, String> {
        let ValKind::Func(f) = callee.kind() else {
            return Err(format!(
                "Callee is not a function: got {}",
                val_type_name(callee)
            ));
        };
        match &f.kind {
            FuncKind::Global(g) => global_call(g, &args, env),
            FuncKind::Method { recv, name } => builtin_method(recv, name, &args),
            FuncKind::Macro(def) => {
                // macro closure: a snapshot of the caller's bindings
                // (runtime.cpp:749-758 — context macro_ctx(args.ctx))
                let mut scope = flatten_scopes(&env.scopes);
                // bind_parameters (runtime.cpp:697-741): defaults evaluate in
                // the CALLER's context (args.ctx)
                for (i, (pname, default)) in def.params.iter().enumerate() {
                    let v = if i < args.count() {
                        args.get_kwarg_or_pos(pname, i)
                    } else {
                        match default {
                            Some(d) => eval(d, env)?,
                            None => {
                                return Err(format!(
                                    "Not enough arguments provided to '{}'",
                                    f.name
                                ))
                            }
                        }
                    };
                    scope.set(pname, v);
                }
                env.scopes.push(scope);
                let mut out = String::new();
                let r = exec(&def.body, env, &mut out);
                env.scopes.pop();
                if let Some(_) = r? {
                    // a break/continue inside a macro body propagates to the
                    // caller's loop in minja (C++ exceptions); the vendor
                    // templates never do this — fail loudly rather than lose
                    // the signal
                    return Err(
                        "break/continue inside a macro body cannot reach a loop".to_string()
                    );
                }
                Ok(Val::new_lit_str(out))
            }
        }
    }

    /// `toobject` (value.cpp:376-408) — shared impl of `dict()` and
    /// `namespace()`: an iterable of 2-tuples and/or an object and/or kwargs.
    fn toobject(a: &Args) -> Result<Val, String> {
        let mut fields: Vec<(String, Val)> = Vec::new();
        // insert replaces an existing key in place (value_object_t::insert)
        macro_rules! insert {
            ($key:expr, $val:expr) => {
                if let Some(slot) = fields.iter_mut().find(|(fk, _)| *fk == $key) {
                    slot.1 = $val;
                } else {
                    fields.push(($key, $val));
                }
            };
        }
        let iter = a.get_pos_or(0, Val::undef());
        let mut iter_first = false;
        if iter.is_array() {
            iter_first = true;
            for it in array_items_clone(&iter).unwrap_or_default() {
                let tuple = array_items_clone(&it);
                if let Some(t) = &tuple {
                    if t.len() == 2 {
                        let key = t[0].as_string()?;
                        let val = t[1].clone();
                        insert!(key, val);
                        continue;
                    }
                }
                return Err(format!(
                    "namespace/dict() iterable argument must consist of tuples, not {}",
                    val_type_name(&it)
                ));
            }
        } else if iter.is_object() {
            iter_first = true;
            if let ValKind::Object(pairs) = iter.kind() {
                for (k, v) in pairs.borrow().iter() {
                    let key = k.clone();
                    let val = v.clone();
                    insert!(key, val);
                }
            }
        }
        for arg in &a.items {
            match arg {
                ArgVal::Kwarg(k, v) => {
                    let key = k.clone();
                    let val = v.clone();
                    insert!(key, val);
                }
                ArgVal::Pos(v) => {
                    if !iter_first {
                        return Err(format!(
                            "namespace/dict() arguments must be kwargs, dict and/or iterable of tuples, not {}",
                            val_type_name(v)
                        ));
                    }
                }
            }
            iter_first = false;
        }
        Ok(Val::new(ValKind::Object(RefCell::new(fields))))
    }

    /// global_builtins functions (value.cpp:351-531)
    fn global_call(name: &str, a: &Args, env: &mut Env) -> Result<Val, String> {
        match name {
            "raise_exception" => {
                let msg = a
                    .get_pos(0)
                    .map_err(|_| "raise_exception expects a message")?;
                Err(format!(
                    "Jinja Exception: {}",
                    msg.as_string().unwrap_or_default()
                ))
            }
            "dict" | "namespace" => toobject(a),
            "strftime_now" => {
                let fmt = a.str_pos(0)?;
                Ok(Val::str_(&strftime_utc(env.now, &fmt)))
            }
            "range" => {
                // value.cpp:382-419
                let (start, stop, step) = match a.count() {
                    1 => (0, a.int_pos(0)?, 1),
                    2 => (a.int_pos(0)?, a.int_pos(1)?, 1),
                    n if n >= 3 => (a.int_pos(0)?, a.int_pos(1)?, a.int_pos(2)?),
                    _ => return Err("range() expects between 1 and 3 arguments".to_string()),
                };
                if step == 0 {
                    return Err("range() step argument must not be zero".to_string());
                }
                let mut out = Vec::new();
                if step > 0 {
                    let mut i = start;
                    while i < stop {
                        out.push(Val::int(i));
                        i += step;
                    }
                } else {
                    let mut i = start;
                    while i > stop {
                        out.push(Val::int(i));
                        i += step;
                    }
                }
                Ok(Val::list(out))
            }
            "tojson" => {
                // tojson (value.cpp:235-262)
                let input = a.get_pos(0)?;
                let ensure_ascii = a.get_kwarg_or_pos("ensure_ascii", 1).truthy()?;
                let indent = match a.get_kwarg_or_pos("indent", 2).kind() {
                    ValKind::Int(i) => *i,
                    _ => -1,
                };
                if a.get_kwarg_or_pos("sort_keys", 4).truthy()? {
                    return Err("NotImplemented: tojson sort_keys=true not implemented".to_string());
                }
                let _ = indent; // indent > 0 is not used by the vendor templates
                Ok(Val::new_lit_str(tojson_opts(&input, ensure_ascii)))
            }
            other => Err(format!("function '{other}()' is not supported")),
        }
    }

    /// per-type builtin tables (value.cpp:534-1352). `args` are the explicit
    /// call arguments; the receiver is argument 0 (the filter input).
    fn builtin_method(recv: &Val, name: &str, args: &Args) -> Result<Val, String> {
        if !type_has_builtin(recv, name) {
            return Err(format!(
                "Unknown (built-in) filter '{name}' for type {}",
                val_type_name(recv)
            ));
        }
        // try_builtin_func marking happens at bind time (try_method); for the
        // slice/filter paths that bypass try_method, mark here
        recv.stats().mark_used();
        recv.stats().add_op(name);

        let mut full = Args::default();
        full.push(ArgVal::Pos(recv.clone()));
        for item in &args.items {
            full.push(item.clone());
        }
        let a = &full;

        match name {
            // ---- default_value (value.cpp:341-349) ------------------------
            "default" if !recv.is_object() && !recv.is_string() => {
                let check_bool = a.get_kwarg_or_pos("boolean", 2).truthy()?;
                let no_value = if check_bool {
                    !recv.truthy()?
                } else {
                    recv.is_undefined() || recv.is_none()
                };
                Ok(if no_value {
                    a.get_pos_or(1, Val::undef())
                } else {
                    recv.clone()
                })
            }
            // string's own default (value.cpp:822-837): the input is by
            // construction a string, so only the `boolean` kwarg can swap it
            "default" => {
                let default_val = if a.count() > 1 && !a.get_pos_or(1, Val::undef()).is_undefined()
                {
                    a.get_pos_or(1, Val::undef())
                } else {
                    Val::new_lit_str(String::new())
                };
                let boolean_val = a.get_kwarg_or_pos("boolean", 2);
                if boolean_val.truthy()? && !recv.truthy()? {
                    Ok(default_val)
                } else {
                    Ok(recv.clone())
                }
            }
            "abs" => match recv.kind() {
                ValKind::Int(i) => Ok(Val::int(i.abs())),
                ValKind::Float(f) => Ok(Val::float(f.abs())),
                _ => unreachable!(),
            },
            "int" => match recv.kind() {
                ValKind::Int(_) => Ok(recv.clone()),
                ValKind::Float(f) => Ok(Val::int(*f as i64)),
                ValKind::Bool(b) => Ok(Val::int(*b as i64)),
                ValKind::Str(_) => {
                    // value.cpp:788-806 — std::stoi semantics with base kwarg
                    let s = a.str_pos(0)?;
                    let base = match a.get_kwarg_or_pos("base", 2).kind() {
                        ValKind::Int(b) => *b,
                        _ => 10,
                    };
                    if base != 0 && !(2..=36).contains(&base) {
                        return Err("int() base must be 0 or between 2 and 36".to_string());
                    }
                    let parsed = stoi_like(&s, base);
                    match parsed {
                        Some(v) => Ok(Val::int(v)),
                        None => Ok(Val::int(match a.get_kwarg_or_pos("default", 1).kind() {
                            ValKind::Int(d) => *d,
                            _ => 0,
                        })),
                    }
                }
                _ => unreachable!(),
            },
            "float" => match recv.kind() {
                ValKind::Float(_) => Ok(recv.clone()),
                ValKind::Int(i) => Ok(Val::float(*i as f64)),
                ValKind::Bool(b) => Ok(Val::float(*b as i64 as f64)),
                ValKind::Str(_) => {
                    let s = a.str_pos(0)?;
                    match stod_like(&s) {
                        Some(v) => Ok(Val::float(v)),
                        None => Ok(Val::float(match a.get_kwarg_or_pos("default", 1).kind() {
                            ValKind::Float(d) => *d,
                            ValKind::Int(d) => *d as f64,
                            _ => 0.0,
                        })),
                    }
                }
                _ => unreachable!(),
            },
            // ---- strings ---------------------------------------------------
            "upper" => {
                let s = a.str_pos(0)?;
                Ok(Val::new_lit_str(
                    s.bytes()
                        .map(|c| c.to_ascii_uppercase() as char)
                        .collect::<String>(),
                ))
            }
            "lower" => {
                let s = a.str_pos(0)?;
                Ok(Val::new_lit_str(
                    s.bytes()
                        .map(|c| c.to_ascii_lowercase() as char)
                        .collect::<String>(),
                ))
            }
            "strip" | "rstrip" | "lstrip" => {
                // string.strip(left, right, chars) (string.cpp:163-200)
                let s = a.str_pos(0)?;
                let chars = a.get_kwarg_or_pos("chars", 1);
                let chars = if chars.is_undefined() {
                    None
                } else {
                    Some(chars.as_string()?)
                };
                let left = name == "strip" || name == "lstrip";
                let right = name == "strip" || name == "rstrip";
                let matches = |c: u8| match &chars {
                    Some(set) => set.as_bytes().contains(&c),
                    None => super::c_isspace(c),
                };
                let b = s.as_bytes();
                let mut start = 0usize;
                let mut end = b.len();
                if left {
                    while start < end && matches(b[start]) {
                        start += 1;
                    }
                }
                if right {
                    while end > start && matches(b[end - 1]) {
                        end -= 1;
                    }
                }
                Ok(Val::new_lit_str(s[start..end].to_string()))
            }
            "title" => {
                let s = a.str_pos(0)?;
                let mut out = String::with_capacity(s.len());
                let mut cap_next = true;
                for c in s.bytes() {
                    if super::c_isspace(c) {
                        cap_next = true;
                        out.push(c as char);
                    } else if cap_next {
                        out.push(c.to_ascii_uppercase() as char);
                        cap_next = false;
                    } else {
                        out.push(c.to_ascii_lowercase() as char);
                    }
                }
                Ok(Val::new_lit_str(out))
            }
            "capitalize" => {
                let s = a.str_pos(0)?;
                let mut out = String::with_capacity(s.len());
                for (i, c) in s.bytes().enumerate() {
                    out.push(if i == 0 {
                        c.to_ascii_uppercase() as char
                    } else {
                        c.to_ascii_lowercase() as char
                    });
                }
                Ok(Val::new_lit_str(out))
            }
            "length" => match recv.kind() {
                // byte length for strings (string.h:56-62)
                ValKind::Str(s) => Ok(Val::int(s.len() as i64)),
                ValKind::List(items) => Ok(Val::int(items.borrow().len() as i64)),
                ValKind::Tuple(items) => Ok(Val::int(items.len() as i64)),
                ValKind::Object(fields) => Ok(Val::int(fields.borrow().len() as i64)),
                ValKind::Undefined => Ok(Val::int(0)),
                _ => unreachable!(),
            },
            "wordcount" => {
                let s = a.str_pos(0)?;
                Ok(Val::int(s.split_ascii_whitespace().count() as i64))
            }
            "startswith" => {
                let s = a.str_pos(0)?;
                let p = a.str_pos(1)?;
                Ok(Val::bool_(s.starts_with(&p)))
            }
            "endswith" => {
                let s = a.str_pos(0)?;
                let p = a.str_pos(1)?;
                Ok(Val::bool_(s.ends_with(&p)))
            }
            "split" | "rsplit" => {
                // value.cpp:667-721
                let s = a.str_pos(0)?;
                let delim = if a.count() > 1 {
                    a.str_pos(1)?
                } else {
                    " ".to_string()
                };
                if delim.is_empty() {
                    return Err("empty separator".to_string());
                }
                let maxsplit: i64 = if a.count() > 2 { a.int_pos(2)? } else { -1 };
                let mut parts: Vec<String> = Vec::new();
                let mut rest: &str = &s;
                let mut budget = maxsplit;
                if name == "split" {
                    while let Some(pos) = rest.find(&delim) {
                        if budget == 0 {
                            break;
                        }
                        parts.push(rest[..pos].to_string());
                        rest = &rest[pos + delim.len()..];
                        budget -= 1;
                    }
                    parts.push(rest.to_string());
                } else {
                    while let Some(pos) = rest.rfind(&delim) {
                        if budget == 0 {
                            break;
                        }
                        parts.push(rest[pos + delim.len()..].to_string());
                        rest = &rest[..pos];
                        budget -= 1;
                    }
                    parts.push(rest.to_string());
                    parts.reverse();
                }
                Ok(Val::list(parts.into_iter().map(Val::new_lit_str).collect()))
            }
            "replace" => {
                // value.cpp:722-752
                let s = a.str_pos(0)?;
                let old = a.str_pos(1)?;
                let new = a.str_pos(2)?;
                if a.count() > 4 {
                    return Err("replace: too many arguments".to_string());
                }
                if let ArgVal::Pos(v) = a.items.get(3).unwrap_or(&ArgVal::Pos(Val::undef())) {
                    if v.as_int_val().unwrap_or(-1) > 0 {
                        return Err(
                            "NotImplemented: String replace with count argument not implemented"
                                .to_string(),
                        );
                    }
                }
                if old == new {
                    return Ok(Val::new_lit_str(s));
                }
                let out = if old.is_empty() {
                    let mut r = new.clone();
                    for c in s.chars() {
                        r.push(c);
                        r.push_str(&new);
                    }
                    r
                } else {
                    s.replace(&old, &new)
                };
                Ok(Val::new_lit_str(out))
            }
            "format" => {
                // value.cpp:753-787 — only '{}' placeholders
                let fmt = a.str_pos(0)?;
                let mut out = String::new();
                let mut arg_idx = 1usize;
                let mut chars = fmt.chars().peekable();
                while let Some(c) = chars.next() {
                    if c != '{' {
                        out.push(c);
                        continue;
                    }
                    if chars.peek() != Some(&'}') {
                        return Err(
                            "NotImplemented: format() only supports simple '{}' placeholders"
                                .to_string(),
                        );
                    }
                    chars.next();
                    out.push_str(&a.get_pos(arg_idx)?.as_string()?);
                    arg_idx += 1;
                }
                Ok(Val::new_lit_str(out))
            }
            "indent" => {
                // value.cpp:875-915
                let input = a.str_pos(0)?;
                let width = a.get_kwarg_or_pos("width", 1);
                let first = a.get_kwarg_or_pos("first", 2).truthy()?;
                let blank = a.get_kwarg_or_pos("blank", 3).truthy()?;
                let indent = match width.kind() {
                    ValKind::Int(w) => " ".repeat((*w).max(0) as usize),
                    ValKind::Str(s) => s.clone(),
                    _ => "    ".to_string(),
                };
                // std::getline semantics: trailing '\n' does not yield a final
                // empty line
                let mut lines: Vec<&str> = input.split('\n').collect();
                if input.ends_with('\n') {
                    lines.pop();
                }
                let mut indented = String::new();
                for line in lines {
                    if !indented.is_empty() {
                        indented.push('\n');
                    }
                    if if indented.is_empty() {
                        first
                    } else {
                        !line.is_empty() || blank
                    } {
                        indented.push_str(&indent);
                    }
                    indented.push_str(line);
                }
                if !input.is_empty() && input.ends_with('\n') {
                    indented.push('\n');
                    if blank {
                        indented.push_str(&indent);
                    }
                }
                Ok(Val::new_lit_str(indented))
            }
            "string" => match recv.kind() {
                // int/float → tojson; bool → "True"/"False"; none → "None";
                // undefined → ""; containers → repr (deep-marks)
                ValKind::Int(_) | ValKind::Float(_) => Ok(Val::new_lit_str(tojson(recv))),
                ValKind::Bool(_) | ValKind::None => Ok(Val::new_lit_str(recv.as_string()?)),
                ValKind::Undefined => Ok(Val::new_lit_str(String::new())),
                ValKind::Str(_) => Ok(recv.clone()),
                _ => {
                    mark_deep_used(recv);
                    Ok(Val::new_lit_str(recv.as_string()?))
                }
            },
            "safe" => match recv.kind() {
                ValKind::Str(_) => Ok(recv.clone()),
                ValKind::Int(_) | ValKind::Float(_) => Ok(Val::new_lit_str(tojson(recv))),
                ValKind::Bool(_) | ValKind::None => Ok(Val::new_lit_str(recv.as_string()?)),
                ValKind::Undefined => Ok(Val::new_lit_str(String::new())),
                _ => Ok(recv.clone()),
            },
            "tojson" => {
                let ensure_ascii = a.get_kwarg_or_pos("ensure_ascii", 1).truthy()?;
                if a.get_kwarg_or_pos("sort_keys", 4).truthy()? {
                    return Err("NotImplemented: tojson sort_keys=true not implemented".to_string());
                }
                Ok(Val::new_lit_str(tojson_opts(recv, ensure_ascii)))
            }
            "join" => {
                // array join (value.cpp:1017-1050); string/object join is not
                // implemented in minja (value.cpp:593-595, 1199-1201)
                let items = array_items_clone(recv)
                    .ok_or("NotImplemented: join builtin not implemented for this type")?;
                let val_delim = a.get_kwarg_or_pos("d", 1);
                let attribute = a.get_kwarg_or_pos("attribute", 2);
                let undef = Val::undef();
                let delim = if val_delim.is_undefined() {
                    String::new()
                } else {
                    val_delim.as_string()?
                };
                let mut result = String::new();
                let n = items.len();
                for (i, item) in items.iter().enumerate() {
                    let mut val_arr = item.clone();
                    if !attribute.is_undefined() {
                        val_arr = get_attribute(item, &attribute, &undef);
                    }
                    if !matches!(
                        val_arr.kind(),
                        ValKind::Str(_) | ValKind::Int(_) | ValKind::Float(_)
                    ) {
                        return Err(
                            "join() can only join arrays of strings or numerics".to_string()
                        );
                    }
                    result.push_str(&val_arr.as_string()?);
                    if i + 1 < n {
                        result.push_str(&delim);
                    }
                }
                Ok(Val::new_lit_str(result))
            }
            // ---- arrays ----------------------------------------------------
            "list" => {
                let items = array_items_clone(recv).ok_or("list: not an array")?;
                Ok(Val::list(items))
            }
            "first" => {
                let items = array_items_clone(recv).ok_or("first: not an array")?;
                Ok(items.first().cloned().unwrap_or_else(Val::undef))
            }
            "last" => {
                let items = array_items_clone(recv).ok_or("last: not an array")?;
                Ok(items.last().cloned().unwrap_or_else(Val::undef))
            }
            "reverse" => {
                let mut items = array_items_clone(recv).ok_or("reverse: not an array")?;
                items.reverse();
                Ok(
                    if recv.is_array() && matches!(recv.kind(), ValKind::Tuple(_)) {
                        Val::tuple(items)
                    } else {
                        Val::list(items)
                    },
                )
            }
            "unique" => Err("NotImplemented: Array unique builtin not implemented".to_string()),
            "min" | "max" => {
                // value.cpp:1195-1247 — attribute support via get_attribute;
                // the loop re-compares the current result every iteration
                let items = array_items_clone(recv).ok_or("min/max: not an array")?;
                let attribute = a.get_kwarg_or_pos("attribute", 2);
                let undef = Val::undef();
                if items.is_empty() {
                    return Ok(undef);
                }
                let lt = name == "min";
                let mut result = items[0].clone();
                for item in &items {
                    let val_arr = if !attribute.is_undefined() {
                        get_attribute(item, &attribute, &undef)
                    } else {
                        item.clone()
                    };
                    let val_cmp = if !attribute.is_undefined() {
                        get_attribute(&result, &attribute, &undef)
                    } else {
                        result.clone()
                    };
                    if value_compare(&val_arr, &val_cmp, lt).unwrap_or(false) {
                        result = item.clone();
                    }
                }
                Ok(result)
            }
            "append" => {
                let ValKind::List(items) = recv.kind() else {
                    // tuples are immutable (value.h:371-377)
                    return Err("Attempting to modify immutable type".to_string());
                };
                let v = a.get_pos(1)?;
                items.borrow_mut().push(v);
                Ok(recv.clone())
            }
            "pop" => {
                let ValKind::List(items) = recv.kind() else {
                    return Err("Attempting to modify immutable type".to_string());
                };
                let mut index = if a.count() > 1 { a.int_pos(1)? } else { -1 };
                let mut b = items.borrow_mut();
                if index < 0 {
                    index += b.len() as i64;
                }
                if index < 0 || index as usize >= b.len() {
                    return Err(format!(
                        "Index {index} out of bounds for array of size {}",
                        b.len()
                    ));
                }
                Ok(b.remove(index as usize))
            }
            "sort" => {
                // value.cpp:1110-1142 — returns a sorted COPY; attribute keys
                // go through get_attribute (mismatches yield undefined, no
                // throw)
                let items = array_items_clone(recv).ok_or("sort: not an array")?;
                let reverse = a.get_kwarg_or_pos("reverse", 1).truthy()?;
                let attribute = a.get_kwarg_or_pos("attribute", 3);
                let undef = Val::undef();
                let key = |item: &Val| -> Val {
                    if attribute.is_undefined() {
                        return item.clone();
                    }
                    get_attribute(item, &attribute, &undef)
                };
                // stable sort by the minja comparator (std::sort is unstable,
                // but equal keys keep adjacent positions in practice)
                let mut idx: Vec<usize> = (0..items.len()).collect();
                idx.sort_by(|&x, &y| {
                    let (vx, vy) = (key(&items[x]), key(&items[y]));
                    match value_compare(&vx, &vy, reverse) {
                        Ok(true) => std::cmp::Ordering::Less,
                        Ok(false) => std::cmp::Ordering::Greater,
                        Err(_) => std::cmp::Ordering::Equal,
                    }
                });
                let sorted: Vec<Val> = idx.into_iter().map(|i| items[i].clone()).collect();
                Ok(if matches!(recv.kind(), ValKind::Tuple(_)) {
                    Val::tuple(sorted)
                } else {
                    Val::list(sorted)
                })
            }
            "selectattr" | "select" | "rejectattr" | "reject" => {
                selectattr_impl(recv, a, name.starts_with("reject"))
            }
            "map" => {
                // value.cpp:1061-1089 — attribute mapping only
                if a.count() < 2 {
                    return Err("map: not enough arguments".to_string());
                }
                let Some(ArgVal::Kwarg(k, _)) = a.items.get(1) else {
                    return Err("NotImplemented: map: filter-mapping not implemented".to_string());
                };
                if k != "attribute" {
                    return Err("map: unexpected keyword argument".to_string());
                }
                let attribute = a.get_kwarg_or_pos("attribute", 1);
                let default_val = a.get_kwarg("default").unwrap_or_else(Val::undef);
                let items = array_items_clone(recv).ok_or("map: not an array")?;
                let mut out = Vec::with_capacity(items.len());
                for item in &items {
                    out.push(get_attribute(item, &attribute, &default_val));
                }
                Ok(if matches!(recv.kind(), ValKind::Tuple(_)) {
                    Val::tuple(out)
                } else {
                    Val::list(out)
                })
            }
            // ---- objects ---------------------------------------------------
            "get" => {
                let key = a.str_pos(1)?;
                let default_val = if a.count() > 2 {
                    a.get_pos(2)?
                } else {
                    Val::null_()
                };
                let ValKind::Object(fields) = recv.kind() else {
                    return Err("get: first argument must be an object".to_string());
                };
                Ok(fields
                    .borrow()
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(default_val))
            }
            "keys" => {
                let ValKind::Object(fields) = recv.kind() else {
                    return Err("keys: not an object".to_string());
                };
                Ok(Val::list(
                    fields.borrow().iter().map(|(k, _)| Val::str_(k)).collect(),
                ))
            }
            "values" => {
                let ValKind::Object(fields) = recv.kind() else {
                    return Err("values: not an object".to_string());
                };
                Ok(Val::list(
                    fields.borrow().iter().map(|(_, v)| v.clone()).collect(),
                ))
            }
            "items" => {
                // object: (key, value) tuples; none/undefined: empty array
                match recv.kind() {
                    ValKind::Object(fields) => Ok(Val::list(
                        fields
                            .borrow()
                            .iter()
                            .map(|(k, v)| Val::tuple(vec![Val::str_(k), v.clone()]))
                            .collect(),
                    )),
                    ValKind::None | ValKind::Undefined => Ok(Val::list(Vec::new())),
                    _ => Err("items: not an object".to_string()),
                }
            }
            "dictsort" => {
                // value.cpp:1275-1293 — sorted object copy
                let ValKind::Object(fields) = recv.kind() else {
                    return Err("dictsort: not an object".to_string());
                };
                let by = a.get_kwarg_or_pos("by", 2);
                let reverse = a.get_kwarg_or_pos("reverse", 3).truthy()?;
                let by_value = matches!(by.as_str_val().as_deref(), Some("value"));
                let mut pairs: Vec<(String, Val)> = fields.borrow().clone();
                // std::sort with value_compare(reverse ? gt : lt) as the
                // strict-weak "less" (value.cpp:1285-1291)
                let cmp_lt = |x: &Val, y: &Val| value_compare(x, y, !reverse);
                pairs.sort_by(|(ka, va), (kb, vb)| {
                    let r = if by_value {
                        cmp_lt(va, vb)
                    } else {
                        cmp_lt(&Val::str_(ka), &Val::str_(kb))
                    };
                    match r {
                        Ok(true) => std::cmp::Ordering::Less,
                        Ok(false) => std::cmp::Ordering::Greater,
                        Err(_) => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Val::new(ValKind::Object(RefCell::new(pairs))))
            }
            // ---- slices (member_expression translation) ---------------------
            "slice" => {
                // value.cpp:838-868 / 984-1012 — the member-slice path always
                // passes all three indexes (input at pos 0)
                let (start, stop, step) = if a.count() >= 4 {
                    (a.int_pos(1)?, a.int_pos(2)?, a.int_pos(3)?)
                } else {
                    // minja's count<4 branches die on undefined args (as_int
                    // throws) — surface the same failure
                    return Err("slice: expected start, stop and step".to_string());
                };
                if step == 0 {
                    return Err("slice step cannot be zero".to_string());
                }
                match recv.kind() {
                    ValKind::Str(s) => {
                        let bytes = s.as_bytes();
                        let idxs = py_slice_indexes(bytes.len() as i64, start, stop, step);
                        let out: Vec<u8> = idxs.iter().map(|&i| bytes[i as usize]).collect();
                        Ok(Val::new_lit_str(String::from_utf8_lossy(&out).into_owned()))
                    }
                    ValKind::List(items) => {
                        let b = items.borrow();
                        let idxs = py_slice_indexes(b.len() as i64, start, stop, step);
                        Ok(Val::list(
                            idxs.iter().map(|&i| b[i as usize].clone()).collect(),
                        ))
                    }
                    ValKind::Tuple(items) => {
                        let idxs = py_slice_indexes(items.len() as i64, start, stop, step);
                        Ok(Val::tuple(
                            idxs.iter().map(|&i| items[i as usize].clone()).collect(),
                        ))
                    }
                    other => Err(format!("slice: cannot slice {other:?}")),
                }
            }
            // none's empty-array builtins
            "sum" => Ok(Val::int(0)),
            "truncate" => Ok(Val::new_lit_str(String::new())),
            other => Err(format!(
                "Unknown (built-in) filter '{other}' for type {}",
                val_type_name(recv)
            )),
        }
    }

    /// `get_attribute(val, attr, default)` (value.cpp:271-293) — the shared
    /// attribute access behind selectattr/join/map/sort/min/max. String
    /// attributes on arrays are coerced to an index when all-digits
    /// (std::stoll, overflow → undefined); type mismatches yield the default
    /// instead of throwing.
    fn get_attribute(val: &Val, attr: &Val, default_val: &Val) -> Val {
        if !attr.is_undefined() {
            if val.is_array() {
                let mut idx = attr.clone();
                if let ValKind::Str(s) = attr.kind() {
                    if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) {
                        match s.parse::<i64>() {
                            Ok(i) => idx = Val::int(i),
                            Err(_) => idx = Val::undef(),
                        }
                    }
                }
                return array_at(val, &idx, default_val);
            } else if val.is_object() {
                // port objects keep string keys; non-string attrs miss (C++
                // hashes the typed value — an int key never matches a
                // JSON-string key)
                if let ValKind::Str(key) = attr.kind() {
                    if let Some(v) = val.field(key) {
                        return v;
                    }
                }
                return default_val.clone();
            }
        }
        default_val.clone()
    }

    /// `value_array_t::at(const value &, value &)` (value.h:436-440): int or
    /// bool index only, negative wraps, anything else / out of range → default.
    fn array_at(arr: &Val, index: &Val, default_val: &Val) -> Val {
        let i = match index.kind() {
            ValKind::Int(_) | ValKind::Bool(_) => index.num_pair().unwrap().0,
            _ => return default_val.clone(),
        };
        let items = array_items_clone(arr).unwrap_or_default();
        let mut i = i;
        if i < 0 {
            i += items.len() as i64;
        }
        if i >= 0 && (i as usize) < items.len() {
            items[i as usize].clone()
        } else {
            default_val.clone()
        }
    }

    /// `selectattr`/`select`/`rejectattr`/`reject` (value.cpp:264-339)
    fn selectattr_impl(recv: &Val, a: &Args, reject: bool) -> Result<Val, String> {
        // count includes the piped input at pos 0
        let items = array_items_clone(recv).ok_or("selectattr: not an array")?;
        let mut out = Vec::new();
        match a.count() {
            2 => {
                // array | selectattr("active") — truthiness of item.<attr>
                let attribute = a.get_pos(1)?;
                let undef = Val::undef();
                for item in &items {
                    let attr_val = get_attribute(item, &attribute, &undef);
                    let mut selected = attr_val.truthy()?;
                    if reject {
                        selected = !selected;
                    }
                    if selected {
                        out.push(item.clone());
                    }
                }
            }
            3 => {
                // array | selectattr("equalto", "text") — test on the ITEM
                let test_name = a.str_pos(1)?;
                let test_val = a.get_pos(2)?;
                let full = format!("test_is_{test_name}");
                if !is_test(&full) {
                    return Err(format!("selectattr: unknown test '{test_name}'"));
                }
                for item in &items {
                    let mut argv = Args::default();
                    argv.push(ArgVal::Pos(item.clone()));
                    argv.push(ArgVal::Pos(test_val.clone()));
                    let mut selected = run_test(&full, &argv)?.truthy()?;
                    if reject {
                        selected = !selected;
                    }
                    if selected {
                        out.push(item.clone());
                    }
                }
            }
            n if n >= 4 => {
                // array | selectattr("status", "equalto", "active") — test the
                // attribute value
                let attribute = a.get_pos(1)?;
                let test_name = a.str_pos(2)?;
                let extra = a.get_pos(3)?;
                let full = format!("test_is_{test_name}");
                if !is_test(&full) {
                    return Err(format!("selectattr: unknown test '{test_name}'"));
                }
                let undef = Val::undef();
                for item in &items {
                    let attr_val = get_attribute(item, &attribute, &undef);
                    let mut argv = Args::default();
                    argv.push(ArgVal::Pos(attr_val));
                    argv.push(ArgVal::Pos(extra.clone()));
                    let mut selected = run_test(&full, &argv)?.truthy()?;
                    if reject {
                        selected = !selected;
                    }
                    if selected {
                        out.push(item.clone());
                    }
                }
            }
            _ => return Err("selectattr: invalid number of arguments".to_string()),
        }
        Ok(Val::list(out))
    }

    /// `std::stoi(str, nullptr, base)` subset: optional sign + digits in the
    /// base, leading whitespace skipped, stops at the first invalid char.
    fn stoi_like(s: &str, base: i64) -> Option<i64> {
        let base = if base == 0 { 10 } else { base };
        let s = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
        let (neg, rest) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let mut val: i64 = 0;
        let mut any = false;
        for c in rest.chars() {
            let d = c.to_digit(36)? as i64;
            if d >= base {
                break;
            }
            val = val.checked_mul(base)?.checked_add(d)?;
            any = true;
        }
        if !any {
            return None;
        }
        Some(if neg { -val } else { val })
    }

    /// `std::stod` subset: leading valid double, else None
    fn stod_like(s: &str) -> Option<f64> {
        let s = s.trim_start();
        let bytes = s.as_bytes();
        let mut end = 0usize;
        let mut seen_digit = false;
        let mut seen_dot = false;
        let mut seen_exp = false;
        let mut i = 0usize;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_digit() {
                seen_digit = true;
                i += 1;
                end = i;
            } else if c == b'.' && !seen_dot && !seen_exp {
                seen_dot = true;
                i += 1;
            } else if (c == b'e' || c == b'E') && seen_digit && !seen_exp {
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_digit() {
                    seen_exp = true;
                    i = j;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        if !seen_digit {
            return None;
        }
        s[..end.max(1)].parse::<f64>().ok()
    }

    fn now_epoch() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    // ---- strftime (UTC subset) ---------------------------------------------

    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

    /// Days since 1970-01-01 → (y, m, d) — Howard Hinnant's civil_from_days.
    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097; // [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
        let mp = (5 * doy + 2) / 153; // [0, 11]
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
        let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    fn is_leap(y: i64) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }

    /// Minimal `strftime` for `strftime_now` (value.cpp:370-381 renders with
    /// `std::localtime`; we render in UTC for determinism). Supported: %Y %y
    /// %m %d %H %M %S %B %b %A %a %j %p %I %e %%; unknown specifiers pass
    /// through (C strftime behavior).
    pub fn strftime_utc(epoch: i64, fmt: &str) -> String {
        let days = epoch.div_euclid(86400);
        let secs = epoch.rem_euclid(86400);
        let (y, m, d) = civil_from_days(days);
        let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
        let dow = (days + 3).rem_euclid(7) as usize; // 1970-01-01 = Thursday
        let mdays = [
            31,
            if is_leap(y) { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        let doy: i64 = mdays[..(m - 1) as usize].iter().sum::<i64>() + d as i64;

        let mut out = String::new();
        let mut it = fmt.chars();
        while let Some(c) = it.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match it.next() {
                Some('Y') => {
                    let _ = write!(out, "{:04}", y);
                }
                Some('y') => {
                    let _ = write!(out, "{:02}", y.rem_euclid(100));
                }
                Some('m') => {
                    let _ = write!(out, "{:02}", m);
                }
                Some('d') => {
                    let _ = write!(out, "{:02}", d);
                }
                Some('e') => {
                    let _ = write!(out, "{:2}", d);
                }
                Some('H') => {
                    let _ = write!(out, "{:02}", hh);
                }
                Some('M') => {
                    let _ = write!(out, "{:02}", mm);
                }
                Some('S') => {
                    let _ = write!(out, "{:02}", ss);
                }
                Some('I') => {
                    let h12 = if hh % 12 == 0 { 12 } else { hh % 12 };
                    let _ = write!(out, "{:02}", h12);
                }
                Some('p') => out.push_str(if hh < 12 { "AM" } else { "PM" }),
                Some('B') => out.push_str(MONTHS[(m - 1) as usize]),
                Some('b') => out.push_str(&MONTHS[(m - 1) as usize][..3]),
                Some('A') => out.push_str(DAYS[dow]),
                Some('a') => out.push_str(&DAYS[dow][..3]),
                Some('j') => {
                    let _ = write!(out, "{:03}", doy);
                }
                Some('%') => out.push('%'),
                Some(other) => {
                    out.push('%');
                    out.push(other);
                }
                None => out.push('%'),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// caps — `jinja::caps` / `jinja::caps_get` (common/jinja/caps.h + caps.cpp:111-571)
//
// Infers template capabilities by executing the template with synthetic
// inputs and inspecting which input values were read (`stats.used`) and how
// (`stats.ops`: test_is_string / selectattr / array_access). Ported against
// the mini-jinja engine above; `selectattr` itself is not evaluated (unknown
// filter ⇒ execution failure ⇒ the conservative default), a documented gap.
// ---------------------------------------------------------------------------

/// `struct jinja::caps` (caps.h:9-27)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JinjaCaps {
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub supports_system_role: bool,
    pub supports_parallel_tool_calls: bool,
    /// supports preserve reasoning trace in the full history
    pub supports_preserve_reasoning: bool,
    /// supports reasoning effort levels
    pub supports_reasoning_effort: bool,
    pub supports_string_content: bool,
    pub supports_typed_content: bool,
    pub supports_object_arguments: bool,
}

impl Default for JinjaCaps {
    fn default() -> Self {
        JinjaCaps {
            supports_tools: true,
            supports_tool_calls: true,
            supports_system_role: true,
            supports_parallel_tool_calls: true,
            supports_preserve_reasoning: false,
            supports_reasoning_effort: false,
            supports_string_content: true,
            supports_typed_content: false,
            supports_object_arguments: false,
        }
    }
}

impl JinjaCaps {
    /// `caps::to_map()` (caps.cpp:87-99)
    pub fn to_map(&self) -> Vec<(String, bool)> {
        vec![
            (
                "supports_string_content".to_string(),
                self.supports_string_content,
            ),
            (
                "supports_typed_content".to_string(),
                self.supports_typed_content,
            ),
            ("supports_tools".to_string(), self.supports_tools),
            ("supports_tool_calls".to_string(), self.supports_tool_calls),
            (
                "supports_parallel_tool_calls".to_string(),
                self.supports_parallel_tool_calls,
            ),
            (
                "supports_system_role".to_string(),
                self.supports_system_role,
            ),
            (
                "supports_preserve_reasoning".to_string(),
                self.supports_preserve_reasoning,
            ),
            (
                "supports_reasoning_effort".to_string(),
                self.supports_reasoning_effort,
            ),
            (
                "supports_object_arguments".to_string(),
                self.supports_object_arguments,
            ),
        ]
    }
}

mod caps_impl {
    use super::mini_jinja::{render_inputs, RenderInputs, Val, ValKind};

    /// item field of a list-of-objects value (stats introspection)
    fn msg_field(messages: &Val, idx: usize, name: &str) -> Val {
        let ValKind::List(items) = messages.kind() else {
            unreachable!()
        };
        items.borrow()[idx].field(name).unwrap()
    }
    fn tools_field(tools: &Val, idx: usize, path: &[&str]) -> Val {
        let ValKind::List(items) = tools.kind() else {
            unreachable!()
        };
        let mut v = items.borrow()[idx].clone();
        for p in path {
            v = v.field(p).unwrap();
        }
        v
    }

    /// `caps_try_execute` (caps.cpp:35-93, 462524043): render with synthetic
    /// inputs, ignore execution errors, then analyze the value stats. Some
    /// templates require a thinking field on every assistant turn (e.g. K2
    /// Horizon): a failure retries once with an empty `reasoning_content` on
    /// the assistant turns that lack one.
    fn try_execute(
        prog: &[super::mini_jinja::Node],
        mut inputs: RenderInputs,
    ) -> (bool, String, RenderInputs) {
        inputs.bos_token = String::new();
        inputs.eos_token = String::new();
        inputs.add_generation_prompt = true;
        for attempt in 0..2 {
            let (success, result) = match render_inputs(prog, &inputs) {
                Ok(s) => (true, s),
                Err(_) => (false, String::new()),
            };
            if success || attempt == 1 {
                return (success, result, inputs);
            }
            // retry once with an empty reasoning_content on the assistant
            // turns that lack one (only when one was actually added)
            let mut added = false;
            if let ValKind::List(items) = inputs.messages.kind() {
                for msg in items.borrow().iter() {
                    if let ValKind::Object(fields) = msg.kind() {
                        let is_assistant = fields
                            .borrow()
                            .iter()
                            .any(|(k, v)| {
                                k == "role" && v.as_str_val().as_deref() == Some("assistant")
                            });
                        let has_reasoning = fields.borrow().iter().any(|(k, _)| k == "reasoning_content");
                        if is_assistant && !has_reasoning {
                            fields
                                .borrow_mut()
                                .push(("reasoning_content".to_string(), json_str("")));
                            added = true;
                        }
                    }
                }
            }
            if !added {
                return (false, String::new(), inputs);
            }
        }
        unreachable!()
    }

    fn json_str(s: &str) -> Val {
        Val::str_(s)
    }

    /// `caps_get(prog)` (caps.cpp:111-571)
    pub fn caps_get(source: &str) -> Result<super::JinjaCaps, String> {
        let toks = super::mini_jinja::lex(source)?;
        let prog = super::mini_jinja::parse(&toks)?;
        let mut result = super::JinjaCaps::default();

        // ---- typed content support (caps.cpp:120-184) ----------------------
        const CONTENT_MARKER: &str = "STRING_MARKER";
        let mut inputs = RenderInputs::default();
        inputs.messages = Val::list(vec![Val::object(vec![
            ("role", json_str("user")),
            ("content", json_str(CONTENT_MARKER)),
        ])]);
        let messages = inputs.messages.clone();
        let (success, rendered, _) = try_execute(&prog, inputs);
        let content = msg_field(&messages, 0, "content");
        let mut checks_for_string = false;
        if content.stats().has_op("test_is_string") {
            checks_for_string = true;
        }
        let used_as_array =
            content.stats().has_op("selectattr") || content.stats().has_op("array_access");
        if used_as_array {
            result.supports_typed_content = true;
        }
        if !success {
            result.supports_string_content = false;
        } else if used_as_array && !rendered.contains(CONTENT_MARKER) {
            result.supports_string_content = false;
        }

        if checks_for_string {
            let mut inputs = RenderInputs::default();
            inputs.messages = Val::list(vec![Val::object(vec![
                ("role", json_str("user")),
                ("content", Val::list(Vec::new())),
            ])]);
            let messages = inputs.messages.clone();
            let (success, _, _) = try_execute(&prog, inputs);
            let content = msg_field(&messages, 0, "content");
            let used_as_array =
                content.stats().has_op("selectattr") || content.stats().has_op("array_access");
            if used_as_array && success {
                result.supports_typed_content = true;
            }
        }

        // ---- system prompt support (caps.cpp:188-213) ----------------------
        {
            let mut inputs = RenderInputs::default();
            inputs.messages = Val::list(vec![
                Val::object(vec![
                    ("role", json_str("system")),
                    ("content", json_str("System message")),
                ]),
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
            ]);
            let messages = inputs.messages.clone();
            let (_, _, _) = try_execute(&prog, inputs);
            let content = msg_field(&messages, 0, "content");
            if !content.stats().used.get() {
                result.supports_system_role = false;
            }
        }

        // ---- tools support: single call with object arguments (caps.cpp:217-307)
        let tools_json = || -> Val {
            Val::list(vec![Val::object(vec![
                ("name", json_str("tool")),
                ("type", json_str("function")),
                (
                    "function",
                    Val::object(vec![
                        ("name", json_str("tool1")),
                        ("description", json_str("Tool description")),
                        (
                            "parameters",
                            Val::object(vec![
                                ("type", json_str("object")),
                                (
                                    "properties",
                                    Val::object(vec![(
                                        "arg",
                                        Val::object(vec![
                                            ("type", json_str("string")),
                                            ("description", json_str("Arg description")),
                                        ]),
                                    )]),
                                ),
                                ("required", Val::list(vec![json_str("arg")])),
                            ]),
                        ),
                    ]),
                ),
            ])])
        };
        let messages_with_tool_calls = |arguments: Val| -> Val {
            Val::list(vec![
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("")),
                    (
                        "tool_calls",
                        Val::list(vec![Val::object(vec![
                            ("id", json_str("call00001")),
                            ("type", json_str("function")),
                            (
                                "function",
                                Val::object(vec![
                                    ("name", json_str("tool1")),
                                    ("arguments", arguments),
                                ]),
                            ),
                        ])]),
                    ),
                ]),
                Val::object(vec![
                    ("role", json_str("tool")),
                    ("content", json_str("Tool response")),
                    ("tool_call_id", json_str("call00001")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("The tool response was 'tool response'")),
                ]),
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
            ])
        };

        {
            let mut inputs = RenderInputs::default();
            inputs.messages =
                messages_with_tool_calls(Val::object(vec![("arg", json_str("value"))]));
            inputs.tools = tools_json();
            let messages = inputs.messages.clone();
            let tools = inputs.tools.clone();
            let (success, _, _) = try_execute(&prog, inputs);
            if success {
                let tool_name = tools_field(&tools, 0, &["function", "name"]);
                if !tool_name.stats().used.get() {
                    result.supports_tools = false;
                }
                let tool_calls = msg_field(&messages, 1, "tool_calls");
                if !tool_calls.stats().used.get() {
                    result.supports_tool_calls = false;
                } else {
                    let tool_arg = {
                        let ValKind::List(calls) = tool_calls.kind() else {
                            unreachable!()
                        };
                        let call = calls.borrow()[0].clone();
                        call.field("function")
                            .unwrap()
                            .field("arguments")
                            .unwrap()
                            .field("arg")
                            .unwrap()
                    };
                    if tool_arg.stats().used.get() {
                        result.supports_object_arguments = true;
                    }
                }
            }
        }

        if !result.supports_object_arguments {
            // ---- single tool with string arguments (caps.cpp:309-397) -----
            let mut inputs = RenderInputs::default();
            inputs.messages = messages_with_tool_calls(json_str(r#"{"arg": "value"}"#));
            inputs.tools = tools_json();
            let messages = inputs.messages.clone();
            let tools = inputs.tools.clone();
            let (success, _, _) = try_execute(&prog, inputs);
            if !success {
                result.supports_tool_calls = false;
                result.supports_tools = false;
            } else {
                let tool_name = tools_field(&tools, 0, &["function", "name"]);
                if !tool_name.stats().used.get() {
                    result.supports_tools = false;
                }
                let tool_calls = msg_field(&messages, 1, "tool_calls");
                if !tool_calls.stats().used.get() {
                    result.supports_tool_calls = false;
                }
            }
        }

        // ---- parallel tool support (caps.cpp:401-493) ----------------------
        {
            let args = if result.supports_object_arguments {
                Val::object(vec![("arg", json_str("value"))])
            } else {
                json_str(r#"{"arg": "value"}"#)
            };
            let call = |id: &str| {
                Val::object(vec![
                    ("id", json_str(id)),
                    ("type", json_str("function")),
                    (
                        "function",
                        Val::object(vec![
                            ("name", json_str("tool1")),
                            ("arguments", args.clone()),
                        ]),
                    ),
                ])
            };
            let mut inputs = RenderInputs::default();
            inputs.messages = Val::list(vec![
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("")),
                    (
                        "tool_calls",
                        Val::list(vec![call("call00001"), call("call00002")]),
                    ),
                ]),
                Val::object(vec![
                    ("role", json_str("tool")),
                    ("content", json_str("Tool response")),
                    ("tool_call_id", json_str("call00001")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("The tool response was 'tool response'")),
                ]),
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
            ]);
            inputs.tools = tools_json();
            let messages = inputs.messages.clone();
            let (success, _, _) = try_execute(&prog, inputs);
            if !success {
                result.supports_parallel_tool_calls = false;
            } else {
                let tool_calls = msg_field(&messages, 1, "tool_calls");
                let tool_call_1 = {
                    let ValKind::List(calls) = tool_calls.kind() else {
                        unreachable!()
                    };
                    calls.borrow()[1].field("function").unwrap()
                };
                if !tool_call_1.stats().used.get() {
                    result.supports_parallel_tool_calls = false;
                }
            }
        }

        // ---- preserve reasoning in history (caps.cpp:497-540) --------------
        const REASONING_PLACEHOLDER: &str = "<REASONING_CONTENT_PLACEHOLDER>";
        {
            let mut inputs = RenderInputs::default();
            inputs.messages = Val::list(vec![
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("Assistant message")),
                    ("reasoning_content", json_str(REASONING_PLACEHOLDER)),
                ]),
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
                Val::object(vec![
                    ("role", json_str("assistant")),
                    ("content", json_str("Assistant message")),
                    ("reasoning_content", json_str("Reasoning content")),
                ]),
                Val::object(vec![
                    ("role", json_str("user")),
                    ("content", json_str("User message")),
                ]),
            ]);
            // ctx_fn: enable_thinking=true + caps_apply_preserve_reasoning(true)
            inputs.enable_thinking = Some(true);
            inputs.extra = vec![
                ("preserve_thinking".to_string(), Val::bool_(true)),
                ("clear_thinking".to_string(), Val::bool_(false)),
                ("truncate_history_thinking".to_string(), Val::bool_(false)),
                ("drop_thinking".to_string(), Val::bool_(false)),
            ];
            let (_, output, _) = try_execute(&prog, inputs);
            if output.contains(REASONING_PLACEHOLDER) {
                result.supports_preserve_reasoning = true;
            }
        }

        // ---- reasoning effort level (caps.cpp:544-566) ----------------------
        {
            let mut inputs = RenderInputs::default();
            inputs.messages = Val::list(vec![Val::object(vec![
                ("role", json_str("user")),
                ("content", json_str("User message")),
            ])]);
            inputs.enable_thinking = Some(true);
            // caps_apply_reasoning_effort: one shared value for both names
            let effort = Val::str_("low");
            inputs.extra = vec![
                ("reasoning_effort".to_string(), effort.clone()),
                ("reasoning_strength".to_string(), effort.clone()),
            ];
            let _ = try_execute(&prog, inputs);
            if effort.stats().used.get() {
                result.supports_reasoning_effort = true;
            }
        }

        Ok(result)
    }
}

/// `jinja::caps_get` (caps.cpp:111)
pub fn caps_get(source: &str) -> Result<JinjaCaps, String> {
    caps_impl::caps_get(source)
}

// ---------------------------------------------------------------------------
// tests — builtin formatters verified against the pinned llama-chat.cpp
// (baseline bd4f514db1) plus a differential run against an extracted copy of
// `llm_chat_apply_template` / `llm_chat_detect_template`:
//   * apply:    55 templates x 9 message sets x add_ass(+/-) = 990 cases, 0 diffs
//   * detect:   93 template sources (incl. the real qwen2.5 GGUF template), 0 diffs
// The differential fixtures are reproduced by
//   crates/llama/tests/chat_fixture.rs (env CHAT_FIXTURE / CHAT_DETECT_FIXTURE).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn m(role: Role, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    // ---- Chatml (llama-chat.cpp:250-257) ----------------------------------
    //   for m: "<|im_start|>" role "\n" content "<|im_end|>\n"
    //   add_ass -> "<|im_start|>assistant\n"

    #[test]
    fn chatml_user_only() {
        let out = apply(ChatTemplate::Chatml, &[m(Role::User, "Hello")], true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_system_and_user() {
        let msgs = [m(Role::System, "You are helpful."), m(Role::User, "Hi")];
        let out = apply(ChatTemplate::Chatml, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_consecutive_assistant() {
        let msgs = [
            m(Role::User, "u"),
            m(Role::Assistant, "a1"),
            m(Role::Assistant, "a2"),
        ];
        let out = apply(ChatTemplate::Chatml, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nu<|im_end|>\n\
             <|im_start|>assistant\na1<|im_end|>\n\
             <|im_start|>assistant\na2<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn chatml_no_trim_and_no_gen_prompt() {
        // content is copied verbatim (no trim in the C++ branch)
        let out = apply(ChatTemplate::Chatml, &[m(Role::User, "  padded  ")], false).unwrap();
        assert_eq!(out, "<|im_start|>user\n  padded  <|im_end|>\n");
    }

    #[test]
    fn chatml_unknown_role_passthrough() {
        let out = apply(ChatTemplate::Chatml, &[m(Role::Tool, "t1")], false).unwrap();
        assert_eq!(out, "<|im_start|>tool\nt1<|im_end|>\n");
    }

    // ---- Llama 3 (llama-chat.cpp:485-493) ---------------------------------
    //   for m: "<|start_header_id|>" role "<|end_header_id|>\n\n" trim(content) "<|eot_id|>"
    //   add_ass -> "<|start_header_id|>assistant<|end_header_id|>\n\n"

    #[test]
    fn llama3_user_only() {
        let out = apply(ChatTemplate::Llama3, &[m(Role::User, "Hello")], true).unwrap();
        assert_eq!(
            out,
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama3_system_and_user() {
        let msgs = [m(Role::System, "You are helpful."), m(Role::User, "Hi")];
        let out = apply(ChatTemplate::Llama3, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama3_consecutive_assistant_and_trim() {
        let msgs = [
            m(Role::User, "  u  "),
            m(Role::Assistant, "a1"),
            m(Role::Assistant, "a2"),
        ];
        let out = apply(ChatTemplate::Llama3, &msgs, true).unwrap();
        // trim() (llama-chat.cpp:16-26) strips "  u  " -> "u"
        assert_eq!(
            out,
            "<|start_header_id|>user<|end_header_id|>\n\nu<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\na1<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\na2<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    // ---- Gemma (llama-chat.cpp:379-400) -----------------------------------
    //   system content is accumulated (trim) and prepended to the first
    //   non-"model" turn followed by "\n\n"; "assistant" -> "model";
    //   turn = "<start_of_turn>" role "\n" [system "\n\n"] trim(content) "<end_of_turn>\n"
    //   add_ass -> "<start_of_turn>model\n"

    #[test]
    fn gemma_user_only() {
        let out = apply(ChatTemplate::Gemma, &[m(Role::User, "Hello")], true).unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_system_merged_into_user() {
        let msgs = [m(Role::System, "You are helpful."), m(Role::User, "Hi")];
        let out = apply(ChatTemplate::Gemma, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\nYou are helpful.\n\nHi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_consecutive_assistant() {
        let msgs = [
            m(Role::User, "u"),
            m(Role::Assistant, "a1"),
            m(Role::Assistant, "a2"),
        ];
        let out = apply(ChatTemplate::Gemma, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\nu<end_of_turn>\n\
             <start_of_turn>model\na1<end_of_turn>\n\
             <start_of_turn>model\na2<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_system_trim_and_merge() {
        // two system messages concatenate (llama-chat.cpp:386 system_prompt += trim(...))
        // and content is trimmed (llama-chat.cpp:396)
        let msgs = [
            m(Role::System, "  s1  "),
            m(Role::System, " s2 "),
            m(Role::User, "  u  "),
        ];
        let out = apply(ChatTemplate::Gemma, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\ns1s2\n\nu<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    // ---- Phi-3 (llama-chat.cpp:336-344) -----------------------------------
    //   for m: "<|" role "|>\n" content "<|end|>\n"   (no trim)
    //   add_ass -> "<|assistant|>\n"

    #[test]
    fn phi3_user_only() {
        let out = apply(ChatTemplate::Phi3, &[m(Role::User, "Hello")], true).unwrap();
        assert_eq!(out, "<|user|>\nHello<|end|>\n<|assistant|>\n");
    }

    #[test]
    fn phi3_system_and_user() {
        let msgs = [m(Role::System, "You are helpful."), m(Role::User, "Hi")];
        let out = apply(ChatTemplate::Phi3, &msgs, true).unwrap();
        assert_eq!(
            out,
            "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
        );
    }

    #[test]
    fn phi3_consecutive_assistant_no_trim() {
        let msgs = [
            m(Role::User, "  u  "),
            m(Role::Assistant, "a1"),
            m(Role::Assistant, "a2"),
        ];
        let out = apply(ChatTemplate::Phi3, &msgs, true).unwrap();
        // Phi-3 does NOT trim (contrast llama3/gemma above)
        assert_eq!(
            out,
            "<|user|>\n  u  <|end|>\n\
             <|assistant|>\na1<|end|>\n\
             <|assistant|>\na2<|end|>\n\
             <|assistant|>\n"
        );
    }

    // ---- detect: real qwen2.5-0.5b-instruct GGUF ---------------------------
    // `tokenizer.chat_template` read from
    //   qwen2.5-0.5b-instruct-q4_k_m.gguf (2509 bytes, verbatim below)
    // Reference `llm_chat_detect_template` returns LLM_CHAT_TEMPLATE_CHATML (0).

    const QWEN25_TEMPLATE: &str = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen25_gguf_template_detects_as_chatml() {
        assert_eq!(QWEN25_TEMPLATE.len(), 2509);
        assert_eq!(detect(QWEN25_TEMPLATE), ChatTemplate::Chatml);
        // builtin Chatml is what the non-jinja path formats the conversation with
        let out = apply(detect(QWEN25_TEMPLATE), &[m(Role::User, "Hello")], true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// `apply_str` (mini-jinja) on the real qwen2.5 template: the tool preamble
    /// branch is untaken and the taken branches stay inside the supported
    /// subset. Expected value == the pinned `common/jinja` engine output
    /// (verified with a standalone build of common/jinja/*.cpp).
    #[test]
    fn qwen25_template_apply_str() {
        let msgs = [m(Role::User, "Hi")];
        let ctx = ChatTemplateCtx::new(&msgs, true, "", "");
        let out = apply_str(QWEN25_TEMPLATE, &ctx).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. \
             You are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen25_template_apply_str_system() {
        let msgs = [m(Role::System, "Be terse."), m(Role::User, "Hi")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let out = apply_str(QWEN25_TEMPLATE, &ctx).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n"
        );
    }

    // ---- name table / detection API ---------------------------------------

    #[test]
    fn builtin_template_names_sorted_and_complete() {
        let names = builtin_templates();
        assert_eq!(names.len(), 54);
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "std::map iteration order = lexical");
        // Dots1 is detectable but has no name (C++ map has no entry)
        assert_eq!(ChatTemplate::Dots1.name(), None);
        assert_eq!(ChatTemplate::Unknown.name(), None);
    }

    #[test]
    fn template_name_roundtrip() {
        for name in builtin_templates() {
            let t = ChatTemplate::from_name(name).unwrap();
            assert_eq!(t.name(), Some(name));
        }
        assert_eq!(ChatTemplate::from_name("not-a-template"), None);
        assert_eq!(
            ChatTemplate::from_name("chatml"),
            Some(ChatTemplate::Chatml)
        );
    }

    #[test]
    fn detect_exact_name_beats_substring() {
        // llama-chat.cpp:90-94 — exact name lookup wins
        assert_eq!(detect("llama2"), ChatTemplate::Llama2);
        assert_eq!(detect("llama2-sys"), ChatTemplate::Llama2Sys);
        // substring sniffing
        assert_eq!(detect("<|im_start|>"), ChatTemplate::Chatml);
        assert_eq!(
            detect("<|start_header_id|>x<|end_header_id|>"),
            ChatTemplate::Llama3
        );
        assert_eq!(detect("<start_of_turn>"), ChatTemplate::Gemma);
        assert_eq!(detect("<|assistant|><|end|>"), ChatTemplate::Phi3);
        assert_eq!(detect("random text"), ChatTemplate::Unknown);
    }

    #[test]
    fn apply_unknown_template_errors() {
        let e = apply(ChatTemplate::Unknown, &[m(Role::User, "hi")], true);
        assert!(e.is_err(), "llama-chat.cpp:940-942 returns -1 for unknown");
    }

    #[test]
    fn apply_named_defaults_to_chatml() {
        // llama.cpp:511-536 — tmpl == NULL -> "chatml"
        let out = apply_named(None, &[m(Role::User, "Hi")], true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let out = apply_named(Some("chatml"), &[m(Role::User, "Hi")], true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // ---- eot_prefix -------------------------------------------------------

    #[test]
    fn eot_prefix_matches_formatter_markers() {
        // each value is the marker the builtin formatter emits (line refs in
        // the eot_prefix doc comment)
        assert_eq!(eot_prefix(ChatTemplate::Chatml), "<|im_end|>");
        assert_eq!(eot_prefix(ChatTemplate::Llama3), "<|eot_id|>");
        assert_eq!(eot_prefix(ChatTemplate::Gemma), "<end_of_turn>");
        assert_eq!(eot_prefix(ChatTemplate::Phi3), "<|end|>");
        assert_eq!(eot_prefix(ChatTemplate::Llama2), "</s>");
        assert_eq!(eot_prefix(ChatTemplate::CommandR), "<|END_OF_TURN_TOKEN|>");
        assert_eq!(eot_prefix(ChatTemplate::Unknown), "");
    }

    #[test]
    fn eot_prefix_is_emitted_by_the_formatter() {
        // sanity: the returned marker really appears in a formatted assistant turn
        for (t, msgs) in [
            (ChatTemplate::Chatml, &[m(Role::Assistant, "a")][..]),
            (ChatTemplate::Llama3, &[m(Role::Assistant, "a")][..]),
            (ChatTemplate::Gemma, &[m(Role::Assistant, "a")][..]),
            (ChatTemplate::Phi3, &[m(Role::Assistant, "a")][..]),
        ] {
            let out = apply(t, msgs, false).unwrap();
            assert!(
                out.contains(eot_prefix(t)),
                "{t:?}: {out:?} lacks {:?}",
                eot_prefix(t)
            );
        }
    }

    // ---- strftime_now (%d %b %Y style date placeholders) ------------------

    #[test]
    fn strftime_utc_known_epoch() {
        // oracle: python3 -c "import time; print(time.strftime(F, time.gmtime(1727000000)))"
        // 1727000000 = 2024-09-22 10:13:20 UTC (a Sunday)
        let t = 1_727_000_000i64;
        assert_eq!(mini_jinja::strftime_utc(t, "%Y-%m-%d"), "2024-09-22");
        assert_eq!(mini_jinja::strftime_utc(t, "%d %b %Y"), "22 Sep 2024");
        assert_eq!(
            mini_jinja::strftime_utc(t, "%A, %B %d"),
            "Sunday, September 22"
        );
        assert_eq!(mini_jinja::strftime_utc(t, "%H:%M:%S"), "10:13:20");
        assert_eq!(mini_jinja::strftime_utc(t, "%I:%M %p"), "10:13 AM");
        assert_eq!(mini_jinja::strftime_utc(t, "%a %e"), "Sun 22");
        assert_eq!(mini_jinja::strftime_utc(t, "%j"), "266");
        assert_eq!(mini_jinja::strftime_utc(t, "%y"), "24");
        assert_eq!(
            mini_jinja::strftime_utc(t, "%d %B %Y %H:%M:%S"),
            "22 September 2024 10:13:20"
        );
        assert_eq!(mini_jinja::strftime_utc(t, "%%"), "%");
        // C strftime: unknown specifiers are copied through
        assert_eq!(mini_jinja::strftime_utc(t, "%Q"), "%Q");
    }

    #[test]
    fn strftime_utc_edges() {
        // 1970-01-01 00:00:00 UTC = Thursday, day-of-year 001
        assert_eq!(
            mini_jinja::strftime_utc(0, "%A %Y-%m-%d %j %H"),
            "Thursday 1970-01-01 001 00"
        );
        // 12:00 -> 12 PM, 00:00 -> 12 AM
        assert_eq!(mini_jinja::strftime_utc(43_200, "%I %p"), "12 PM");
        assert_eq!(mini_jinja::strftime_utc(0, "%I %p"), "12 AM");
        // leap day 2024-02-29 (+08:00 wall clock is not relevant; UTC)
        assert_eq!(
            mini_jinja::strftime_utc(1_709_208_000, "%Y-%m-%d %j"),
            "2024-02-29 060"
        );
        // pre-epoch
        assert_eq!(
            mini_jinja::strftime_utc(-1, "%Y-%m-%d %H:%M:%S"),
            "1969-12-31 23:59:59"
        );
    }

    /// llama3.1-style templates embed `strftime_now('%d %b %Y')`; with a pinned
    /// clock the render is deterministic (ChatTemplateCtx::now).
    #[test]
    fn strftime_now_in_template() {
        let tmpl = "Knowledge: {{ strftime_now('%d %b %Y') }}<|eot_id|>";
        let msgs: [ChatMessage; 0] = [];
        let mut ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        ctx.now = Some(1_727_000_000);
        assert_eq!(
            apply_str(tmpl, &ctx).unwrap(),
            "Knowledge: 22 Sep 2024<|eot_id|>"
        );
    }

    // ---- mini-jinja whitespace control ------------------------------------

    #[test]
    fn mini_jinja_trims_trailing_newline() {
        // jinja: keep_trailing_newline=false (common/jinja/lexer.cpp:54-57) —
        // one trailing template newline is stripped
        let msgs = [m(Role::User, "Hi")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        assert_eq!(
            apply_str("{{ messages[0].content }}\n", &ctx).unwrap(),
            "Hi"
        );
        assert_eq!(
            apply_str("{{ messages[0].content }}\n\n", &ctx).unwrap(),
            "Hi\n"
        );
    }

    #[test]
    fn mini_jinja_for_loop_and_whitespace_markers() {
        // lstrip_blocks + trim_blocks semantics: block tags on their own lines
        // contribute no whitespace
        let msgs = [m(Role::User, "a"), m(Role::Assistant, "b")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = "{% for x in messages %}\n  {{- x.role }}={{ x.content }}\n{% endfor %}";
        // The body's trailing "\n" is generated CONTENT, so the
        // keep_trailing_newline strip (which only touches the template's own
        // final characters) does not remove it — same as reference lexer.cpp
        // (lstrip_blocks keeps the newline before {% endfor %}).
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "user=a\nassistant=b\n");
    }

    // ---- mini-jinja engine: statements (parser.cpp / runtime.cpp) ----------

    /// `{% macro %}` with default args, kwargs, recursion and caller-scope
    /// closure (runtime.cpp:697-764). NOTE: macros render to *strings* —
    /// minja has no value-returning macros, so `+` on results concatenates
    /// (the gpt-oss/functionary32 templates only ever string-build with them)
    #[test]
    fn mini_jinja_macro_kwargs_and_recursion() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- macro wrap(inner, depth=0) -%}",
            "{%- if depth > 0 -%}<w>{{ wrap(inner, depth - 1) }}</w>",
            "{%- else -%}{{ inner }}",
            "{%- endif -%}",
            "{%- endmacro -%}",
            "{%- macro greet(name, punct='!') -%}hi {{ name }}{{ punct }}{%- endmacro -%}",
            "{{ greet('a') }}|{{ greet('b', punct='?') }}|{{ wrap('x', 2) }}"
        );
        assert_eq!(
            apply_str(tmpl, &ctx).unwrap(),
            "hi a!|hi b?|<w><w>x</w></w>"
        );
    }

    /// `namespace()` + `{% set ns.attr = v %}` — the shared-state workaround
    /// for minja's loop scoping (runtime.cpp:672-689, value.cpp:358-369)
    #[test]
    fn mini_jinja_namespace_mutation() {
        let msgs = [m(Role::User, "a"), m(Role::User, "b"), m(Role::User, "c")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- set ns = namespace(count=0) -%}",
            "{%- for x in messages -%}",
            "{%- set ns.count = ns.count + 1 -%}",
            "{%- endfor -%}",
            "{{ ns.count }}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "3");
    }

    /// block `{% set x %}…{% endset %}` binds the body's rendered string
    /// (runtime.cpp:644-646)
    #[test]
    fn mini_jinja_block_set() {
        let msgs = [m(Role::User, "U")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- set greeting %}",
            "Hello {{ messages[0].content }}!",
            "{%- endset -%}",
            "[{{ greeting }}]"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "[Hello U!]");
    }

    /// `{% set %}` inside a loop body persists across that loop's iterations
    /// but does not escape it (runtime.cpp:484-490 — one scope per loop)
    #[test]
    fn mini_jinja_loop_set_scoping() {
        let msgs = [m(Role::User, "a"), m(Role::User, "b")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- set total = 0 -%}",
            "{%- for x in messages -%}",
            "{%- set total = total + loop.index -%}{{ total }}",
            "{%- endfor -%}",
            "|after={{ total }}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "13|after=0");
    }

    /// `{% break %}` / `{% continue %}` (runtime.h:228-254) and the inline
    /// for-filter `{% for x in xs if test %}` (parser.cpp:346-348)
    #[test]
    fn mini_jinja_break_continue_and_for_filter() {
        let msgs = [
            m(Role::User, "a"),
            m(Role::User, "b"),
            m(Role::Assistant, "c"),
        ];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- for x in messages if x.role == 'user' -%}",
            "{{ x.content }}",
            "{%- endfor -%}",
            "|",
            "{%- for x in messages -%}",
            "{%- if x.content == 'b' -%}{%- continue -%}{%- endif -%}",
            "{%- if x.role == 'assistant' -%}{%- break -%}{%- endif -%}",
            "{{ x.content }}",
            "{%- endfor -%}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "ab|a");
    }

    /// `loop.previtem`/`nextitem` (runtime.cpp:613-614)
    #[test]
    fn mini_jinja_loop_previtem_nextitem() {
        let msgs = [m(Role::User, "a"), m(Role::User, "b")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- for x in messages -%}",
            "{{ 'p' if loop.previtem else '-' }}{{ 'n' if loop.nextitem else '-' }}",
            "{%- endfor -%}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "-np-");
    }

    // ---- mini-jinja engine: expressions (parser.cpp / runtime.cpp) ----------

    /// Python slices `[a:b:c]` incl. `[::-1]` and negative clamping
    /// (value.cpp:72-117)
    #[test]
    fn mini_jinja_slices() {
        let msgs = [
            m(Role::User, "a"),
            m(Role::User, "b"),
            m(Role::User, "c"),
            m(Role::User, "d"),
        ];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{{ messages[1:] | map(attribute='content') | join(',') }}|",
            "{{ messages[:-1] | map(attribute='content') | join(',') }}|",
            "{{ messages[::-1] | map(attribute='content') | join(',') }}|",
            "{{ messages[1:3] | map(attribute='content') | join(',') }}|",
            "{{ messages[-2:] | map(attribute='content') | join(',') }}"
        );
        assert_eq!(
            apply_str(tmpl, &ctx).unwrap(),
            "b,c,d|a,b,c|d,c,b,a|b,c|c,d"
        );
    }

    /// string subscripts are byte indexes; negative indexes do NOT wrap for
    /// strings (runtime.cpp:902-907)
    #[test]
    fn mini_jinja_string_subscript_and_methods() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{{ 'hello'[1] }}|",
            "{{ 'a-b-c'.split('-') | join('.') }}|",
            "{{ '  x  '.strip() }}|",
            "{{ 'aXbXc'.replace('X', '-') }}|",
            "{{ 'hello'.startswith('he') }}|",
            "{{ 'hello'.endswith('lo') }}|",
            "{{ 'ab' * 3 }}|",
            "{%- set d = {'k': 'v', 'n': 1} -%}",
            "{{ d.get('k') }}|{{ d.get('missing', 'dflt') }}|{{ d['n'] }}|{{ 'k' in d }}"
        );
        assert_eq!(
            apply_str(tmpl, &ctx).unwrap(),
            "e|a.b.c|x|a-b-c|True|True|ababab|v|dflt|1|True"
        );
    }

    /// ternary expressions (parser.cpp:332-351), value-returning and/or
    /// (runtime.cpp:116-121), `in` membership and conditional filters
    #[test]
    fn mini_jinja_ternary_in_and_or() {
        let msgs = [m(Role::User, "a")];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{{ 'yes' if messages[0].role == 'user' else 'no' }}|",
            "{{ (messages[0].content or 'empty') }}|",
            "{{ (none or 'fallback') }}|",
            "{{ 'a' in 'abc' }}|",
            "{{ 'z' not in 'abc' }}|",
            "{{ 1 if true else 2 }}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "yes|a|fallback|True|True|1");
    }

    /// `selectattr` in its 2/3/4-argument forms (value.cpp:264-339). The
    /// 3-arg form's FIRST argument is the test name — `selectattr("type",
    /// "defined")` asks for test `type`, which does not exist and errors in
    /// minja too (functionary32 only reaches it on oneOf branches).
    #[test]
    fn mini_jinja_selectattr() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- set xs = [",
            "{'type': 'code_interpreter', 'name': 'ci'},",
            "{'type': 'function', 'name': 'f'}] -%}",
            "{{ xs | selectattr('type', 'equalto', 'code_interpreter') | map(attribute='name') | join(',') }}|",
            "{%- set ys = [{'a': true}, {'a': false}] -%}",
            "{{ ys | selectattr('a') | length }}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "ci|1");
        assert!(apply_str("{{ [{}] | selectattr('type', 'defined') | list }}", &ctx).is_err());
    }

    /// non-call test statements with an argument — `x is divisibleby 3`
    /// (parser.cpp:432-439, #29443)
    #[test]
    fn mini_jinja_noncall_test_arg() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        assert_eq!(
            apply_str("{% if 6 is divisibleby 3 %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "y"
        );
        assert_eq!(
            apply_str("{% if 7 is divisibleby 3 %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "n"
        );
        assert_eq!(
            apply_str("{% if 7 is not divisibleby 3 %}y{% endif %}", &ctx).unwrap(),
            "y"
        );
        // `and`/`or`/`else` terminate the test — `x is defined and y is defined`
        assert_eq!(
            apply_str("{% if a is defined and b is defined %}y{% else %}n{% endif %}", &ctx)
                .unwrap(),
            "n"
        );
    }

    /// `sameas` (value.cpp:570-592, #29448): none/none, bools by value, ints
    /// only within CPython's small-int cache
    #[test]
    fn mini_jinja_sameas() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        assert_eq!(
            apply_str("{% if none is sameas none %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "y"
        );
        assert_eq!(
            apply_str("{% if true is sameas true %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "y"
        );
        assert_eq!(
            apply_str("{% if true is sameas false %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "n"
        );
        assert_eq!(
            apply_str("{% if 5 is sameas 5 %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "y"
        );
        // outside the [-5, 256] small-int cache, identity is pointer-based
        assert_eq!(
            apply_str("{% if 300 is sameas 300 %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "n"
        );
        // distinct literals of container type are never identical
        assert_eq!(
            apply_str("{% if 'a' is sameas 'a' %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "n"
        );
        // undefined is never sameas anything
        assert_eq!(
            apply_str("{% if a is sameas none %}y{% else %}n{% endif %}", &ctx).unwrap(),
            "n"
        );
    }

    /// `dict()` / `namespace()` share `toobject` (value.cpp:376-408, #29477):
    /// kwargs, an iterable of 2-tuples, or an object
    #[test]
    fn mini_jinja_dict_builtin() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        assert_eq!(
            apply_str("{% set d = dict(a=1, b=2) %}{{ d.a }}{{ d.b }}", &ctx).unwrap(),
            "12"
        );
        assert_eq!(
            apply_str(
                "{% set d = dict([('a', 1), ('b', 2)]) %}{{ d.a }}{{ d.b }}",
                &ctx
            )
            .unwrap(),
            "12"
        );
        assert_eq!(
            apply_str(
                "{% set s = {'x': 9} %}{% set d = dict(s, y=8) %}{{ d.x }}{{ d.y }}",
                &ctx
            )
            .unwrap(),
            "98"
        );
        // a later kwarg replaces an earlier key in place
        assert_eq!(
            apply_str("{% set d = dict(a=1, a=2) %}{{ d.a }}", &ctx).unwrap(),
            "2"
        );
        // a non-tuple iterable entry throws
        assert!(apply_str("{% set d = dict([1, 2]) %}", &ctx).is_err());
        // a positional arg without an iterable head throws
        assert!(apply_str("{% set d = dict(5) %}", &ctx).is_err());
        // namespace() keeps the kwargs-only contract through the shared impl
        assert_eq!(
            apply_str("{% set ns = namespace(c=0) %}{{ ns.c }}", &ctx).unwrap(),
            "0"
        );
    }

    /// coerced array attributes (value.cpp:271-293 + value.h:436-440, #29574):
    /// selectattr/join/map/min/max accept all-digit string attributes on
    /// arrays, and bool indexes are ints
    #[test]
    fn mini_jinja_coerced_array_attributes() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        // string attribute "1" coerces to index 1
        assert_eq!(
            apply_str("{{ [[1, 2], [3, 4]] | join(',', attribute='1') }}", &ctx).unwrap(),
            "2,4"
        );
        assert_eq!(
            apply_str("{{ [[1, 2], [3, 4]] | map(attribute='0') | join('+') }}", &ctx).unwrap(),
            "1+3"
        );
        // int attribute keeps working
        assert_eq!(
            apply_str("{{ [[5, 6]] | join(',', attribute=1) }}", &ctx).unwrap(),
            "6"
        );
        // a non-digit string attribute on arrays yields the default
        assert_eq!(
            apply_str("{{ [[1, 2]] | map(attribute='x', default='?') | join(',') }}", &ctx)
                .unwrap(),
            "?"
        );
        // min/max with an attribute key (upstream tests chain with tojson;
        // the set-capture keeps the port's printer out of the picture)
        assert_eq!(
            apply_str(
                "{% set m = [{'v': 3}, {'v': 1}, {'v': 2}] | min(attribute='v') %}{{ m.v }}",
                &ctx
            )
            .unwrap(),
            "1"
        );
        assert_eq!(
            apply_str(
                "{% set m = [{'v': 3}, {'v': 1}, {'v': 2}] | max(attribute='v') %}{{ m.v }}",
                &ctx
            )
            .unwrap(),
            "3"
        );
        // bool index on an array member access
        assert_eq!("{{ [10, 20][true] }}", "{{ [10, 20][true] }}");
        assert_eq!(apply_str("{{ [10, 20][true] }}", &ctx).unwrap(), "20");
        // a non-object item in selectattr is now skipped via get_attribute
        // instead of throwing
        assert_eq!(
            apply_str("{{ [1, {'a': true}] | selectattr('a') | length }}", &ctx).unwrap(),
            "1"
        );
    }

    /// `range()` with negative step (value.cpp:382-419) + dict iteration
    /// (`for k, v in obj.items()` / bare `for k in obj`, runtime.cpp:513-523)
    #[test]
    fn mini_jinja_range_and_dict_items() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        let tmpl = concat!(
            "{%- for i in range(3, 0, -1) -%}{{ i }}{%- endfor -%}|",
            "{%- set d = {'x': 1, 'y': 2} -%}",
            "{%- for k in d -%}{{ k }}{%- endfor -%}|",
            "{%- for k, v in d.items() -%}{{ k }}={{ v }};{%- endfor -%}|",
            "{%- for k, v in d -%}{{ k }}:{{ v }},{%- endfor -%}"
        );
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "321|xy|x=1;y=2;|x:1,y:2,");
    }

    /// `{% generation %}` markers are ignored; content renders in place
    /// (parser.cpp:207-212)
    #[test]
    fn mini_jinja_generation_markers_ignored() {
        let msgs = [m(Role::User, "a")];
        let ctx = ChatTemplateCtx::new(&msgs, true, "", "");
        let tmpl = "{% generation %}{{ messages[0].content }}{% endgeneration %}";
        assert_eq!(apply_str(tmpl, &ctx).unwrap(), "a");
    }

    /// unsupported statements remain loud parse errors (loud-fail policy)
    #[test]
    fn mini_jinja_unsupported_constructs_error() {
        let msgs: [ChatMessage; 0] = [];
        let ctx = ChatTemplateCtx::new(&msgs, false, "", "");
        for tmpl in [
            "{% call x() %}{% endcall %}",
            "{% filter upper %}a{% endfilter %}",
            "{% raw %}a{% endraw %}",
            "{% include 'x' %}",
        ] {
            assert!(apply_str(tmpl, &ctx).is_err(), "{tmpl} should not parse");
        }
    }
}
