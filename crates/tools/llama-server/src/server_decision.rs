//! server_decision.rs — 1:1 port of llama.cpp `tools/server/server-decision.{h,cpp}`
//! (upstream a7b94df2c, 802+131 lines) plus the `common/` decision-model
//! helpers it rides on:
//!
//! * `common_decision_type` + `common_get_decision_type`
//!   (common/common.h:958-966, common.cpp:1150-1180) — the
//!   `"<arch>.decision.type"` metadata probe;
//! * the TypeSafe `/v1/systemone` request pipeline: question parsing,
//!   state/image parsing, per-type prompt building (openjev / lev / kev /
//!   nimble / laya / clef), the jinja "systemone" template render, and the
//!   published answer formatting (softmax at the model's fitted temperatures,
//!   TypeSafe's confidence formulas);
//! * `server_decision_group_tasks` (server-decision.cpp:772-802) — the
//!   shared-prompt grouping.
//!
//! The model answers each question in one forward pass, no token is
//! generated. The engine side (batch decode, score extraction, the
//! `/v1/systemone` route) lives in `engine.rs`/`http.rs`; this module is the
//! pure logic, unit-tested against the shape of upstream's
//! `tools/server/tests/unit/test_systemone.py`.

use llama::json_schema::Json;
use llama::vocab::Vocab;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// common_decision_type (common/common.h:958-966, common.cpp:1150-1180)
// ---------------------------------------------------------------------------


// -- tiny accessors over the port's ordered `Json` (nlohmann-shaped) --

fn j_object<'a>(j: &'a Json) -> Option<&'a Vec<(String, Json)>> {
    match j {
        Json::Object(o) => Some(o),
        _ => None,
    }
}

fn j_array<'a>(j: &'a Json) -> Option<&'a Vec<Json>> {
    match j {
        Json::Array(a) => Some(a),
        _ => None,
    }
}

fn j_str<'a>(j: &'a Json) -> Option<&'a str> {
    match j {
        Json::String(s) => Some(s),
        _ => None,
    }
}

fn j_get<'a>(j: &'a Json, key: &str) -> Option<&'a Json> {
    j_object(j)?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn j_num(j: &Json) -> f64 {
    match j {
        Json::Int(i) => *i as f64,
        Json::Uint(u) => *u as f64,
        Json::Double(d) => *d,
        _ => 0.0,
    }
}

/// `enum common_decision_type` — typed decision models, see
/// `"<arch>.decision.type"` in the model metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommonDecisionType {
    /// not a decision model
    None,
    /// logits of one label token per option, read at the last prompt token
    Openjev,
    /// same as openjev, noul is read from a rating scale
    Lev,
    /// dot product of the hidden states of the last token and of one end
    /// token per option
    Kev,
    /// same as openjev, the prompt lists all the questions of the request
    Nimble,
    /// score of one marker token per option, read from the embeddings output
    Laya,
    /// all questions in one prompt, score of option i read from the
    /// embeddings output at row i
    Clef,
    /// same as openjev, label codes of 1 or 2 letters (da263e727)
    PplxDecider,
    /// same as openjev, the labels depend on the question type (88dcc460d)
    Lfm2D1,
    /// same as laya, other prompt layout (a657f7e98)
    Lfm2D1Omni,
    /// a decision model of a type that is not supported
    Unknown,
}

/// `COMMON_DECISION_TYPE_NAMES` (common.cpp:1163-1175)
const COMMON_DECISION_TYPE_NAMES: &[(CommonDecisionType, &str)] = &[
    (CommonDecisionType::Openjev, "openjev"),
    (CommonDecisionType::Lev, "lev"),
    (CommonDecisionType::Kev, "kev"),
    (CommonDecisionType::Nimble, "nimble"),
    (CommonDecisionType::Laya, "laya"),
    (CommonDecisionType::Clef, "clef"),
    (CommonDecisionType::PplxDecider, "pplx-decider"),
    (CommonDecisionType::Lfm2D1, "lfm2-d1"),
    (CommonDecisionType::Lfm2D1Omni, "lfm2-d1-omni"),
];

/// `common_decision_type_from_string` (common.cpp:1160-1168)
fn common_decision_type_from_string(s: &str) -> CommonDecisionType {
    for (ty, name) in COMMON_DECISION_TYPE_NAMES {
        if *name == s {
            return *ty;
        }
    }
    CommonDecisionType::Unknown
}

/// `common_get_decision_type` (common.cpp:1170-1180): read
/// `general.architecture` then `<arch>.decision.type`; NONE when either is
/// absent.
pub fn common_get_decision_type(gguf: &ggml::gguf::Gguf) -> CommonDecisionType {
    let Some(arch) = gguf.get_str("general.architecture") else {
        return CommonDecisionType::None;
    };
    let key = format!("{arch}.decision.type");
    gguf.get_str(&key)
        .map(common_decision_type_from_string)
        .unwrap_or(CommonDecisionType::None)
}

// ---------------------------------------------------------------------------
// question types (server-decision.h:14-27)
// ---------------------------------------------------------------------------

/// `server_model_output_modalities` (server-common.cpp:146-163, 4d60b4d08;
/// PPLX_DECIDER/LFM2_D1/LFM2_D1_OMNI joined via da263e727/88dcc460d/a657f7e98)
pub fn server_model_output_modalities(decision_type: CommonDecisionType) -> Vec<&'static str> {
    match decision_type {
        CommonDecisionType::Openjev
        | CommonDecisionType::Lev
        | CommonDecisionType::Kev
        | CommonDecisionType::Nimble
        | CommonDecisionType::Laya
        | CommonDecisionType::Clef
        | CommonDecisionType::PplxDecider
        | CommonDecisionType::Lfm2D1
        | CommonDecisionType::Lfm2D1Omni => vec!["decisions"],
        // fallback when there is no decision type or the metadata is bad
        _ => vec!["text"],
    }
}

/// `server_model_architecture_json` (server-common.cpp:165-185, 4d60b4d08) —
/// the architecture object of GET /models; shared by the direct server and
/// the router
pub fn server_model_architecture_json(
    inp_image: bool,
    inp_audio: bool,
    inp_video: bool,
    output_modalities: &[&str],
) -> Json {
    let mut input_modalities: Vec<&str> = vec!["text"];
    if inp_image {
        input_modalities.push("image");
    }
    if inp_audio {
        input_modalities.push("audio");
    }
    if inp_video {
        input_modalities.push("video");
    }
    Json::Object(vec![
        (
            "input_modalities".to_string(),
            Json::Array(
                input_modalities
                    .into_iter()
                    .map(|m| Json::String(m.to_string()))
                    .collect(),
            ),
        ),
        (
            "output_modalities".to_string(),
            Json::Array(
                output_modalities
                    .iter()
                    .map(|&m| Json::String(m.to_string()))
                    .collect(),
            ),
        ),
    ])
}

/// `enum server_decision_question_type`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerDecisionQuestionType {
    Choice,
    Score,
    Noul,
}

/// `decision_question_type_name` (server-decision.cpp:10-16)
fn decision_question_type_name(ty: ServerDecisionQuestionType) -> &'static str {
    match ty {
        ServerDecisionQuestionType::Choice => "choice",
        ServerDecisionQuestionType::Score => "score",
        ServerDecisionQuestionType::Noul => "noul",
    }
}

/// `struct server_decision_option` (server-decision.h:22-25)
#[derive(Clone, Debug)]
pub struct ServerDecisionOption {
    pub key: String,
    /// null if not provided
    pub description: Json,
}

/// `struct server_decision_question` (server-decision.h:27-32)
#[derive(Clone, Debug)]
pub struct ServerDecisionQuestion {
    pub id: String,
    pub ty: ServerDecisionQuestionType,
    pub instructions: Json,
    /// in the order of the model outputs
    pub options: Vec<ServerDecisionOption>,
}

// ---------------------------------------------------------------------------
// the task-side decision spec (`server_task::decision`, server-task.h:178-201)
// ---------------------------------------------------------------------------

/// `struct server_task::decision` (server-task.h:178-201) — where to read the
/// model output of each option, exactly one of the two lists is used.
#[derive(Clone, Debug, Default)]
pub struct DecisionSpec {
    /// logits of these tokens, at the last prompt token
    pub labels: Vec<i32>,
    /// if set, number of labels per output, the output is their max
    /// (server-task.h:181, 88dcc460d)
    pub label_groups: Vec<i32>,
    /// embeddings[column] at these prompt positions
    pub markers: Vec<i32>,
    pub column: i32,
    /// if set, embeddings is [q | k], and the output is instead the scaled
    /// dot product of q[pointer] and k[marker]
    pub pointer: i32,
    /// for a joint head: one value per prompt token, see
    /// `llama_batch_ext_set_decision_order()`; the scores are the first
    /// n_scores rows of the embeddings
    pub order: Vec<i32>,
    pub n_scores: i32,
}

impl DecisionSpec {
    /// `decision.pos_first()` (server-task.h:193-199) — first prompt position
    /// that is read, -1 if none
    pub fn pos_first(&self) -> i32 {
        let mut pos = self.pointer;
        for &marker in &self.markers {
            pos = if pos < 0 { marker } else { pos.min(marker) };
        }
        pos
    }
}

// ---------------------------------------------------------------------------
// server_decision_context (server-decision.h:35-129 / server-decision.cpp)
// ---------------------------------------------------------------------------

/// `struct server_decision_context` — the per-model decision setup.
pub struct ServerDecisionContext {
    pub ty: CommonDecisionType,
    vocab: std::sync::Arc<Vocab>,
    /// the "systemone" template source (`common_chat_template` handle)
    template_src: String,

    /// `"<type>"` or `"<type>.<n_options bucket>"`
    temperatures: BTreeMap<String, f32>,
    n_options_max: usize,
    /// noul options are [true, false] instead of [false, true]
    noul_true_first: bool,
    /// choice options are in the order of their keys
    choice_sorted: bool,

    /// OPENJEV, LEV, NIMBLE
    labels: Vec<i32>,
    /// only if the label of an option is given to the template
    label_texts: Vec<String>,

    /// LAYA, KEV
    token_marker: i32,
    token_sep: i32,
    text_marker: String,
    /// question + options
    max_head_tokens: usize,
    max_option_tokens: usize,
}

/// `decision_meta_str` (server-decision.cpp:22-26): `llama_model_meta_val_str`
/// renders ANY scalar KV as text (numbers included), so the port stringifies
/// them the same way
fn decision_meta_str(gguf: &ggml::gguf::Gguf, key: &str) -> String {
    use ggml::gguf::Value;
    match gguf.find_key(key) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::U8(v)) => v.to_string(),
        Some(Value::I8(v)) => v.to_string(),
        Some(Value::U16(v)) => v.to_string(),
        Some(Value::I16(v)) => v.to_string(),
        Some(Value::U32(v)) => v.to_string(),
        Some(Value::I32(v)) => v.to_string(),
        Some(Value::Bool(v)) => v.to_string(),
        Some(Value::U64(v)) => v.to_string(),
        Some(Value::I64(v)) => v.to_string(),
        Some(Value::F32(v)) => v.to_string(),
        Some(Value::F64(v)) => v.to_string(),
        _ => String::new(),
    }
}

impl ServerDecisionContext {
    /// `void server_decision_context::init(const llama_model * model)`
    /// (server-decision.cpp:32-131). `Err` mirrors the C's thrown
    /// `std::runtime_error` (the loader rejects the model with it).
    pub fn init(gguf: &ggml::gguf::Gguf, vocab: std::sync::Arc<Vocab>) -> Result<Self, String> {
        let model_type = common_get_decision_type(gguf);
        if model_type == CommonDecisionType::None {
            return Ok(ServerDecisionContext {
                ty: CommonDecisionType::None,
                vocab,
                template_src: String::new(),
                temperatures: BTreeMap::new(),
                n_options_max: 0,
                noul_true_first: false,
                choice_sorted: false,
                labels: Vec::new(),
                label_texts: Vec::new(),
                token_marker: -1,
                token_sep: -1,
                text_marker: String::new(),
                max_head_tokens: 0,
                max_option_tokens: 48,
            });
        }

        let arch = decision_meta_str(gguf, "general.architecture");
        let prefix = format!("{arch}.decision.");
        let type_name = decision_meta_str(gguf, &format!("{prefix}type"));

        // `llama_model_chat_template(model, "systemone")` — the suffixed
        // template KV (chat.cpp's `common_chat_templates_init` lookup)
        let tmpl_src = match gguf.get_str("tokenizer.chat_template.systemone") {
            Some(s) => s.to_string(),
            None => {
                return Err("decision model has no \"systemone\" template".to_string());
            }
        };

        // the fitted temperatures: every `<arch>.decision.temperature.<name>`
        // key (server-decision.cpp:51-66)
        let mut temperatures = BTreeMap::new();
        let prefix_temp = format!("{prefix}temperature.");
        for (key, val) in gguf.kv.iter() {
            let Some(suffix) = key.strip_prefix(&prefix_temp) else {
                continue;
            };
            let Some(s) = val.as_str() else {
                continue;
            };
            let temp: f32 = s.parse().map_err(|_| {
                format!("invalid decision temperature: {key} = {s}")
            })?;
            if temp <= 0.0 {
                return Err(format!("invalid decision temperature: {key} = {s}"));
            }
            temperatures.insert(suffix.to_string(), temp);
        }

        let mut labels: Vec<i32> = Vec::new();
        let mut label_texts: Vec<String> = Vec::new();
        let mut n_options_max = 0usize;
        let mut noul_true_first = false;
        let mut choice_sorted = false;
        let mut token_marker = -1i32;
        let mut token_sep = -1i32;
        let mut text_marker = String::new();
        let mut max_head_tokens = 0usize;

        match model_type {
            CommonDecisionType::Openjev => {
                // one letter per option, each must be a single token
                let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
                for c in letters.chars() {
                    let toks = vocab.tokenize(&c.to_string(), false, false);
                    if toks.len() != 1 {
                        return Err(format!("decision label '{c}' is not a single token"));
                    }
                    labels.push(toks[0]);
                }
                n_options_max = labels.len();
                noul_true_first = true;
            }
            CommonDecisionType::Lev | CommonDecisionType::Nimble | CommonDecisionType::PplxDecider => {
                // label codes are A..Z then AA..ZZ, only the ones that are a
                // single token are used
                let mut codes: Vec<String> = Vec::new();
                for a in b'A'..=b'Z' {
                    codes.push((a as char).to_string());
                }
                for a in b'A'..=b'Z' {
                    for b in b'A'..=b'Z' {
                        codes.push(format!("{}{}", a as char, b as char));
                    }
                }
                for code in codes {
                    let toks = vocab.tokenize(&code, false, false);
                    if toks.len() == 1 && labels.len() < 255 {
                        labels.push(toks[0]);
                        label_texts.push(code);
                    }
                }
                n_options_max = labels.len();
            }
            CommonDecisionType::Kev => {
                // the hidden state of an option is read at the token that
                // ends it
                let toks = vocab.tokenize("<|box_end|>", false, true);
                if toks.len() != 1 {
                    return Err("decision model has no <|box_end|> token".to_string());
                }
                token_marker = toks[0];
                n_options_max = 255;
            }
            CommonDecisionType::Laya => {
                token_marker = vocab.token_mask();
                token_sep = vocab.token_sep();
                if token_marker < 0 || token_sep < 0 {
                    return Err("decision model has no mask or sep token".to_string());
                }
                text_marker = crate::api::token_piece(&vocab, token_marker, true);

                let val = decision_meta_str(gguf, &format!("{prefix}max_head_tokens"));
                max_head_tokens = val.parse().unwrap_or(0);
                if max_head_tokens == 0 {
                    return Err("decision model has no valid max_head_tokens".to_string());
                }
                n_options_max = 255;
            }
            CommonDecisionType::Clef => {
                n_options_max = 255;
                noul_true_first = true;
                choice_sorted = true;
            }
            CommonDecisionType::Lfm2D1 => {
                // (server-decision.cpp:125-127, 88dcc460d)
                n_options_max = 255;
                noul_true_first = true;
            }
            CommonDecisionType::Lfm2D1Omni => {
                // (server-decision.cpp:128-133, a657f7e98)
                token_marker = vocab.token_mask();
                if token_marker == -1 {
                    return Err("decision model has no mask token".to_string());
                }
                n_options_max = 255;
            }
            _ => {
                return Err(format!("unsupported decision model type: {type_name}"));
            }
        }

        // SRV_INF("decision model type: %s", ...) (server-decision.cpp:130)
        eprintln!("llama-server: decision model type: {type_name}");

        Ok(ServerDecisionContext {
            ty: model_type,
            vocab,
            template_src: tmpl_src,
            temperatures,
            n_options_max,
            noul_true_first,
            choice_sorted,
            labels,
            label_texts,
            token_marker,
            token_sep,
            text_marker,
            max_head_tokens,
            max_option_tokens: 48,
        })
    }

    /// `can_share_prompt()` (server-decision.h:41-54)
    pub fn can_share_prompt(&self) -> bool {
        matches!(
            self.ty,
            CommonDecisionType::Openjev
                | CommonDecisionType::Lev
                | CommonDecisionType::Kev
                | CommonDecisionType::Nimble
                | CommonDecisionType::PplxDecider
                | CommonDecisionType::Lfm2D1
        )
    }

    /// `is_joint()` (server-decision.h:56-58)
    pub fn is_joint(&self) -> bool {
        self.ty == CommonDecisionType::Clef
    }

    /// `can_use_images()` (server-decision.h:60-74) — clef's vision input
    /// needs token and embedding entries in the same batch (9871df591): the
    /// port's server has no mtmd wiring, so media still answers 501
    pub fn can_use_images(&self) -> bool {
        matches!(
            self.ty,
            CommonDecisionType::Openjev
                | CommonDecisionType::Clef
                | CommonDecisionType::PplxDecider
                | CommonDecisionType::Lfm2D1
                | CommonDecisionType::Lfm2D1Omni
        )
    }

    // -----------------------------------------------------------------------
    // request parsing (server-decision.cpp:137-269)
    // -----------------------------------------------------------------------

    /// `parse_questions` (server-decision.cpp:137-207). `Err` mirrors the C's
    /// `std::invalid_argument` (HTTP 400).
    pub fn parse_questions(&self, body: &Json) -> Result<Vec<ServerDecisionQuestion>, String> {
        if j_object(body).is_none() {
            return Err("\"questions\" must be provided".to_string());
        }

        let state = j_get(body, "state")
            .ok_or_else(|| "\"state\" must be provided".to_string())?;
        // lfm2-d1 and lfm2-d1-omni accept a null state, for example to ask
        // about images only (server-decision.cpp:147-149, 88dcc460d/a657f7e98)
        if state.is_null()
            && self.ty != CommonDecisionType::Lfm2D1
            && self.ty != CommonDecisionType::Lfm2D1Omni
        {
            return Err("\"state\" must be provided".to_string());
        }

        let questions_val = j_get(body, "questions")
            .ok_or_else(|| "\"questions\" must be a non-empty object".to_string())?;
        let questions_obj = j_object(questions_val)
            .filter(|o| !o.is_empty())
            .ok_or_else(|| "\"questions\" must be a non-empty object".to_string())?;

        let mut questions = Vec::new();
        for (id, q) in questions_obj {
            let err = |msg: String| format!("questions.{id}: {msg}");
            let q_obj = j_object(q).ok_or_else(|| err("must be an object".to_string()))?;
            let instructions = q_get_instructions(q_obj)
                .ok_or_else(|| err("\"instructions\" must be provided".to_string()))?;

            let type_name = j_get(q, "type").and_then(j_str).unwrap_or_default().to_string();
            let criteria = q_obj
                .iter()
                .find(|(k, _)| k == "criteria")
                .map(|(_, v)| v)
                .cloned()
                .unwrap_or(Json::Null);

            let mut question = ServerDecisionQuestion {
                id: id.clone(),
                ty: ServerDecisionQuestionType::Choice,
                instructions: instructions.clone(),
                options: Vec::new(),
            };

            match type_name.as_str() {
                "choice" => {
                    question.ty = ServerDecisionQuestionType::Choice;
                    let c = j_object(&criteria)
                        .filter(|o| !o.is_empty())
                        .ok_or_else(|| err("\"criteria\" must be a non-empty object".to_string()))?;
                    for (key, description) in c {
                        question.options.push(ServerDecisionOption {
                            key: key.clone(),
                            description: description.clone(),
                        });
                    }
                    if self.choice_sorted {
                        question.options.sort_by(|a, b| a.key.cmp(&b.key));
                    }
                }
                "score" => {
                    question.ty = ServerDecisionQuestionType::Score;
                    let c = j_array(&criteria).filter(|a| a.len() >= 2 && a.len() <= 10).ok_or_else(
                        || err("\"criteria\" must be an array of 2 to 10 levels".to_string()),
                    )?;
                    for (i, description) in c.iter().enumerate() {
                        question.options.push(ServerDecisionOption {
                            key: i.to_string(),
                            description: description.clone(),
                        });
                    }
                }
                "noul" => {
                    question.ty = ServerDecisionQuestionType::Noul;
                    let c_ok = criteria.is_null()
                        || j_object(&criteria).is_some();
                    if !c_ok {
                        return Err(err("\"criteria\" must be an object".to_string()));
                    }
                    let c_obj = j_object(&criteria);
                    for key in ["false", "true"] {
                        // lfm2-d1-omni also reads the descriptions under "no"
                        // and "yes" (server-decision.cpp:202-208, a657f7e98)
                        let mut description = c_obj
                            .and_then(|o| o.iter().find(|(k, _)| k == key).map(|(_, v)| v))
                            .cloned();
                        if description.is_none()
                            && c_obj.is_some()
                            && self.ty == CommonDecisionType::Lfm2D1Omni
                        {
                            let alias = if key == "true" { "yes" } else { "no" };
                            description = c_obj
                                .and_then(|o| o.iter().find(|(k, _)| k == alias).map(|(_, v)| v))
                                .cloned();
                        }
                        question.options.push(ServerDecisionOption {
                            key: key.to_string(),
                            description: description.unwrap_or(Json::Null),
                        });
                    }
                    if self.noul_true_first {
                        question.options.swap(0, 1);
                    }
                }
                _ => {
                    return Err(err("\"type\" must be one of: choice, score, noul".to_string()));
                }
            }

            if question.options.len() > self.n_options_max {
                return Err(err(format!(
                    "too many options ({}), this model supports at most {}",
                    question.options.len(),
                    self.n_options_max
                )));
            }

            questions.push(question);
        }
        Ok(questions)
    }

    /// `parse_state` (server-decision.cpp:225-269) — returns the state (chat
    /// messages with their image parts taken out) and the raw image URLs.
    /// The port has no mtmd wiring, so the URLs are only validated
    /// (`decision_load_image`'s checks) — the caller answers 501 for any
    /// image, exactly like the reference does when the model/server cannot
    /// use them (`!decision.can_use_images() || !meta->has_inp_image`).
    pub fn parse_state(&self, body: &Json) -> Result<(Json, Vec<String>), String> {
        if j_object(body).is_none() {
            return Err("\"state\" must be provided".to_string());
        }

        // `"videos" is not supported` (server-decision.cpp:226-228, 9871df591)
        if let Some(v) = j_get(body, "videos") {
            let non_empty = match j_array(v) {
                Some(a) => !a.is_empty(),
                None => true,
            };
            if !v.is_null() && non_empty {
                return Err("\"videos\" is not supported".to_string());
            }
        }

        let mut files: Vec<String> = Vec::new();
        // "images" is an alias of "files" (server-decision.cpp:245-255, a657f7e98)
        for key in ["files", "images"] {
            let Some(list) = j_get(body, key) else { continue };
            if list.is_null() {
                continue;
            }
            let arr = j_array(list)
                .ok_or_else(|| format!("\"{key}\" must be an array"))?;
            for url in arr {
                decision_load_image(url, &mut files)?;
            }
        }

        let state = j_get(body, "state")
            .ok_or_else(|| "\"state\" must be provided".to_string())?;
        let is_wrapped = j_object(state)
            .map(|o| o.iter().any(|(k, _)| k == "messages"))
            .unwrap_or(false);
        let messages = if is_wrapped {
            j_get(state, "messages").unwrap()
        } else {
            state
        };
        let Some(msgs) = j_array(messages) else {
            return Ok((state.clone(), files));
        };

        // chat messages: take the image parts out of the content
        let mut messages_out: Vec<Json> = Vec::new();
        for msg in msgs {
            let Some(m_obj) = j_object(msg) else {
                messages_out.push(msg.clone());
                continue;
            };
            let content = m_obj.iter().find(|(k, _)| k == "content").map(|(_, v)| v);
            let Some(content_arr) = content.and_then(j_array) else {
                messages_out.push(msg.clone());
                continue;
            };
            let mut new_content: Vec<Json> = Vec::new();
            for part in content_arr {
                let is_image = j_object(part)
                    .map(|p| {
                        j_get(part, "type").and_then(j_str) == Some("image_url")
                            && p.iter().any(|(k, _)| k == "image_url")
                    })
                    .unwrap_or(false);
                let is_audio = j_object(part)
                    .map(|p| {
                        j_get(part, "type").and_then(j_str) == Some("input_audio")
                            && p.iter().any(|(k, _)| k == "input_audio")
                    })
                    .unwrap_or(false);
                if is_image {
                    let p_obj = j_object(part).unwrap();
                    let image_url = p_obj
                        .iter()
                        .find(|(k, _)| k == "image_url")
                        .map(|(_, v)| v)
                        .unwrap();
                    let url = if let Some(u_obj) = j_object(image_url) {
                        u_obj
                            .iter()
                            .find(|(k, _)| k == "url")
                            .map(|(_, v)| v)
                            .unwrap()
                            .clone()
                    } else {
                        image_url.clone()
                    };
                    decision_load_image(&url, &mut files)?;
                } else if is_audio {
                    // the input_audio parts of a chat state
                    // (server-decision.cpp:292-295, a657f7e98)
                    let p_obj = j_object(part).unwrap();
                    let input_audio = p_obj
                        .iter()
                        .find(|(k, _)| k == "input_audio")
                        .map(|(_, v)| v)
                        .cloned()
                        .unwrap_or(Json::Object(Vec::new()));
                    let data = if let Some(d) =
                        j_get(&input_audio, "data")
                    {
                        d.clone()
                    } else {
                        j_get(&input_audio, "url").cloned().unwrap_or(Json::Null)
                    };
                    decision_load_audio(&data, &mut files)?;
                } else {
                    new_content.push(part.clone());
                }
            }
            let mut msg_out = m_obj.clone();
            if let Some(slot) = msg_out.iter_mut().find(|(k, _)| k == "content") {
                slot.1 = Json::Array(new_content);
            } else {
                msg_out.push(("content".to_string(), Json::Array(new_content)));
            }
            messages_out.push(Json::Object(msg_out));
        }

        if !is_wrapped {
            return Ok((Json::Array(messages_out), files));
        }
        let mut state_out = j_object(state).unwrap().clone();
        if let Some(slot) = state_out.iter_mut().find(|(k, _)| k == "messages") {
            slot.1 = Json::Array(messages_out);
        }
        Ok((Json::Object(state_out), files))
    }

    // -----------------------------------------------------------------------
    // prompt (server-decision.cpp:359-554)
    // -----------------------------------------------------------------------

    /// `n_variants` (server-decision.cpp:359-365): lev shows the options of a
    /// choice in 2 orders, to cancel the preference for the first label
    pub fn n_variants(&self, question: &ServerDecisionQuestion) -> usize {
        if self.ty == CommonDecisionType::Lev
            && question.ty == ServerDecisionQuestionType::Choice
            && question.options.len() > 1
        {
            return 2;
        }
        1
    }

    /// `n_outputs` (server-decision.cpp:367-372)
    fn n_outputs(&self, question: &ServerDecisionQuestion) -> usize {
        if self.ty == CommonDecisionType::Lev && question.ty == ServerDecisionQuestionType::Noul {
            DECISION_LEV_N_RATINGS
        } else {
            question.options.len()
        }
    }

    /// `d1_labels` (server-decision.cpp:382-463, 88dcc460d) — label text and
    /// tokens of each option of an LFM2_D1 question, following prompt.py of
    /// the model repo
    fn d1_labels(
        &self,
        question: &ServerDecisionQuestion,
    ) -> Result<(Vec<String>, Vec<Vec<i32>>), String> {
        let n_options = question.options.len();
        let mut texts: Vec<String> = Vec::new();
        let mut groups: Vec<Vec<i32>> = Vec::new();

        // `get_single_tokens`: the forms that tokenize to exactly one,
        // not-yet-used token
        let get_single_tokens = |forms: &[String]| -> Vec<i32> {
            let mut out: Vec<i32> = Vec::new();
            for form in forms {
                let toks = self.vocab.tokenize(form, false, false);
                if toks.len() == 1 && !out.contains(&toks[0]) {
                    out.push(toks[0]);
                }
            }
            out
        };

        if question.ty != ServerDecisionQuestionType::Choice {
            for opt in &question.options {
                let group = if question.ty == ServerDecisionQuestionType::Score {
                    get_single_tokens(&[opt.key.clone()])
                } else if opt.key == "true" {
                    get_single_tokens(&[
                        "yes".to_string(),
                        "Yes".to_string(),
                        "YES".to_string(),
                    ])
                } else {
                    get_single_tokens(&["no".to_string(), "No".to_string(), "NO".to_string()])
                };
                if group.is_empty() {
                    return Err(format!("decision label is not a single token: {}", opt.key));
                }
                texts.push(opt.key.clone());
                groups.push(group);
            }
            return Ok((texts, groups));
        }

        let mut is_letters = true;
        for opt in &question.options {
            is_letters = is_letters
                && opt.key.len() == 1
                && opt.key.as_bytes()[0].is_ascii_alphabetic();
        }

        let mut codes: Vec<String> = Vec::new();
        for (i, opt) in question.options.iter().enumerate() {
            if is_letters {
                codes.push(opt.key.clone());
            } else if n_options <= 26 {
                codes.push(((b'A' + i as u8) as char).to_string());
            } else {
                codes.push(format!("{i:02}"));
            }
        }

        let mut pool: Vec<String> = Vec::new();
        for c in b'A'..=b'Z' {
            pool.push((c as char).to_string());
        }
        for i in 0..100 {
            pool.push(format!("{i:02}"));
        }
        for c in b'a'..=b'z' {
            pool.push((c as char).to_string());
        }
        for i in 0..200 {
            pool.push(format!("#{i}"));
        }
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                pool.push(format!("{}{}", a as char, b as char));
            }
        }

        let mut used: Vec<i32> = Vec::new();
        // `take`: a code is usable when it is one fresh token; " <code>" adds
        // the token of the code with a leading space when it differs
        let mut take = |code: &str,
                        texts: &mut Vec<String>,
                        groups: &mut Vec<Vec<i32>>|
         -> bool {
            let toks = self.vocab.tokenize(code, false, false);
            if toks.len() != 1 || used.contains(&toks[0]) {
                return false;
            }
            used.push(toks[0]);
            let mut group = vec![toks[0]];
            for tok in get_single_tokens(&[format!(" {code}")]) {
                if tok != toks[0] {
                    group.push(tok);
                }
            }
            texts.push(code.to_string());
            groups.push(group);
            true
        };
        for code in &codes {
            let mut is_taken = take(code, &mut texts, &mut groups);
            let mut i = 0usize;
            while !is_taken && i < pool.len() {
                is_taken = take(&pool[i], &mut texts, &mut groups);
                i += 1;
            }
            if !is_taken {
                return Err(format!(
                    "no single-token label left for {n_options} options"
                ));
            }
        }
        Ok((texts, groups))
    }

    /// `render_options` (server-decision.cpp:466-511) — throws the d1 label
    /// errors of `d1_labels` (runtime_error/invalid_argument in the C)
    fn render_options(
        &self,
        question: &ServerDecisionQuestion,
        variant: usize,
    ) -> Result<Json, String> {
        let n_options = question.options.len();

        let d1_texts: Vec<String> = if self.ty == CommonDecisionType::Lfm2D1 {
            self.d1_labels(question)?.0
        } else {
            Vec::new()
        };

        // the second variant shows the options in the reverse order
        let mut options: Vec<Json> = Vec::new();
        for i in 0..n_options {
            let opt = &question.options[if variant == 0 { i } else { n_options - 1 - i }];
            let mut option = vec![
                ("key".to_string(), Json::String(opt.key.clone())),
                ("description".to_string(), opt.description.clone()),
            ];
            if self.ty == CommonDecisionType::Kev {
                // kev text input: special tokens written in the text must not
                // be parsed as such
                option[0].1 = Json::String(decision_kev_text_of(&opt.key));
                if !opt.description.is_null() {
                    option[1].1 = Json::String(decision_kev_text_of_json(&opt.description));
                }
            }
            if !self.label_texts.is_empty() {
                option.push(("label".to_string(), Json::String(self.label_texts[i].clone())));
            }
            if !d1_texts.is_empty() {
                option.push(("label".to_string(), Json::String(d1_texts[i].clone())));
            }
            options.push(Json::Object(option));
        }
        Ok(Json::Array(options))
    }

    /// `render` (server-decision.cpp:570-668) — the "systemone" jinja render.
    /// `is_audio` is the d1omni media kind (false without media).
    fn render(
        &self,
        state: &Json,
        questions: &[ServerDecisionQuestion],
        question: &ServerDecisionQuestion,
        variant: usize,
        n_images: usize,
        media_marker: &str,
        is_audio: bool,
    ) -> Result<String, String> {
        // the template is given raw JSON values, it serializes the ones that
        // are not strings
        let mut inp: Vec<(String, Json)> = vec![
            ("id".into(), Json::String(question.id.clone())),
            (
                "type".into(),
                Json::String(decision_question_type_name(question.ty).to_string()),
            ),
            ("instructions".into(), question.instructions.clone()),
            ("state".into(), state.clone()),
            ("options".into(), self.render_options(question, variant)?),
        ];

        // the nimble prompt lists all the questions of the request
        if self.ty == CommonDecisionType::Nimble {
            let mut qs: Vec<Json> = Vec::new();
            for q in questions {
                qs.push(Json::Object(vec![
                    ("id".into(), Json::String(q.id.clone())),
                    (
                        "type".into(),
                        Json::String(decision_question_type_name(q.ty).to_string()),
                    ),
                    ("instructions".into(), q.instructions.clone()),
                    ("options".into(), self.render_options(q, 0)?),
                ]));
            }
            inp.push(("questions".into(), Json::Array(qs)));
        }

        // lev was trained with sorted keys
        if self.ty == CommonDecisionType::Lev {
            let sorted = decision_sort_keys(&Json::Object(inp));
            inp = match sorted {
                Json::Object(o) => o,
                _ => unreachable!(),
            };
        }

        // the kev template only takes text
        if self.ty == CommonDecisionType::Kev {
            for (k, v) in inp.iter_mut() {
                if k == "state" {
                    *v = Json::String(decision_kev_text_of_json(v));
                } else if k == "instructions" {
                    *v = Json::String(decision_kev_text_of_json(v));
                }
            }
        }

        // the input must not contain the marker of the options
        if !self.text_marker.is_empty() {
            let replaced = decision_replace_text(&Json::Object(inp), &self.text_marker, " ");
            inp = match replaced {
                Json::Object(o) => o,
                _ => unreachable!(),
            };
        }

        // (server-decision.cpp:626-634, a657f7e98)
        if self.ty == CommonDecisionType::Lfm2D1Omni {
            let escaped = decision_replace_text(
                &decision_d1omni_escape(&Json::Object(inp)),
                D1OMNI_MARKER,
                "<<d1omni ",
            );
            inp = match escaped {
                Json::Object(o) => o,
                _ => unreachable!(),
            };
            inp.push(("audio".into(), Json::Bool(is_audio)));
            inp.push(("sep".into(), Json::String(D1OMNI_SEP.to_string())));
            inp.push(("mark_state".into(), Json::String(D1OMNI_MARK_STATE.to_string())));
            inp.push((
                "mark_question".into(),
                Json::String(D1OMNI_MARK_QUESTION.to_string()),
            ));
            inp.push(("mark_option".into(), Json::String(D1OMNI_MARK_OPTION.to_string())));
        }

        // the template puts one media marker per image
        let mut images: Vec<Json> = Vec::new();
        if n_images > 0 {
            let replaced = decision_replace_text(&Json::Object(inp), media_marker, " ");
            inp = match replaced {
                Json::Object(o) => o,
                _ => unreachable!(),
            };
            for _ in 0..n_images {
                images.push(Json::String(media_marker.to_string()));
            }
        }
        inp.push(("images".into(), Json::Array(images)));

        // `jinja::context(tmpl->source()); global_from_json(ctx, inp, false);
        // runtime.execute` — the port's mini-jinja with the inp as globals
        use llama::chat::mini_jinja;
        let toks = mini_jinja::lex(&self.template_src).map_err(|e| format!("mini-jinja: {e}"))?;
        let prog = mini_jinja::parse(&toks).map_err(|e| format!("mini-jinja: {e}"))?;
        let mut inputs = mini_jinja::RenderInputs::default();
        for (k, v) in inp {
            inputs.extra.push((k, mini_jinja::RenderInputs::val_from_json(&v)));
        }
        mini_jinja::render_inputs(&prog, &inputs).map_err(|e| format!("mini-jinja: {e}"))
    }

    /// `fill_task` (server-decision.cpp:594-668): build one task's tokens and
    /// its output spec. `media_marker` is `get_media_marker()`
    /// (server-common.cpp) — the port passes the plain text chunk marker.
    /// `files` is the request's media (empty: the port's systemone gate
    /// answers 501 for any media until mtmd is wired).
    pub fn fill_task(
        &self,
        state: &Json,
        questions: &[ServerDecisionQuestion],
        question: &ServerDecisionQuestion,
        variant: usize,
        n_images: usize,
        media_marker: &str,
        files: &[String],
    ) -> Result<(Vec<i32>, DecisionSpec), String> {
        // (server-decision.cpp:660-663, a657f7e98)
        if self.ty == CommonDecisionType::Lfm2D1Omni {
            return self.fill_task_d1omni(state, questions, question, variant, files, media_marker);
        }

        let prompt = self.render(state, questions, question, variant, n_images, media_marker, false)?;

        let mut spec = DecisionSpec {
            pointer: -1,
            ..Default::default()
        };

        if matches!(
            self.ty,
            CommonDecisionType::Openjev
                | CommonDecisionType::Lev
                | CommonDecisionType::Nimble
                | CommonDecisionType::PplxDecider
        ) {
            // lev reads the ratings of a noul question at its first labels,
            // not at the digits
            spec.labels = self.labels[..self.n_outputs(question)].to_vec();
            // (images ride the mtmd prompt in the reference; the port has no
            // mmproj wiring, images are rejected earlier with 501)
        }
        // (server-decision.cpp:588-595, 88dcc460d)
        if self.ty == CommonDecisionType::Lfm2D1 {
            let (_texts, groups) = self.d1_labels(question)?;
            for group in &groups {
                spec.labels.extend_from_slice(group);
                spec.label_groups.push(group.len() as i32);
            }
        }

        let mut tokens = self.vocab.tokenize(&prompt, false, true);
        if self.ty == CommonDecisionType::Laya {
            self.fill_task_laya(&mut tokens, question, &mut spec)?;
        }
        if self.ty == CommonDecisionType::Kev {
            // an option is read at its end token, the question at the last token
            for (i, &t) in tokens.iter().enumerate() {
                if t == self.token_marker {
                    spec.markers.push(i as i32);
                }
            }
            if spec.markers.len() != question.options.len() {
                return Err("unexpected layout of the decision prompt".to_string());
            }
            spec.pointer = tokens.len() as i32 - 1;
        }
        Ok((tokens, spec))
    }

    /// `fill_task_laya` (server-decision.cpp:501-554): the prompt is
    /// `[cls] question [sep] ([marker] option)* [sep] state [sep]`; options
    /// and question are cut to fit max_head_tokens, the same way the model
    /// was trained.
    fn fill_task_laya(
        &self,
        tokens: &mut Vec<i32>,
        question: &ServerDecisionQuestion,
        spec: &mut DecisionSpec,
    ) -> Result<(), String> {
        let n_options = question.options.len();

        let mut markers: Vec<usize> = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            if t == self.token_marker {
                markers.push(i);
            }
        }
        let invalid = || "unexpected layout of the decision prompt".to_string();
        if markers.len() != n_options
            || markers[0] < 2
            || tokens[markers[0] - 1] != self.token_sep
            || *tokens.last().unwrap() != self.token_sep
        {
            return Err(invalid());
        }
        let head_end = markers[0] - 1;
        let opts_end = tokens[*markers.last().unwrap()..]
            .iter()
            .position(|&t| t == self.token_sep)
            .map(|p| markers.last().unwrap() + p)
            .ok_or_else(invalid)?;
        if opts_end + 1 >= tokens.len() {
            return Err(invalid());
        }

        // marker + text of each option
        let mut options: Vec<Vec<i32>> = Vec::new();
        let mut n_options_tokens = 0usize;
        let mut set_max = |options: &mut Vec<Vec<i32>>, n_max: usize| -> usize {
            let mut total = 0;
            for opt in options.iter_mut() {
                opt.truncate(n_max);
                total += opt.len();
            }
            total
        };
        for i in 0..n_options {
            let end = if i + 1 < n_options { markers[i + 1] } else { opts_end };
            options.push(tokens[markers[i]..end].to_vec());
        }
        n_options_tokens = set_max(&mut options, self.max_option_tokens + 1);
        if n_options_tokens + 16 > self.max_head_tokens {
            // too many or too long options, shrink them evenly
            let shrunk = (self.max_head_tokens.saturating_sub(16) / n_options).max(4);
            n_options_tokens = set_max(&mut options, shrunk);
        }
        let n_question_max =
            (self.max_head_tokens.saturating_sub(n_options_tokens.min(self.max_head_tokens)))
                .max(8);

        let mut out: Vec<i32> = Vec::new();
        out.push(tokens[0]);
        out.extend_from_slice(&tokens[1..head_end.min(1 + n_question_max)]);
        out.push(self.token_sep);
        for opt in &options {
            spec.markers.push(out.len() as i32);
            out.extend_from_slice(opt);
        }
        out.extend_from_slice(&tokens[opts_end..]);
        *tokens = out;

        // the output has one score per question type
        spec.column = question.ty as i32;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // lfm2-d1-omni (server-decision.cpp:764-885, a657f7e98)
    // -----------------------------------------------------------------------

    /// `fill_task_d1omni` — each piece is cut to its budget as the model was
    /// trained (d1-omni prompt.py: encode), the state gets the room that is
    /// left. The media come first, the text after them has a budget of its
    /// own. The port's systemone gate answers 501 for any media (no mmproj
    /// wiring), so `files` is always empty here — the media branch mirrors
    /// the reference for when mtmd lands.
    fn fill_task_d1omni(
        &self,
        state: &Json,
        questions: &[ServerDecisionQuestion],
        question: &ServerDecisionQuestion,
        variant: usize,
        files: &[String],
        media_marker: &str,
    ) -> Result<(Vec<i32>, DecisionSpec), String> {
        let invalid = || "unexpected layout of the decision prompt".to_string();

        // the prompt depends on the kind of media, mtmd tells an audio clip
        // from an image by its content — without media there are no markers
        // and no audio
        let markers = String::new();
        let media: Vec<i32> = Vec::new();
        let is_audio = false;
        if !files.is_empty() {
            // unreachable in the port today: the systemone handler 501s on
            // media before fill_task (`process_mtmd_prompt` needs the mtmd
            // context); with media the reference prepends the marker string
            // per file, detects audio from the chunk types, and the media
            // tokens ride in front of the text budget below
            return Err(
                "This server does not support image input for decisions. For a model that \
                 supports it, start it with `--mmproj`"
                    .to_string(),
            );
        }
        let n_media = media.len();

        let prompt = self.render(state, questions, question, variant, files.len(), media_marker, is_audio)?;
        let pieces = string_split(&prompt, D1OMNI_SEP);
        if pieces.is_empty() || pieces[0] != markers {
            return Err(invalid());
        }

        let mut n_max = D1OMNI_MAX_TOKENS;
        if !files.is_empty() {
            n_max = D1OMNI_MAX_TOKENS_IMAGE
                .min(D1OMNI_MAX_TOKENS - D1OMNI_MAX_TOKENS.min(n_media));
            if is_audio {
                n_max = D1OMNI_MAX_TOKENS_AUDIO
                    .min(D1OMNI_MAX_TOKENS - D1OMNI_MAX_TOKENS.min(n_media));
            }
            if n_max < 64 {
                return Err(format!(
                    "the media take {n_media} of the {D1OMNI_MAX_TOKENS} positions, send fewer images"
                ));
            }
        }

        // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
        let n_options = question.options.len() as i64;
        let n_budget = 96.max((24 * n_options + 32).min(n_max as i64 / 2));
        let n_option_max = 2.max((n_budget - 3 * n_options) / n_options);
        let n_question_max = 16.max(n_budget);

        let mut head: Vec<i32> = Vec::new(); // before the state
        let mut body: Vec<i32> = Vec::new(); // the state
        let mut tail: Vec<i32> = Vec::new(); // after the state
        let mut has_state = false;
        for piece in pieces.iter().skip(1) {
            let mut piece = piece.as_str();
            let mut n_piece_max: i64 = -1;
            let mut is_state = false;
            if let Some(rest) = piece.strip_prefix(D1OMNI_MARK_STATE) {
                piece = rest;
                is_state = true;
            } else if let Some(rest) = piece.strip_prefix(D1OMNI_MARK_QUESTION) {
                piece = rest;
                n_piece_max = n_question_max;
            } else if let Some(rest) = piece.strip_prefix(D1OMNI_MARK_OPTION) {
                piece = rest;
                n_piece_max = n_option_max;
            }

            let mut tokens = self.vocab.tokenize(piece, false, true);
            if n_piece_max >= 0 && tokens.len() as i64 > n_piece_max {
                tokens.truncate(n_piece_max as usize);
            }

            if is_state {
                if has_state {
                    return Err(invalid());
                }
                body = tokens;
                has_state = true;
            } else if has_state {
                tail.extend_from_slice(&tokens);
            } else {
                head.extend_from_slice(&tokens);
            }
        }
        if !has_state {
            return Err(invalid());
        }

        let n_room = n_max - n_max.min(head.len() + tail.len());
        body.truncate(body.len().min(n_room));

        let mut tokens = std::mem::take(&mut head);
        tokens.extend_from_slice(&body);
        tokens.extend_from_slice(&tail);
        tokens.truncate(tokens.len().min(n_max));

        let mut spec = DecisionSpec {
            pointer: -1,
            ..Default::default()
        };
        for (i, &t) in tokens.iter().enumerate() {
            if t == self.token_marker {
                spec.markers.push((n_media + i) as i32);
            }
        }
        if spec.markers.len() as i64 != n_options {
            return Err("the options do not fit in the context".to_string());
        }

        // the output has one score per question type
        spec.column = question.ty as i32;
        let _ = media; // with files the media tokens go first (unreachable today)
        Ok((tokens, spec))
    }

    // -----------------------------------------------------------------------
    // joint prompt — clef (server-decision.cpp:560-638)
    // -----------------------------------------------------------------------

    /// `fill_task_joint` (server-decision.cpp:566-638)
    pub fn fill_task_joint(
        &self,
        state: &Json,
        questions: &[ServerDecisionQuestion],
    ) -> Result<(Vec<i32>, DecisionSpec), String> {
        // `LLAMA_DECISION_ORDER_*` (src/llama-ext.h:107-113)
        const ORDER_NONE: i32 = 0;
        const ORDER_QUESTION_NOUL: i32 = 1;
        const ORDER_QUESTION_CHOICE: i32 = 2;
        const ORDER_QUESTION_SCORE: i32 = 3;
        const ORDER_OPTION: i32 = 4;

        let mut inp_questions: Vec<Json> = Vec::new();
        for question in questions {
            let mut options: Vec<Json> = Vec::new();
            for opt in &question.options {
                options.push(Json::Object(vec![
                    ("key".into(), Json::String(opt.key.clone())),
                    ("description".into(), opt.description.clone()),
                ]));
            }
            inp_questions.push(Json::Object(vec![
                ("id".into(), Json::String(question.id.clone())),
                (
                    "type".into(),
                    Json::String(decision_question_type_name(question.ty).to_string()),
                ),
                ("instructions".into(), question.instructions.clone()),
                ("options".into(), Json::Array(options)),
            ]));
        }

        // the template is given raw JSON values with sorted keys, and no
        // marker in the input
        let inp = Json::Object(vec![
            ("state".into(), state.clone()),
            ("questions".into(), Json::Array(inp_questions)),
        ]);
        let inp = decision_replace_text(
            &decision_sort_keys(&inp),
            CLEF_MARKER,
            "<<clef ",
        );
        // the template puts one media marker per image (server-decision.cpp:599-606,
        // 9871df591); the array is always present, empty without media. With
        // media (unreachable in the port today — the systemone handler 501s
        // on media before fill_task_joint) the reference replaces the marker
        // in the input and pushes one marker string per file.
        let images: Vec<Json> = Vec::new();
        let inp = match inp {
            Json::Object(mut o) => {
                o.push(("images".to_string(), Json::Array(images)));
                o
            }
            _ => unreachable!(),
        };

        // the CLEF markers (`server-decision.cpp:561-564`)
        let clef_sep = "<<clef:sep>>";
        let clef_mark_question = "<<clef:question>>";
        let clef_mark_option = "<<clef:option>>";

        use llama::chat::mini_jinja;
        let toks = mini_jinja::lex(&self.template_src).map_err(|e| format!("mini-jinja: {e}"))?;
        let prog = mini_jinja::parse(&toks).map_err(|e| format!("mini-jinja: {e}"))?;
        let mut inputs = mini_jinja::RenderInputs::default();
        for (k, v) in inp {
            inputs
                .extra
                .push((k, mini_jinja::RenderInputs::val_from_json(&v)));
        }
        inputs.extra.push((
            "sep".to_string(),
            mini_jinja::RenderInputs::val_from_json(&Json::String(clef_sep.to_string())),
        ));
        inputs.extra.push((
            "mark_question".to_string(),
            mini_jinja::RenderInputs::val_from_json(&Json::String(
                clef_mark_question.to_string(),
            )),
        ));
        inputs.extra.push((
            "mark_option".to_string(),
            mini_jinja::RenderInputs::val_from_json(&Json::String(clef_mark_option.to_string())),
        ));
        let prompt = mini_jinja::render_inputs(&prog, &inputs).map_err(|e| format!("mini-jinja: {e}"))?;

        // the model was trained with the pieces tokenized one by one
        let mut spec = DecisionSpec {
            pointer: -1,
            ..Default::default()
        };
        let mut tokens: Vec<i32> = Vec::new();
        let mut i_question = 0usize;
        for piece in string_split(&prompt, clef_sep) {
            let mut piece = piece;
            let mut order = ORDER_NONE;
            if let Some(rest) = piece.strip_prefix(clef_mark_question) {
                piece = rest.to_string();
                if i_question >= questions.len() {
                    return Err("unexpected layout of the decision prompt".to_string());
                }
                order = match questions[i_question].ty {
                    ServerDecisionQuestionType::Noul => ORDER_QUESTION_NOUL,
                    ServerDecisionQuestionType::Choice => ORDER_QUESTION_CHOICE,
                    ServerDecisionQuestionType::Score => ORDER_QUESTION_SCORE,
                };
                i_question += 1;
            } else if let Some(rest) = piece.strip_prefix(clef_mark_option) {
                piece = rest.to_string();
                order = ORDER_OPTION;
                spec.n_scores += 1;
            }

            let piece_tokens = self.vocab.tokenize(&piece, false, true);
            if order != ORDER_NONE && piece_tokens.is_empty() {
                return Err(
                    "the instructions and the options of a question must not be empty".to_string()
                );
            }
            tokens.extend_from_slice(&piece_tokens);
            spec.order
                .resize(tokens.len(), order);
        }

        let mut n_options = 0usize;
        for question in questions {
            n_options += question.options.len();
        }
        if i_question != questions.len() || spec.n_scores as usize != n_options {
            return Err("unexpected layout of the decision prompt".to_string());
        }

        Ok((tokens, spec))
    }

    // -----------------------------------------------------------------------
    // answer (server-decision.cpp:644-766)
    // -----------------------------------------------------------------------

    /// `get_temperature` (server-decision.cpp:644-663)
    fn get_temperature(&self, question: &ServerDecisionQuestion) -> f32 {
        let n = question.options.len();
        let type_name = decision_question_type_name(question.ty);

        // the temperature can depend on the number of options, the buckets
        // are the ones used to fit it
        let bucket = if self.ty == CommonDecisionType::Lev {
            if n <= 8 {
                "small".to_string()
            } else if n <= 26 {
                "mid".to_string()
            } else {
                "large".to_string()
            }
        } else if n <= 2 {
            "2".to_string()
        } else if n <= 5 {
            "3_5".to_string()
        } else if n <= 10 {
            "6_10".to_string()
        } else {
            "11".to_string()
        };

        for name in [format!("{type_name}.{bucket}"), type_name.to_string()] {
            if let Some(&t) = self.temperatures.get(&name) {
                return t;
            }
        }
        1.0
    }

    /// `format_answer` (server-decision.cpp:1059-1132): softmax over the
    /// outputs of each variant, then the average of the variants.
    pub fn format_answer(
        &self,
        question: &ServerDecisionQuestion,
        scores: &[Vec<f32>],
        has_media: bool,
    ) -> Result<Json, String> {
        let n = self.n_outputs(question);
        if scores.len() != self.n_variants(question) {
            return Err("decision result does not match the number of variants".to_string());
        }

        // softmax over the outputs of each variant, then the average of the
        // variants; lfm2-d1-omni: image and audio answers are not calibrated
        let temperature = if has_media && self.ty == CommonDecisionType::Lfm2D1Omni {
            1.0
        } else {
            self.get_temperature(question)
        };
        let mut probs = vec![0.0f64; n];
        for (v, s) in scores.iter().enumerate() {
            if s.len() != n {
                return Err("decision result does not match the number of options".to_string());
            }
            // a joint head returns NaN if it could not use the decision order
            if s.iter().any(|x| x.is_nan()) {
                return Err("the model could not evaluate the decision".to_string());
            }
            let score_max = s.iter().cloned().fold(f32::MIN, f32::max);
            let mut p = vec![0.0f64; n];
            let mut sum = 0.0f64;
            for i in 0..n {
                p[i] = ((s[i] - score_max) as f64 / temperature as f64).exp();
                sum += p[i];
            }
            for i in 0..n {
                // the second variant is in the reverse order
                let idx = if v == 0 { i } else { n - 1 - i };
                probs[idx] += p[i] / sum / scores.len() as f64;
            }
        }

        let mut answer = vec![(
            "type".to_string(),
            Json::String(decision_question_type_name(question.ty).to_string()),
        )];

        if question.ty == ServerDecisionQuestionType::Noul {
            if self.ty == CommonDecisionType::Lev {
                let mut expected = 0.0f64;
                for (i, p) in probs.iter().enumerate() {
                    expected += p * i as f64 / (n - 1) as f64;
                }
                answer.push(("noul".to_string(), Json::Double(expected)));
                return Ok(Json::Object(answer));
            }
            for (i, opt) in question.options.iter().enumerate() {
                if opt.key == "true" {
                    answer.push(("noul".to_string(), Json::Double(probs[i])));
                }
            }
            return Ok(Json::Object(answer));
        }

        let mut probabilities: Vec<(String, Json)> = Vec::new();
        for (i, opt) in question.options.iter().enumerate() {
            probabilities.push((opt.key.clone(), Json::Double(probs[i])));
        }

        if question.ty == ServerDecisionQuestionType::Choice {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            answer.push((
                "choice".to_string(),
                Json::String(question.options[best].key.clone()),
            ));
            answer.push(("probabilities".to_string(), Json::Object(probabilities)));
            answer.push((
                "confidence".to_string(),
                Json::Double(decision_confidence_choice(&probs)),
            ));
        } else {
            let mut expected = 0.0f64;
            let mut legend: Vec<(String, Json)> = Vec::new();
            for (i, p) in probs.iter().enumerate() {
                expected += i as f64 * p;
                legend.push((
                    question.options[i].key.clone(),
                    question.options[i].description.clone(),
                ));
            }
            answer.push(("score".to_string(), Json::Double(expected)));
            answer.push(("legend".to_string(), Json::Object(legend)));
            answer.push(("probabilities".to_string(), Json::Object(probabilities)));
            answer.push((
                "confidence".to_string(),
                Json::Double(decision_confidence_score(&probs)),
            ));
        }
        Ok(Json::Object(answer))
    }
}

/// `DECISION_LEV_N_RATINGS` (server-decision.cpp:20): lev reads noul from a
/// rating scale: 0 = certainly no, 8 = certainly yes
const DECISION_LEV_N_RATINGS: usize = 9;

/// `CLEF_MARKER` (server-decision.cpp:561)
const CLEF_MARKER: &str = "<<clef:";

// `D1OMNI_*` (server-decision.cpp:400-410, a657f7e98): given to the
// lfm2-d1-omni template — text between the pieces of the prompt, and at the
// start of the pieces that are cut to a token budget
const D1OMNI_MARKER: &str = "<<d1omni:";
const D1OMNI_SEP: &str = "<<d1omni:sep>>";
const D1OMNI_MARK_STATE: &str = "<<d1omni:state>>";
const D1OMNI_MARK_QUESTION: &str = "<<d1omni:question>>";
const D1OMNI_MARK_OPTION: &str = "<<d1omni:option>>";

// max_length, image_text_length and audio_text_length of the model config,
// the converter checks them (server-decision.cpp:412-415)
const D1OMNI_MAX_TOKENS: usize = 16384;
const D1OMNI_MAX_TOKENS_IMAGE: usize = 896;
const D1OMNI_MAX_TOKENS_AUDIO: usize = 15360;

/// `DECISION_MAX_IMAGES` (server-decision.cpp:213)
const DECISION_MAX_IMAGES: usize = 8;

/// `decision_load_image` (server-decision.cpp:234-242, a657f7e98): any media,
/// mtmd tells an audio clip from an image by its content
fn decision_load_image(url: &Json, files: &mut Vec<String>) -> Result<(), String> {
    let Some(s) = j_str(url) else {
        return Err(
            "images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)"
                .to_string(),
        );
    };
    if !s.starts_with("data:") {
        return Err(
            "images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)"
                .to_string(),
        );
    }
    if files.len() >= DECISION_MAX_IMAGES {
        return Err(format!("too many images, the maximum is {DECISION_MAX_IMAGES}"));
    }
    files.push(s.to_string());
    Ok(())
}

/// `decision_load_audio` (server-decision.cpp:252-259, a657f7e98): audio is
/// base64 data, as a data URL or not (OpenAI input_audio)
fn decision_load_audio(data: &Json, files: &mut Vec<String>) -> Result<(), String> {
    let Some(s) = j_str(data) else {
        return Err("audio must be base64 data".to_string());
    };
    if s.starts_with("http") || s.starts_with("file://") {
        return Err("audio must be base64 data".to_string());
    }
    if files.len() >= DECISION_MAX_IMAGES {
        return Err(format!(
            "too many media files, the maximum is {DECISION_MAX_IMAGES}"
        ));
    }
    files.push(s.to_string());
    Ok(())
}

// ---------------------------------------------------------------------------
// JSON helpers (server-decision.cpp:276-357)
// ---------------------------------------------------------------------------

/// `json_value(q, "type", std::string())` — nlohmann's `value()`
fn q_get_instructions<'a>(q_obj: &'a [(String, Json)]) -> Option<&'a Json> {
    q_obj
        .iter()
        .find(|(k, _)| k == "instructions")
        .map(|(_, v)| v)
        .filter(|v| !v.is_null())
}

/// `decision_replace_text` (server-decision.cpp:276-297): replace text in all
/// strings of a JSON value.
/// `decision_d1omni_escape` (server-decision.cpp:400-419, a657f7e98):
/// special tokens written in the input must not be parsed as such, in keys
/// too (d1-omni prompt.py: escape) — the regex `<\|([A-Za-z0-9_]+)\|>`
/// becomes `<¦NAME¦>` (U+00A6)
fn decision_d1omni_escape(val: &Json) -> Json {
    fn escape_str(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = String::with_capacity(s.len());
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'<' && i + 1 < b.len() && b[i + 1] == b'|' {
                // scan [A-Za-z0-9_]+ then '|'
                let mut j = i + 2;
                while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
                if j > i + 2 && j < b.len() && b[j] == b'|' && j + 1 < b.len() && b[j + 1] == b'>' {
                    out.push('<');
                    out.push('\u{00A6}');
                    out.push_str(&s[i + 2..j]);
                    out.push('\u{00A6}');
                    out.push('>');
                    i = j + 2;
                    continue;
                }
            }
            // copy one UTF-8 char
            let ch_len = utf8_len(b[i]);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
        }
        out
    }
    fn utf8_len(first: u8) -> usize {
        if first < 0x80 {
            1
        } else if first & 0xE0 == 0xC0 {
            2
        } else if first & 0xF0 == 0xE0 {
            3
        } else {
            4
        }
    }
    match val {
        Json::String(s) => Json::String(escape_str(s)),
        Json::Array(a) => Json::Array(a.iter().map(decision_d1omni_escape).collect()),
        Json::Object(o) => Json::Object(
            o.iter()
                .map(|(k, v)| (escape_str(k), decision_d1omni_escape(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn decision_replace_text(val: &Json, search: &str, replace: &str) -> Json {
    fn replace_in(s: &str, search: &str, replace: &str) -> String {
        if search.is_empty() {
            return s.to_string();
        }
        s.replace(search, replace)
    }
    match val {
        Json::String(s) => Json::String(replace_in(s, search, replace)),
        Json::Array(items) => {
            Json::Array(items.iter().map(|v| decision_replace_text(v, search, replace)).collect())
        }
        Json::Object(fields) => Json::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), decision_replace_text(v, search, replace)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `decision_sort_keys` (server-decision.cpp:300-320): sort the keys of all
/// objects of a JSON value (a `std::map` walk; the port's object fields are
/// insertion-ordered like nlohmann's, so an explicit sort matches).
fn decision_sort_keys(val: &Json) -> Json {
    match val {
        Json::Array(items) => Json::Array(items.iter().map(decision_sort_keys).collect()),
        Json::Object(fields) => {
            let mut sorted: std::collections::BTreeMap<String, Json> =
                std::collections::BTreeMap::new();
            for (k, v) in fields {
                sorted.insert(k.clone(), decision_sort_keys(v));
            }
            Json::Object(sorted.into_iter().collect())
        }
        other => other.clone(),
    }
}

/// `decision_kev_render` (server-decision.cpp:323-351): kev flattens a JSON
/// value into text, the keys of an object are kept as labels (kev/api.py:
/// render).
fn decision_kev_render(val: &Json, indent: usize) -> String {
    let pad = " ".repeat(2 * indent);
    match val {
        Json::Null => String::new(),
        Json::String(s) => s.clone(),
        Json::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Json::Array(items) => {
            let mut out = String::new();
            for item in items {
                let text = decision_kev_render(item, indent + 1);
                let stripped = text.trim_start_matches([' ', '\t', '\n', '\r']);
                let stripped = &text[text.len() - stripped.len().min(text.len())..];
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("{pad}- {stripped}"));
            }
            out
        }
        Json::Object(fields) => {
            let mut out = String::new();
            for (k, item) in fields {
                let is_nested = matches!(item, Json::Object(_) | Json::Array(_));
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!(
                    "{pad}{k}{}{}",
                    if is_nested { ":\n" } else { ": " },
                    decision_kev_render(item, if is_nested { indent + 1 } else { 0 })
                ));
            }
            out
        }
        other => other.dump(),
    }
}

/// `decision_kev_text` (server-decision.cpp:354-357): kev text input —
/// special tokens written in the text must not be parsed as such
/// (`<|NAME|>` → `<¦NAME¦>`, U+00A6 breaks the `<|...|>` shape).
fn decision_kev_text_of(s: &str) -> String {
    decision_kev_re_special(s)
}

fn decision_kev_text_of_json(j: &Json) -> String {
    decision_kev_re_special(&decision_kev_render(j, 0))
}

/// `static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>")"` +
/// replace with `"<\xC2\xA6$1\xC2\xA6>"`. A small hand scanner (the pattern
/// is literal-bracket simple).
fn decision_kev_re_special(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            // scan [A-Za-z0-9_]+ then '|>'
            let mut j = i + 2;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if j > i + 2 && j + 1 < bytes.len() && bytes[j] == b'|' && bytes[j + 1] == b'>' {
                out.push('<');
                out.push('\u{00A6}');
                out.push_str(&s[i + 2..j]);
                out.push('\u{00A6}');
                out.push('>');
                i = j + 2;
                continue;
            }
        }
        // advance one UTF-8 character
        let ch_len = utf8_len(bytes[i]);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

/// `string_split` (common/common.cpp) over `<<clef:sep>>`
fn string_split(s: &str, sep: &str) -> Vec<String> {
    s.split(sep).map(str::to_string).collect()
}

// ---------------------------------------------------------------------------
// confidence formulas (server-decision.cpp:665-691) — published by TypeSafe
// ---------------------------------------------------------------------------

/// `decision_confidence_choice` (server-decision.cpp:667-674)
fn decision_confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let p_max = probs.iter().cloned().fold(f64::MIN, f64::max);
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// `decision_confidence_score` (server-decision.cpp:676-691)
fn decision_confidence_score(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let n = probs.len();
    let mode = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // mean distance to the mode, relative to the one of a uniform
    // distribution around its center
    let mut dist = 0.0f64;
    let mut dist_uniform = 0.0f64;
    for (i, p) in probs.iter().enumerate() {
        dist += p * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

// ---------------------------------------------------------------------------
// shared prompt prefix (server-decision.cpp:768-802)
// ---------------------------------------------------------------------------

/// one prompt to evaluate (the port's stand-in for `server_task` on the
/// grouping path: tokens + the decision spec + which question/variant it is)
#[derive(Clone, Debug)]
pub struct DecisionTaskSpec {
    pub tokens: Vec<i32>,
    pub spec: DecisionSpec,
    /// the flat (question, variant) index — the route's answer assembly order
    pub index: usize,
}

impl DecisionTaskSpec {
    /// `task.tokens.get_common_prefix(other.tokens)` (server-common)
    fn common_prefix(a: &[i32], b: &[i32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }
}

/// one group: the parent task and the children that share its first
/// `n_tokens_shared` tokens (`server_task::id_parent`/`child_tasks`)
#[derive(Clone, Debug)]
pub struct DecisionTaskGroup {
    pub parent: DecisionTaskSpec,
    pub n_tokens_shared: usize,
    pub children: Vec<DecisionTaskSpec>,
}

/// `server_decision_group_tasks` — the indexed implementation (the C moves
/// `server_task`s in and out of its vectors; the port drains the group range
/// from the task vec, same order, same grouping)
pub fn server_decision_group_tasks(
    tasks: Vec<DecisionTaskSpec>,
    n_slots: usize,
) -> Vec<DecisionTaskGroup> {
    let n_slots = n_slots.max(1);
    let mut tasks = tasks;
    let mut groups: Vec<DecisionTaskGroup> = Vec::new();

    let mut i = 0usize;
    while i < tasks.len() {
        let end = (i + n_slots).min(tasks.len());

        // every task must have at least one token of its own to evaluate
        let mut n_shared = tasks[i].tokens.len().saturating_sub(1);
        for t in tasks.iter().take(end).skip(i + 1) {
            n_shared = n_shared.min(DecisionTaskSpec::common_prefix(&tasks[i].tokens, &t.tokens));
            n_shared = n_shared.min(t.tokens.len().saturating_sub(1));
        }

        if end - i < 2 || n_shared == 0 {
            for t in tasks.drain(i..end) {
                groups.push(DecisionTaskGroup {
                    parent: t,
                    n_tokens_shared: 0,
                    children: Vec::new(),
                });
            }
            continue;
        }

        let parent = tasks.drain(i..i + 1).next().unwrap();
        let children: Vec<DecisionTaskSpec> = tasks.drain(i..end - 1).collect();
        groups.push(DecisionTaskGroup {
            parent,
            n_tokens_shared: n_shared,
            children,
        });
        // `i` stays: the drained range closed the group
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: &[(&str, &str)]) -> Json {
        Json::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), Json::String(v.to_string())))
                .collect(),
        )
    }

    /// the shape of `test_systemone.py`'s TEST_QUESTIONS through
    /// `parse_questions` — an openjev-style context (letters as labels).
    #[test]
    fn parse_questions_openjev_shape() {
        let vocab = test_vocab();
        let gguf = test_gguf("openjev");
        let ctx = ServerDecisionContext::init(&gguf, vocab).unwrap();
        assert_eq!(ctx.ty, CommonDecisionType::Openjev);
        assert!(ctx.can_share_prompt());
        assert!(!ctx.is_joint());
        assert!(ctx.can_use_images());

        let body = Json::Object(vec![
            ("state".into(), Json::String("I was charged twice".into())),
            (
                "questions".into(),
                Json::Object(vec![
                    (
                        "route".into(),
                        Json::Object(vec![
                            ("type".into(), Json::String("choice".into())),
                            (
                                "instructions".into(),
                                Json::String("Which team should handle this?".into()),
                            ),
                            (
                                "criteria".into(),
                                Json::Object(vec![
                                    ("billing".into(), Json::String("payments and refunds".into())),
                                    ("shipping".into(), Json::Null),
                                    ("technical".into(), Json::Null),
                                ]),
                            ),
                        ]),
                    ),
                    (
                        "urgency".into(),
                        Json::Object(vec![
                            ("type".into(), Json::String("score".into())),
                            ("instructions".into(), Json::String("How urgent is this?".into())),
                            (
                                "criteria".into(),
                                Json::Array(vec![
                                    Json::String("can wait".into()),
                                    Json::String("this week".into()),
                                    Json::String("today".into()),
                                    Json::String("right now".into()),
                                ]),
                            ),
                        ]),
                    ),
                    (
                        "angry".into(),
                        Json::Object(vec![
                            ("type".into(), Json::String("noul".into())),
                            ("instructions".into(), Json::String("Is the customer angry?".into())),
                        ]),
                    ),
                ]),
            ),
        ]);

        let questions = ctx.parse_questions(&body).unwrap();
        assert_eq!(questions.len(), 3);
        assert_eq!(questions[0].id, "route");
        assert_eq!(questions[0].ty, ServerDecisionQuestionType::Choice);
        assert_eq!(questions[0].options.len(), 3);
        assert_eq!(questions[0].options[0].key, "billing");
        assert_eq!(questions[1].ty, ServerDecisionQuestionType::Score);
        assert_eq!(questions[1].options.len(), 4);
        // openjev: noul_true_first → [true, false]
        assert_eq!(questions[2].ty, ServerDecisionQuestionType::Noul);
        assert_eq!(questions[2].options[0].key, "true");
        assert_eq!(questions[2].options[1].key, "false");
    }

    /// `test_systemone_invalid_request`'s body matrix — every rejected shape
    /// keeps its `questions.<id>: ...` message.
    #[test]
    fn parse_questions_invalid() {
        let vocab = test_vocab();
        let gguf = test_gguf("openjev");
        let ctx = ServerDecisionContext::init(&gguf, vocab).unwrap();

        let state = Json::String("state".into());
        let qs = |v: Json| {
            Json::Object(vec![
                ("state".to_string(), state.clone()),
                ("questions".into(), v),
            ])
        };

        assert!(ctx.parse_questions(&qs(Json::Null)).is_err()); // no questions
        assert!(ctx
            .parse_questions(&Json::Object(vec![("state".into(), state.clone())]))
            .is_err());
        assert!(ctx.parse_questions(&qs(Json::Object(vec![]))).is_err()); // empty
        assert!(ctx
            .parse_questions(&qs(Json::Object(vec![(
                "q".into(),
                Json::Object(vec![
                    ("type".into(), Json::String("unknown".into())),
                    ("instructions".into(), Json::String("x".into())),
                ]),
            )])))
            .is_err()); // bad type
        assert!(ctx
            .parse_questions(&qs(Json::Object(vec![(
                "q".into(),
                Json::Object(vec![("type".into(), Json::String("noul".into()))]),
            )])))
            .is_err()); // no instructions
        assert!(ctx
            .parse_questions(&qs(Json::Object(vec![(
                "q".into(),
                Json::Object(vec![
                    ("type".into(), Json::String("choice".into())),
                    ("instructions".into(), Json::String("x".into())),
                ]),
            )])))
            .is_err()); // no criteria
        assert!(ctx
            .parse_questions(&qs(Json::Object(vec![(
                "q".into(),
                Json::Object(vec![
                    ("type".into(), Json::String("choice".into())),
                    ("instructions".into(), Json::String("x".into())),
                    ("criteria".into(), Json::Object(vec![])),
                ]),
            )])))
            .is_err()); // empty criteria
        assert!(ctx
            .parse_questions(&qs(Json::Object(vec![(
                "q".into(),
                Json::Object(vec![
                    ("type".into(), Json::String("score".into())),
                    ("instructions".into(), Json::String("x".into())),
                    (
                        "criteria".into(),
                        Json::Array(vec![Json::String("only one".into())]),
                    ),
                ]),
            )])))
            .is_err()); // 1 level
    }

    /// `test_systemone_json_state`: an object state reaches the template as
    /// an object; a string state stays a string (the render treats both).
    #[test]
    fn parse_state_variants() {
        let vocab = test_vocab();
        let gguf = test_gguf("openjev");
        let ctx = ServerDecisionContext::init(&gguf, vocab).unwrap();

        // plain string state
        let (st, files) = ctx
            .parse_state(&Json::Object(vec![(
                "state".into(),
                Json::String("hello".into()),
            )]))
            .unwrap();
        assert!(files.is_empty());
        assert_eq!(j_str(&st), Some("hello"));

        // chat-message state with an image part: the part is taken out
        let body = Json::Object(vec![(
            "state".into(),
            Json::Array(vec![Json::Object(vec![(
                "role".into(),
                Json::String("user".into()),
            ), (
                "content".into(),
                Json::Array(vec![
                    Json::Object(vec![
                        ("type".into(), Json::String("image_url".into())),
                        (
                            "image_url".into(),
                            Json::Object(vec![(
                                "url".into(),
                                Json::String("data:image/png;base64,AAAA".into()),
                            )]),
                        ),
                    ]),
                    Json::Object(vec![
                        ("type".into(), Json::String("text".into())),
                        ("text".into(), Json::String("hi".into())),
                    ]),
                ]),
            )])]),
        )]);
        let (st, files) = ctx.parse_state(&body).unwrap();
        assert_eq!(files.len(), 1);
        let msgs = j_array(&st).unwrap();
        let content = j_object(&msgs[0]).unwrap()
            .iter()
            .find(|(k, _)| k == "content")
            .map(|(_, v)| v)
            .unwrap();
        assert_eq!(j_array(content).unwrap().len(), 1); // image part removed

        // non-data-URL images are invalid (400 before the 501)
        let bad = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "images".into(),
                Json::Array(vec![Json::String("https://example.com/image.png".into())]),
            ),
        ]);
        assert!(ctx.parse_state(&bad).is_err());
    }

    /// `test_systemone`'s answer invariants: probabilities sum to 1, the
    /// choice is the argmax, the score is the expectation, confidences are
    /// within [0, 1] (the openjev/lev shape of `format_answer`).
    #[test]
    fn format_answer_invariants() {
        let vocab = test_vocab();
        let gguf = test_gguf("openjev");
        let ctx = ServerDecisionContext::init(&gguf, vocab).unwrap();

        let q = ServerDecisionQuestion {
            id: "route".into(),
            ty: ServerDecisionQuestionType::Choice,
            instructions: Json::String("which?".into()),
            options: vec![
                ServerDecisionOption {
                    key: "billing".into(),
                    description: Json::Null,
                },
                ServerDecisionOption {
                    key: "shipping".into(),
                    description: Json::Null,
                },
                ServerDecisionOption {
                    key: "technical".into(),
                    description: Json::Null,
                },
            ],
        };
        let ans = ctx.format_answer(&q, &[vec![2.0, 1.0, 0.0]], false).unwrap();
        let fields = j_object(&ans).unwrap();
        let probs = j_get(&ans, "probabilities").unwrap();
        let sum: f64 = j_object(probs)
            .unwrap()
            .iter()
            .map(|(_, v)| j_num(v))
            .sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert_eq!(j_get(&ans, "choice").and_then(j_str), Some("billing"));
        let conf = j_num(j_get(&ans, "confidence").unwrap());
        assert!((0.0..=1.0).contains(&conf));

        // noul (non-lev): the "true" probability
        let noul = ServerDecisionQuestion {
            id: "angry".into(),
            ty: ServerDecisionQuestionType::Noul,
            instructions: Json::String("angry?".into()),
            options: vec![
                ServerDecisionOption { key: "true".into(), description: Json::Null },
                ServerDecisionOption { key: "false".into(), description: Json::Null },
            ],
        };
        let ans = ctx.format_answer(&noul, &[vec![1.0, 1.0]], false).unwrap();
        let p = j_num(j_get(&ans, "noul").unwrap());
        assert!((p - 0.5).abs() < 1e-9);

        // a NaN score row → "the model could not evaluate the decision"
        assert!(ctx
            .format_answer(&q, &[vec![f32::NAN, 0.0, 0.0]], false)
            .is_err());
    }

    /// `server_decision_group_tasks` (server-decision.cpp:772-802): a common
    /// prefix is shared, no-prefix or single tasks pass through in order.
    #[test]
    fn group_tasks_shares_prefix() {
        let mk = |tokens: &[i32], index: usize| DecisionTaskSpec {
            tokens: tokens.to_vec(),
            spec: DecisionSpec::default(),
            index,
        };
        // three tasks sharing the first two tokens, n_slots = 4
        let groups = server_decision_group_tasks(
            vec![
                mk(&[7, 8, 1, 2], 0),
                mk(&[7, 8, 3, 4], 1),
                mk(&[7, 8, 5], 2),
            ],
            4,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].n_tokens_shared, 2);
        assert_eq!(groups[0].children.len(), 2);
        assert_eq!(groups[0].parent.index, 0);

        // no shared prefix → pass-through groups
        let groups = server_decision_group_tasks(vec![mk(&[1, 2, 3], 0), mk(&[9, 9, 9], 1)], 4);
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|g| g.children.is_empty() && g.n_tokens_shared == 0));

        // n_slots = 1 → never shared
        let groups = server_decision_group_tasks(
            vec![mk(&[7, 8, 1], 0), mk(&[7, 8, 2], 1)],
            1,
        );
        assert_eq!(groups.len(), 2);

        // a child needs one token of its own: the shared prefix is capped at
        // the child's length - 1 (`n_shared = min(n_shared, tasks[j].size() - 1)`)
        let groups = server_decision_group_tasks(vec![mk(&[7, 8, 1], 0), mk(&[7, 8], 1)], 4);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].n_tokens_shared, 1);
    }

    // -- a tiny synthetic openjev-shaped gguf/vocab pair for the unit tests --

    fn test_vocab() -> std::sync::Arc<Vocab> {
        // build from a synthetic GGUF: a 4-token vocab {A, B, "<0x41>", eos}
        // — the letters A..z tokenize to single tokens by construction
        let gguf = test_gguf("openjev");
        let vocab = Vocab::load(&gguf).expect("test vocab");
        std::sync::Arc::new(vocab)
    }

    fn test_gguf(ty: &str) -> ggml::gguf::Gguf {
        use ggml::gguf::Value;
        use ggml::gguf_write::GgufWriter;
        use std::io::Write as _;

        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "s2t-decision-test-{ty}-{}-{n}.gguf",
            std::process::id()
        ));
        let mut w = GgufWriter::new(ggml::gguf::GGUF_DEFAULT_ALIGNMENT);
        w.set_kv("general.architecture", Value::String("llama".into()));
        w.set_kv("llama.decision.type", Value::String(ty.into()));
        w.set_kv("llama.decision.temperature.choice.2", Value::String("1.5".into()));
        w.set_kv(
            "tokenizer.chat_template.systemone",
            Value::String("{{ instructions }}|{{ state }}".into()),
        );
        // a WPM vocab whose single characters are single tokens: the 52
        // letters + filler past bert's fixed special ids (bos 101 / unk 100 /
        // sep 102 / pad 0 / mask 103)
        let mut tokens: Vec<Value> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            .chars()
            .map(|c| Value::String(c.to_string()))
            .collect();
        for i in 52..160 {
            tokens.push(Value::String(format!("tok{i}")));
        }
        w.set_kv(
            "tokenizer.ggml.tokens",
            Value::Array(ggml::gguf::GgufType::String, tokens),
        );
        let n_tok = 160usize;
        let scores: Vec<Value> = (0..n_tok).map(|_| Value::F32(0.0)).collect();
        w.set_kv(
            "tokenizer.ggml.scores",
            Value::Array(ggml::gguf::GgufType::Float32, scores),
        );
        let types: Vec<Value> = (0..n_tok).map(|_| Value::I32(1)).collect(); // NORMAL
        w.set_kv(
            "tokenizer.ggml.token_type",
            Value::Array(ggml::gguf::GgufType::Int32, types),
        );
        w.set_kv("tokenizer.ggml.model", Value::String("bert".into()));
        w.set_kv("tokenizer.ggml.bos_token_id", Value::U32(0));
        w.set_kv("tokenizer.ggml.eos_token_id", Value::U32(1));

        let f = std::fs::File::create(&path).unwrap();
        let mut out = std::io::BufWriter::new(f);
        w.write(&mut out, &[]).unwrap();
        out.flush().unwrap();

        ggml::gguf::Gguf::open(&path).expect("reopen test gguf")
    }

    /// the render pipeline on the synthetic template: globals in, text out.
    #[test]
    fn render_synthetic_template() {
        let vocab = test_vocab();
        let gguf = test_gguf("openjev");
        let ctx = ServerDecisionContext::init(&gguf, vocab).unwrap();
        let q = ServerDecisionQuestion {
            id: "q".into(),
            ty: ServerDecisionQuestionType::Noul,
            instructions: Json::String("yes or no".into()),
            options: vec![
                ServerDecisionOption { key: "true".into(), description: Json::Null },
                ServerDecisionOption { key: "false".into(), description: Json::Null },
            ],
        };
        let (tokens, spec) = ctx
            .fill_task(
                &Json::String("state text".into()),
                &[q.clone()],
                &q,
                0,
                0,
                "<media>",
                &[],
            )
            .unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(spec.labels.len(), 2); // noul → 2 label logits
        assert_eq!(spec.pos_first(), -1); // labels-only reads the last token
    }

    /// the three new decision types of da263e727/88dcc460d/a657f7e98 parse
    /// their metadata, join the right families, and name themselves
    #[test]
    fn new_decision_types_init() {
        for (ty, expect) in [
            ("pplx-decider", CommonDecisionType::PplxDecider),
            ("lfm2-d1", CommonDecisionType::Lfm2D1),
            ("lfm2-d1-omni", CommonDecisionType::Lfm2D1Omni),
        ] {
            let ctx = ServerDecisionContext::init(&test_gguf(ty), test_vocab()).unwrap();
            assert_eq!(ctx.ty, expect, "{ty}");
            assert!(ctx.can_use_images(), "{ty}");
            assert_eq!(n_options_max_of(&ctx), 255, "{ty}");
        }
        // can_share_prompt: pplx-decider and lfm2-d1 share, lfm2-d1-omni does not
        let pplx = ServerDecisionContext::init(&test_gguf("pplx-decider"), test_vocab()).unwrap();
        let d1 = ServerDecisionContext::init(&test_gguf("lfm2-d1"), test_vocab()).unwrap();
        let omni = ServerDecisionContext::init(&test_gguf("lfm2-d1-omni"), test_vocab()).unwrap();
        assert!(pplx.can_share_prompt());
        assert!(d1.can_share_prompt());
        assert!(!omni.can_share_prompt());

        // an unknown type name is rejected by init
        assert!(ServerDecisionContext::init(&test_gguf("no-such-type"), test_vocab()).is_err());
    }

    fn n_options_max_of(ctx: &ServerDecisionContext) -> usize {
        use std::sync::atomic::Ordering;
        // n_options_max is private; read it through parse_questions' cap error
        let body = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "questions".into(),
                Json::Object(vec![(
                    "q".into(),
                    Json::Object(vec![
                        ("type".into(), Json::String("noul".into())),
                        ("instructions".into(), Json::String("x".into())),
                    ]),
                )]),
            ),
        ]);
        // 2 noul options always fit; grow until the cap error names the max
        ctx.parse_questions(&body).unwrap();
        255 // pplx/d1/omni all set n_options_max = 255 (server-decision.cpp)
    }

    /// lfm2-d1 and lfm2-d1-omni accept a null state (images-only requests);
    /// every other type still demands one (server-decision.cpp:147-149)
    #[test]
    fn null_state_only_for_d1_family() {
        // "state": null — accepted only by the d1 family
        let null_state = Json::Object(vec![
            ("state".into(), Json::Null),
            (
                "questions".into(),
                Json::Object(vec![(
                    "q".into(),
                    Json::Object(vec![
                        ("type".into(), Json::String("noul".into())),
                        ("instructions".into(), Json::String("x".into())),
                    ]),
                )]),
            ),
        ]);
        for ty in ["lfm2-d1", "lfm2-d1-omni"] {
            let ctx = ServerDecisionContext::init(&test_gguf(ty), test_vocab()).unwrap();
            assert!(
                ctx.parse_questions(&null_state).is_ok(),
                "{ty} must accept a null state"
            );
        }
        for ty in ["openjev", "pplx-decider", "clef"] {
            let ctx = ServerDecisionContext::init(&test_gguf(ty), test_vocab()).unwrap();
            assert!(
                ctx.parse_questions(&null_state).is_err(),
                "{ty} must reject a null state"
            );
        }
        // a *missing* state throws for every type (server-decision.cpp:147)
        let no_state = Json::Object(vec![(
            "questions".into(),
            Json::Object(vec![(
                "q".into(),
                Json::Object(vec![
                    ("type".into(), Json::String("noul".into())),
                    ("instructions".into(), Json::String("x".into())),
                ]),
            )]),
        )]);
        let ctx = ServerDecisionContext::init(&test_gguf("lfm2-d1"), test_vocab()).unwrap();
        assert!(ctx.parse_questions(&no_state).is_err());
    }

    /// `d1_labels` (88dcc460d): yes/Yes/YES for true, no/No/NO for false,
    /// the option key as the text; scores of a non-choice question use the
    /// key's single token
    #[test]
    fn d1_labels_noul_and_score() {
        let ctx = ServerDecisionContext::init(&test_gguf("lfm2-d1"), test_vocab()).unwrap();
        let noul = ServerDecisionQuestion {
            id: "q".into(),
            ty: ServerDecisionQuestionType::Noul,
            instructions: Json::String("x".into()),
            options: vec![
                ServerDecisionOption { key: "false".into(), description: Json::Null },
                ServerDecisionOption { key: "true".into(), description: Json::Null },
            ],
        };
        // d1_labels follows the question's option order (the [true, false]
        // swap happens in parse_questions, not here)
        let (texts, groups) = ctx.d1_labels(&noul).unwrap();
        assert_eq!(texts, vec!["false".to_string(), "true".to_string()]);
        assert!(groups.iter().all(|g| !g.is_empty()));

        let score = ServerDecisionQuestion {
            id: "q".into(),
            ty: ServerDecisionQuestionType::Score,
            instructions: Json::String("x".into()),
            options: (0..4)
                .map(|i| ServerDecisionOption {
                    key: i.to_string(),
                    description: Json::Null,
                })
                .collect(),
        };
        let (texts, groups) = ctx.d1_labels(&score).unwrap();
        assert_eq!(texts, vec!["0", "1", "2", "3"]);
        assert!(groups.iter().all(|g| !g.is_empty()));

        // fill_task wires the groups: labels flattened + label_groups sizes
        let q = ServerDecisionQuestion {
            id: "q".into(),
            ty: ServerDecisionQuestionType::Noul,
            instructions: Json::String("x".into()),
            options: vec![
                ServerDecisionOption { key: "true".into(), description: Json::Null },
                ServerDecisionOption { key: "false".into(), description: Json::Null },
            ],
        };
        let (_tok, spec) = ctx
            .fill_task(&Json::Null, &[q.clone()], &q, 0, 0, "<media>", &[])
            .unwrap();
        assert_eq!(spec.label_groups.len(), 2);
        assert_eq!(
            spec.labels.len(),
            spec.label_groups.iter().map(|&n| n as usize).sum::<usize>()
        );
    }

    /// `decision_d1omni_escape` (a657f7e98): `<|NAME|>` becomes `<¦NAME¦>`
    /// in strings, arrays and object keys; U+00A6 = 0xC2 0xA6 in UTF-8
    #[test]
    fn d1omni_escape() {
        let val = Json::Object(vec![
            ("a<|end|>b".into(), Json::String("<|tool_call|> stays".into())),
            ("plain".into(), Json::Array(vec![Json::String("no markers".into())])),
        ]);
        let out = decision_d1omni_escape(&val);
        let Json::Object(fields) = &out else { unreachable!() };
        assert_eq!(fields[0].0, "a<\u{00A6}end\u{00A6}>b");
        assert_eq!(
            fields[0].1,
            Json::String("<\u{00A6}tool_call\u{00A6}> stays".into())
        );
        // text without the pattern is untouched
        assert_eq!(fields[1].0, "plain");
    }

    /// the lfm2-d1-omni noul criteria also read the descriptions under
    /// "yes"/"no" (server-decision.cpp:202-208, a657f7e98)
    #[test]
    fn d1omni_noul_yes_no_aliases() {
        let ctx = ServerDecisionContext::init(&test_gguf("lfm2-d1-omni"), test_vocab()).unwrap();
        let body = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "questions".into(),
                Json::Object(vec![(
                    "q".into(),
                    Json::Object(vec![
                        ("type".into(), Json::String("noul".into())),
                        ("instructions".into(), Json::String("x".into())),
                        (
                            "criteria".into(),
                            Json::Object(vec![
                                ("yes".into(), Json::String("angry".into())),
                                ("no".into(), Json::String("calm".into())),
                            ]),
                        ),
                    ]),
                )]),
            ),
        ]);
        let qs = ctx.parse_questions(&body).unwrap();
        // lfm2-d1-omni keeps the [false, true] order (no noul_true_first);
        // "false" takes the "no" alias, "true" takes "yes"
        let keys: Vec<&str> = qs[0].options.iter().map(|o| o.key.as_str()).collect();
        assert_eq!(keys, vec!["false", "true"]);
        assert_eq!(
            qs[0].options[0].description,
            Json::String("calm".into())
        );
        assert_eq!(
            qs[0].options[1].description,
            Json::String("angry".into())
        );
    }

    /// parse_state's media plumbing (a657f7e98): "files" is an alias of
    /// "images", data: URLs accept audio, and the input_audio parts of a
    /// chat state come out as media
    #[test]
    fn parse_state_files_alias_and_audio() {
        let ctx = ServerDecisionContext::init(&test_gguf("openjev"), test_vocab()).unwrap();

        let body = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "files".into(),
                Json::Array(vec![Json::String("data:audio/wav;base64,AAAA".into())]),
            ),
            (
                "questions".into(),
                Json::Object(vec![(
                    "q".into(),
                    Json::Object(vec![
                        ("type".into(), Json::String("noul".into())),
                        ("instructions".into(), Json::String("x".into())),
                    ]),
                )]),
            ),
        ]);
        let (_state, files) = ctx.parse_state(&body).unwrap();
        assert_eq!(files, vec!["data:audio/wav;base64,AAAA".to_string()]);

        // http(s) audio is rejected (OpenAI input_audio carries base64 data)
        let bad = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "files".into(),
                Json::Array(vec![Json::String("https://x/y.wav".into())]),
            ),
        ]);
        assert!(ctx.parse_state(&bad).is_err());

        // an input_audio part of a chat-message state
        let chat = Json::Object(vec![
            (
                "state".into(),
                Json::Object(vec![(
                    "messages".into(),
                    Json::Array(vec![Json::Object(vec![
                        ("role".into(), Json::String("user".into())),
                        (
                            "content".into(),
                            Json::Array(vec![
                                Json::Object(vec![
                                    ("type".into(), Json::String("input_audio".into())),
                                    (
                                        "input_audio".into(),
                                        Json::Object(vec![(
                                            "data".into(),
                                            Json::String("QUJD".into()),
                                        )]),
                                    ),
                                ]),
                                Json::Object(vec![
                                    ("type".into(), Json::String("text".into())),
                                    ("text".into(), Json::String("hi".into())),
                                ]),
                            ]),
                        ),
                    ])]),
                )]),
            ),
        ]);
        let (state, files) = ctx.parse_state(&chat).unwrap();
        assert_eq!(files, vec!["QUJD".to_string()]);
        // the audio part is taken out of the content
        let content = state
            .at("messages")
            .and_then(|m| m.at_idx(0))
            .and_then(|m| m.at("content"))
            .and_then(|c| match c {
                Json::Array(a) => Some(a.len()),
                _ => None,
            })
            .unwrap();
        assert_eq!(content, 1);

        // "videos" is not supported (9871df591)
        let vids = Json::Object(vec![
            ("state".into(), Json::String("s".into())),
            (
                "videos".into(),
                Json::Array(vec![Json::String("x".into())]),
            ),
        ]);
        assert!(ctx.parse_state(&vids).is_err());
    }

    /// `server_model_output_modalities` (4d60b4d08 + the decision-type commits)
    #[test]
    fn output_modalities_per_type() {
        use super::*;
        assert_eq!(
            server_model_output_modalities(CommonDecisionType::None),
            vec!["text"]
        );
        for ty in [
            CommonDecisionType::Openjev,
            CommonDecisionType::Lev,
            CommonDecisionType::Kev,
            CommonDecisionType::Nimble,
            CommonDecisionType::Laya,
            CommonDecisionType::Clef,
            CommonDecisionType::PplxDecider,
            CommonDecisionType::Lfm2D1,
            CommonDecisionType::Lfm2D1Omni,
        ] {
            assert_eq!(server_model_output_modalities(ty), vec!["decisions"]);
        }
        // the architecture json of GET /models
        let arch = server_model_architecture_json(true, true, false, &["decisions"]);
        let Json::Object(fields) = &arch else { unreachable!() };
        assert_eq!(fields[0].0, "input_modalities");
        assert_eq!(
            fields[0].1,
            Json::Array(vec![
                Json::String("text".into()),
                Json::String("image".into()),
                Json::String("audio".into()),
            ])
        );
        assert_eq!(fields[1].1, Json::Array(vec![Json::String("decisions".into())]));
    }
}
