// repro: run apply for case N
use llama::chat_tools::*;
use llama::json_schema::Json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cases_text = std::fs::read_to_string("parity/chat_tools_cases.json").unwrap();
    let all = Json::parse(&cases_text).unwrap();
    let case = all.at("cases").unwrap().at_idx(n).unwrap().clone();
    let name = case.at("template").unwrap().get_str().unwrap().to_string();
    let src = all
        .at("templates")
        .unwrap()
        .iter()
        .find(|t| t.at("name").and_then(|v| v.get_str().ok()) == Some(name.as_str()))
        .and_then(|t| t.at("src"))
        .unwrap()
        .get_str()
        .unwrap()
        .to_string();
    let tmpls = ChatTemplates::init(&ChatTemplatesInit {
        chat_template_override: src,
        chat_template_tool_use: String::new(),
        bos_token: String::new(),
        eos_token: String::new(),
        add_bos: false,
        add_eos: false,
    })
    .unwrap();
    let messages = msgs_parse_oaicompat(case.at("messages").unwrap()).unwrap();
    let tools = tools_parse_oaicompat(case.at("tools").unwrap_or(&Json::Null)).unwrap();
    let tc = tool_choice_parse_oaicompat(
        case.at("tool_choice")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("auto"),
    )
    .unwrap();
    let inputs = TemplatesInputs {
        messages,
        tools,
        tool_choice: tc,
        add_generation_prompt: matches!(case.at("add_generation_prompt"), Some(Json::Bool(true))),
        now: Some(1727000000),
        ..TemplatesInputs::default()
    };
    let p = chat_templates_apply(&tmpls, &inputs).unwrap();
    println!("{}", p.prompt);
}
