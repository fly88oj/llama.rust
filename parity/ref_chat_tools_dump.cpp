// ref_chat_tools_dump.cpp — reference-side ground truth for the chat
// tool-calling port (agent: chat tools). Reads a JSON "cases" file and, for
// each case, runs the pinned `common_chat_templates_apply` (jinja path) and
// `common_chat_parse` with a pinned clock, and dumps everything the port
// compares against:
//
//   parity/chat_tools_ref.json   (via gen_chat_tools_ref.sh)
//
// Input file format (see parity/chat_tools_cases.json):
//   {
//     "templates": [{"name": "...", "src": "<jinja template>"}],
//     "cases": [{
//        "template": "name",
//        "messages": [... oaicompat ...],
//        "tools": [...] | null,
//        "tool_choice": "auto"|"required"|"none",
//        "parallel_tool_calls": false,
//        "add_generation_prompt": true,
//        "reasoning_format": "none",
//        "json_schema": "" | "<schema json>",
//        "parse_input": "optional raw model output to run common_chat_parse on",
//        "parse_partial": false
//     }]
//   }
//
// Build (see gen_chat_tools_ref.sh):
//   g++ -O2 -std=c++17 ref_chat_tools_dump.cpp -Ipinned/src ... -lllama-common -lllama

#include "chat.h"
#include "common.h"
#include "json.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = common_json;

// pinned clock so `datetime`/`date_string` render deterministically (run the
// probe with TZ=UTC: the reference formats with std::localtime)
static const int64_t PINNED_EPOCH = 1727000000;

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s CASES_JSON OUT_JSON\n", argv[0]);
        return 1;
    }
    json input = json::parse(read_file(argv[1]));
    json out    = json::array();

    std::map<std::string, common_chat_templates_ptr> tmpls;
    for (const auto & t : input.at("templates").items()) {
        const std::string name = t.value().at("name");
        const std::string src  = t.value().at("src");
        // model == nullptr: the template override keeps init() model-free
        // (chat.cpp:765-779); bos/eos tokens are not referenced by these
        // templates
        tmpls[name] = common_chat_templates_init(/* model = */ nullptr, src);
    }

    for (const auto & c : input.at("cases").items()) {
        const json & cj = c.value();
        json result;

        common_chat_templates_inputs inputs;
        inputs.use_jinja              = true;
        inputs.messages               = common_chat_msgs_parse_oaicompat(cj.at("messages"));
        inputs.tools                  = common_chat_tools_parse_oaicompat(cj.value("tools", json()));
        inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(cj.value("tool_choice", "auto"));
        inputs.parallel_tool_calls    = cj.value("parallel_tool_calls", false);
        inputs.add_generation_prompt  = cj.value("add_generation_prompt", true);
        inputs.reasoning_format       = common_reasoning_format_from_name(cj.value("reasoning_format", "none"));
        inputs.json_schema            = cj.value("json_schema", std::string());
        inputs.enable_thinking        = cj.value("enable_thinking", true);
        inputs.now = std::chrono::system_clock::time_point(std::chrono::seconds(PINNED_EPOCH));

        common_chat_templates_ptr & tmpls_case = tmpls.at(cj.at("template").get<std::string>());

        try {
            common_chat_params params = common_chat_templates_apply(tmpls_case.get(), inputs);
            result["format"]            = common_chat_format_name(params.format);
            result["prompt"]            = params.prompt;
            result["generation_prompt"] = params.generation_prompt;
            result["grammar"]           = params.grammar;
            result["grammar_lazy"]      = params.grammar_lazy;
            json triggers = json::array();
            for (const auto & t : params.grammar_triggers) {
                triggers.push_back({{"type", (int) t.type}, {"value", t.value}});
            }
            result["grammar_triggers"] = triggers;
            result["preserved_tokens"] = params.preserved_tokens;
            result["additional_stops"] = params.additional_stops;
            result["parser"]           = params.parser;
            {
                // canonical structural dump: allocation ids in the serialized
                // arena depend on the C++ evaluation order; the dump from the
                // root is deterministic and graph-structural
                common_peg_arena arena;
                arena.load(params.parser);
                result["parser_dump"] = arena.dump(arena.root());
            }
            result["supports_thinking"] = params.supports_thinking;
            result["thinking_start_tag"] = params.thinking_start_tag;
            result["thinking_end_tags"]  = params.thinking_end_tags;
            result["message_delimiters"] = params.message_delimiters.to_json();
            result["ok"] = true;

            if (cj.contains("parse_input")) {
                common_chat_parser_params parse_params(params);
                // the server loads the serialized arena into the parse params
                // before parsing (see server-context.cpp chat slot handling)
                common_peg_arena arena;
                arena.load(params.parser);
                parse_params.parser = arena;
                common_chat_msg msg = common_chat_parse(cj.at("parse_input").get<std::string>(),
                                                        cj.value("parse_partial", false), parse_params);
                result["parse"] = common_chat_msgs_to_json_oaicompat({msg}).at(0);
            }
        } catch (const std::exception & e) {
            result["ok"] = false;
            result["error"] = e.what();
        }
        out.push_back(result);
    }

    std::ofstream of(argv[2]);
    of << out.dump() << std::endl;
    return 0;
}
