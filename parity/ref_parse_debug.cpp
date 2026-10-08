// ref_parse_debug.cpp — one-off: build the specialized parser for a template
// and dump the parse failure position for a given model output.
#include "chat.h"
#include "common.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using json = common_json;

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s TEMPLATE_FILE PARSE_INPUT TOOLS_JSON RF\n", argv[0]);
        return 1;
    }
    const std::string src         = read_file(argv[1]);
    const std::string parse_input = argv[2];
    json tools = json::parse(argv[3]);
    const std::string rf = argv[4];

    common_chat_templates_ptr tmpls = common_chat_templates_init(nullptr, src);
    common_chat_templates_inputs inputs;
    inputs.use_jinja       = true;
    inputs.messages        = common_chat_msgs_parse_oaicompat(json::parse(R"([{"role":"user","content":"hi"}])"));
    inputs.tools           = tools.is_array() ? common_chat_tools_parse_oaicompat(tools) : std::vector<common_chat_tool>{};
    inputs.reasoning_format = common_reasoning_format_from_name(rf);
    inputs.now = std::chrono::system_clock::time_point(std::chrono::seconds(1727000000));

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
    printf("generation_prompt: %s\n", params.generation_prompt.c_str());

    common_peg_arena arena;
    arena.load(params.parser);
    common_chat_parser_params pp(params);
    pp.parser   = arena;
    pp.debug    = true;
    try {
        common_chat_msg msg = common_chat_parse(parse_input, false, pp);
        printf("parsed ok: %s\n", common_chat_msgs_to_json_oaicompat({msg})[0].dump().c_str());
    } catch (const std::exception & e) {
        printf("parse failed: %s\n", e.what());
    }
    return 0;
}
