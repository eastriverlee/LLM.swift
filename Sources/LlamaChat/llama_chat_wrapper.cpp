#include "llama_chat_wrapper.h"
#include "chat.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

struct llm_chat_session {
    common_chat_templates_ptr templates;
    common_chat_parser_params parserParams;
};

static char * copyString(const std::string & text) {
    char * result = static_cast<char *>(std::malloc(text.size() + 1));
    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

static char * errorJSON(const std::exception & exception) {
    return copyString(json{ { "error", exception.what() } }.dump());
}

llm_chat_session * llm_chat_session_create(const void * model) {
    try {
        auto session = new llm_chat_session();
        session->templates = common_chat_templates_init(static_cast<const llama_model *>(model), "");
        return session;
    } catch (...) {
        return nullptr;
    }
}

void llm_chat_session_free(llm_chat_session * session) {
    delete session;
}

char * llm_chat_render(llm_chat_session * session, const char * messagesJSON, const char * toolsJSON, bool addAssistantPrefix, bool enableThinking) {
    try {
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messagesJSON));
        if (toolsJSON != nullptr) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(toolsJSON));
        }
        inputs.add_generation_prompt = addAssistantPrefix;
        inputs.enable_thinking = enableThinking;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

        auto params = common_chat_templates_apply(session->templates.get(), inputs);

        common_chat_parser_params parserParams(params);
        parserParams.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        parserParams.parser.load(params.parser);
        session->parserParams = std::move(parserParams);

        json result = {
            { "prompt", params.prompt },
            { "additionalStops", params.additional_stops },
        };
        return copyString(result.dump());
    } catch (const std::exception & exception) {
        return errorJSON(exception);
    }
}

char * llm_chat_parse(const llm_chat_session * session, const char * text, bool isPartial) {
    try {
        auto message = common_chat_parse(text, isPartial, session->parserParams);
        return copyString(message.to_json_oaicompat().dump());
    } catch (const std::exception & exception) {
        return errorJSON(exception);
    }
}

char * llm_chat_grammar_from_json_schema(const char * schemaJSON) {
    try {
        return copyString(json_schema_to_grammar(json::parse(schemaJSON)));
    } catch (...) {
        return nullptr;
    }
}

void llm_chat_string_free(char * string) {
    std::free(string);
}
