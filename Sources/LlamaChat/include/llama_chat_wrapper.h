#ifndef LLAMA_CHAT_WRAPPER_H
#define LLAMA_CHAT_WRAPPER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llm_chat_session llm_chat_session;

llm_chat_session * llm_chat_session_create(const void * model);
void llm_chat_session_free(llm_chat_session * session);
char * llm_chat_render(llm_chat_session * session, const char * messagesJSON, const char * toolsJSON, bool addAssistantPrefix, bool enableThinking);
char * llm_chat_parse(const llm_chat_session * session, const char * text, bool isPartial);
char * llm_chat_grammar_from_json_schema(const char * schemaJSON);
void llm_chat_string_free(char * string);

#ifdef __cplusplus
}
#endif

#endif
