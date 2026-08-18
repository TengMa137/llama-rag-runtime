#ifndef LRS_BRIDGE_H
#define LRS_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lrs_index lrs_index;

typedef struct {
    const char* database_path;
    const char* embedding_host;
    uint16_t embedding_port;
    size_t embedding_dimension;
    int deterministic_embeddings;
    const char* embedding_model;
    size_t max_input_tokens;
    size_t reserved_tokens;
    size_t (*measure_tokens)(void* context, const char* text, size_t length);
    void* token_measurer_context;
} lrs_index_options;

int lrs_index_open(const lrs_index_options* options, lrs_index** out, char** error);
int lrs_index_stage_upsert(const lrs_index* current, const lrs_index_options* options,
                           const char* document_id, const char* title, const char* content,
                           lrs_index** candidate, int* unchanged, char** error);
int lrs_index_stage_upsert_with_metadata(const lrs_index* current, const lrs_index_options* options,
                                         const char* document_id, const char* title,
                                         const char* content, const char* const* metadata_keys,
                                         const char* const* metadata_values, size_t metadata_count,
                                         lrs_index** candidate, int* unchanged, char** error);
int lrs_index_stage_delete(const lrs_index* current, const lrs_index_options* options,
                           const char* document_id, lrs_index** candidate, int* deleted,
                           char** error);
int lrs_index_search_json(const lrs_index* index, const char* query, const char* mode, size_t top_k,
                          char** json, char** error);
int lrs_index_search_filtered_json(const lrs_index* index, const char* query, const char* mode,
                                   size_t top_k, const char* const* filter_keys,
                                   const char* const* filter_values, size_t filter_count,
                                   char** json, char** error);
void lrs_index_destroy(lrs_index* index);
void lrs_string_destroy(char* value);

#ifdef __cplusplus
}
#endif
#endif
