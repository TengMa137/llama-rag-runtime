/* Stable C ABI for the owned retrieval core. */
#ifndef RAG_CORE_C_API_H
#define RAG_CORE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAG_C_API_VERSION_MAJOR 1
#define RAG_C_API_VERSION_MINOR 1
#define RAG_C_ABI_VERSION 1u

typedef enum rag_status {
    RAG_OK = 0,
    RAG_ERR_NOT_FOUND = 1,
    RAG_ERR_INVALID_ARGUMENT = 2,
    RAG_ERR_DIMENSION_MISMATCH = 3,
    RAG_ERR_IO = 4,
    RAG_ERR_PARSE = 5,
    RAG_ERR_TRANSPORT = 6,
    RAG_ERR_UNAVAILABLE = 7,
    RAG_ERR_EMPTY_CORPUS = 8,
    RAG_ERR_ALREADY_EXISTS = 9,
    RAG_ERR_CORRUPT = 10,
    RAG_ERR_UNKNOWN = 255
} rag_status;

typedef struct rag_engine rag_engine;
typedef struct rag_embedder rag_embedder;
typedef struct rag_results rag_results;

typedef struct rag_engine_options {
    uint32_t abi_version;
    size_t struct_size;
    size_t reader_threads;
    size_t writer_threads;
    size_t memory_budget_bytes;
} rag_engine_options;

typedef struct rag_local_http_embedder_options {
    uint32_t abi_version;
    size_t struct_size;
    const char* host;
    uint16_t port;
    const char* path;
    const char* model;
    size_t dimension;
    uint32_t connect_timeout_ms;
    uint32_t read_timeout_ms;
    size_t max_response_bytes;
    size_t concurrency;
} rag_local_http_embedder_options;

const char* rag_version(void);
const char* rag_last_error(void);

rag_embedder* rag_embedder_hash(size_t dimension);
rag_status rag_embedder_local_http_create(const rag_local_http_embedder_options* options,
                                          rag_embedder** out);
void rag_embedder_free(rag_embedder* embedder);

rag_engine* rag_engine_new(void);
rag_status rag_engine_create(const rag_engine_options* options, rag_engine** out);
void rag_engine_free(rag_engine* engine);
rag_status rag_engine_set_embedder(rag_engine* engine, rag_embedder* embedder);
rag_status rag_engine_add(rag_engine* engine, const char* uri, const char* text, const char* title,
                          const char** metadata_keys, const char** metadata_values,
                          size_t metadata_count, uint32_t* out_document_id);
rag_status rag_engine_build(rag_engine* engine);
rag_status rag_engine_search(rag_engine* engine, const char* query, size_t count,
                             const char** metadata_keys, const char** metadata_values,
                             size_t metadata_count, rag_results** out);

size_t rag_results_count(const rag_results* results);
float rag_results_score(const rag_results* results, size_t index);
uint32_t rag_results_chunk_id(const rag_results* results, size_t index);
uint32_t rag_results_doc_id(const rag_results* results, size_t index);
uint32_t rag_results_start_line(const rag_results* results, size_t index);
uint32_t rag_results_end_line(const rag_results* results, size_t index);
const char* rag_results_uri(const rag_results* results, size_t index);
const char* rag_results_text(const rag_results* results, size_t index);
const char* rag_results_context(const rag_results* results, size_t index);
void rag_results_free(rag_results* results);

rag_status rag_engine_save(rag_engine* engine, const char* path);
rag_engine* rag_engine_load(const char* path, rag_status* out_status);
void rag_string_free(char* string);

#ifdef __cplusplus
}
#endif
#endif
