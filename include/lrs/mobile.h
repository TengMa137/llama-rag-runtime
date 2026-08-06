#ifndef LRS_MOBILE_H
#define LRS_MOBILE_H

#include <stddef.h>

#if defined(_WIN32)
#define LRS_MOBILE_API __declspec(dllexport)
#elif defined(__GNUC__)
#define LRS_MOBILE_API __attribute__((visibility("default")))
#else
#define LRS_MOBILE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lrs_mobile_index lrs_mobile_index;

LRS_MOBILE_API int lrs_mobile_open(const char* database_path, lrs_mobile_index** out, char** error);

LRS_MOBILE_API int lrs_mobile_prepare_document_json(const lrs_mobile_index* index,
                                                    const char* document_id, const char* content,
                                                    char** json, char** error);

LRS_MOBILE_API int lrs_mobile_upsert_lexical(lrs_mobile_index* index, const char* document_id,
                                             const char* title, const char* content, int* unchanged,
                                             char** error);

LRS_MOBILE_API int lrs_mobile_upsert_vectors(lrs_mobile_index* index, const char* document_id,
                                             const char* title, const char* content,
                                             const float* embeddings, size_t embedding_count,
                                             size_t embedding_dimension, int* unchanged,
                                             char** error);

LRS_MOBILE_API int lrs_mobile_search_json(lrs_mobile_index* index, const char* query,
                                          const float* query_embedding, size_t embedding_dimension,
                                          const char* mode, size_t top_k, char** json,
                                          char** error);

LRS_MOBILE_API void lrs_mobile_destroy(lrs_mobile_index* index);
LRS_MOBILE_API void lrs_mobile_string_destroy(char* value);

#ifdef __cplusplus
}
#endif

#endif
