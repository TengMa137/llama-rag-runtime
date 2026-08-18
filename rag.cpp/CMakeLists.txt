cmake_minimum_required(VERSION 3.24)

project(rag_core VERSION 0.2.0 LANGUAGES CXX)

include(cmake/OwnedCoreGuard.cmake)

# Repository-owned product core.  Keep this list explicit: experimental
# research systems and backend hosts are intentionally not pulled into the
# desktop or mobile runtime by discovering a new source file.
set(RAG_CORE_SOURCES
    src/backend/candidate_backend.cpp
    src/backend/embedded_backend.cpp
    src/backend/embedded_checkpoint.cpp
    src/c/rag_c.cpp
    src/core/keys.cpp
    src/dense/backends.cpp
    src/dense/cache_fingerprint.cpp
    src/dense/exact_sidecar.cpp
    src/dense/hnsw_sidecar.cpp
    src/dense/index.cpp
    src/dense/native_hnsw.cpp
    src/dense/policy.cpp
    src/dense/http_transport.cpp
    src/dense/simd.cpp
    src/dense/tiered_index.cpp
    src/fusion/fuse.cpp
    src/gpu/device.cpp
    src/index/corpus.cpp
    src/index/hnsw.cpp
    src/index/pq.cpp
    src/ingestion/coordinator.cpp
    src/ingestion/embedded_durability.cpp
    src/ingestion/embedded_runtime.cpp
    src/ingestion/job.cpp
    src/ingestion/job_store.cpp
    src/lexical/bm25.cpp
    src/pipeline/pipeline.cpp
    src/preparation/document_preparer.cpp
    src/rerank/mmr.cpp
    src/retrieval/runtime.cpp
    src/store/container.cpp
    src/store/container_layout.cpp
    src/store/container_view.cpp
    src/store/wal.cpp
    src/text/chunker.cpp
    src/text/tokenizer.cpp)

if(NOT ANDROID)
    list(APPEND RAG_CORE_SOURCES
        src/migration/contract.cpp
        src/migration/embedded_endpoint.cpp)
endif()

if(LRS_ENABLE_FAISS)
    if(ANDROID)
        message(FATAL_ERROR "LRS_ENABLE_FAISS is desktop-only; Android uses native dense indexes")
    endif()
    find_package(faiss CONFIG REQUIRED)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(TARGET OpenMP::OpenMP_CXX)
        set(RAG_FAISS_OPENMP_TARGET OpenMP::OpenMP_CXX)
    elseif(APPLE)
        find_path(RAG_FAISS_OPENMP_INCLUDE omp.h
            HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
            REQUIRED)
        find_library(RAG_FAISS_OPENMP_LIBRARY omp
            HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
            REQUIRED)
        add_library(rag_faiss_openmp INTERFACE)
        target_include_directories(rag_faiss_openmp INTERFACE "${RAG_FAISS_OPENMP_INCLUDE}")
        target_compile_options(rag_faiss_openmp INTERFACE -Xpreprocessor -fopenmp)
        target_link_libraries(rag_faiss_openmp INTERFACE "${RAG_FAISS_OPENMP_LIBRARY}")
        set(RAG_FAISS_OPENMP_TARGET rag_faiss_openmp)
    else()
        message(FATAL_ERROR "Installed FAISS requires an OpenMP C++ development package")
    endif()
    list(APPEND RAG_CORE_SOURCES src/dense/faiss_index.cpp src/dense/faiss_sidecar.cpp)
endif()

if(LRS_ENABLE_POSTGRES)
    if(ANDROID)
        message(FATAL_ERROR "LRS_ENABLE_POSTGRES is a parent desktop/runtime backend")
    endif()
    list(APPEND RAG_CORE_SOURCES
        src/backend/postgres_backend.cpp
        src/backend/postgres_config.cpp
        src/ingestion/postgres_job_store.cpp
        src/ingestion/postgres_runtime.cpp
        src/migration/postgres_endpoint.cpp
        src/postgres/connection_pool.cpp
        src/postgres/migrations.cpp)
endif()

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    # llama.cpp already vendors the same single-header dependency. Reuse it so
    # normal/offline product builds never fetch code while configuring.
    set(RAG_CORE_JSON_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor")
    if(NOT EXISTS "${RAG_CORE_JSON_DIR}/nlohmann/json.hpp")
        message(FATAL_ERROR "nlohmann/json.hpp was not found")
    endif()
    add_library(rag_core_json INTERFACE)
    target_include_directories(rag_core_json INTERFACE "${RAG_CORE_JSON_DIR}")
    add_library(nlohmann_json::nlohmann_json ALIAS rag_core_json)
endif()

add_library(rag_core STATIC ${RAG_CORE_SOURCES})
add_library(rag::core ALIAS rag_core)

target_include_directories(rag_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(rag_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(rag_core PUBLIC cxx_std_20)
target_compile_definitions(rag_core PUBLIC RAG_USE_SIMD=1)
if(LRS_ENABLE_FAISS)
    target_compile_definitions(rag_core PUBLIC LRS_ENABLE_FAISS=1)
    if(TARGET faiss::faiss)
        target_link_libraries(rag_core PRIVATE faiss::faiss ${RAG_FAISS_OPENMP_TARGET})
    elseif(TARGET faiss)
        target_link_libraries(rag_core PRIVATE faiss ${RAG_FAISS_OPENMP_TARGET})
    else()
        message(FATAL_ERROR "Installed FAISS package did not export faiss or faiss::faiss")
    endif()
else()
    target_compile_definitions(rag_core PUBLIC LRS_ENABLE_FAISS=0)
endif()
if(LRS_ENABLE_POSTGRES)
    target_compile_definitions(rag_core PUBLIC LRS_ENABLE_POSTGRES=1)
    target_link_libraries(rag_core PRIVATE PostgreSQL::PostgreSQL)
else()
    target_compile_definitions(rag_core PUBLIC LRS_ENABLE_POSTGRES=0)
endif()

if(NOT MSVC)
    target_compile_options(rag_core PRIVATE -Wall -Wextra -Wpedantic)
    if(LRS_WARNINGS_AS_ERRORS)
        target_compile_options(rag_core PRIVATE -Werror)
    endif()
endif()
