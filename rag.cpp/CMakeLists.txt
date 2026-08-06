cmake_minimum_required(VERSION 3.24)

project(rag_cpp VERSION 0.1.1 LANGUAGES CXX)

# Repository-owned product core.  Keep this list explicit: experimental
# research systems and backend hosts are intentionally not pulled into the
# desktop or mobile runtime by discovering a new source file.
set(RAG_CPP_SOURCES
    src/c/rag_c.cpp
    src/dense/backends.cpp
    src/dense/http_transport.cpp
    src/dense/simd.cpp
    src/fusion/fuse.cpp
    src/gpu/device.cpp
    src/index/corpus.cpp
    src/index/hnsw.cpp
    src/index/pq.cpp
    src/lexical/bm25.cpp
    src/pipeline/pipeline.cpp
    src/rerank/mmr.cpp
    src/rerank/reranker.cpp
    src/store/container.cpp
    src/store/wal.cpp
    src/text/chunker.cpp
    src/text/contextual.cpp
    src/text/semantic_chunker.cpp
    src/text/tokenizer.cpp)

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    # llama.cpp already vendors the same single-header dependency. Reuse it so
    # normal/offline product builds never fetch code while configuring.
    set(RAG_CPP_JSON_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor")
    if(NOT EXISTS "${RAG_CPP_JSON_DIR}/nlohmann/json.hpp")
        message(FATAL_ERROR "nlohmann/json.hpp was not found")
    endif()
    add_library(rag_cpp_json INTERFACE)
    target_include_directories(rag_cpp_json INTERFACE "${RAG_CPP_JSON_DIR}")
    add_library(nlohmann_json::nlohmann_json ALIAS rag_cpp_json)
endif()

add_library(rag_cpp STATIC ${RAG_CPP_SOURCES})
add_library(rag::rag ALIAS rag_cpp)
# Transitional spelling retained for existing consumers.
add_library(ragcpp::ragcpp ALIAS rag_cpp)

target_include_directories(rag_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(rag_cpp PUBLIC nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
target_compile_features(rag_cpp PUBLIC cxx_std_20)
target_compile_definitions(rag_cpp PUBLIC RAGCPP_USE_SIMD=1)

if(NOT MSVC)
    target_compile_options(rag_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()
