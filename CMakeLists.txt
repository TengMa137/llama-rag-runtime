cmake_minimum_required(VERSION 3.24)
project(llama-rag-runtime VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries so bin contains applications only" FORCE)

option(LRS_BUILD_LLAMA_SERVER "Build the pinned llama.cpp llama-server" OFF)
option(LRS_BUILD_TESTS "Build native tests" ON)
option(LRS_BUILD_COORDINATOR "Build the desktop HTTP coordinator" ON)
option(LRS_BUILD_MOBILE "Build the in-process mobile C ABI shared library" ON)
option(LRS_BUILD_MOBILE_SMOKE "Build the Android/device mobile smoke executable" OFF)
option(LRS_BUILD_SPEC_CHECK "Build the native specification checker" ON)
option(LRS_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
option(LRS_ENABLE_FAISS "Enable an installed desktop FAISS package" OFF)
option(LRS_ENABLE_POSTGRES "Enable an installed libpq PostgreSQL backend" OFF)
if(LRS_ENABLE_POSTGRES)
  find_package(PostgreSQL REQUIRED)
endif()
add_compile_definitions(
  LRS_ENABLE_FAISS=$<BOOL:${LRS_ENABLE_FAISS}>
  LRS_ENABLE_POSTGRES=$<BOOL:${LRS_ENABLE_POSTGRES}>)

set(LRS_FORMAT_SOURCES
  rag.cpp/include/rag/backend/candidate_backend.hpp
  rag.cpp/include/rag/backend/embedded_checkpoint.hpp
  rag.cpp/include/rag/backend/embedded_backend.hpp
  rag.cpp/include/rag/backend/postgres_backend.hpp
  rag.cpp/include/rag/backend/postgres_config.hpp
  rag.cpp/include/rag/core/keys.hpp
  rag.cpp/include/rag/dense/cache_fingerprint.hpp
  rag.cpp/include/rag/dense/exact_sidecar.hpp
  rag.cpp/include/rag/dense/faiss_index.hpp
  rag.cpp/include/rag/dense/faiss_sidecar.hpp
  rag.cpp/include/rag/dense/hnsw_sidecar.hpp
  rag.cpp/include/rag/dense/index.hpp
  rag.cpp/include/rag/dense/native_hnsw.hpp
  rag.cpp/include/rag/dense/policy.hpp
  rag.cpp/include/rag/dense/tiered_index.hpp
  rag.cpp/include/rag/engine.hpp
  rag.cpp/include/rag/ingestion/coordinator.hpp
  rag.cpp/include/rag/ingestion/embedded_durability.hpp
  rag.cpp/include/rag/ingestion/embedded_runtime.hpp
  rag.cpp/include/rag/ingestion/job.hpp
  rag.cpp/include/rag/ingestion/job_store.hpp
  rag.cpp/include/rag/ingestion/postgres_job_store.hpp
  rag.cpp/include/rag/ingestion/postgres_runtime.hpp
  rag.cpp/include/rag/ingestion/runtime.hpp
  rag.cpp/include/rag/migration/contract.hpp
  rag.cpp/include/rag/migration/embedded_endpoint.hpp
  rag.cpp/include/rag/migration/postgres_endpoint.hpp
  rag.cpp/include/rag/preparation/document_preparer.hpp
  rag.cpp/include/rag/retrieval/runtime.hpp
  rag.cpp/src/backend/embedded_backend.cpp
  rag.cpp/src/backend/candidate_backend.cpp
  rag.cpp/src/backend/embedded_checkpoint.cpp
  rag.cpp/src/backend/postgres_backend.cpp
  rag.cpp/src/backend/postgres_config.cpp
  rag.cpp/src/core/keys.cpp
  rag.cpp/src/dense/cache_fingerprint.cpp
  rag.cpp/src/dense/exact_sidecar.cpp
  rag.cpp/src/dense/faiss_index.cpp
  rag.cpp/src/dense/faiss_sidecar.cpp
  rag.cpp/src/dense/hnsw_sidecar.cpp
  rag.cpp/src/dense/index.cpp
  rag.cpp/src/dense/native_hnsw.cpp
  rag.cpp/src/dense/policy.cpp
  rag.cpp/src/dense/tiered_index.cpp
  rag.cpp/src/postgres/connection_pool.cpp
  rag.cpp/src/postgres/connection_pool.hpp
  rag.cpp/src/postgres/migrations.cpp
  rag.cpp/src/postgres/migrations.hpp
  rag.cpp/src/migration/contract.cpp
  rag.cpp/src/migration/embedded_endpoint.cpp
  rag.cpp/src/migration/postgres_endpoint.cpp
  rag.cpp/src/store/container.cpp
  rag.cpp/src/store/container_layout.cpp
  rag.cpp/src/store/container_view.cpp
  rag.cpp/src/ingestion/coordinator.cpp
  rag.cpp/src/ingestion/embedded_durability.cpp
  rag.cpp/src/ingestion/embedded_runtime.cpp
  rag.cpp/src/ingestion/job.cpp
  rag.cpp/src/ingestion/job_store.cpp
  rag.cpp/src/ingestion/postgres_job_store.cpp
  rag.cpp/src/ingestion/postgres_runtime.cpp
  rag.cpp/src/preparation/document_preparer.cpp
  rag.cpp/src/retrieval/runtime.cpp
  include/lrs/bridge.h
  include/lrs/config.hpp
  include/lrs/model_client.hpp
  include/lrs/mobile.h
  include/lrs/service.hpp
  src/bridge.cpp
  src/config.cpp
  src/main.cpp
  src/model_client.cpp
  src/mobile.cpp
  src/service.cpp
  tests/backend_contract_suite.cpp
  tests/backend_contract_suite.hpp
  tests/tests.cpp
  tests/rag_core_tests.cpp
  tests/rag_integration_tests.cpp
  tools/mobile_smoke.cpp
  tools/eval/common.cpp
  tools/eval/common.hpp
  tools/eval/corpus.cpp
  tools/eval/corpus.hpp
  tools/eval/dense.cpp
  tools/eval/dense.hpp
  tools/eval/qrels.cpp
  tools/eval/qrels.hpp
  tools/eval/runtime.cpp
  tools/eval/runtime.hpp
  tools/rag_eval.cpp
  tools/rag_migrate.cpp
  tools/spec_check.cpp)
list(TRANSFORM LRS_FORMAT_SOURCES PREPEND "${CMAKE_SOURCE_DIR}/")

find_program(LRS_CLANG_FORMAT NAMES clang-format)
if(NOT LRS_CLANG_FORMAT AND APPLE)
  execute_process(
    COMMAND xcrun -f clang-format
    OUTPUT_VARIABLE LRS_CLANG_FORMAT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(LRS_CLANG_FORMAT)
  add_custom_target(format
    COMMAND "${LRS_CLANG_FORMAT}" -i --style=file ${LRS_FORMAT_SOURCES}
    COMMENT "Formatting project-owned C and C++ sources")
  add_custom_target(format-check
    COMMAND "${LRS_CLANG_FORMAT}" --dry-run --Werror --style=file ${LRS_FORMAT_SOURCES}
    COMMENT "Checking project-owned C and C++ formatting")
else()
  message(STATUS "clang-format not found; format and format-check targets are unavailable")
endif()

add_subdirectory(rag.cpp)

if(LRS_BUILD_LLAMA_SERVER)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
  set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
  add_custom_target(lrs_llama_server ALL DEPENDS llama-server)
endif()

if(LRS_BUILD_COORDINATOR)
  add_library(lrs_bridge STATIC src/bridge.cpp)
  target_include_directories(lrs_bridge PUBLIC include)
  target_link_libraries(lrs_bridge PRIVATE rag::core)
  target_compile_features(lrs_bridge PRIVATE cxx_std_20)

  find_package(Threads REQUIRED)
  add_library(lrs_httplib STATIC third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp)
  target_include_directories(lrs_httplib PUBLIC third_party/llama.cpp/vendor/cpp-httplib)
  target_link_libraries(lrs_httplib PUBLIC Threads::Threads)
  target_compile_features(lrs_httplib PUBLIC cxx_std_17)
  if(NOT MSVC)
    target_compile_options(lrs_httplib PRIVATE -w)
  endif()

  add_library(lrs_core STATIC src/config.cpp src/service.cpp src/model_client.cpp)
  target_include_directories(lrs_core PUBLIC include PRIVATE
    rag.cpp/include
    third_party/llama.cpp/vendor
    third_party/llama.cpp/vendor/cpp-httplib)
  target_link_libraries(lrs_core PUBLIC lrs_bridge lrs_httplib PRIVATE rag::core)
  target_compile_features(lrs_core PUBLIC cxx_std_17 PRIVATE cxx_std_20)

  add_executable(llama-rag-server src/main.cpp)
  target_include_directories(llama-rag-server PRIVATE third_party/llama.cpp/vendor)
  target_link_libraries(llama-rag-server PRIVATE lrs_core)
  target_compile_features(llama-rag-server PRIVATE cxx_std_17)
endif()

if(LRS_BUILD_MOBILE)
  add_library(lrs_mobile SHARED src/mobile.cpp)
  set_target_properties(lrs_mobile PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME rag_mobile
    VISIBILITY_INLINES_HIDDEN YES)
  target_include_directories(lrs_mobile PUBLIC include)
  target_link_libraries(lrs_mobile PRIVATE rag::core)
  target_compile_features(lrs_mobile PRIVATE cxx_std_20)
endif()

if(LRS_BUILD_MOBILE AND LRS_BUILD_MOBILE_SMOKE)
  add_executable(lrs-mobile-smoke tools/mobile_smoke.cpp)
  target_include_directories(lrs-mobile-smoke PRIVATE include)
  target_link_libraries(lrs-mobile-smoke PRIVATE lrs_mobile)
  target_compile_features(lrs-mobile-smoke PRIVATE cxx_std_17)
  set_target_properties(lrs-mobile-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
endif()

if(LRS_BUILD_SPEC_CHECK)
  add_executable(lrs-spec-check tools/spec_check.cpp)
  target_include_directories(lrs-spec-check PRIVATE third_party/llama.cpp/vendor)
  target_compile_definitions(lrs-spec-check PRIVATE
    LRS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    LRS_LLAMA_PIN="7bd8282c37fcd9c4d7236106d664761a23318f18")
  target_compile_features(lrs-spec-check PRIVATE cxx_std_17)
  set_target_properties(lrs-spec-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
endif()

add_executable(lrs-rag-eval
  tools/rag_eval.cpp
  tools/eval/common.cpp
  tools/eval/corpus.cpp
  tools/eval/dense.cpp
  tools/eval/qrels.cpp
  tools/eval/runtime.cpp)

target_link_libraries(lrs-rag-eval PRIVATE rag::core)
target_compile_features(lrs-rag-eval PRIVATE cxx_std_20)
target_compile_definitions(lrs-rag-eval PRIVATE LRS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
set_target_properties(lrs-rag-eval PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
if(LRS_BUILD_TESTS)
  add_test(NAME lrs-eval-qrels COMMAND lrs-rag-eval qrels)
  add_test(NAME lrs-eval-dense-smoke COMMAND lrs-rag-eval dense
    --vectors 256 --dimension 32 --queries 5 --k 10
    --implementation native --algorithm exact)
  add_test(NAME lrs-eval-runtime-smoke COMMAND lrs-rag-eval runtime
    --vectors 256 --dimension 32 --queries 5 --k 10
    --implementation native --algorithm exact)
endif()

if(LRS_ENABLE_POSTGRES)
  add_executable(lrs-rag-migrate tools/rag_migrate.cpp)
  target_link_libraries(lrs-rag-migrate PRIVATE rag::core)
  target_compile_features(lrs-rag-migrate PRIVATE cxx_std_20)
  set_target_properties(lrs-rag-migrate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  if(LRS_BUILD_TESTS)
    add_test(NAME lrs-migration-cli-help COMMAND lrs-rag-migrate --help)
  endif()
endif()

set(LRS_WARNING_TARGETS lrs-rag-eval)
if(LRS_ENABLE_POSTGRES)
  list(APPEND LRS_WARNING_TARGETS lrs-rag-migrate)
endif()
if(LRS_BUILD_SPEC_CHECK)
  list(APPEND LRS_WARNING_TARGETS lrs-spec-check)
endif()
if(LRS_BUILD_COORDINATOR)
  list(APPEND LRS_WARNING_TARGETS lrs_core llama-rag-server)
endif()
if(LRS_BUILD_MOBILE)
  list(APPEND LRS_WARNING_TARGETS lrs_mobile)
endif()
if(LRS_BUILD_MOBILE AND LRS_BUILD_MOBILE_SMOKE)
  list(APPEND LRS_WARNING_TARGETS lrs-mobile-smoke)
endif()
foreach(t IN LISTS LRS_WARNING_TARGETS)
  if(MSVC)
    target_compile_options(${t} PRIVATE /W4)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Wpedantic)
    if(LRS_WARNINGS_AS_ERRORS)
      target_compile_options(${t} PRIVATE -Werror)
    endif()
  endif()
endforeach()

if(LRS_BUILD_TESTS)
  add_executable(rag-core-tests
    tests/rag_core_tests.cpp
    tests/backend_contract_suite.cpp)
  target_link_libraries(rag-core-tests PRIVATE rag::core)
  target_compile_features(rag-core-tests PRIVATE cxx_std_20)
  target_compile_definitions(rag-core-tests PRIVATE LRS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(LRS_ENABLE_POSTGRES)
    target_include_directories(rag-core-tests PRIVATE "${CMAKE_SOURCE_DIR}/rag.cpp/src")
    target_link_libraries(rag-core-tests PRIVATE PostgreSQL::PostgreSQL)
  endif()
  set_target_properties(rag-core-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
  add_test(NAME rag-owned-core-behavior COMMAND rag-core-tests)
  if(LRS_BUILD_SPEC_CHECK)
    add_test(NAME lrs-specification-traceability COMMAND lrs-spec-check)
  endif()
endif()

if(LRS_BUILD_TESTS AND LRS_BUILD_COORDINATOR AND LRS_BUILD_MOBILE)
  add_executable(rag-integration-tests tests/rag_integration_tests.cpp)
  target_link_libraries(rag-integration-tests PRIVATE lrs_bridge lrs_mobile rag::core)
  target_include_directories(rag-integration-tests PRIVATE
    third_party/llama.cpp/vendor)
  target_compile_features(rag-integration-tests PRIVATE cxx_std_20)
  set_target_properties(rag-integration-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
  add_test(NAME rag-desktop-mobile-integration COMMAND rag-integration-tests)

  find_package(Catch2 3 REQUIRED)
  add_executable(lrs-tests tests/tests.cpp)
  target_link_libraries(lrs-tests PRIVATE lrs_core lrs_mobile Catch2::Catch2WithMain)
  target_include_directories(lrs-tests PRIVATE
    third_party/llama.cpp/vendor
    rag.cpp/include)
  target_compile_features(lrs-tests PRIVATE cxx_std_20)
  set_target_properties(lrs-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
  add_test(NAME lrs-unit-component-behavior COMMAND lrs-tests)
endif()
