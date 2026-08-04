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
option(LRS_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

set(RAGCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RAGCPP_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(RAGCPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAGCPP_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(RAGCPP_WITH_LLAMA OFF CACHE BOOL "" FORCE)
set(RAGCPP_WITH_RCP OFF CACHE BOOL "" FORCE)
set(RAGCPP_WITH_METAL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rag-cpp)

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

add_library(lrs_bridge STATIC src/bridge.cpp)
target_include_directories(lrs_bridge PUBLIC include)
target_link_libraries(lrs_bridge PRIVATE ragcpp::ragcpp)
target_compile_features(lrs_bridge PRIVATE cxx_std_23)

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
  third_party/llama.cpp/vendor
  third_party/llama.cpp/vendor/cpp-httplib)
target_link_libraries(lrs_core PUBLIC lrs_bridge lrs_httplib)
target_compile_features(lrs_core PUBLIC cxx_std_17)

add_executable(llama-rag-server src/main.cpp)
target_link_libraries(llama-rag-server PRIVATE lrs_core)
target_compile_features(llama-rag-server PRIVATE cxx_std_17)

add_executable(lrs-spec-check tools/spec_check.cpp)
target_include_directories(lrs-spec-check PRIVATE third_party/llama.cpp/vendor)
target_compile_definitions(lrs-spec-check PRIVATE
  LRS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  LRS_LLAMA_PIN="7bd8282c37fcd9c4d7236106d664761a23318f18"
  LRS_RAGCPP_PIN="cfe46cee87fccb9ca5dee68d416229489285fdea")
target_compile_features(lrs-spec-check PRIVATE cxx_std_17)
set_target_properties(lrs-spec-check PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")

foreach(t lrs_core llama-rag-server lrs-spec-check)
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
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  add_executable(lrs-tests tests/tests.cpp)
  target_link_libraries(lrs-tests PRIVATE lrs_core ragcpp::ragcpp Catch2::Catch2WithMain)
  target_include_directories(lrs-tests PRIVATE third_party/llama.cpp/vendor)
  target_compile_features(lrs-tests PRIVATE cxx_std_17)
  set_target_properties(lrs-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
  add_test(NAME lrs-unit-component-behavior COMMAND lrs-tests)
  add_test(NAME lrs-specification-traceability COMMAND lrs-spec-check)
endif()
