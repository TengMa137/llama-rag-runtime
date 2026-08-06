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

set(LRS_FORMAT_SOURCES
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
  tests/tests.cpp
  tools/mobile_smoke.cpp
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

if(LRS_BUILD_COORDINATOR)
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
  target_include_directories(llama-rag-server PRIVATE third_party/llama.cpp/vendor)
  target_link_libraries(llama-rag-server PRIVATE lrs_core)
  target_compile_features(llama-rag-server PRIVATE cxx_std_17)
endif()

if(LRS_BUILD_MOBILE)
  add_library(lrs_mobile SHARED src/mobile.cpp)
  set_target_properties(lrs_mobile PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME ragcpp_mobile
    VISIBILITY_INLINES_HIDDEN YES)
  target_include_directories(lrs_mobile PUBLIC include)
  target_link_libraries(lrs_mobile PRIVATE ragcpp::ragcpp)
  target_compile_features(lrs_mobile PRIVATE cxx_std_23)
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
    LRS_LLAMA_PIN="7bd8282c37fcd9c4d7236106d664761a23318f18"
    LRS_RAGCPP_PIN="cfe46cee87fccb9ca5dee68d416229489285fdea")
  target_compile_features(lrs-spec-check PRIVATE cxx_std_17)
  set_target_properties(lrs-spec-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
endif()

set(LRS_WARNING_TARGETS)
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

if(LRS_BUILD_TESTS AND LRS_BUILD_COORDINATOR AND LRS_BUILD_MOBILE)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  add_executable(lrs-tests tests/tests.cpp)
  target_link_libraries(lrs-tests PRIVATE lrs_core lrs_mobile Catch2::Catch2WithMain)
  target_include_directories(lrs-tests PRIVATE
    third_party/llama.cpp/vendor
    third_party/rag-cpp/include)
  target_compile_features(lrs-tests PRIVATE cxx_std_23)
  set_target_properties(lrs-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libexec")
  add_test(NAME lrs-unit-component-behavior COMMAND lrs-tests)
  if(LRS_BUILD_SPEC_CHECK)
    add_test(NAME lrs-specification-traceability COMMAND lrs-spec-check)
  endif()
endif()
