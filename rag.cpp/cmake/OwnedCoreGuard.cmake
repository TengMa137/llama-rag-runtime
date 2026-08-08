set(_rag_removed_markers
    "rag/bridge/" "rag/cache/" "rag/cascade/" "rag/crag/" "rag/eval/"
    "rag/graph/" "rag/late/" "rag/loaders/" "rag/plugin/" "rag/query/"
    "rag/ralm/" "rag/raptor/" "rag/rcp/" "rag/sparse/"
    "semantic_chunker" "contextual")

get_filename_component(_rag_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

file(GLOB_RECURSE _rag_public_headers
     "${_rag_root}/include/rag/*.h"
     "${_rag_root}/include/rag/*.hpp")
set(_rag_guard_files "${_rag_root}/CMakeLists.txt" ${_rag_public_headers})
foreach(_file IN LISTS _rag_guard_files)
    file(READ "${_file}" _contents)
    foreach(_marker IN LISTS _rag_removed_markers)
        string(FIND "${_contents}" "${_marker}" _found)
        if(NOT _found EQUAL -1)
            message(FATAL_ERROR "Owned-core guard: ${_file} references removed capability ${_marker}")
        endif()
    endforeach()
endforeach()

file(GLOB_RECURSE _rag_code
     "${_rag_root}/src/*"
     "${_rag_root}/include/*")
foreach(_file IN LISTS _rag_code)
    file(READ "${_file}" _contents)
    if(_contents MATCHES "dlopen[ \t\r\n]*\\(" OR
       _contents MATCHES "popen[ \t\r\n]*\\(" OR
       _contents MATCHES "(^|[^A-Za-z_])exec[ \t\r\n]*\\(" OR
       _contents MATCHES "listen[ \t\r\n]*\\(" OR
       _contents MATCHES "api[_-]?key" OR
       _contents MATCHES "api\\.(openai|cohere|jina|together)" OR
       _contents MATCHES "FetchContent|ExternalProject|download[_-]?(file|url)")
        message(FATAL_ERROR "Owned-core guard: forbidden runtime behavior in ${_file}")
    endif()
endforeach()

file(READ "${_rag_root}/include/rag/rag.hpp" _rag_umbrella)
foreach(_marker IN LISTS _rag_removed_markers)
    string(FIND "${_rag_umbrella}" "${_marker}" _found)
    if(NOT _found EQUAL -1)
        message(FATAL_ERROR "Owned-core guard: rag/rag.hpp exports ${_marker}")
    endif()
endforeach()

set(_rag_docs README.md ARCHITECTURE.md FORMAT.md PROVENANCE.md CAPABILITIES.md)
foreach(_doc IN LISTS _rag_docs)
    file(READ "${_rag_root}/${_doc}" _contents)
    if(_contents MATCHES "FetchContent|build/(cli|examples|bench)/")
        message(FATAL_ERROR "Owned-core guard: ${_doc} advertises an unavailable build option or binary")
    endif()
endforeach()
