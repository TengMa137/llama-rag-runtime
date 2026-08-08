# rag.cpp

`rag.cpp/` is the independently maintained retrieval-and-persistence core used
by llama-rag-runtime desktop, mobile, and local-agent paths. It owns documents,
structural chunks, stable IDs, retrieval indexes, ranking pipelines, and
`.ragdb` persistence. It does not own agents, chat, prompts, generation, model
processes, or user-facing services.

Supported features are deterministic token-budgeted UTF-8-safe chunking with
line citations; BM25; exact cosine and HNSW dense search; lexical, dense, and
hybrid retrieval; RRF fusion; metadata filters; MMR; adjacent-chunk
stitching; retrieval profiles; tombstones; WAL recovery; atomic saves; caller-
supplied embedders; and one loopback-only HTTP embedding client.

The public C++ API is C++20. Link the `rag::core` target.

```cmake
add_subdirectory(rag.cpp)
target_link_libraries(my_app PRIVATE rag::core)
target_compile_features(my_app PRIVATE cxx_std_20)
```

```cpp
#include <rag/rag.hpp>

rag::Engine engine;
engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{256}});
engine.add("notes/one", "A local document.");
engine.build();
auto hits = engine.search("local", 5);
```

`LocalHttpEmbedder` accepts only hosts resolving entirely to loopback, validated
relative paths, fixed timeouts, bounded responses, and an expected dimension.
It has no credential or arbitrary-header surface. The parent coordinator owns
the model process and configures this seam. Mobile supplies vectors through its
own C boundary.

Build and test from the parent repository:

```sh
cmake --preset macos-dev
cmake --build --preset macos-dev
ctest --preset macos-dev
./build/macos-dev/libexec/lrs-rag-eval
```

See [ARCHITECTURE.md](ARCHITECTURE.md), [FORMAT.md](FORMAT.md),
[CAPABILITIES.md](CAPABILITIES.md), and [PROVENANCE.md](PROVENANCE.md).
