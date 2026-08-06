# rag.cpp

`rag.cpp/` is the repository-owned retrieval core of llama-rag-runtime. It was
derived from upstream rag-cpp v0.1.0 and is maintained here as ordinary tracked
source rather than a submodule. See [PROVENANCE.md](PROVENANCE.md) for the exact
source revision, license, and owned changes.

## Shipped product core

The explicit CMake target builds the production retrieval path:

- deterministic token-aware, UTF-8-safe document chunking;
- BM25 lexical search;
- normalized exact cosine and HNSW dense search;
- RRF, RSF, and convex hybrid fusion;
- metadata filtering and soft deletion;
- MMR diversity and adjacent-parent stitching;
- `.ragdb` v1 persistence, CRC validation, atomic publication, and tombstones;
- injected HTTP or caller-computed embeddings;
- the C++20 `rag::Engine` facade and versioned opaque C ABI.

New consumers should link `rag::rag`. `ragcpp::ragcpp` remains only as a
transitional alias for existing integration code.

```cmake
add_subdirectory(rag.cpp)
target_link_libraries(my_target PRIVATE rag::rag)
target_compile_features(my_target PRIVATE cxx_std_20)
```

The desktop and mobile compatibility adapters live in the parent repository's
`src/bridge.cpp` and `src/mobile.cpp`. C++ types and STL ownership do not cross
those embedded boundaries.

## Retrieval profiles

`rag::retrieval::SearchOptions` selects one pipeline profile:

| Profile | Candidate pool | Extra stages |
|---|---:|---|
| `efficiency` | `max(3 * top_k, 24)` | none |
| `balanced` | `max(6 * top_k, 60)` | default |
| `quality` | `max(20 * top_k, 200)` | MMR plus adjacent-parent stitching |

Every profile setting can be overridden. `rag::retrieval::Diagnostics` reports
the selected profile, candidate pool, stages, elapsed time, result count, and
fallback reasons.

## Chunking and embedding contract

`rag::text::EmbeddingPolicy` records model and tokenizer identities, vector
dimension, target/hard/overlap/reserved token budgets, task prefixes, counting
mode, and invalid UTF-8 behavior. All fields contribute to the persisted
chunking fingerprint.

Desktop injects llama.cpp's exact `/tokenize` result and refuses estimated
counts. Mobile uses an explicit conservative UTF-8 byte policy until its model
stack exposes the exact tokenizer. Callers providing vectors must embed the
exact `embedding_text` returned by native preparation, in the same order.

## Persistence compatibility

The owned implementation continues reading and writing `.ragdb` major version
1. Format minor 2 adds policy identity and chunking fingerprint fields to the
existing `META` JSON; older readers ignore those additive fields. Newly written
indexes use embedded metadata as the authority, while legacy desktop indexes
may still be validated through their sidecar manifest.

Version 2 is intentionally a roadmap, not an implemented format. See
[CAPABILITIES.md](CAPABILITIES.md) for the capability matrix and v2 criteria.

## Build and verification

Build and test from the repository root:

```bash
cmake --preset macos-dev
cmake --build --preset macos-dev
ctest --preset macos-dev
./build/macos-dev/libexec/lrs-rag-eval
```

The qrels evaluator reports Recall@k, MRR, and nDCG@k for all three profiles.
The parent [README](../README.md) contains the complete server, TypeScript,
Python, Android, and real-model smoke-test instructions.

## Retained reference sources

Some upstream research sources and documents remain for provenance and future
evaluation, but are not part of the explicit product target. This includes
GraphRAG, RAPTOR, HyDE, CRAG, learned sparse retrieval, ColBERT, plugins, RCP,
CLI, ONNX, GPU, and in-process llama.cpp embedding. They must meet the evidence
requirements in [CAPABILITIES.md](CAPABILITIES.md) before entering the shipped
runtime.

The original MIT license is retained in [LICENSE](LICENSE).
