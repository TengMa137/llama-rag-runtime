# llama-rag-runtime Technical Overview

**Document role:** current implementation walkthrough  
**Repository version:** v0.1 development snapshot  
**Last reviewed:** 2026-08-05

This document explains what is implemented in this repository today and how the pieces fit together. It is deliberately different from the other two entry points:

- [`README.md`](../README.md) is the user guide for building, starting, and calling the service.
- [`llama-rag-server-spec.md`](llama-rag-server-spec.md) is the broader product specification and future vision.
- This document is the maintainer's map of the code that currently exists, including its boundaries and known gaps.

When code and the specification differ, this document describes the code and calls out the difference. The specification is not evidence that a feature has already been implemented.

## 1. Current system at a glance

The repository currently builds a native local RAG vertical slice for macOS. The installed command-line surface is intentionally small:

```text
build/macos-dev/bin/llama-rag-server
build/macos-dev/bin/llama-server
```

`llama-rag-server` is a C++17 HTTP coordinator. It owns configuration, model-process supervision, request validation, index-generation publication, retrieval orchestration, context construction, and RAG SSE translation.

`llama-server` comes from the pinned llama.cpp submodule. Two instances provide embedding and generation. They can be children of the coordinator or independently managed OpenAI-compatible loopback backends.

rag-cpp is embedded as a C++23 static library. It owns document/chunk records, embeddings, BM25, dense retrieval, hybrid retrieval, and `.ragdb` persistence. A project-owned C ABI keeps its C++23 types out of the C++17 coordinator.

```text
client
  |
  | HTTP JSON / SSE on 127.0.0.1:8080
  v
llama-rag-server (C++17)
  |-- lrs_core: config, model HTTP client, service logic
  |-- lrs_bridge: C ABI boundary compiled as C++23
  |     `-- rag-cpp --> data/*.ragdb
  |-- embedding HTTP --> llama-server on 127.0.0.1:8081
  `-- generation HTTP --> llama-server on 127.0.0.1:8082
```

No SQLite, FAISS, Python service, Node service, or external vector database is required.

## 2. Repository map

| Path | Current responsibility |
|---|---|
| `CMakeLists.txt` | Superbuild, target language boundaries, dependency options, output layout, and test registration |
| `CMakePresets.json` | The `macos-dev` configure/build/test workflow |
| `config/` | Spawned-model, manually managed backend, and real-model smoke configurations |
| `include/lrs/bridge.h` | C-compatible ownership and retrieval boundary around rag-cpp |
| `include/lrs/config.hpp` | Runtime configuration model |
| `include/lrs/model_client.hpp` | Private llama-server generation/tokenization client |
| `include/lrs/service.hpp` | Coordinator service API and active-index ownership |
| `src/bridge.cpp` | rag-cpp configuration, persistence, upsert, search, and stable public chunk IDs |
| `src/config.cpp` | JSON loading and v0.1 safety validation |
| `src/main.cpp` | CLI, child processes, readiness loop, and HTTP routes |
| `src/model_client.cpp` | Health, tokenization, OpenAI chat-completion SSE parsing, and cancellation |
| `src/service.cpp` | Ingestion transaction, retrieval, context budgeting, and public RAG SSE |
| `tests/tests.cpp` | Deterministic native unit/component/behavior/fault scenarios |
| `tools/spec_check.cpp` | Requirement-to-test and pinned-dependency consistency checks |
| `requirements.json` | Machine-readable active/deferred requirement catalog |
| `third_party/llama.cpp` | Pinned upstream inference engine and `llama-server` |
| `third_party/rag-cpp` | Pinned upstream retrieval and persistence engine |

Generated applications go to `build/<preset>/bin`, static libraries to `lib`, and developer-only test/check executables to `libexec`.

## 3. Build graph and language boundary

The top-level project requires CMake 3.24 or newer and uses target-scoped language features:

```text
llama-rag-server (C++17)
  `-- lrs_core (C++17)
       |-- lrs_httplib (C++17)
       `-- lrs_bridge (implementation C++23, public header is C)
            `-- ragcpp::ragcpp (C++23)

llama-server (llama.cpp C++17 baseline)
lrs-tests (native Catch2 test executable)
lrs-spec-check (native traceability executable)
```

The bridge exposes opaque `lrs_index*` handles, plain option structures, integer status values, and explicit string destructors. Neither rag-cpp headers nor C++23 library types cross into coordinator headers.

The pinned build disables rag-cpp's llama integration, RCP server, Metal backend, examples, CLI, and upstream tests for the product target. The coordinator talks to an embedding service over HTTP instead of loading an embedding GGUF inside rag-cpp.

Catch2 v3.8.1 is fetched at configure time when `LRS_BUILD_TESTS=ON`. A clean configure therefore needs network access unless Catch2 is already populated in the build cache.

## 4. Configuration and process lifecycle

`Config` has three endpoints:

- `listen`, defaulting to coordinator port 8080;
- `inference.embedding`, defaulting to port 8081;
- `inference.generation`, defaulting to port 8082.

All three hosts must currently be loopback addresses and their ports must be distinct. The embedding dimension and context sizes are explicit configuration; pooling is restricted to `mean`.

With `inference.spawn=true`, `src/main.cpp` forks two children and executes the configured `llama-server` binary. The embedding child receives `--embeddings`, `--pooling`, context, batch, and model flags. The generation child receives its model and context flags. The coordinator owns the child PIDs and sends `SIGTERM`, then waits for them during normal stack unwinding.

With `spawn=false`, the same service attaches to already-running compatible backends. There is no child lifecycle management in that mode.

Startup then performs up to 120 initialization attempts at one-second intervals. Initialization opens or creates the rag-cpp engine, attaches its embedder, and checks both model health endpoints. Only after initialization succeeds does the coordinator bind its public HTTP listener.

Current lifecycle limitations:

- child supervision is POSIX-only and Windows rejects `spawn=true`;
- there is no signal handler, restart backoff, child liveness monitor, or configurable startup timeout;
- startup validates configured dimension and pooling values but does not probe the embedding server to prove them;
- `/health` and `/ready` currently return the same small process-level status object.

## 5. Index and persistence model

The configured `index.path` is the active rag-cpp `.ragdb`. Its sibling `<path>.manifest.json` records the schema, rag-cpp version, chunking fingerprint, chunk settings, and embedding dimension. An existing database is rejected when the manifest is missing or incompatible.

Current chunk settings in `src/bridge.cpp` are:

| Setting | Value |
|---|---:|
| Algorithm | rag-cpp fixed chunker |
| Maximum lines | 40 |
| Maximum characters | 384 |
| Overlap | 4 lines |
| Markdown heading context | disabled |

The 384-character bound was selected for the bundled Granite model's 512-token ceiling. These settings differ from the older 1,600-character/heading-context defaults still described in parts of the product specification; the implementation and manifest values are authoritative for existing indexes.

Public chunk IDs are not rag-cpp numeric IDs. The bridge computes a stable FNV-1a-derived ID from document URI, source line range, and normalized chunk text. Replacing text therefore produces new chunk IDs while repeated searches of the same generation retain the same IDs.

### Ingestion and publication

`POST /v1/rag/documents` implements text-document upsert:

1. Parse and validate `id`, `content`, optional `title`, and content type.
2. Serialize writers with `Service::mutation_`.
3. Retain the currently active shared index generation.
4. Return `unchanged` if normalized content and title already match.
5. Reopen the persisted generation as a candidate.
6. Upsert the document, chunk it, embed pending chunks, and build indexes.
7. Save to a uniquely named staging database.
8. Reopen the staging database to validate it.
9. Rename it to the configured active database path and write the manifest through a temporary file.
10. Atomically swap the service's `shared_ptr` to the new in-memory generation.

Embedding or candidate-save failure leaves the active in-memory generation unchanged. Searches that have already loaded a `shared_ptr` finish against the generation on which they started.

There is a remaining durability gap: database publication and manifest publication are two separate filesystem renames, so a crash between them is not a fully atomic two-file transaction. WAL recovery, snapshotting, and compaction policy remain future work.

## 6. Retrieval path

`POST /v1/rag/search` accepts text, not a raw vector. `src/bridge.cpp` supports:

- `lexical`: rag-cpp BM25;
- `dense`: embed the query and run dense similarity search;
- `hybrid`: run rag-cpp's standard hybrid pipeline with reciprocal-rank fusion.

The response contains sorted results with document ID, stable chunk ID, one-based rank, mode-specific score, line offsets, and source text. `top_k` must be between 1 and 100.

Within rag-cpp, hybrid lexical and dense work runs concurrently. Small dense corpora use vectorized exact similarity; larger corpora use HNSW according to rag-cpp's corpus threshold. The normal single-query path is CPU-oriented. The optional rag-cpp GPU implementation is Metal-only and accelerates very large batched flat scans, so it is neither enabled here nor applicable to Android.

Every dense or hybrid chat query still requires one query embedding. Document embedding is normally paid during ingestion, but on-device query embedding may dominate the much smaller BM25/HNSW similarity cost.

## 7. Grounded generation path

`POST /v1/rag/query` currently uses the final message's `content` as the retrieval query. It emits:

1. `rag.started`;
2. `rag.retrieval.completed` with the authoritative ordered sources;
3. zero or more `rag.generation.delta` events;
4. `rag.completed`, or `rag.error` on failure.

The service builds a prompt that labels retrieved chunks as untrusted data, asks the model to use only those sources, and assigns `[n]` citation positions. It calls the generation backend's `/tokenize` route while adding sources. If tokenization is unavailable, `ModelClient` uses a conservative four-bytes-per-token estimate.

Generation uses streaming `POST /v1/chat/completions`. The client translates OpenAI-style `choices[0].delta.content` records into the RAG event contract. A failed or malformed generation stream does not damage the index and subsequent search requests remain available. A closed downstream `DataSink` causes the upstream callback to stop, providing best-effort cancellation.

Current query limitations:

- only the last message content is used; full conversational history and roles are not forwarded;
- non-streaming RAG responses are not implemented;
- context packing is ordered truncation rather than a dedicated relevance/budget optimizer;
- citation claims in generated prose are not validated against emitted source numbers;
- the public coordinator does not expose a general OpenAI-compatible chat endpoint.

Applications that need their own agent loop can call `/v1/rag/search` and then call an OpenAI-compatible generation backend themselves. The private ports are intentionally loopback-only.

## 8. Concurrency and ownership

The HTTP listener can dispatch requests concurrently. The service uses different coordination for reads and writes:

- `active_` is an atomically loaded/stored `shared_ptr`, giving each search or query a stable generation lifetime.
- `mutation_` serializes ingestion requests.
- candidate ingestion does its expensive work outside the active generation and only swaps after successful persistence.
- rag-cpp's corpus permits concurrent read searches through shared locking and excludes mutation/build operations.
- hybrid retrieval internally overlaps its BM25 and dense branches.

The current server does not yet expose queue sizes, work priorities, admission control, or `429` backpressure. On a mobile target, rag-cpp's process-wide worker pool also needs a configurable thread cap because its desktop default uses approximately all available CPU cores. A practical first mobile policy is one serialized writer and one or two bounded concurrent readers.

## 9. Model backend contract

The product build uses rag-cpp's OpenAI embedder adapter against a local HTTP server. A manually managed embedding backend must provide:

- `GET /health`;
- `POST /v1/embeddings` with the configured model name and vector dimension.

The generation backend must provide:

- `GET /health`;
- streaming `POST /v1/chat/completions` in OpenAI delta format;
- optionally `POST /tokenize` for exact context budgeting.

llama.cpp is the built and tested backend. Other local OpenAI-compatible servers can be attached through `config/server.backends.json`, subject to those route and response contracts. Authentication headers, TLS, non-loopback backends, and provider-specific compatibility are not implemented in v0.1.

## 10. Test and specification enforcement

The native test suite uses deterministic rag-cpp hash embeddings, temporary `.ragdb` files, and a small in-process HTTP generation stub. There is no `fake-llama-server` executable and no test-only public route.

Current scenarios cover:

- C++17/C++23 boundary intent and loopback validation;
- persistence and reopen;
- identical upsert and replacement;
- failed embedding isolation;
- lexical, dense, and hybrid search;
- stable source fields;
- SSE ordering and formatting;
- context-budget rejection;
- basic identifier and log-content safety.

`lrs-spec-check` verifies pinned dependency revisions and ensures every active catalog entry names an existing test tag. It also scans normative specification lines and acceptance/exit criteria for stable requirement IDs.

The suite is valuable but not yet the complete matrix described by the specification. In particular, there is no automated end-to-end test that starts the public coordinator executable with real GGUF models, no disconnect/capacity assertion through the public socket, and no sanitizer/coverage CI configuration committed here yet. `config/server.models.json` is the manual real-model smoke path.

## 11. Mobile and Android implementation

The normative integration handoff for the adjacent application is
[`MOBILE_RAG_CONTRACT.md`](MOBILE_RAG_CONTRACT.md). Start there when wiring
`../mobileAgent`; this section explains the implementation architecture.

Android now has an initial `arm64-v8a` build target. The artifact is the in-process `libragcpp_mobile.so`, not an Android command-line server. Flutter calls it through Dart FFI and keeps the `.ragdb` in app-private storage.

The intended mobile split is:

```text
Flutter application
  |-- existing Flutter LLM engine --> answer generation
  |-- LiteRT/Flutter embedding model --> document and query vectors
  `-- libragcpp_mobile.so --> chunk/index/search/persist
```

llama.cpp is unnecessary in that topology when the Flutter stack supplies both generation and embedding inference. If it only supplies generation, a separate embedding model/runtime is still required because every dense query needs an embedding.

The current mobileAgent generation models do not expose an embedding API. Vector retrieval therefore uses a separate TFLite embedding model through Flutter Gemma's existing LiteRT worker. The vendored catalog includes quantized Gecko-110M models for token-free English retrieval and EmbeddingGemma-300M models for broader language coverage. This is an in-process model, not a localhost backend. The model is installed once, can embed documents ahead of chat, and remains available to embed each live query.

Without that model, the mobile bridge remains useful as a persisted BM25 store through `upsertLexical` and lexical search. It does not substitute hash vectors or generation-model output for semantic embeddings.

The adjacent `../mobileAgent` repository already has a Dart lexical retrieval service and, through Flutter Gemma, LiteRT embeddings plus a qdrant-edge native vector store. rag-cpp should replace those retrieval layers if adopted, rather than becoming a third independent index.

The implemented mobile C ABI can:

1. open or create a persisted `.ragdb`;
2. return the exact native chunks and embedding text as JSON;
3. upsert a lexical-only document;
4. accept row-major precomputed document vectors from Flutter/LiteRT;
5. accept a precomputed query vector and run dense or hybrid retrieval;
6. return stable source records as JSON;
7. reopen and search the persisted index.

The precomputed-vector C ABI keeps asynchronous Flutter inference outside C++. Flutter asks the bridge to prepare chunks, embeds every returned `embedding_text` with `TaskType.retrievalDocument`, and passes the vectors back in the same order. Query search uses `TaskType.retrievalQuery`. The bridge normalizes vectors before storing or scoring them.

For ordinary mobile chatbot retrieval, use CPU NEON plus HNSW for similarity. GPU/NPU effort should focus on embedding inference. A Vulkan/OpenCL vector-search backend is unlikely to improve single-query latency enough to justify its complexity.

The pinned rag-cpp core and mobile bridge have been built with Android NDK 28.2, API 24, and run on a PLQ110 (`arm64-v8a`, Android API 36). A native on-device smoke test persisted a document and returned it through hybrid vector search with supplied vectors. A Flutter integration test then loaded the packaged library from mobileAgent, persisted a lexical document through Dart FFI, and retrieved it successfully. The stripped shared library is approximately 1.6 MB in the current build.

`../mobileAgent/lib/retrieval/native_rag_index.dart` is the initial Dart wrapper. The Android APK packages the library through `android/app/src/main/jniLibs/arm64-v8a`. `tools/sync_mobile_agent_android.sh` rebuilds and copies the artifact.

Current mobile limitations:

- operations on one native handle are serialized;
- lexical-only and vector documents cannot be mixed in one index;
- deletion, metadata filters, stats, and compaction are not exposed yet;
- the bridge currently compiles the complete rag-cpp source target rather than a size-minimized source subset;
- mobileAgent's existing `RetrievalService` still owns production call sites; the native wrapper is available but has not replaced the Dart/qdrant paths;
- the application must retain one embedding model identity and dimension for the lifetime of an index; a mobile manifest check is still needed;
- Dart FFI calls that can persist or rebuild a large index should move to a dedicated isolate before production use.

## 12. Known gaps and next engineering steps

The current vertical slice is intentionally narrower than the engineering specification. Important outstanding work includes:

- public deletion, document listing, raw/precomputed vector insertion, and metadata filters;
- exact embedding-server capability and dimension probing;
- bounded queues, request size limits, priorities, and overload responses;
- authentication, TLS termination policy, and explicit CORS controls;
- robust process supervision, signal forwarding, restart policy, and degraded health;
- transactional database/manifest publication and crash-recovery exercises;
- full conversation-aware prompt construction and citation verification;
- structured metrics and privacy-reviewed logging;
- real-model automated behavior tests, sanitizers, coverage, quality fixtures, and performance baselines;
- Linux validation and production hardening of the Android shared-library target.

Changes should update behavior tests and `requirements.json` alongside implementation. Normative future behavior belongs in the specification; verified current behavior belongs here only after the code and tests exist.
