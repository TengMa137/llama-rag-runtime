# llama-rag-runtime Technical Overview

**Document role:** current implementation walkthrough  
**Repository version:** v0.1 development snapshot  
**Last reviewed:** 2026-08-11

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

The repository-owned rag.cpp is embedded as a C++20 static library. It owns
document/chunk records, embeddings, BM25, dense retrieval, hybrid retrieval,
and `.ragdb` persistence. A project-owned C ABI keeps C++ types out of the
C++17 coordinator.

```text
client
  |
  | HTTP JSON / SSE on 127.0.0.1:8080
  v
llama-rag-server (C++17)
  |-- lrs_core: config, model HTTP client, service logic
  |-- lrs_bridge: C ABI boundary compiled as C++20
  |     `-- rag.cpp --> data/*.ragdb
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
| `include/lrs/bridge.h` | C-compatible ownership and retrieval boundary around rag.cpp |
| `include/lrs/config.hpp` | Runtime configuration model |
| `include/lrs/model_client.hpp` | Private llama-server generation/tokenization client |
| `include/lrs/service.hpp` | Coordinator service API and active-index ownership |
| `src/bridge.cpp` | rag.cpp configuration, persistence, upsert, deletion, search, and stable public chunk IDs |
| `src/config.cpp` | JSON loading and v0.1 safety validation |
| `src/main.cpp` | CLI, child processes, readiness loop, and HTTP routes |
| `src/model_client.cpp` | Health, tokenization, OpenAI chat-completion SSE parsing, and cancellation |
| `src/service.cpp` | Ingestion transaction, retrieval, context budgeting, and public RAG SSE |
| `tests/tests.cpp` | Deterministic native unit/component/behavior/fault scenarios |
| `tools/spec_check.cpp` | Requirement-to-test and pinned-dependency consistency checks |
| `requirements.json` | Machine-readable active/deferred requirement catalog |
| `third_party/llama.cpp` | Pinned upstream inference engine and `llama-server` |
| `rag.cpp` | Repository-owned retrieval and persistence core |

The backend-adaptation work inside `rag.cpp` is divided by responsibility:

| Core path | Responsibility |
|---|---|
| `core/keys.*` | Canonical source normalization, document fingerprints, and stable public chunk keys |
| `preparation/document_preparer.*` | Storage-free normalization, chunking, batched embedding, and validation |
| `backend/candidate_backend.hpp` | Portable records and the thread-safe storage/candidate contract |
| `backend/embedded_backend.*` | Correctness-first immutable in-process generations with revision-checked atomic publication |
| `backend/embedded_checkpoint.*` | Validated v1.2 checkpoint mapping for active revisions, stable IDs, vectors, and represented WAL position |
| `backend/postgres_backend.*` | Optional transactional candidate backend with exact JSONB filters, owned BM25, and pgvector exact/HNSW search |
| `postgres/connection_pool.*` | Bounded libpq connections, prepared statements, acquisition/statement timeouts, and cancellation |
| `postgres/migrations.*` | Advisory-locked, project-owned schema versions for corpora, revisions, chunks, postings, and jobs |
| `dense/index.*` | Replaceable dense-index contract and native exact recall oracle |
| `dense/tiered_index.*` | Immutable dense base, exact mutable-generation delta, base tombstones, and deterministic merged search |
| `dense/faiss_index.*` | Optional desktop-only FAISS adapter behind explicit policies |
| `dense/*_sidecar.*` | Disposable validated native/FAISS index caches |
| `ingestion/job.*` | Durable job states and content-free status projections |
| `ingestion/job_store.*` | Private, CRC-framed append-only job records with torn-tail recovery |
| `ingestion/coordinator.*` | Bounded workers, revision arbitration, supersession, recovery, and synchronous/asynchronous execution |
| `ingestion/embedded_durability.*` | Snapshot-before-log-prefix-truncation ordering for embedded durability |
| `ingestion/embedded_runtime.*` | Composite checkpoint, job-log, coordinator, embedding, and backend-neutral search ownership |
| `ingestion/postgres_job_store.*` | Corpus-scoped durable job records using the shared ingestion-job codec |
| `ingestion/postgres_runtime.*` | Composite PostgreSQL backend, job store, coordinator, embedding, and search ownership |
| `ingestion/runtime.hpp` | Backend-neutral application ownership and ingestion/search lifecycle contract |
| `migration/contract.*` | Ordered batch, audit, progress, resume, checksum, and sampled-search validation contracts |
| `migration/embedded_endpoint.*` | Read-only checkpoint/ready-WAL export and resumable atomic checkpoint destination |
| `migration/postgres_endpoint.*` | Repeatable-read export and corpus-scoped durable migration progress/import |
| `retrieval/runtime.*` | Backend-independent filtering, RRF, feature reranking, MMR, stitching, and result resolution |
| `tools/eval/corpus.*` | Reproducible normalized vector-corpus contract shared by quality and runtime benchmarks |
| `tools/eval/dense.*` | Exact-oracle recall, overlap, filtered completeness, build, latency, and memory evaluator |
| `tools/eval/qrels.*` | Backend-neutral ranking and immediate-visibility evaluator over public retrieval contracts |
| `tools/eval/runtime.*` | Exact-delta overhead and concurrent-compaction availability evaluator |
| `tests/backend_contract_suite.*` | Reusable candidate-backend activation, filtering, revision, deletion, fetch, and result-shape contract |
| `index/corpus.*` and `pipeline/*` | Existing `.ragdb` compatibility implementation used by current desktop/mobile entry points |

The new backend path is intentionally layered rather than coupled to the HTTP
service. Each published generation has an active document/chunk catalog and
BM25 index, a shared immutable dense base, an exact dense delta, and tombstones
for base rows removed by replacement or deletion. The catalog owns metadata and
revision visibility; `TieredDenseIndex` owns only vector representations and
requires the catalog to supply prefiltered allowed IDs. Search scores base and
delta independently, merges them deterministically, and never exposes a
tombstoned or superseded row.

Activation rebuilds the current correctness-first BM25/catalog snapshot and
only the small exact delta; it does not rebuild the dense base. Manual
compaction builds a new exact base outside the writer lock, checks the configured
1 GiB memory budget, and publishes only if the source generation is still
current. The default event-driven maintenance worker schedules that operation
when a threshold is crossed; deterministic hosts may disable it. Existing
readers retain the old generation through shared ownership.
Maintenance statistics expose base rows, delta rows, tombstones, generation,
estimated build bytes, and threshold state, including a maintenance signal when
the estimated build exceeds its configured budget. Portable checkpoints retain
active prepared rows, deletion revisions, stable public IDs, normalized vectors,
generation, and represented job-log position in the existing v1.2 container.
They intentionally exclude optimized dense state. Checkpoint publication occurs
before atomic job-log prefix compaction; compaction retains all nonterminal jobs
and a bounded recent terminal-status history before every newer frame. Torn
tails are repaired before a newly opened job log accepts appends.
`EmbeddedRuntime` restores the checkpoint first, then replays that surviving
log through the same revision/idempotency checks, and owns query embedding for
backend-neutral search. Ready-job completion triggers checkpointing at the
configured 256 MiB WAL threshold for both synchronous and asynchronous callers;
maintenance serialization prevents competing workers from publishing the same
checkpoint concurrently. `rag::Engine::open_runtime` exposes this ownership
through stable ingestion, job, deletion, search, and checkpoint entry points;
the old constructors and `corpus()` accessor remain the compatibility path.
The complete ordered checklist is maintained in `PORTABLE_RUNTIME_ROADMAP.md`.

The active chunk catalog stores text, metadata, source ranges, and revision
identity but no vector copy. Prepared documents remain the durable vector
source, dense indexes own the representation they search, and only bounded
`fetch(... include_embedding=true)` calls materialize vectors for MMR. Container
publication writes the header/table and each section sequentially while updating
CRC incrementally, avoiding a second full-file serialization buffer; output is
byte-identical to the v1.2 serializer.

`store::ContainerView` is the read-side contract: it owns a read-only file
mapping, validates magic/version/CRC/table bounds/unique tags/non-overlap once,
and returns section views that are valid for exactly the mapping lifetime. Both
portable checkpoint recovery and the owning compatibility reader use this same
validator. The compatibility reader copies validated sections individually
(including unknown tags), while portable recovery parses directly from mapped
views and avoids a whole-file staging allocation. Atomic rename publication
means an already-open view safely continues reading its original inode while a
new generation becomes visible to later opens.

The native exact base is cached beside the checkpoint as
`.dense.native-exact-v1`. Its fingerprint covers the implementation/algorithm
version, embedding identity, ordered public chunk keys, and normalized vector
bytes. The cache is atomically replaced and never authoritative: reopen accepts
it only after container, fingerprint, count, dimension, key-uniqueness, finite,
and unit-norm checks. Missing or corrupt caches are rebuilt from `.ragdb` and
republished. Accepted vector matrices remain mapped; only public keys and the
small exact-search scratch row are resident allocations. The mutable delta
continues to use an owned exact index.

Prepared revisions are immutable shared objects. Publishing a generation,
retaining an old reader generation, and taking a checkpoint snapshot therefore
copy document-map references rather than document content or embeddings.
`dense::VectorSource` is explicitly a call-lifetime view of keys and vector
spans; dense builds, compaction, fingerprints, and sidecar publication consume
those views synchronously and copy only the representation they own. This keeps
the input corpus from being duplicated merely to cross a module boundary.

`dense::NativeHnswIndex` is the approximate native implementation behind the
same `DenseIndex` contract. It assigns generation-local numeric graph IDs and
maps results back to stable `ChunkKey` values. Allowed IDs are applied during
the graph walk; if an approximate walk cannot fill the required filtered count,
the graph scans its own exact matrix rather than post-filtering or consulting a
second vector copy. The adapter deliberately disables the legacy SQ8 walk
mirror, reports primary/compressed vector memory separately, and keeps one f32
representation available as the recall and selective-filter oracle.

`dense::DensePolicy` is the construction contract; arbitrary factory strings
are not accepted. Native policies resolve to `exact` or `hnsw`, while
`automatic` selects exact below 2,000 vectors and HNSW at or above that
threshold. `TieredDenseIndex` uses the factory only for immutable base builds;
its delta is always native exact, so updates become searchable without waiting
for an approximate rebuild. Backend statistics expose the resolved
implementation, algorithm, exactness, resident bytes, and mapped bytes.

Native HNSW generations use `.dense.native-hnsw-v1` disposable sidecars. Exact
and HNSW caches share one fingerprint implementation covering the ordered
public keys, normalized f32 bytes, embedding identity, algorithm version, and
validated HNSW parameters. HNSW cache loading additionally validates sequential
numeric IDs, graph dimensions, key order, and absence of SQ8/PQ mirrors.
Corrupt or stale graphs rebuild from `.ragdb` and are atomically republished.
Serializing a sealed graph temporarily materializes its link lists and releases
them again before returning, so cache publication does not leave both adjacency
layouts resident.

An optional installed FAISS package can provide `flat`, `hnsw`, `ivf-sq8`, or
`ivf-pq` immutable bases when the build enables `LRS_ENABLE_FAISS`. FAISS types
do not cross the adapter header. Public chunk keys map to stable sequential
numeric row IDs, vectors and queries are unit-normalized for inner-product
cosine search, and selective filters are resolved before final top-k. Filtered
search exhausts the relevant representation when an approximate selector cannot
prove completeness. IVF construction rejects undersized training sets.

FAISS state never enters `.ragdb`. A CRC-protected
`.dense.faiss-<algorithm>-v1` sidecar is accepted only when its corpus,
embedding identity, key order, algorithm, parameters, dimension, and serialized
index agree. Missing or corrupt caches rebuild from the portable checkpoint;
the exact mutable delta continues to serve updates independently of that base.

The desktop service now uses `rag::Engine::open_runtime` for ingestion,
deletion, filtering, and retrieval. New mobile indexes use synchronous prepared
activation through the same revision, exact-delta, and v1.2 checkpoint
primitives. The bridge and mobile opener retain a legacy `Corpus` fallback for
already-published databases.

The ingestion coordinator now implements the shared internal lifecycle
`queued -> chunking -> embedding -> publishing -> ready`, with terminal
`failed`, `superseded`, and `cancelled` states. Complete ready records retain
prepared chunks and normalized vectors, allowing an empty backend to recover
without calling the embedding provider. Nonterminal records are requeued after
restart, deletion supersedes pending work, and a bounded asynchronous queue
rejects excess submissions. HTTP remains synchronous by default; exact
`Prefer: respond-async` returns `202` with a `Location`, and
`GET /v1/rag/jobs/{job_id}` exposes only the content-free job projection.

Generated applications go to `build/<preset>/bin`, static libraries to `lib`, and developer-only test/check executables to `libexec`.

## 3. Build graph and language boundary

The top-level project requires CMake 3.24 or newer and uses target-scoped language features:

```text
llama-rag-server (C++17)
  `-- lrs_core (C++17)
       |-- lrs_httplib (C++17)
       `-- lrs_bridge (implementation C++20, public header is C)
            `-- rag::core (C++20)

llama-server (llama.cpp C++17 baseline)
lrs-tests (native Catch2 test executable)
lrs-spec-check (native traceability executable)
```

The bridge exposes opaque `lrs_index*` handles, plain option structures, integer status values, and explicit string destructors. No rag.cpp header or C++ ABI type crosses into coordinator headers. rag.cpp also provides a versioned opaque C ABI whose extensible options begin with `abi_version` and `struct_size`.

The owned rag.cpp core contains no llama integration, agent protocol, Metal backend, examples, CLI, or research modules. The coordinator talks to a loopback embedding service instead of loading a model inside rag.cpp.

Catch2 3 is a system build dependency for the parent desktop suite when
`LRS_BUILD_TESTS=ON`; CMake never downloads it. On macOS, install it with
`brew install catch2` before configuring the development preset. The owned
rag.cpp behavior tests themselves remain framework-free.

## 4. Configuration and process lifecycle

`Config` has three endpoints:

- `listen`, defaulting to coordinator port 8080;
- `inference.embedding`, defaulting to port 8081;
- `inference.generation`, defaulting to port 8082.

The private inference hosts must be loopback addresses and all three ports must
be distinct. The embedding dimension and context sizes are explicit
configuration; pooling is restricted to `mean`.

The embedding and generation hosts are always loopback-only. The coordinator
may bind beyond loopback only when `authentication.api_key` contains at least
16 non-whitespace bytes. When configured, a pre-routing handler requires an
exact `X-API-Key` match on every endpoint and compares it without an
early-exit. Authentication does not provide transport encryption; non-loopback
deployments must use TLS termination or a private encrypted network.

With `inference.spawn=true`, `src/main.cpp` forks two children and executes the configured `llama-server` binary. The embedding child receives `--embeddings`, `--pooling`, context, batch, and model flags. The generation child receives its model and context flags. The coordinator owns the child PIDs and sends `SIGTERM`, then waits for them during normal stack unwinding.

With `spawn=false`, the same service attaches to already-running compatible backends. There is no child lifecycle management in that mode.

Startup then performs up to 120 initialization attempts at one-second intervals. Initialization opens or creates the rag.cpp engine, attaches its embedder, and checks both model health endpoints. Only after initialization succeeds does the coordinator bind its public HTTP listener.

Current lifecycle limitations:

- child supervision is POSIX-only and Windows rejects `spawn=true`;
- there is no signal handler, restart backoff, child liveness monitor, or configurable startup timeout;
- startup validates configured dimension and pooling values but does not probe the embedding server to prove them;
- `/health` and `/ready` currently return the same small process-level status object.

## 5. Index and persistence model

`retrieval.backend` selects `embedded` (the default) or `postgres` in a build
configured with `LRS_ENABLE_POSTGRES=ON`. For embedded storage,
`retrieval.embedded.path` is the active `.ragdb`; legacy `index.path` remains a
compatibility fallback. Its typed dense policy selects `native` plus
`automatic|exact|hnsw`, or, in an enabled desktop build, `faiss` plus
`flat|hnsw|ivf-sq8|ivf-pq`. Invalid implementation/algorithm pairs fail during
configuration validation. The checkpoint's sibling `<path>.manifest.json`
records the schema, retrieval-core version, chunking fingerprint, chunk
settings, and embedding dimension. An existing database is rejected when the
manifest is missing or incompatible.

Current chunk settings in `src/bridge.cpp` are:

| Setting | Value |
|---|---:|
| Algorithm | hierarchical, UTF-8-safe, token-measured |
| Maximum lines | 40 |
| Desktop hard budget | configured context minus 8 reserved exact tokens |
| Desktop target/overlap | 75% / 12.5% of usable exact tokens |
| Mobile hard/target/overlap | 384 / 320 / 32 conservative UTF-8 bytes |
| Markdown heading context | disabled |

Desktop requires the embedding backend's exact `/tokenize` endpoint; startup
fails instead of estimating when it is unavailable. Mobile remains conservative
until its embedding stack exposes an exact tokenizer. The core policy
distinguishes target, hard, overlap, and reserved budgets and includes
model/tokenizer identities, dimension, prefixes, counting mode, and
invalid-UTF-8 policy in the persisted fingerprint.

The C++ facade also exposes named retrieval profiles over one pipeline:
`efficiency` uses `max(3 * top_k, 24)` candidates, `balanced` (the default)
uses `max(6 * top_k, 60)`, and `quality` uses `max(20 * top_k, 200)` plus MMR
and adjacent-parent stitching. All use RRF by default and accept explicit
overrides. Diagnostics report the selected profile, stages, candidate pool,
elapsed time, and fallback reasons.

Public chunk IDs are not rag.cpp numeric IDs. The bridge computes a stable FNV-1a-derived ID from document URI, source line range, and normalized chunk text. Replacing text therefore produces new chunk IDs while repeated searches of the same generation retain the same IDs.

### Ingestion and publication

`POST /v1/rag/documents` implements text-document upsert:

1. Parse and validate `id`, `content`, optional `title`, string metadata tags,
   content type, and the optional async preference.
2. Deduplicate identical ready or pending input by document fingerprint.
3. Persist a revisioned queued job before acknowledging async acceptance.
4. Chunk and embed the complete document without changing searchable state.
5. Persist the complete publishing record, including normalized vectors.
6. Atomically activate the revision in BM25 and the exact dense delta after a
   last-write-wins revision check.
7. Persist `ready`; synchronous callers wait for this state, while async callers
   poll the job resource.
8. Checkpoint and compact the represented job-log prefix at the configured WAL
   threshold.

Embedding or publication failure leaves the previous active revision unchanged.
Searches that already loaded a shared generation finish against it.

`DELETE /v1/rag/documents/{id}` uses the same revision/job path. It immediately
publishes a deletion revision so the document is excluded from lexical, dense,
and hybrid results. Deleting an absent ID remains an idempotent success with
`deleted: false`.

For embedded storage, the durable ingestion unit is the synced sibling job/WAL
record; `.ragdb` itself is the latest atomic v1.2 checkpoint. Checkpoint rename
precedes compaction of the represented log prefix. A crash before the rename
replays the old checkpoint plus the full log; a crash after it may replay
retained duplicates, which activation treats idempotently. Thus acknowledged
state is durable across the checkpoint and adjacent log together, not in the
`.ragdb` file alone until the next checkpoint.

The PostgreSQL runtime uses the same `PreparedDocument`, revision, job, and
candidate contracts. It commits an active document, chunks, vectors, postings,
revision ledger, and generation increment in one serializable transaction.
Ready job states are durable rows using the shared job codec, so restart can
recover without embedding again. Hybrid lexical and dense candidate reads share
one repeatable-read transaction. Exact JSONB predicates are bound inside both
queries before `LIMIT`; the runtime never implements permissions as overfetch
and post-filter. BM25 uses the repository tokenizer and formula rather than
PostgreSQL full-text ranking. Exact pgvector inner-product search is the
default; HNSW uses strict iterative scans and is explicit opt-in.

PostgreSQL configuration names a connection-string environment variable rather
than carrying credentials in JSON. Non-loopback hosts require
`sslmode=verify-ca` or `verify-full`; pools are bounded and operations have
acquisition and server statement timeouts. Connection or backend failure is
returned to the caller and never falls back to local state. The PostgreSQL
backend is excluded from Android.

### Administrative migration

`lrs-rag-migrate` is built only when PostgreSQL support is enabled. Both
directions use the same ordered `RevisionedDocument` contract. An embedded
source loads `.ragdb` and replays only complete `ready` records from the sibling
`.jobs` log through a read-only parser; it never opens either source file for
write or repairs a torn tail. A PostgreSQL source holds one repeatable-read,
read-only transaction and does not invoke schema migration.

The transfer is ordered by public document key and bounded by the configured
batch size. PostgreSQL progress is stored in `migration_runs`; embedded progress
is a CRC-protected atomic `<destination>.migration` sidecar. Destination data is
published before progress. On restart, the runner independently proves the
actual destination is an exact source prefix, which safely handles a crash
between those two writes. Conflicting content fails closed.

Preflight and final audits validate normalized source content, document hashes,
stable chunk IDs, ordinals, dimensions, embedding/chunking identities, and unit
vectors. Completion requires equal live counts, canonical document checksums,
bitwise vector checksums, and sampled exact top-k rankings. The JSON report is
written to stdout and optionally to an atomic report file. The tool does not
modify runtime configuration; backend cutover is an explicit operator action
after a complete report.

## 6. Retrieval path

`POST /v1/rag/search` accepts text, not a raw vector. `src/bridge.cpp` supports:

- `lexical`: rag.cpp BM25;
- `dense`: embed the query and run dense similarity search;
- `hybrid`: run rag.cpp's standard hybrid pipeline with reciprocal-rank fusion.

An optional `filter` object accepts a string or string array per key. Values are
sorted and deduplicated; values within one key are ORed and keys are ANDed.
Missing keys, unknown keys, empty value arrays, and an explicitly empty filter
match nothing. Only an absent `filter` is unrestricted. Filters retain the
64-key/65,536-byte bounds and additionally allow at most 256 values per key and
1,024 values total. The predicate is applied before lexical truncation, inside dense candidate
selection, and before/after hybrid fusion so disallowed documents cannot leak
through a semantic path. The response contains sorted results with document ID,
stable chunk ID, title, metadata, one-based rank, mode-specific score, line
offsets, and source text. Filtered responses also contain a normalized
`filter_ack` with contract `metadata-any-of-v1`; callers requiring enforcement
must compare it exactly and fail closed. `top_k` must be between 1 and 100.

Within rag.cpp, hybrid lexical and dense work runs concurrently. Small dense
corpora use SIMD-accelerated exact similarity; larger corpora use HNSW according
to rag.cpp's corpus threshold. The retrieval path is CPU-oriented on desktop
and Android.

Every dense or hybrid chat query still requires one query embedding. Document embedding is normally paid during ingestion, but on-device query embedding may dominate the much smaller BM25/HNSW similarity cost.

## 7. Grounded generation path

`POST /v1/rag/query` currently uses the final message's `content` as the retrieval query. It emits:

1. `rag.started`;
2. `rag.retrieval.completed` with the authoritative ordered sources and, for a
   filtered request, `filter_ack`;
3. zero or more `rag.generation.delta` events;
4. `rag.completed`, or `rag.error` on failure.

The service builds a prompt that labels retrieved chunks as untrusted data, asks the model to use only those sources, and assigns `[n]` citation positions. It calls the generation backend's `/tokenize` route while adding sources. If tokenization is unavailable, `ModelClient` uses a conservative four-bytes-per-token estimate.

Generation uses streaming `POST /v1/chat/completions`. The client translates OpenAI-style `choices[0].delta.content` records into the RAG event contract. A failed or malformed generation stream does not damage the index and subsequent search requests remain available. A closed downstream `DataSink` causes the upstream callback to stop, providing best-effort cancellation.

The coordinator also exposes OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`. Chat completions support streaming and non-streaming responses and use the same single-pass grounded retrieval pipeline. A `rag_sources` response extension carries the ranked chunks. Filtered non-streaming responses carry `rag_filter_ack`; streaming responses carry it in the first chunk beside `rag_sources`.

Current query limitations:

- only the last message content is used; full conversational history and roles are not forwarded;
- context packing is ordered truncation rather than a dedicated relevance/budget optimizer;
- citation claims in generated prose are not validated against emitted source numbers;
- OpenAI chat currently accepts string content in the final message rather than the full multimodal content-part schema.

Applications that need their own agent loop can call `/v1/rag/search` and then call an OpenAI-compatible generation backend themselves. The private ports are intentionally loopback-only.

Agentic RAG is not implemented by the coordinator. Those research systems are not part of rag.cpp, and the runtime does not expose agent-run endpoints. The current path retrieves once and makes one generation request; bounded agent behavior belongs in a future parent-runtime phase.

## 8. Concurrency and ownership

The HTTP listener can dispatch requests concurrently. The service uses different coordination for reads and writes:

- `active_` is an atomically loaded/stored `shared_ptr`, giving each search or query a stable generation lifetime.
- `mutation_` serializes ingestion requests.
- candidate ingestion does its expensive work outside the active generation and only swaps after successful persistence.
- rag.cpp's corpus permits concurrent read searches through shared locking and excludes mutation/build operations.
- hybrid retrieval internally overlaps its BM25 and dense branches.

The current server does not yet expose queue sizes, work priorities, admission control, or `429` backpressure. rag.cpp uses operation-local bounded workers and creates no process-global pool. Writers remain serialized by the compatibility adapters; the versioned C ABI reserves per-engine reader/writer and memory-budget fields for host policy.

## 9. Model backend contract

The product build uses the neutral loopback-only `LocalHttpEmbedder` against a
local HTTP server. A manually managed embedding backend must provide:

- `GET /health`;
- `POST /v1/embeddings` with the configured model name and vector dimension.

The generation backend must provide:

- `GET /health`;
- streaming `POST /v1/chat/completions` in OpenAI delta format;
- optionally `POST /tokenize` for exact context budgeting.

llama.cpp is the built and tested backend. Other local OpenAI-compatible servers
can be attached through `config/server.backends.json`, subject to those route
and response contracts. Model-backend authentication headers, TLS, non-loopback
model backends, and provider-specific compatibility are not implemented. This
is separate from the coordinator's inbound `X-API-Key` authentication.

## 10. Test and specification enforcement

The native test suite uses deterministic rag.cpp hash embeddings, temporary `.ragdb` files, and a small in-process HTTP generation stub. There is no `fake-llama-server` executable and no test-only public route.

Current scenarios cover:

- C++17/C++20 boundary intent and loopback validation;
- persistence and reopen;
- identical upsert and replacement;
- idempotent deletion and exclusion from search;
- failed embedding isolation;
- lexical, dense, and hybrid search;
- stable source fields;
- SSE ordering and formatting;
- context-budget rejection;
- basic identifier and log-content safety;
- one shared candidate-backend contract over native exact/HNSW, every FAISS
  policy, and PostgreSQL;
- embedded cache/restart/compaction failure races and PostgreSQL rollback,
  timeout, and pool saturation; and
- migration equivalence, exact/filtered recall gates, sanitizer builds, and an
  Android arm64 cross-build.

`lrs-spec-check` verifies pinned dependency revisions and ensures every active catalog entry names an existing test tag. It also scans normative specification lines and acceptance/exit criteria for stable requirement IDs.

`RELEASE_VERIFICATION.md` records the completed deterministic backend/platform
matrix. There is still no automated CI job that downloads and starts real GGUF
models; `config/server.models.json` remains the operator-controlled real-model
smoke path. On 2026-08-07 the root-owned rag.cpp build passed that path with the
configured Granite embedding and Qwen generation models: readiness, ingestion,
hybrid retrieval, grounded completion with citations, full-process restart, and
persisted reopen all succeeded.

## 11. Mobile and Android implementation

The normative integration handoff for the adjacent application is
[`MOBILE_RAG_CONTRACT.md`](MOBILE_RAG_CONTRACT.md). Start there when wiring
`../mobileAgent`; this section explains the implementation architecture.

Android now has an initial `arm64-v8a` build target. The artifact is the in-process `librag_mobile.so`, not an Android command-line server. Flutter calls it through Dart FFI and keeps the `.ragdb` in app-private storage.

The intended mobile split is:

```text
Flutter application
  |-- existing Flutter LLM engine --> answer generation
  |-- LiteRT/Flutter embedding model --> document and query vectors
  `-- librag_mobile.so --> chunk/index/search/persist
```

llama.cpp is unnecessary in that topology when the Flutter stack supplies both generation and embedding inference. If it only supplies generation, a separate embedding model/runtime is still required because every dense query needs an embedding.

The current mobileAgent generation models do not expose an embedding API. Vector retrieval therefore uses a separate TFLite embedding model through Flutter Gemma's existing LiteRT worker. The vendored catalog includes quantized Gecko-110M models for token-free English retrieval and EmbeddingGemma-300M models for broader language coverage. This is an in-process model, not a localhost backend. The model is installed once, can embed documents ahead of chat, and remains available to embed each live query.

Without that model, the mobile bridge remains useful as a persisted BM25 store through `upsertLexical` and lexical search. It does not substitute hash vectors or generation-model output for semantic embeddings.

The adjacent `../mobileAgent` repository already has a Dart lexical retrieval service and, through Flutter Gemma, LiteRT embeddings plus a qdrant-edge native vector store. rag.cpp should replace those retrieval layers if adopted, rather than becoming a third independent index.

The implemented mobile C ABI can:

1. open or create a persisted `.ragdb`;
2. return the exact native chunks and embedding text as JSON;
3. upsert a lexical-only document;
4. accept row-major precomputed document vectors from Flutter/LiteRT;
5. accept a precomputed query vector and run dense or hybrid retrieval;
6. return stable source records as JSON;
7. reopen and search the persisted index.

The precomputed-vector C ABI keeps asynchronous Flutter inference outside C++. Flutter asks the bridge to prepare chunks, embeds every returned `embedding_text` with `TaskType.retrievalDocument`, and passes the vectors back in the same order. Query search uses `TaskType.retrievalQuery`. The bridge normalizes vectors before storing or scoring them.

For a new database, mobile prepares the complete document, validates the
precomputed row count/dimension, assigns a monotonic revision, activates it in
the exact delta, and atomically checkpoints the portable v1.2 sections. Reopen
restores the revision catalog and builds an exact base; replacement cannot
resurrect an older revision. Existing legacy `.ragdb` files continue through a
compatibility path until explicit migration.

For ordinary mobile chatbot retrieval, use CPU NEON plus HNSW for similarity. GPU/NPU effort should focus on embedding inference. A Vulkan/OpenCL vector-search backend is unlikely to improve single-query latency enough to justify its complexity.

The owned rag.cpp core and mobile bridge have been built with Android NDK 28.2, API 24, and run on a PLQ110 (`arm64-v8a`, Android API 36). A native on-device smoke test persisted a document and returned it through hybrid vector search with supplied vectors. A Flutter integration test then loaded the packaged library from mobileAgent, persisted a lexical document through Dart FFI, and retrieved it successfully. The stripped shared library is approximately 1.6 MB in the current build.

`../mobileAgent/lib/retrieval/native_rag_index.dart` is the initial Dart wrapper. The Android APK packages the library through `android/app/src/main/jniLibs/arm64-v8a`. `tools/sync_mobile_agent_android.sh` rebuilds and copies the artifact.

Current mobile limitations:

- operations on one native handle are serialized;
- lexical-only and vector documents cannot be mixed in one index;
- deletion, metadata filters, stats, and compaction are not exposed yet;
- the explicit rag.cpp product source list should continue to be checked against mobile binary-size budgets;
- mobileAgent's existing `RetrievalService` still owns production call sites; the native wrapper is available but has not replaced the Dart/qdrant paths;
- the application must retain one embedding model identity and dimension for the lifetime of an index; a mobile manifest check is still needed;
- Dart FFI calls that can persist or rebuild a large index should move to a dedicated isolate before production use.

## 12. Known gaps and next engineering steps

The current vertical slice is intentionally narrower than the engineering specification. Important outstanding work includes:

- document listing and raw/precomputed vector insertion on the desktop API;
- exact embedding-server capability and dimension probing;
- bounded queues, request size limits, priorities, and overload responses;
- TLS termination deployment and explicit CORS controls;
- robust process supervision, signal forwarding, restart policy, and degraded health;
- transactional database/manifest publication and crash-recovery exercises;
- full conversation-aware prompt construction and citation verification;
- structured metrics and privacy-reviewed logging;
- real-model automated behavior tests, sanitizers, coverage, quality fixtures, and performance baselines;
- Linux validation and production hardening of the Android shared-library target.

Changes should update behavior tests and `requirements.json` alongside implementation. Normative future behavior belongs in the specification; verified current behavior belongs here only after the code and tests exist.
