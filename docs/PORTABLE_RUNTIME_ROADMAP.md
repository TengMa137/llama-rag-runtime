# Portable runtime completion roadmap

This file is the implementation checklist for the local-first, incrementally
indexable, backend-adaptable RAG runtime. A checked item requires code, contract
tests, maintainer documentation, full native tests, and a milestone commit.

## Completed foundations

- [x] Backend-neutral records, exact metadata filters, and `CandidateBackend`.
- [x] Portable preparation and retrieval pipelines with fake-backend contracts.
- [x] Revisioned synchronous/asynchronous ingestion coordinator and durable job log.
- [x] Immutable native exact base, exact delta, tombstones, filtered merged search,
      maintenance statistics, and atomic manual compaction.
- [x] Event-driven single-worker automatic compaction with bounded build-memory policy.

## Remaining milestones

### Embedded durability

- [x] Persist active revisions, prepared chunks, vectors, generation fingerprint,
      and represented WAL position in a validated `.ragdb` v1.2 checkpoint.
- [x] Recover the checkpoint and replay only newer complete activation records.
- [x] Truncate only the WAL prefix represented by a durably published checkpoint.
- [x] Bound or compact the append-only job log without losing nonterminal jobs.
- [x] Preserve search availability if compaction, checkpoint validation, or
      publication fails.

### Application integration

- [x] Make `rag::Engine` use the backend-neutral preparation, ingestion, and
      retrieval interfaces while retaining the deprecated embedded-only accessor.
- [x] Move desktop HTTP ingestion/search/deletion off direct `Corpus` access.
- [x] Keep synchronous `POST /v1/rag/documents` response compatibility.
- [x] Implement `Prefer: respond-async`, bounded admission, `202`/`Location`, and
      `GET /v1/rag/jobs/{job_id}` with content-free errors.
- [x] Reuse revision/delta/checkpoint primitives in the synchronous mobile ABI.

### Embedded memory and index caches

- [x] Remove vector ownership from general stored chunk records.
- [x] Replace whole-file staging reads with one shared layout validator and
      lifetime-bound mapped section views on supported platforms.
- [x] Let dense indexes search mapped checkpoint/sidecar vectors without first
      materializing a second full f32 corpus.
- [x] Stream checkpoint output and CRC calculation.
- [x] Store dense indexes as disposable atomic sidecars keyed by corpus,
      embedding, algorithm, and implementation fingerprints.
- [x] Share immutable prepared revisions between generations/checkpoints and use
      non-owning vector-source views for builds, compaction, and cache writes.
- [x] Adapt native HNSW to `DenseIndex` with stable public-key mapping, one f32
      representation, pre-filtered search, and exact completeness fallback.
- [x] Add validated explicit dense policies and automatic native exact/HNSW
      selection while keeping every mutable update in the exact delta.
- [x] Eliminate simultaneous corpus-f32, HNSW-f32, and SQ8 ownership; portable
      HNSW keeps one f32 search matrix and no automatic SQ8/PQ mirror.

### Optional FAISS

- [x] Add `LRS_ENABLE_FAISS=OFF`; find an installed package without downloads.
- [x] Implement explicit validated `FlatIP`, `HNSWFlat`, `IVF-SQ8`, and `IVF-PQ`
      policies with stable numeric-ID mapping.
- [x] Use selectors only when completeness is proven and exact allowed-subset
      fallback otherwise.
- [x] Validate IVF training size, disposable cache recovery, recall gates, and
      filtered complete top-k. Keep Android native-only.

### PostgreSQL and pgvector

- [x] Add libpq backend configuration, bounded connection pool, prepared
      statements, timeouts, cancellation, and verified TLS requirements.
- [x] Own migrations for corpora, documents, chunks, terms/postings, jobs, and
      migration versions.
- [x] Activate complete prepared revisions transactionally and run lexical/dense
      candidate queries in one repeatable-read snapshot.
- [x] Reproduce owned BM25, bind exact JSONB predicates in every candidate query,
      and default to exact pgvector search.
- [x] Add opt-in HNSW iterative scans, filtered-recall tests, rollback/failure
      contracts, and fail-closed behavior.

### Explicit migration

- [x] Add `lrs-rag-migrate embedded-to-postgres` and `postgres-to-embedded`.
- [x] Validate dimensions, identities, normalization, fingerprints, stable IDs,
      and migrate only ready active revisions.
- [x] Use bounded resumable batches with destination progress and never mutate
      the source.
- [x] Validate counts, hashes, vector checksums, sampled exact searches, and emit
      a machine-readable report. Configuration cutover remains operator-owned.

### Evaluation and release matrix

- [x] Extend `lrs-rag-eval` with recall, nDCG, filtered recall, query p50/p95,
      build time, update-to-search latency, dense memory, and process RSS.
- [x] Generate reproducible 25k, 100k, and 500k 384-dimensional corpora and
      record native exact/HNSW baselines.
- [x] Run the reusable backend contract suite against embedded native, every
      enabled FAISS policy, and PostgreSQL, including recovery/failure races.
- [x] Verify stable HTTP/C/C++/mobile result contracts, `.ragdb` fixtures,
      malicious persistence inputs, round-trip migration, macOS debug/release,
      sanitizers, and Android native builds.

## Dependency order

Embedded durability precedes product integration so acknowledged jobs and active
generations share one recovery contract. Memory ownership is fixed before FAISS
so optional indexes do not cement duplicate storage. PostgreSQL follows the
backend-neutral product path, migration follows both durable backends, and the
final evaluation matrix validates every implementation against the exact oracle.
