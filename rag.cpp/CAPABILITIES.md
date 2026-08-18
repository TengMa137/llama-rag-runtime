# Supported contract

| Capability | Status | Public entry point |
|---|---|---|
| Documents, metadata, chunks, line ranges, stable IDs | Supported | core types, `CandidateBackend` |
| Deterministic token-budgeted UTF-8 structural chunking | Supported | `text::ChunkOptions`, fingerprint |
| BM25 and metadata-filtered lexical search | Supported | `CandidateBackend` |
| Exact cosine and HNSW dense search | Supported | `DenseIndex`, explicit policy |
| Optional FAISS Flat/HNSW/IVF-SQ8/IVF-PQ | Supported on desktop | `DenseIndex`, explicit policy |
| Embedded and PostgreSQL/pgvector storage | Supported | `CandidateBackend` |
| Lexical, dense, hybrid retrieval | Supported | `CandidateBackend`, `Engine`, retrieval runtime |
| RRF fusion | Supported | fusion and pipeline config |
| MMR and adjacent-chunk stitching | Supported | pipeline/profile config |
| Efficiency, balanced, quality profiles | Supported | retrieval options/diagnostics |
| Caller-supplied embeddings | Supported | `AnyEmbedder` |
| Deterministic hash embeddings | Supported | `HashEmbedder` |
| Bounded loopback HTTP embeddings | Supported | `LocalHttpEmbedder` |
| `.ragdb` v1.0–v1.2, tombstones, atomic publication | Supported | `Engine`, `Container` |
| Revisioned jobs, WAL recovery, and checkpointing | Supported | ingestion runtime, embedded durability |
| Resumable embedded/PostgreSQL migration | Supported | migration endpoints and CLI |
| C++20 target and stable opaque C ABI | Supported | `rag::core`, `rag/c/rag.h` |
| SIMD CPU acceleration | Supported | automatic |
| GPU execution | Not supported | CPU-only device seam |

Agents, prompts, chat, generation, user-facing servers, model lifecycle,
general remote providers, dynamic extensions, content loaders, and research
retrieval systems are parent-runtime concerns or intentionally out of scope.
They are not a roadmap for this library. New features enter this contract only
when they are fundamental retrieval primitives with product-level tests and a
format/compatibility analysis.
