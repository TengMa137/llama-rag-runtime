# Supported contract

| Capability | Status | Public entry point |
|---|---|---|
| Documents, metadata, chunks, line ranges, stable IDs | Supported | core types, `Corpus` |
| Deterministic token-budgeted UTF-8 structural chunking | Supported | `text::ChunkOptions`, fingerprint |
| BM25 and metadata-filtered lexical search | Supported | `Corpus` |
| Exact cosine and HNSW dense search | Supported | `Corpus`, HNSW config |
| Lexical, dense, hybrid retrieval | Supported | `Corpus`, `Engine`, `Pipeline` |
| RRF fusion | Supported | fusion and pipeline config |
| MMR and adjacent-chunk stitching | Supported | pipeline/profile config |
| Efficiency, balanced, quality profiles | Supported | retrieval options/diagnostics |
| Caller-supplied embeddings | Supported | `AnyEmbedder` |
| Deterministic hash embeddings | Supported | `HashEmbedder` |
| Bounded loopback HTTP embeddings | Supported | `LocalHttpEmbedder` |
| `.ragdb` v1.0–v1.2, tombstones, atomic publication | Supported | `Engine`, `Container` |
| WAL recovery and checkpointing | Supported | `Corpus`, `Wal` |
| C++20 target and stable opaque C ABI | Supported | `rag::core`, `rag/c/rag.h` |
| SIMD CPU acceleration | Supported | automatic |
| GPU execution | Not supported | CPU-only device seam |

Agents, prompts, chat, generation, user-facing servers, model lifecycle,
general remote providers, dynamic extensions, content loaders, and research
retrieval systems are parent-runtime concerns or intentionally out of scope.
They are not a roadmap for this library. New features enter this contract only
when they are fundamental retrieval primitives with product-level tests and a
format/compatibility analysis.
