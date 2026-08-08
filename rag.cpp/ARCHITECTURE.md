# Architecture

The owned data flow is:

```text
Document + metadata
        │
        ▼
deterministic structural chunker ──► chunks + line ranges + fingerprint
        │
        ├──► tokenizer ──► BM25
        │
        └──► caller or loopback embedder ──► exact vectors / HNSW
                                                │
query ──► lexical + dense candidates ──► fusion/filter/MMR/stitch ──► results
                                                │
                                                └──► .ragdb + tombstones/WAL
```

`Corpus` owns documents, chunks, stable 32-bit internal IDs, embeddings, BM25,
HNSW, tombstones, and persistence state. `Pipeline` owns retrieval composition.
`Engine` is the facade used by the desktop and mobile adapters.

Chunking is structural and deterministic. A policy records model/tokenizer
identity, token budgets, prefixes, dimension, counting mode, and invalid UTF-8
behavior. Its fingerprint is persisted so a coordinator can reject incompatible
reopening. Desktop can inject an exact token counter; mobile uses the documented
conservative byte-counting policy.

Embeddings are an input boundary, not model ownership. `AnyEmbedder` accepts a
caller implementation. `HashEmbedder` is deterministic for tests and fallback.
`LocalHttpEmbedder` is the sole built-in network path and is restricted to
loopback. The parent runtime starts/stops models, owns generation, exposes HTTP,
and assembles agent context.

Persistence treats every file and WAL record as untrusted input. Container and
section bounds are checked before allocation, IDs and parallel arrays are
ordered, vectors are finite/unit-normalized/single-dimension, and parse or
allocation exceptions become `Result` errors. Atomic replacement publishes a
complete old or new file. WAL replay tolerates only a torn final record.

The C ABI contains opaque handles and status codes. Every output is initialized
before fallible work and exceptions are contained at the boundary.
