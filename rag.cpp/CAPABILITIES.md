# Runtime capability matrix

| Capability | Runtime status | Inclusion requirement |
|---|---|---|
| Deterministic structural/token-aware chunking | Shipped | Compatibility and hard-limit tests |
| BM25, exact cosine, HNSW, filtering, tombstones | Shipped | Retrieval and persistence tests |
| RRF, RSF, convex fusion | Shipped | Deterministic ranking tests |
| Efficiency, balanced, quality profiles | Shipped policy bundles | qrels quality/latency baseline |
| MMR and adjacent-parent stitching | Quality profile | Coverage evaluation |
| Injected HTTP or caller-computed embeddings | Shipped | Dimension/norm/order validation |
| GraphRAG, RAPTOR, HyDE, CRAG, Self-RAG | Roadmap | Corpus-specific quality gain and bounded cost |
| Semantic/contextual chunking | Roadmap for product target | Re-index migration and retrieval evaluation |
| SPLADE/learned sparse, ColBERT, model rerankers | Roadmap | Model lifecycle, size, latency, and qrels gain |
| Plugins, RCP, CLI, ONNX, GPU, in-process llama.cpp | Not in runtime target | Concrete product requirement and platform budget |

Research sources may remain in the owned tree for provenance and later study,
but the explicit CMake source list is the shipped boundary.

## `.ragdb` v2 roadmap

Version 2 is intentionally not implemented. It requires a concrete incompatible
need and a migration tool. Candidate work includes wider stable IDs,
transactional generation directories, an explicit WAL/checkpoint policy,
richer byte/source offsets, and resumable migrations. Version 1 remains the
portable single-file format; additive metadata belongs in its `META` JSON.
