# Provenance

This directory was originally derived from rag-cpp v0.1.0, commit
`cfe46cee87fccb9ca5dee68d416229489285fdea`. The original work is copyright
rag-cpp contributors and remains licensed under the retained [MIT license](LICENSE).

This fork is independently maintained as source owned by llama-rag-runtime. It
does not track upstream and makes no claim that later upstream APIs, features,
formats, or behavior are present here.

The fork moved the supported contract to C++20, retained `.ragdb` v1
compatibility, added deterministic token-aware UTF-8 chunking and policy
fingerprints, hardened persistence/C boundaries, restricted built-in network
access to loopback embeddings, and removed executable extensions, protocols,
serving, generation-adjacent research systems, provider clients, loaders, and
in-process model backends.
