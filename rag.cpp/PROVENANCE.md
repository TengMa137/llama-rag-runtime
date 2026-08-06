# Provenance and ownership

`rag.cpp/` is the repository-owned retrieval core, initially derived from
rag-cpp v0.1.0 at
commit `cfe46cee87fccb9ca5dee68d416229489285fdea`.

The original work is copyright its contributors and remains available under
the MIT license in `LICENSE`. The upstream history is
<https://github.com/1ay1/rag-cpp>.

## Product changes

- Converted the former git submodule into tracked source.
- Limited the runtime target to explicit product-core sources.
- Replaced the C++23 `std::expected` dependency with a compact C++20 result
  value for NDK and broad desktop toolchain support.
- Replaced line-indivisible chunking and the local 384-byte patch with a
  deterministic, token-measured, UTF-8-safe hierarchical chunker.
- Added an embedding policy and complete chunking fingerprint to additive
  `.ragdb` v1 metadata (format minor 2).
- Kept `ragcpp::ragcpp` as a transitional CMake alias; new code should use
  `rag::rag`.

Future changes must be recorded here or in `CHANGELOG.md`, retain the MIT
notice, and preserve `.ragdb` compatibility unless a documented major-format
migration is introduced.
