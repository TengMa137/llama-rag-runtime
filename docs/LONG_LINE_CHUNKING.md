# Long-line chunking patch

## Status

The pinned rag-cpp revision and current upstream `main` treat each newline-delimited source line as indivisible. `ChunkOptions::max_chars` therefore acts as a soft target: one Markdown paragraph, minified JSON object, OCR/PDF extraction line, table row, log record, URL, or encoded value can produce a chunk larger than the configured embedding budget.

This working tree carries a local modification to `third_party/rag-cpp/src/text/chunker.cpp`. Before the normal line-window chunker runs, it splits an oversized source line into UTF-8-safe pieces no larger than `max_chars`. Every piece retains the original source-line number, so citations still refer to the submitted document rather than synthetic line numbers.

## Client contract

Desktop clients should continue sending complete documents to `POST /v1/rag/documents`. They must not independently choose chunks or compute document vectors. The coordinator passes the document to rag-cpp, which chunks it and embeds the resulting chunk text through the configured embedding backend.

The Android precomputed-vector API is different: callers must first ask the native bridge to prepare its exact chunks, embed each returned `embedding_text`, and pass the vectors back in the same order.

## Submodule warning

`third_party/rag-cpp` is a Git submodule. A dirty source file inside it is not stored by a commit in this parent repository. Running commands such as `git submodule update --checkout`, cleaning the submodule, or replacing it with another checkout removes the patch.

Before distributing this behavior, commit and test the fix in the rag-cpp repository, push that commit, update this repository's submodule pointer and recorded dependency pin, and update the chunking fingerprint so existing indexes are not silently opened under different chunking behavior.

## Required upstream tests

An upstream version of the patch should cover:

- an ordinary multiline document whose lines fit the limit;
- a long line containing spaces and punctuation;
- multibyte UTF-8 near every hard boundary;
- a long string with no natural separator;
- empty input and `max_chars` edge values;
- preservation of original `start_line` and `end_line` values.
