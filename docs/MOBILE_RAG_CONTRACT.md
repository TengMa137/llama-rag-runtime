# mobileAgent RAG integration contract

This is the starting point for using this repository from the adjacent
`../mobileAgent` application. It describes the code that exists now, rather
than the future server specification.

## Status

The current Android path has been built for `arm64-v8a` with Android NDK 28.2,
integrated into mobileAgent, and exercised on the PLQ110:

- native rag-cpp persistence plus hybrid search passed with supplied vectors;
- Flutter loaded the packaged library and passed lexical persistence/search;
- the real EmbeddingGemma CPU path produced 768-dimensional vectors, indexed
  documents, passed hybrid ranking, and preserved results after reopen;
- production memory/history retrieval initializes the embedder and a
  workspace-scoped native index without a localhost service;
- the PLQ110 profile application completed a cold start with the hybrid index
  ready and remained alive after Supertonic warm-up was deferred to first use.

The existing Dart `RetrievalService` remains as the lexical fallback and the
document-loading index. Memory and prior-chat retrieval use the native hybrid
path through:

```text
../mobileAgent/lib/retrieval/memory_history_rag.dart
../mobileAgent/lib/retrieval/native_rag_index.dart
```

## Component ownership

```text
Flutter Gemma / LiteRT
  downloads and runs the embedding .tflite model
              |
              | 768-float document/query vectors
              v
NativeRagIndex (Dart FFI facade)
              |
              v
libragcpp_mobile.so
  chunks, indexes, ranks, returns sources, persists .ragdb

mobileAgent's existing Flutter LLM engine
  receives selected source text and generates the answer
```

There is no localhost process in this topology. The mobile RAG path needs
neither `llama-server` nor the coordinator executable, and rag-cpp does not
load an embedding model. SQLite is also unnecessary: the `.ragdb` file is the
rag-cpp index and persistence format.

## Recommended embedding model

Use the generic seq=512 mixed-precision LiteRT conversion from
`litert-community/embeddinggemma-300m`:

```text
embeddinggemma-300M_seq512_mixed-precision.tflite
sentencepiece.model
```

It is a 768-dimensional, multilingual model. The `.tflite` file is about
179 MB. Seq=512 is a good initial fit for the mobile bridge's short chunks;
do not use the original Transformers `model.safetensors` file with Flutter
Gemma. Access is gated: accept the Gemma license on Hugging Face and provide a
read token while downloading both files. Do not hard-code that token in the
application or commit it.

The current vendored Flutter Gemma embedding worker runs LiteRT embeddings on
CPU. Its GPU/NPU selection is intentionally disabled until those paths return
verified, nonzero vectors. The PLQ110 has an SM8750 and ample RAM/storage for
the generic model, so CPU is the correct first correctness target. Hardware
accelerator work can follow a measured real-model baseline.

## Install and open

First package the native library:

```bash
cd ../llama-rag-runtime
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/28.2.13676358"
./tools/sync_mobile_agent_android.sh ../mobileAgent
```

In mobileAgent, install the embedding model once and obtain the active model:

```dart
import 'package:flutter_gemma/flutter_gemma.dart';

const modelUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/'
    'resolve/main/embeddinggemma-300M_seq512_mixed-precision.tflite';
const tokenizerUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/'
    'resolve/main/sentencepiece.model';

await FlutterGemma.installEmbedder()
    .modelFromNetwork(modelUrl, token: hfReadToken)
    .tokenizerFromNetwork(tokenizerUrl, token: hfReadToken)
    .install();

final embeddingModel = await FlutterGemma.getActiveEmbedder();
```

Store the index in application-private support storage:

```dart
import 'package:mobile_agent/retrieval/native_rag_index.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

final support = await getApplicationSupportDirectory();
final rag = NativeRagIndex.open(p.join(support.path, 'knowledge.ragdb'));
```

Create the parent directory before opening the index. Production mobileAgent
versions the filename by embedding policy and hashes the active File access
workspace into the filename, preventing retrieval across workspace boundaries.

Model storage and File access are intentionally separate. Model storage holds
runtime weights and must be readable by the current app UID; File access is the
tool-visible workspace containing papers, skills, memory, history, and uploads.
Do not point both settings at the same raw Downloads folder. On Android, files
that survive reinstall can retain the previous app UID and be visible while
opening them fails with `EACCES`. Reset/import model storage to create an
app-owned runtime copy; do not expose model weights through `/fs`.

## Index and search

Index documents before or between chats. `documentId` is the stable upsert
key: reusing it replaces the old document atomically, and an identical upsert
returns `false` (unchanged).

```dart
final changed = await rag.upsertWithEmbeddings(
  documentId: 'manual/getting-started',
  title: 'Getting started',
  content: markdown,
  embeddingModel: embeddingModel,
);
```

For each user question, retrieve sources and pass their text to the existing
agent/generation loop:

```dart
final hits = await rag.search(
  query: userQuestion,
  embeddingModel: embeddingModel,
  mode: NativeRagSearchMode.hybrid,
  topK: 8,
);

final context = hits
    .map((hit) => '[${hit.documentId}:${hit.startLine}-${hit.endLine}]\n${hit.text}')
    .join('\n\n');
```

`lexical` is BM25, `dense` is vector similarity, and `hybrid` fuses both
rankings. Results are already ordered best-first and carry stable document and
chunk IDs, line offsets, score, and source text. The application—not the
native index—currently owns prompt construction, context budgeting, citations,
and answer generation.

If no embedding model is installed, the valid fallback is lexical-only:

```dart
rag.upsertLexical(
  documentId: 'manual/getting-started',
  title: 'Getting started',
  content: markdown,
);

final hits = await rag.search(
  query: userQuestion,
  mode: NativeRagSearchMode.lexical,
);
```

Do not mix lexical-only and vector documents in one index with the current
bridge.

## Invariants and lifecycle

- Always embed the exact `embeddingText` returned by native chunk preparation.
  `upsertWithEmbeddings` does this automatically.
- Documents must use `TaskType.retrievalDocument`; questions must use
  `TaskType.retrievalQuery`. The wrapper applies these task prefixes.
- One `.ragdb` must use one embedding model, tokenizer, output dimension, and
  task-prefix policy. Delete/rebuild or version the database path when any of
  those change.
- Keep the embedding model alive for dense/hybrid queries. Close the native
  index first, then close the embedding model when the owning service stops.
- A native handle serializes its operations. Move large persistence/rebuild
  calls to a dedicated Dart isolate before production integration.
- Completed chat turns are persisted before they are embedded. Durable-memory
  files use stable IDs, and changed content replaces the prior vectors.
  Process-local SHA-256 fingerprints avoid re-embedding unchanged text.
- Every user request searches memory/history before generation. Retrieved
  excerpts are bounded, treated as untrusted context, and injected only when
  relevant. The current request is indexed after its answer, avoiding a
  self-match during that turn.
- The Dart lexical index remains the safe fallback when EmbeddingGemma is not
  installed or cannot initialize. Native hits are filtered against currently
  present memory/history IDs so removed artifacts cannot be injected.

## Verification

The device test verifies lexical persistence and, when the staged model pair is
present, the real EmbeddingGemma-to-native hybrid path:

```bash
cd ../mobileAgent
flutter devices
flutter test integration_test/native_rag_index_test.dart -d <device-id>
```

`integration_test/native_rag_index_test.dart` expects the model and tokenizer
under the app-specific external files directory, generates a correctness
probe, indexes semantically distinct documents, asserts hybrid order, closes
and reopens the `.ragdb`, and asserts the persisted result. Staging the gated
files remains an external setup step; no Hugging Face token is committed.

## Current API surface

The mobile bridge currently supports open/create, document chunk preparation,
lexical upsert, vector upsert, lexical/dense/hybrid search, persistence/reopen,
and close. It does not yet expose delete, list, metadata filters, statistics,
compaction, or a complete RAG answer endpoint. Those are application or future
bridge work, not hidden capabilities.
