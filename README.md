# llama-rag-runtime

`llama-rag-runtime` is a native local retrieval-augmented generation server. It combines:

- `llama.cpp` for local embedding and text-generation models;
- `rag-cpp` for document chunking, persistence, lexical search, vector search, and hybrid ranking;
- a small coordinator that retrieves relevant sources and sends grounded prompts to the generation model.

The desktop runtime exposes an OpenAI-compatible chat-completions API, so an existing OpenAI client can use local RAG by changing its base URL. Document ingestion and direct retrieval use the project-specific `/v1/rag/*` endpoints. An Android C ABI is also available for applications that provide embeddings in-process.

## Build

Clone the submodules, configure, build, and test on macOS:

```bash
git submodule update --init --recursive
cmake --preset macos-dev
cmake --build --preset macos-dev
ctest --preset macos-dev
```

The main executables are:

```text
build/macos-dev/bin/llama-server
build/macos-dev/bin/llama-rag-server
```

## Configure models

Edit `config/server.models.json` and set the two GGUF paths:

```json
{
  "listen": { "host": "127.0.0.1", "port": 8080 },
  "index": { "path": "data/knowledge.ragdb" },
  "inference": {
    "spawn": true,
    "llama_server": "build/macos-dev/bin/llama-server",
    "embedding": {
      "model": "model/granite-embedding.gguf",
      "api_model": "granite-embedding",
      "host": "127.0.0.1",
      "port": 8081,
      "pooling": "mean",
      "dimension": 384,
      "context_size": 512,
      "batch_size": 2048
    },
    "generation": {
      "model": "model/qwen-generation.gguf",
      "api_model": "qwen-generation",
      "host": "127.0.0.1",
      "port": 8082,
      "context_size": 8192
    }
  },
  "rag": { "context_tokens": 6144 }
}
```

The configured embedding dimension must match the embedding model. `rag.context_tokens` must be smaller than the generation model context size.

## Run

Start the complete runtime in one terminal:

```bash
./build/macos-dev/bin/llama-rag-server --config config/server.models.json
```

With `inference.spawn` enabled, the coordinator starts and supervises both `llama-server` processes:

| Address | Service | OpenAI-compatible endpoint |
|---|---|---|
| `http://127.0.0.1:8080` | RAG coordinator | `POST /v1/chat/completions` |
| `http://127.0.0.1:8081` | llama.cpp embedding server | `POST /v1/embeddings` |
| `http://127.0.0.1:8082` | llama.cpp generation server | `POST /v1/chat/completions` |

The llama.cpp listeners are loopback-only model backends. Use port 8080 for grounded answers and port 8082 when you intentionally want raw generation without retrieval.

To manage the model servers yourself, start compatible embedding and generation servers on ports 8081 and 8082, set `inference.spawn` to `false`, and run the coordinator with `config/server.backends.json`.

Check startup state with:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

## Add documents

Add a new document or replace an existing document with the same `id`:

```bash
curl http://127.0.0.1:8080/v1/rag/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "docs/storage",
    "title": "Storage",
    "content_type": "text/markdown",
    "content": "# Storage\nThe index is persisted in a rag-cpp database."
  }'
```

The index is stored at `index.path`. Re-ingesting identical content returns `unchanged`; changed content replaces the document atomically.

For the desktop HTTP API, send the complete document to `/v1/rag/documents`. Do not split it into chunks or vectorize it first: the coordinator asks rag-cpp for the chunks, sends those chunks to the configured embedding endpoint, and commits the document only after embedding succeeds. `/v1/rag/search` likewise accepts query text and computes its query embedding internally.

The pinned rag-cpp fixed chunker splits on line boundaries. A long multi-line document is chunked normally, but a single line longer than the configured chunk size remains one oversized chunk. Add line breaks before ingesting minified JSON, OCR output, or other unusually long unbroken text. This is an upstream chunker limitation; the runtime intentionally does not carry a local patch in `third_party/rag-cpp`.

## Chat with RAG through an OpenAI client

The coordinator accepts the standard chat-completions fields `model`, `messages`, `stream`, `temperature`, `max_tokens`, and `max_completion_tokens`. The final message must contain string content. An optional `rag` object controls retrieval:

```json
{
  "rag": { "mode": "hybrid", "top_k": 8 }
}
```

Python with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

response = client.chat.completions.create(
    model="qwen-generation",
    messages=[{"role": "user", "content": "How is the index persisted?"}],
    temperature=0.2,
    max_tokens=256,
    extra_body={"rag": {"mode": "hybrid", "top_k": 8}},
)

print(response.choices[0].message.content)
```

Streaming uses normal OpenAI `chat.completion.chunk` events and terminates with `data: [DONE]`. Responses also contain a `rag_sources` extension with the ranked source records; OpenAI clients that do not expose unknown response fields can use `/v1/rag/search` to obtain them directly.

The same request with raw llama.cpp generation uses `base_url="http://127.0.0.1:8082/v1"`. Embeddings use `base_url="http://127.0.0.1:8081/v1"` and `client.embeddings.create(...)`.

## Retrieve without generation

```bash
curl http://127.0.0.1:8080/v1/rag/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How is the index persisted?",
    "mode": "hybrid",
    "top_k": 8
  }'
```

Search supports `lexical`, `dense`, and `hybrid` modes. Results include `document_id`, `chunk_id`, line offsets, score, rank, and source text.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Report process health |
| `GET /ready` | Report whether the index and model backends are ready |
| `GET /v1/models` | List the RAG chat model in OpenAI format |
| `POST /v1/chat/completions` | Retrieve sources and return an OpenAI-compatible grounded completion |
| `POST /v1/rag/documents` | Add or replace a document |
| `POST /v1/rag/search` | Retrieve ranked source chunks |
| `POST /v1/rag/query` | Legacy RAG event stream for project-specific clients |

## Android library

The Android build produces an in-process `arm64-v8a` RAG library. It does not include llama.cpp or the HTTP coordinator; the application supplies precomputed document and query embeddings. This is the only supported flow where the caller participates in vectorization: first ask the native API to prepare its exact chunks, embed every returned `embedding_text`, then pass those vectors back in the same order. Do not independently choose chunk boundaries, because persisted vectors must correspond exactly to rag-cpp's chunks.

```bash
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/28.2.13676358"
cmake --preset android-arm64
cmake --build --preset android-arm64
```

Output:

```text
build/android-arm64/lib/libragcpp_mobile.so
```

See `docs/MOBILE_RAG_CONTRACT.md` for the C ABI lifecycle and embedding contract. Use `tools/sync_mobile_agent_android.sh` to copy a built library into the adjacent `mobileAgent` checkout.

## Development

Format project-owned C and C++ files with:

```bash
cmake --build --preset macos-dev --target format
```

Check formatting without changing files:

```bash
cmake --build --preset macos-dev --target format-check
```

Implementation details are in `docs/TECHNICAL_OVERVIEW.md`; the broader design and requirement IDs are in `docs/llama-rag-server-spec.md` and `requirements.json`.
