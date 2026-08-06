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

The pinned upstream rag-cpp fixed chunker normally treats line boundaries as indivisible. This checkout currently carries a local UTF-8-safe patch that splits oversized source lines while preserving their original line numbers. See [`docs/LONG_LINE_CHUNKING.md`](docs/LONG_LINE_CHUNKING.md) before updating or cleaning the submodule.

## How retrieval works

rag-cpp stores the original document, its chunks, a BM25 lexical index over the chunk text, and an embedding vector for each chunk. A search does not scan or send every complete document to the model:

| Mode | Query work | Ranking |
|---|---|---|
| `lexical` | Tokenize the query | BM25 matches query terms against indexed chunk text; no query embedding is created |
| `dense` | Embed the query once | Vector similarity against stored chunk embeddings |
| `hybrid` | Run both paths | Reciprocal-rank fusion combines the BM25 and dense result lists |

All three modes return the stored text of only the best chunks. `/v1/chat/completions` places those returned chunks in the generation prompt; it does not give the generation model the complete vector store or every full document.

RAG chat defaults to `hybrid` retrieval with `top_k: 8`. A direct REST request may override this with a top-level `"rag": {"mode": "lexical|dense|hybrid", "top_k": 8}` extension; the standard OpenAI SDK examples below use the defaults so their request body remains portable.

## TypeScript: manage documents, search, and chat

Use Node.js 18 or newer and install the OpenAI client:

```bash
npm install openai
```

This example inserts a document, replaces it by reusing its ID, performs lexical and hybrid searches, asks a grounded question, and finally removes the document:

```ts
import OpenAI from "openai";

const server = "http://127.0.0.1:8080";

async function jsonRequest(path: string, init: RequestInit = {}) {
  const response = await fetch(`${server}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(body));
  return body;
}

async function upsertDocument(id: string, title: string, content: string) {
  return jsonRequest("/v1/rag/documents", {
    method: "POST",
    body: JSON.stringify({ id, title, content, content_type: "text/markdown" }),
  });
}

async function removeDocument(id: string) {
  return jsonRequest(`/v1/rag/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

async function search(query: string, mode: "lexical" | "dense" | "hybrid") {
  return jsonRequest("/v1/rag/search", {
    method: "POST",
    body: JSON.stringify({ query, mode, top_k: 8 }),
  });
}

async function main() {
  // Insert. A new ID returns status "indexed".
  console.log(await upsertDocument(
    "storage-guide",
    "Storage guide",
    "# Storage\nThe index is persisted in a rag-cpp database.",
  ));

  // Replace atomically by sending changed content with the same ID.
  console.log(await upsertDocument(
    "storage-guide",
    "Storage guide",
    "# Storage\nThe index is persisted atomically in a rag-cpp database.",
  ));

  // Lexical is appropriate for exact names, identifiers, and keywords.
  console.log((await search("rag-cpp database", "lexical")).results);

  // Hybrid combines exact term matching with semantic similarity.
  console.log((await search("how durable storage works", "hybrid")).results);

  const openai = new OpenAI({
    baseURL: `${server}/v1`,
    apiKey: "local", // The local coordinator does not currently validate this value.
  });
  const answer = await openai.chat.completions.create({
    model: "qwen-generation",
    messages: [{ role: "user", content: "How is the index persisted?" }],
    max_tokens: 256,
  });
  console.log(answer.choices[0].message.content);

  // Deletion is idempotent: deleted is false if the ID is already absent.
  console.log(await removeDocument("storage-guide"));
}

await main();
```

## Python: manage documents, search, and chat

Install the OpenAI client:

```bash
python -m pip install openai
```

The standard library handles the project-specific document and search endpoints; the OpenAI client handles grounded chat:

```python
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

from openai import OpenAI

SERVER = "http://127.0.0.1:8080"


def request_json(path, payload=None, method="POST"):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        SERVER + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        return json.load(response)


def upsert_document(document_id, title, content):
    return request_json("/v1/rag/documents", {
        "id": document_id,
        "title": title,
        "content": content,
        "content_type": "text/markdown",
    })


def remove_document(document_id):
    return request_json(
        "/v1/rag/documents/" + quote(document_id, safe=""),
        method="DELETE",
    )


def search(query, mode):
    return request_json("/v1/rag/search", {
        "query": query,
        "mode": mode,
        "top_k": 8,
    })


# Insert a new document.
print(upsert_document(
    "storage-guide",
    "Storage guide",
    "# Storage\nThe index is persisted in a rag-cpp database.",
))

# Replace it atomically by reusing the ID with changed content.
print(upsert_document(
    "storage-guide",
    "Storage guide",
    "# Storage\nThe index is persisted atomically in a rag-cpp database.",
))

# Use lexical for exact terminology and hybrid for general questions.
print(search("rag-cpp database", "lexical")["results"])
print(search("how durable storage works", "hybrid")["results"])

client = OpenAI(base_url=SERVER + "/v1", api_key="local")
answer = client.chat.completions.create(
    model="qwen-generation",
    messages=[{"role": "user", "content": "How is the index persisted?"}],
    max_tokens=256,
)
print(answer.choices[0].message.content)

# Remove it. Repeating this call succeeds with deleted=false.
print(remove_document("storage-guide"))
```

Streaming RAG chat uses normal OpenAI `chat.completion.chunk` events and terminates with `data: [DONE]`. Responses also contain a `rag_sources` extension with the ranked source records. The raw generation backend uses `base_url="http://127.0.0.1:8082/v1"`; the raw embedding backend uses `base_url="http://127.0.0.1:8081/v1"`.

## Agentic RAG status

Agentic RAG is not implemented in the coordinator today. The current flow is one bounded pass: retrieve once, build one grounded prompt, and make one generation request. The vendored rag-cpp contains optional components such as HyDE, CRAG, Self-RAG gates, and RAPTOR, but `llama-rag-runtime` does not configure or call them.

`docs/llama-rag-server-spec.md` describes bounded agentic retrieval as a future phase, not current behavior. There are no tool-selection loops, iterative query rewriting, retrieval grading/retry, web fallback, or `/v1/agents/*` endpoints yet. Applications can build their own agent loop by calling `/v1/rag/search` and deciding when to retrieve or generate.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Report process health |
| `GET /ready` | Report whether the index and model backends are ready |
| `GET /v1/models` | List the RAG chat model in OpenAI format |
| `POST /v1/chat/completions` | Retrieve sources and return an OpenAI-compatible grounded completion |
| `POST /v1/rag/documents` | Add or replace a document |
| `DELETE /v1/rag/documents/{id}` | Idempotently remove a document |
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
