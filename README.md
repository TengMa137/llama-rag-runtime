# llama-rag-runtime

A native local RAG coordinator built from pinned llama.cpp and rag-cpp dependencies. It stores documents, chunks, lexical indexes, embeddings, and vector-search data in one `.ragdb` file and uses OpenAI-compatible local model endpoints for embedding and generation.

## Build on macOS

```bash
cmake --preset macos-dev
cmake --build --preset macos-dev
ctest --preset macos-dev
```

Only the user-facing programs are placed in `build/macos-dev/bin`:

```text
build/macos-dev/bin/llama-server
build/macos-dev/bin/llama-rag-server
```

Static libraries are under `build/macos-dev/lib`. Test and specification-check executables are under `build/macos-dev/libexec`.

## Easiest startup: one terminal

With `"spawn": true`, `llama-rag-server` starts and supervises both private llama-server processes. You run only this command:

```bash
./build/macos-dev/bin/llama-rag-server \
  --config config/server.models.json
```

This creates three processes:

```text
127.0.0.1:8080  llama-rag-server       public RAG coordinator
127.0.0.1:8081  llama-server           private embedding model
127.0.0.1:8082  llama-server           private generation model
```

The coordinator waits for both models before `/ready` returns HTTP 200. Stopping the coordinator also stops its two children.

## Manual startup: three terminals

Use this when you want to manage or replace the model backends yourself.

Terminal 1—embedding server:

```bash
./build/macos-dev/bin/llama-server \
  --host 127.0.0.1 --port 8081 \
  --model model/granite-embedding-107m-multilingual-Q6_K_L.gguf \
  --alias granite-embedding \
  --embeddings --pooling mean \
  --ctx-size 512 --batch-size 2048 --ubatch-size 2048
```

Terminal 2—generation server:

```bash
./build/macos-dev/bin/llama-server \
  --host 127.0.0.1 --port 8082 \
  --model model/Qwen3.5-4B.Q4_K_M.gguf \
  --alias qwen-generation \
  --ctx-size 8192
```

Terminal 3—RAG coordinator configured with `"spawn": false`:

```bash
./build/macos-dev/bin/llama-rag-server \
  --config config/server.backends.json
```

`config/server.backends.json` can point at another local OpenAI-compatible backend. The embedding backend must provide `/health` and `POST /v1/embeddings`. The generation backend must provide `/health` and streaming `POST /v1/chat/completions`. `/tokenize` is used when available; otherwise the coordinator uses a conservative estimate. v0.1 intentionally restricts model backends to loopback.

## What is stored? Do I need SQLite or FAISS?

No. You do not need SQLite, FAISS, a vector database, or another service.

rag-cpp owns the `.ragdb` file configured by `index.path`. It contains:

- normalized documents and deterministic chunks;
- stable document identities and chunk line offsets;
- BM25 lexical index data;
- dense embedding vectors;
- HNSW/vector-search structures when the corpus reaches the configured threshold;
- tombstone and persistence data used by rag-cpp.

`llama-rag-server` also writes `<index>.manifest.json` with the embedding dimension, rag-cpp version, and chunking fingerprint. It refuses to open an incompatible index.

Unlike a low-level FAISS API, clients submit text rather than raw vectors. The server chunks documents, calls the embedding backend, maintains the indexes, and returns resolved source records. New document IDs append to the corpus; an existing ID is atomically replaced; identical content returns `unchanged`.

Current v0.1 operations are:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health |
| `GET /ready` | Models and index are ready |
| `POST /v1/rag/documents` | Add or atomically replace a document |
| `POST /v1/rag/search` | Return ranked chunks using `lexical`, `dense`, or `hybrid` search |
| `POST /v1/rag/query` | Retrieve sources and stream a grounded generated answer |

Document deletion, raw-vector insertion, listing every document, and metadata filtering are not exposed by the current HTTP API yet.

## Index or replace a document

```bash
curl -X POST http://127.0.0.1:8080/v1/rag/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "docs/storage",
    "content_type": "text/markdown",
    "title": "Storage",
    "content": "# Storage\nThe index is persisted in a rag-cpp database."
  }'
```

Use a new `id` to append a document. Reuse an `id` to replace that document atomically.

## Retrieve ranked sources

```bash
curl -X POST http://127.0.0.1:8080/v1/rag/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How is the index persisted?",
    "mode": "hybrid",
    "top_k": 8
  }'
```

Results are already sorted by `rank` and include `document_id`, stable `chunk_id`, line offsets, score, and source text. Scores from different modes or index versions should not be compared directly.

## Let the coordinator retrieve and generate

```bash
curl -N -X POST http://127.0.0.1:8080/v1/rag/query \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"How is the index persisted?"}],
    "retrieval": {"mode":"hybrid","top_k":8},
    "generation": {"max_tokens":256,"temperature":0.2},
    "stream": true
  }'
```

The custom SSE stream emits `rag.started`, `rag.retrieval.completed`, answer deltas, and a terminal `rag.completed` or `rag.error`. Source metadata is emitted before model text.

## Use search from a TypeScript agent loop

Your agent can retrieve from the coordinator and call any chat backend itself:

```ts
const search = await fetch("http://127.0.0.1:8080/v1/rag/search", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ query: "How is the index persisted?", mode: "hybrid", top_k: 8 }),
}).then(r => r.json());

const context = search.results
  .map((source: any, i: number) => `[${i + 1}] ${source.document_id}\n${source.text}`)
  .join("\n\n");

const answer = await fetch("http://127.0.0.1:8082/v1/chat/completions", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model: "qwen-generation",
    messages: [{
      role: "user",
      content: `Answer using only these sources and cite [n].\n\n${context}\n\nQuestion: How is the index persisted?`,
    }],
    stream: false,
  }),
}).then(r => r.json());

console.log(answer.choices[0].message.content);
```

Port 8082 is deliberately private/loopback-only, but a local TypeScript or Python process can call it.

## Use search from a Python agent loop

```python
import json
from urllib.request import Request, urlopen

def post(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urlopen(request))

search = post("http://127.0.0.1:8080/v1/rag/search", {
    "query": "How is the index persisted?",
    "mode": "hybrid",
    "top_k": 8,
})

context = "\n\n".join(
    f"[{i}] {source['document_id']}\n{source['text']}"
    for i, source in enumerate(search["results"], 1)
)

answer = post("http://127.0.0.1:8082/v1/chat/completions", {
    "model": "qwen-generation",
    "messages": [{
        "role": "user",
        "content": f"Answer using only these sources and cite [n].\n\n{context}\n\nQuestion: How is the index persisted?",
    }],
    "stream": False,
})

print(answer["choices"][0]["message"]["content"])
```

For most applications, `/v1/rag/query` is simpler and safer because the coordinator performs context budgeting, source ordering, prompt isolation, and cancellation. Calling `/v1/rag/search` plus `/v1/chat/completions` separately is useful when your own agent loop needs to decide when to retrieve, rerank, or generate.

See [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) for a maintainer-oriented walkthrough of the current implementation, [`docs/llama-rag-server-spec.md`](docs/llama-rag-server-spec.md) for the broader product vision, and `requirements.json` for machine-readable requirement status and test mappings.
