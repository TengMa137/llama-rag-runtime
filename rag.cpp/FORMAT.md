# `.ragdb` format v1.2

`.ragdb` is a single CRC-protected container. Multi-byte values are encoded in
little-endian order and vectors are IEEE-754 binary32. The current writer emits
major `1`, minor `2`.

## Container

The 28-byte header is followed by `section_count` 20-byte table entries,
section payloads, then a four-byte CRC32 over every preceding byte.

| Header offset | Type | Meaning |
|---:|---|---|
| 0 | `u8[8]` | `RAGDB\0\0\0` |
| 8 | `u16` | major (`1`) |
| 10 | `u16` | minor (`0`, `1`, or `2`) |
| 12 | `u32` | flags: bit 0 HNSW, bit 1 embeddings |
| 16 | `u32` | section count |
| 20 | `u64` | reserved |

Each table entry is `{u32 tag, u64 absolute_offset, u64 byte_length}`. Tags must
be unique. Payload ranges must start after the complete table, end before the
CRC, and not overlap. Unknown tags are skipped. `META`, `DOCS`, `CHNK`, and
`BM25` are required; flags must agree with optional `EMBD` and `HNSW` sections.

| ASCII | LE tag | Payload |
|---|---:|---|
| `META` | `0x4154454D` | UTF-8 configuration JSON |
| `DOCS` | `0x53434F44` | documents |
| `CHNK` | `0x4B4E4843` | chunks and source ranges |
| `EMBD` | `0x42444D45` | vectors parallel to chunks |
| `BM25` | `0x35324D42` | versioned inverted-index blob |
| `HNSW` | `0x57534E48` | versioned ANN blob |
| `TOMB` | `0x424D4F54` | deleted document IDs |
| `RTME` | `0x454D5452` | optional portable-runtime revisions and public IDs |

CRC32 uses the reflected IEEE polynomial `0xEDB88320`, initial value
`0xFFFFFFFF`, and final XOR `0xFFFFFFFF`. It detects accidental corruption; it
is not an authenticity or security boundary.

## Records

`str` means `u32 byte_length` followed by that many bytes.

```text
DOCS := u32 count
        count × { u32 id, str uri, str title, str text,
                  u32 metadata_count, metadata_count × { str key, str value } }

CHNK := u32 count
        count × { u32 id, u32 document_id, str text, str heading_context,
                  u32 start_line, u32 end_line }

EMBD := chunk_count × { u32 dimension, f32[dimension] values }
        dimension 0 means that chunk has no vector

TOMB := u32 count, count × u32 document_id

RTME := u32 version, u64 generation, u64 represented_wal_position,
        u32 active_document_count,
        active_document_count × {
          str document_key, u64 revision, str content_hash,
          str chunking_fingerprint, str embedding_identity,
          u32 chunk_count,
          chunk_count × { str public_chunk_key, u64 ordinal, str indexed_text }
        },
        u32 revision_count, revision_count × { str document_key, u64 revision }
```

Document and chunk IDs equal record ordinals. Chunk document IDs and tombstones
must reference existing documents. Embeddings are in chunk order, have one
nonzero dimension across the file, contain only finite values, and have unit
norm. `TOMB` first appeared in minor 1 and is omitted when empty.
`RTME` is emitted by the backend-neutral embedded runtime. Its active rows map
the ordinary `DOCS`/`CHNK`/`EMBD` sections to stable public identities, while
the revision catalog also retains deletion revisions. Legacy v1.2 readers skip
this additive section and can still open the standard sections.

`META` stores `hnsw_threshold`, `embed_batch`, BM25 parameters, chunk geometry,
and the embedding/chunking policy. Minor 2 added policy identity and fingerprint
fields inside this JSON. `BM25` currently uses magic `2BM1`, version 1. `HNSW`
uses magic `HNW1`, version 2. Their blobs are covered by the container CRC.

## Compatibility

Readers reject unknown major versions. Minor releases are additive: v1.2 reads
v1.0 and v1.1 files, where absent tombstones and policy fields take legacy
defaults. Writers preserve stable IDs and serialize tombstones so deletion does
not resurrect after reopening. A format-major change requires an explicit
migration path; none is currently planned.

Saving writes and flushes a temporary file in the destination directory, then
atomically renames and syncs the directory. WAL data is a separate framed file,
not a `.ragdb` section; `RTME` records the exact WAL prefix represented by its
checkpoint. Replay accepts a torn final frame but rejects corruption inside
acknowledged history.

Mapped readers validate the complete container layout before exposing any
payload. A mapped section view is non-owning and must not outlive its reader.
Atomic replacement leaves existing views bound to the old inode, so publishing
a generation cannot invalidate concurrent readers.

Files ending in `.dense.native-exact-v1` are disposable cache sidecars, not
durable `.ragdb` generations. They reuse the CRC-protected section container but
are accepted only when their embedded implementation, algorithm, corpus/vector
fingerprint, row count, dimension, keys, and normalized vectors match the active
checkpoint. Deleting or corrupting a sidecar must only cause deterministic
rebuild; it cannot remove or change indexed documents.

The corresponding native HNSW cache ends in `.dense.native-hnsw-v1`. It stores
the validated stable-key catalog and serialized graph under the same
generation/vector fingerprint rules, extended with HNSW construction/search
parameters. It is accepted only when numeric graph IDs are the sequential row
IDs for that key catalog and the graph contains one f32 matrix with no SQ8/PQ
mirror. It has the same delete-or-rebuild semantics as the exact sidecar.

Optional FAISS caches end in `.dense.faiss-<algorithm>-v1`, where algorithm is
`flat`, `hnsw`, `ivf-sq8`, or `ivf-pq`. They reuse the validated container with
`META`, ordered `CHNK` keys, and an opaque `DIDX` serialized-index section. The
fingerprint includes the FAISS implementation version, explicit construction
and search parameters, embedding identity, key order, and vector bytes. They
are never authoritative: incompatible, missing, truncated, or CRC-invalid
files are rebuilt from normalized vectors in the portable `.ragdb` checkpoint.
