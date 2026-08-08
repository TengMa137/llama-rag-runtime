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
```

Document and chunk IDs equal record ordinals. Chunk document IDs and tombstones
must reference existing documents. Embeddings are in chunk order, have one
nonzero dimension across the file, contain only finite values, and have unit
norm. `TOMB` first appeared in minor 1 and is omitted when empty.

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
not a `.ragdb` section; replay accepts a torn final frame but rejects corruption
inside acknowledged history.
