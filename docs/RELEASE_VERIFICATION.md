# Portable runtime verification matrix

This matrix records the deterministic release verification completed on
2026-08-12. Optional packages were installed independently; CMake did not
download FAISS, PostgreSQL, pgvector, or the Android NDK.

## Shared backend contract

`tests/backend_contract_suite.*` owns one backend-neutral scenario over
`CandidateBackend`. A backend fixture supplies only construction, an isolated
namespace, and an optional immutable-base publication callback. The shared
scenario verifies:

- deterministic preparation and stable public chunk IDs;
- atomic document activation and backend capability reporting;
- exact metadata filtering in lexical, dense, and hybrid modes;
- bounded fetches with and without vectors;
- deterministic resolved result fields and ordering;
- replacement visibility, stale-revision rejection, and failed-vector
  isolation; and
- immediate, idempotent deletion without stale resurrection.

The scenario passes against native exact, native HNSW, FAISS FlatIP, FAISS
HNSWFlat, FAISS IVF-SQ8, FAISS IVF-PQ, and exact PostgreSQL/pgvector. Backend-
specific suites additionally cover embedded concurrent compaction and cache
recovery, ingestion restart/races, PostgreSQL rollback/pool saturation/HNSW,
and embedded-to-PostgreSQL-to-embedded migration.

## Build and test matrix

| Target | Configuration | Result |
|---|---|---|
| macOS arm64 | Debug, native embedded | 7/7 CTest tests passed |
| macOS arm64 | Release, native embedded | 7/7 CTest tests passed |
| macOS arm64 | Release, FAISS enabled | 7/7 CTest tests passed |
| macOS arm64 | Debug, PostgreSQL enabled | 8/8 CTest tests passed against PostgreSQL 17 + pgvector |
| macOS arm64 | ASan + UBSan, native embedded | 7/7 CTest tests passed |
| macOS arm64 | ASan + UBSan, FAISS enabled | 7/7 CTest tests passed |
| macOS arm64 | ASan + UBSan, PostgreSQL enabled | 8/8 CTest tests passed |
| Android arm64-v8a | NDK 28.2, API 24, Release | `librag_mobile.so` and `lrs-mobile-smoke` compiled |

The native/component suites cover the stable C++ API, opaque C ABI, desktop
HTTP response/event contracts, and mobile supplied-vector ABI. Persistence
tests cover v1.0 compatibility, current v1.2 output, legacy-minor defaults,
malformed/overlapping/truncated sections, CRC failures, WAL recovery, and
disposable sidecar rebuild. PostgreSQL tests include the resumable round-trip
migration audit and prove that reading or reverse migration does not mutate the
source.

Quality and performance evidence, including the 100k recall gates and
25k/100k/500k native baselines, is stored in
`benchmarks/macos-arm64-2026-08-12.json`.

## Reproduction

The ordinary local build remains dependency-free:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
ctest --test-dir build/native --output-on-failure
```

FAISS and PostgreSQL are explicit build variants:

```bash
cmake -S . -B build/faiss -DCMAKE_BUILD_TYPE=Release -DLRS_ENABLE_FAISS=ON
cmake -S . -B build/postgres -DLRS_ENABLE_POSTGRES=ON
```

The PostgreSQL contract is enabled only when `LRS_TEST_POSTGRES_URL` names an
operator-provided test database. Android remains native-only:

```bash
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/28.2.13676358"
cmake --preset android-arm64
cmake --build --preset android-arm64
cmake --build build/android-arm64 --target lrs-mobile-smoke
```

The loopback HTTP fixture must be allowed to bind local ephemeral ports. No
test sends document data or credentials to an external service.
