// rag/index/corpus.cpp — document ingest, hybrid indexing, persistence.

#include "rag/index/corpus.hpp"
#include "rag/core/keys.hpp"
#include "rag/dense/simd.hpp"
#include "rag/gpu/device.hpp"
#include "rag/store/container.hpp"
#include "rag/util/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>

#include <nlohmann/json.hpp>

namespace rag::index {

using json = nlohmann::json;

void Corpus::relink_meta() {
    for (auto& ch : chunks_)
        ch.meta = (ch.doc.get() < docs_.size()) ? &docs_[ch.doc.get()].meta : nullptr;
}

void Corpus::move_from(Corpus&& o) {
    cfg_ = std::move(o.cfg_);
    embedder_ = std::move(o.embedder_);
    docs_ = std::move(o.docs_);
    chunks_ = std::move(o.chunks_);
    bm25_ = std::move(o.bm25_);
    hnsw_ = std::move(o.hnsw_);
    dirty_ = o.dirty_;
    epoch_ = o.epoch_;
    deleted_docs_ = std::move(o.deleted_docs_);
    wal_ = std::move(o.wal_);
    replaying_ = o.replaying_;
    relink_meta(); // borrowed pointers now point at OUR docs_ storage
    meta_stale_ = false;
    // The packed mirror is a cache keyed on epoch_, and epoch_ came from `o`.
    // Not invalidating it here would let a stale (or empty) matrix be accepted
    // as current, because the epoch it was stamped with is now ours too.
    packed_valid_ = false;
    packed_.clear();
    packed_ids_.clear();
    packed_dim_ = 0;
}

Result<DocId> Corpus::add_document(std::string uri, std::string text, Metadata meta,
                                   std::string title) {
    std::unique_lock lk(mu_);
    return add_document_locked(std::move(uri), std::move(text), std::move(meta), std::move(title));
}

Result<DocId> Corpus::upsert_document(std::string uri, std::string text, Metadata meta,
                                      std::string title) {
    // ONE lock across find + remove + add, so two threads upserting the same
    // uri cannot both miss and both insert.
    std::unique_lock lk(mu_);
    if (!uri.empty())
        if (auto existing = find_by_uri_locked(uri))
            (void)remove_document_locked(*existing);
    return add_document_locked(std::move(uri), std::move(text), std::move(meta), std::move(title));
}

Result<DocId> Corpus::add_document_locked(std::string uri, std::string text, Metadata meta,
                                          std::string title) {
    if (docs_.size() >= store::kMaxDocuments || docs_.size() >= DocId::invalid().get())
        return fail<DocId>(Errc::invalid_argument, "document limit exceeded");
    if (uri.size() > UINT32_MAX || text.size() > UINT32_MAX || title.size() > UINT32_MAX ||
        meta.size() > store::kMaxMetadataEntries)
        return fail<DocId>(Errc::invalid_argument, "document field limit exceeded");
    for (const auto& [key, value] : meta)
        if (key.size() > UINT32_MAX || value.size() > UINT32_MAX)
            return fail<DocId>(Errc::invalid_argument, "metadata field limit exceeded");
    // Log BEFORE mutating. If the append fails, nothing has changed and the
    // caller gets an honest error; logging after the mutation would leave a
    // corpus that has a document its log does not, so a crash would silently
    // roll it back after the client was told it succeeded.
    if (wal_.is_open() && !replaying_) {
        store::WalRecord rec;
        rec.op = store::WalOp::add_document;
        rec.uri = uri;
        rec.title = title;
        rec.text = text;
        rec.meta = meta;
        if (auto w = wal_.append(rec); !w)
            return unexpected(w.error());
    }

    DocId did{static_cast<std::uint32_t>(docs_.size())};
    Document doc;
    doc.id = did;
    doc.uri = std::move(uri);
    doc.title = std::move(title);
    doc.text = std::move(text);
    doc.meta = std::move(meta);
    docs_.push_back(std::move(doc));
    const Document& stored = docs_.back();

    auto new_chunks = text::chunk_document(did, stored.text, cfg_.chunk);
    if (new_chunks.size() >
            store::kMaxChunks - std::min<std::size_t>(chunks_.size(), store::kMaxChunks) ||
        new_chunks.size() > static_cast<std::size_t>(ChunkId::invalid().get()) -
                                std::min<std::size_t>(chunks_.size(), ChunkId::invalid().get())) {
        docs_.pop_back();
        return fail<DocId>(Errc::invalid_argument, "chunk limit exceeded");
    }

    for (auto& ch : new_chunks) {
        ChunkId cid{static_cast<std::uint32_t>(chunks_.size())};
        ch.id = cid;
        bm25_.add(cid.get(), ch.indexed_text());
        chunks_.push_back(std::move(ch));
    }
    // NOTE: this used to relink_meta() and bm25_.finalize() here — both O(total
    // corpus), per document, which made bulk ingest quadratic (it dominated a
    // 20k-document build at ~1.8s). Neither is needed until something READS:
    //   • chunk meta pointers are borrowed into docs_, which push_back above may
    //     have reallocated — so they are relinked lazily in ensure_linked(),
    //     driven by the `meta_stale_` flag, before any accessor hands one out;
    //   • bm25 idf/avgdl are pure functions of the accumulated counts, so
    //     finalize() is idempotent and only has to run before a query.
    // build() does both; the read paths do them on demand for callers who
    // query without an explicit build().
    meta_stale_ = true;
    dirty_ = true;
    ++epoch_;
    return did;
}

void Corpus::ensure_linked() const {
    // Double-checked: the common case is a clean corpus, where this is a single
    // relaxed read and no lock at all. Only the rare stale case pays for the
    // mutex, and the re-check inside it stops two readers from both relinking.
    if (!meta_stale_)
        return;
    std::lock_guard lk(lazy_mu_);
    if (!meta_stale_)
        return;
    const_cast<Corpus*>(this)->relink_meta();
    meta_stale_ = false;
}

Result<void> Corpus::embed_pending() {
    if (!embedder_)
        return {};
    // Collect chunks with empty embeddings.
    std::vector<std::size_t> pending;
    for (std::size_t i = 0; i < chunks_.size(); ++i)
        if (chunks_[i].embedding.empty())
            pending.push_back(i);
    if (pending.empty())
        return {};

    const std::size_t bs = cfg_.embed_batch ? cfg_.embed_batch : 1;
    const std::size_t nb = (pending.size() + bs - 1) / bs;

    // Batches are independent: each reads a disjoint slice of `pending` and
    // writes only the chunks named by that slice, so they can be in flight
    // concurrently. HOW concurrently is the backend's call — an in-process
    // model already owns every core (hint 1 ⇒ serial), a hosted endpoint is
    // latency-bound (hint 8) — see dense::ConcurrencyAware.
    const std::size_t workers = std::min(embedder_->max_concurrency(), nb);

    // First failure wins; later batches short-circuit rather than pile up
    // retries against a backend that is already known to be down.
    std::atomic<bool> failed{false};
    std::mutex err_mu;
    Error first_err{};

    auto run_batch = [&](std::size_t b) {
        if (failed.load(std::memory_order_relaxed))
            return;
        const std::size_t off = b * bs;
        const std::size_t end = std::min(off + bs, pending.size());
        std::vector<std::string> batch;
        batch.reserve(end - off);
        for (std::size_t j = off; j < end; ++j)
            batch.push_back(chunks_[pending[j]].indexed_text());

        auto res = embedder_->embed(batch);
        if (!res) {
            std::lock_guard lk(err_mu);
            if (!failed.exchange(true, std::memory_order_acq_rel))
                first_err = res.error();
            return;
        }
        auto& vecs = *res;
        if (vecs.size() != end - off) {
            std::lock_guard lk(err_mu);
            if (!failed.exchange(true, std::memory_order_acq_rel))
                first_err = Error{Errc::dimension_mismatch, "embedding response count mismatch"};
            return;
        }
        const std::size_t dimension = embedder_->dimension();
        for (auto& vector : vecs) {
            if (dimension == 0 || dimension > store::kMaxVectorDimension ||
                vector.size() != dimension) {
                std::lock_guard lk(err_mu);
                if (!failed.exchange(true, std::memory_order_acq_rel))
                    first_err =
                        Error{Errc::dimension_mismatch, "embedding response dimension mismatch"};
                return;
            }
            double norm = 0.0;
            for (const float value : vector) {
                if (!std::isfinite(value)) {
                    std::lock_guard lk(err_mu);
                    if (!failed.exchange(true, std::memory_order_acq_rel))
                        first_err =
                            Error{Errc::invalid_argument, "embedding contains a non-finite value"};
                    return;
                }
                norm += static_cast<double>(value) * value;
            }
            if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::min()) {
                std::lock_guard lk(err_mu);
                if (!failed.exchange(true, std::memory_order_acq_rel))
                    first_err =
                        Error{Errc::invalid_argument, "embedding must have a finite non-zero norm"};
                return;
            }
            const float inverse = static_cast<float>(1.0 / std::sqrt(norm));
            for (float& value : vector)
                value *= inverse;
        }
        for (std::size_t j = 0; j < vecs.size(); ++j)
            chunks_[pending[off + j]].embedding = std::move(vecs[j]);
    };

    if (workers <= 1)
        for (std::size_t b = 0; b < nb; ++b)
            run_batch(b);
    else
        util::parallel_for_dynamic(nb, workers, run_batch);

    if (failed.load(std::memory_order_acquire))
        return unexpected(first_err);
    return {};
}

Result<void> Corpus::build() {
    std::unique_lock lk(mu_);
    return build_locked();
}

Result<void> Corpus::build_locked() {
    ensure_linked();
    bm25_.finalize();
    if (embedder_) {
        if (auto r = embed_pending(); !r)
            return r; // degrade: propagate, caller may ignore
        // Build HNSW past threshold.
        if (chunks_.size() >= cfg_.hnsw_threshold) {
            HnswIndex idx(cfg_.hnsw);
            // Collect the embedded chunks, then construct the graph in parallel.
            // build_batch stages all nodes first and links them across every
            // core — the dominant cost of indexing a large corpus.
            std::vector<std::size_t> rows;
            rows.reserve(chunks_.size());
            for (std::size_t i = 0; i < chunks_.size(); ++i)
                if (!chunks_[i].embedding.empty())
                    rows.push_back(i);

            idx.build_batch(
                rows.size(),
                [&](std::size_t i) -> std::span<const float> { return chunks_[rows[i]].embedding; },
                [&](std::size_t i) { return chunks_[rows[i]].id.get(); });
            hnsw_ = std::move(idx);
        }
    }
    dirty_ = false;
    ++epoch_;
    return {};
}

namespace {

// Ranking order for the dense scan: score descending, chunk id ascending.
//
// The id tiebreak is not cosmetic. std::partial_sort is NOT stable, so a
// comparator that only looks at score leaves tied hits in an order decided by
// the algorithm's internals — which means the ranking depends on how the
// candidate list happened to be laid out. That made the parallel scan and the
// batched/GPU scan return the SAME SET in DIFFERENT ORDERS, caught by driving
// both paths on a 200k-chunk corpus where a hash embedder produces many exact
// score ties. Identical scores are common in practice (duplicate passages,
// quantized vectors), so ties must resolve to something stable that does not
// depend on the execution path or the thread count.
constexpr auto hit_order = [](const Hit& a, const Hit& b) {
    if (a.score.get() != b.score.get())
        return a.score.get() > b.score.get();
    return a.chunk.get() < b.chunk.get();
};

} // namespace

void Corpus::ensure_packed() const {
    std::lock_guard lz(lazy_mu_);
    if (packed_valid_ && packed_epoch_ == epoch_)
        return;

    packed_.clear();
    packed_ids_.clear();
    packed_dim_ = 0;
    for (const auto& ch : chunks_)
        if (!ch.embedding.empty()) {
            packed_dim_ = ch.embedding.size();
            break;
        }

    if (packed_dim_ != 0) {
        packed_.reserve(chunks_.size() * packed_dim_);
        packed_ids_.reserve(chunks_.size());
        for (const auto& ch : chunks_) {
            // Skip ragged rows rather than packing a matrix whose stride lies.
            // A mixed-dimension corpus is already broken, but it must not turn
            // into an out-of-bounds GPU read.
            if (ch.embedding.size() != packed_dim_)
                continue;
            packed_.insert(packed_.end(), ch.embedding.begin(), ch.embedding.end());
            packed_ids_.push_back(ch.id.get());
        }
    }
    packed_epoch_ = epoch_;
    packed_valid_ = true;
}

Result<std::vector<std::vector<Hit>>>
Corpus::dense_search_batch(std::span<const std::string> queries, std::size_t k,
                           const MetaFilter& filter) const {
    std::shared_lock lk(mu_);
    // Empty in, empty out — checked BEFORE the embedder, because asking zero
    // questions is not a question a missing embedder can fail to answer.
    if (queries.empty())
        return std::vector<std::vector<Hit>>{};
    if (!embedder_)
        return fail<std::vector<std::vector<Hit>>>(Errc::unavailable, "no embedder");
    ensure_linked();

    // The GPU path is only reachable for a plain, unfiltered, graph-less scan.
    // Everything else falls back to running the existing per-query path, which
    // is exactly what a caller would have written by hand — so this method is
    // never worse than the loop it replaces.
    const bool scan_path = !hnsw_ && !filter;
    if (scan_path) {
        ensure_packed();
        const std::size_t n = packed_ids_.size();
        const std::size_t dim = packed_dim_;
        const std::size_t nq = queries.size();

        if (n > 0 && dim > 0 && gpu::available() && nq * n * dim >= gpu::min_batch_work()) {
            // Embed every query first; a partial batch is not worth salvaging.
            std::vector<float> qmat;
            qmat.reserve(nq * dim);
            bool ok = true;
            for (const auto& q : queries) {
                auto qv = embedder_->embed_one(q);
                if (!qv || qv->size() != dim) {
                    ok = false;
                    break;
                }
                dense::normalize(*qv);
                qmat.insert(qmat.end(), qv->begin(), qv->end());
            }

            if (ok) {
                std::vector<float> scores(nq * n);
                if (gpu::score_batch(packed_, qmat, dim, scores)) {
                    std::vector<std::vector<Hit>> out(nq);
                    for (std::size_t q = 0; q < nq; ++q) {
                        // Partial top-k per query over that query's score row.
                        const float* row = scores.data() + q * n;
                        std::vector<Hit> hits;
                        hits.reserve(n);
                        for (std::size_t i = 0; i < n; ++i)
                            hits.push_back(Hit{ChunkId{packed_ids_[i]}, Score{row[i]}});
                        const std::size_t want = std::min(k, hits.size());
                        std::partial_sort(hits.begin(), hits.begin() + (long)want, hits.end(),
                                          hit_order);
                        hits.resize(want);
                        out[q] = std::move(hits);
                    }
                    return out;
                }
                // score_batch declining is a ROUTING answer, not an error: fall
                // through to the CPU path below rather than failing the query.
            }
        }
    }

    std::vector<std::vector<Hit>> out;
    out.reserve(queries.size());
    for (const auto& q : queries) {
        auto r = dense_search_locked(q, k, filter);
        if (!r)
            return unexpected(r.error());
        out.push_back(std::move(*r));
    }
    return out;
}

std::vector<Hit> Corpus::lexical_search(std::string_view query, std::size_t k) const {
    std::shared_lock lk(mu_);
    return lexical_search_locked(query, k);
}

std::vector<Hit> Corpus::lexical_search(std::string_view query, std::size_t k,
                                        const MetaFilter& filter) const {
    std::shared_lock lk(mu_);
    return lexical_search_locked(query, k, filter);
}

std::vector<Hit> Corpus::lexical_search_locked(std::string_view query, std::size_t k) const {
    // add_document() no longer finalizes on every insert (that was quadratic);
    // a caller may therefore query without an intervening build(). finalize()
    // is idempotent and pure in the accumulated counts, so bringing it up to
    // date here is both cheap and correct.
    //
    // It is also a WRITE performed under a shared lock, so it needs lazy_mu_:
    // without it two concurrent readers would finalize the same index at the
    // same time. Double-checked, so a finalized index costs one bool read.
    if (!bm25_.finalized()) {
        std::lock_guard lz(lazy_mu_);
        if (!bm25_.finalized())
            const_cast<Corpus*>(this)->bm25_.finalize();
    }
    if (deleted_docs_.empty())
        return bm25_.search(query, k);
    // Over-fetch, then drop tombstoned chunks and truncate to k.
    auto hits = bm25_.search(query, k + deleted_docs_.size() * 2 + k);
    std::vector<Hit> out;
    out.reserve(std::min(hits.size(), k));
    for (const auto& h : hits) {
        const Chunk* ch = chunk_locked(h.chunk);
        if (ch && deleted_docs_.count(ch->doc.get()))
            continue;
        out.push_back(h);
        if (out.size() >= k)
            break;
    }
    return out;
}

std::vector<Hit> Corpus::lexical_search_locked(std::string_view query, std::size_t k,
                                               const MetaFilter& filter) const {
    if (!filter)
        return lexical_search_locked(query, k);

    // A selective permission filter must not be applied only to BM25's top-k:
    // doing so can yield no results even when lower-ranked allowed documents
    // exist. Rank all lexical matches, retain only allowed live documents, and
    // then truncate. This is correctness-first; BM25 can later accept an allow
    // predicate directly if filtered lexical latency becomes material.
    auto hits = lexical_search_locked(query, bm25_.size());
    std::vector<Hit> out;
    out.reserve(std::min(hits.size(), k));
    for (const auto& hit : hits) {
        if (!passes_locked(hit.chunk, filter))
            continue;
        out.push_back(hit);
        if (out.size() == k)
            break;
    }
    return out;
}

Result<void> Corpus::remove_document(DocId id) {
    std::unique_lock lk(mu_);
    return remove_document_locked(id);
}

Result<void> Corpus::remove_document_locked(DocId id) {
    ensure_linked();
    if (id.get() >= docs_.size() || deleted_docs_.count(id.get()))
        return fail<void>(Errc::not_found, "remove_document: unknown or already-deleted id");
    // Logged only after the validity check, so a no-op delete writes nothing.
    if (wal_.is_open() && !replaying_) {
        store::WalRecord rec;
        rec.op = store::WalOp::remove_document;
        rec.doc_id = id.get();
        if (auto w = wal_.append(rec); !w)
            return w;
    }
    deleted_docs_.insert(id.get());
    ++epoch_;
    // Tombstone the doc's chunks in the HNSW graph so dense search skips them.
    if (hnsw_)
        for (const auto& ch : chunks_)
            if (ch.doc.get() == id.get())
                hnsw_->remove(ch.id.get());
    return {};
}

bool Corpus::is_deleted(DocId id) const noexcept {
    std::shared_lock lk(mu_);
    return deleted_docs_.count(id.get()) != 0;
}
std::size_t Corpus::live_document_count() const noexcept {
    std::shared_lock lk(mu_);
    return docs_.size() - deleted_docs_.size();
}

Result<std::vector<Hit>> Corpus::dense_search(std::string_view query, std::size_t k) const {
    return dense_search(query, k, MetaFilter{});
}

Result<Vector> Corpus::embed_text(const std::string& text) const {
    if (!embedder_)
        return fail<Vector>(Errc::unavailable, "no embedder");
    auto v = embedder_->embed_one(text);
    if (!v)
        return unexpected(v.error());
    dense::normalize(*v);
    return v;
}

Result<std::vector<Hit>> Corpus::dense_search(std::string_view query, std::size_t k,
                                              const MetaFilter& filter) const {
    std::shared_lock lk(mu_);
    return dense_search_locked(query, k, filter);
}

Result<std::vector<Hit>> Corpus::dense_search_locked(std::string_view query, std::size_t k,
                                                     const MetaFilter& filter) const {
    if (!embedder_)
        return fail<std::vector<Hit>>(Errc::unavailable, "no embedder");
    ensure_linked(); // the allow-predicate below dereferences chunk->meta
    auto qv = embedder_->embed_one(std::string(query));
    if (!qv)
        return unexpected(qv.error());
    dense::normalize(*qv);

    // Build an allow-predicate over chunk id from the metadata filter.
    HnswIndex::AllowFn allow;
    if (filter) {
        allow = [this, &filter](std::uint32_t id) -> bool {
            const Chunk* ch = (id < chunks_.size()) ? &chunks_[id] : nullptr;
            if (!ch || !ch->meta)
                return false;
            return filter(*ch->meta);
        };
    }

    // Use HNSW if built, else brute-force cosine (with the same pre-filter).
    if (hnsw_) {
        return allow ? hnsw_->search_filtered(*qv, k, allow) : hnsw_->search(*qv, k);
    }

    // Brute-force scan. Parallel over contiguous blocks: each worker scores its
    // own range into a private buffer, then we concatenate and select. Scoring
    // is pure (reads immutable embeddings), so no synchronization is needed
    // beyond the join.
    const std::size_t n = chunks_.size();
    std::vector<std::vector<Hit>> parts(util::block_count(n));

    util::parallel_blocks(n, [&](std::size_t lo, std::size_t hi, std::size_t b) {
        auto& local = parts[b];
        local.reserve((hi - lo) / 4 + 8);
        for (std::size_t i = lo; i < hi; ++i) {
            const auto& ch = chunks_[i];
            if (ch.embedding.empty())
                continue;
            if (!deleted_docs_.empty() && deleted_docs_.count(ch.doc.get()))
                continue;
            if (allow && !allow(ch.id.get()))
                continue;
            local.push_back(Hit{ch.id, Score{dense::dot(ch.embedding, *qv)}});
        }
    });

    std::vector<Hit> hits;
    std::size_t total = 0;
    for (const auto& p : parts)
        total += p.size();
    hits.reserve(total);
    for (auto& p : parts)
        hits.insert(hits.end(), p.begin(), p.end());

    const std::size_t kk = std::min(k, hits.size());
    std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk), hits.end(),
                      hit_order);
    hits.resize(kk);
    return hits;
}

const Chunk* Corpus::chunk(ChunkId id) const {
    std::shared_lock lk(mu_);
    return chunk_locked(id);
}
const Chunk* Corpus::chunk_locked(ChunkId id) const {
    ensure_linked();
    return id.get() < chunks_.size() ? &chunks_[id.get()] : nullptr;
}
const Document* Corpus::document(DocId id) const {
    std::shared_lock lk(mu_);
    return document_locked(id);
}
const Document* Corpus::document_locked(DocId id) const {
    return id.get() < docs_.size() ? &docs_[id.get()] : nullptr;
}

std::optional<DocId> Corpus::find_by_uri(std::string_view uri) const {
    std::shared_lock lk(mu_);
    return find_by_uri_locked(uri);
}

std::optional<DocId> Corpus::find_by_uri_locked(std::string_view uri) const {
    if (uri.empty())
        return std::nullopt;
    for (const auto& d : docs_) {
        if (deleted_docs_.count(d.id.get()))
            continue; // skip tombstones
        if (d.uri == uri)
            return d.id;
    }
    return std::nullopt;
}

SearchResult Corpus::resolve(const Hit& h) const {
    std::shared_lock lk(mu_);
    return resolve_locked(h);
}

SearchResult Corpus::resolve_locked(const Hit& h) const {
    SearchResult r;
    const Chunk* ch = chunk_locked(h.chunk);
    if (!ch)
        return r;
    r.chunk = ch->id;
    r.doc = ch->doc;
    r.score = h.score;
    r.text = ch->text;
    r.context = ch->context;
    r.start_line = ch->start_line;
    r.end_line = ch->end_line;
    if (const Document* d = document_locked(ch->doc)) {
        r.uri = d->uri;
        r.document_key = d->uri;
        r.chunk_key = stable_chunk_key(d->uri, ch->start_line, ch->end_line, ch->text);
        r.title = d->title;
        r.metadata = d->meta;
    }
    return r;
}

bool Corpus::passes(ChunkId id, const MetaFilter& f) const {
    std::shared_lock lk(mu_);
    return passes_locked(id, f);
}

bool Corpus::passes_locked(ChunkId id, const MetaFilter& f) const {
    const Chunk* ch = chunk_locked(id);
    if (!ch || !ch->meta)
        return !f; // no metadata: pass only if no filter
    return f ? f(*ch->meta) : true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence — the stable, versioned .ragdb container (see rag/store + FORMAT.md).
// Sections: META (config json), DOCS, CHNK (chunk records), EMBD (embeddings),
// BM25 (inverted index), HNSW (ANN graph). CRC-verified on load.
// ─────────────────────────────────────────────────────────────────────────────
Result<void> Corpus::save(const std::string& path) const {
    try {
        // save() splits into two phases that want very different lock treatment.
        //
        //   1. SNAPSHOT — walk docs_/chunks_/bm25_/hnsw_ and pack them into section
        //      blobs. This touches corpus state, so it must hold a lock, and the
        //      lock is what makes the snapshot consistent: no document can be
        //      appended halfway through serializing the arrays.
        //
        //   2. WRITE — concatenate those blobs, CRC the result, write it to a temp
        //      file, fsync, rename, fsync the directory. This touches NO corpus
        //      state whatsoever; the Container owns its own copies.
        //
        // Phase 2 is the expensive one — measured on a 20k-doc corpus it is ~24 of
        // the ~28 ms, almost all of it CRC and the big concatenating copy — and it
        // used to run with the shared lock still held. Readers did not care (they
        // share it too), but every WRITER blocked for the whole thing: an
        // add_document() concurrent with a save loop measured 6.0 ms mean against
        // 0.001 ms uncontended. Dropping the lock between the phases takes that
        // stall down to just the snapshot.
        store::Container snap;
        std::uint64_t at_epoch = 0;
        {
            std::shared_lock lk(mu_);
            auto s = snapshot_locked();
            if (!s)
                return unexpected(s.error());
            snap = std::move(*s);
            at_epoch = epoch_;
        }

        // PUBLISH, in epoch order.
        //
        // Splitting the phases means two concurrent save()s can interleave as:
        // A snapshots at epoch 10, B snapshots at epoch 20, B renames, A renames —
        // and the file on disk goes BACKWARDS to epoch 10.
        //
        // This is rare but real, and it took an honest harness to see it. A loop
        // that loads the file while savers and a writer run showed nothing at all
        // with 3-8 savers on 8 cores: the savers stay roughly in lockstep, so their
        // renames happen to come out in snapshot order. Only when the savers are
        // OVERSUBSCRIBED (16 and 32 threads on 8 cores, so the scheduler preempts
        // one between its snapshot and its rename) does the on-disk doc count go
        // backwards — 2 and 4 times per 300 loads. With this gate: 0, at 16, 32 and
        // 64 savers.
        //
        // The rename is therefore serialized and gated on the epoch that last
        // reached this path. A snapshot that lost the race is DROPPED, not written:
        // its content is a strict prefix of what is already there, so discarding it
        // loses nothing, and reporting success is honest — the caller asked for the
        // state at their save() call to be durable, and a newer state that contains
        // it is on disk.
        auto& st = path_state(path);
        std::lock_guard pk(st.mu);
        if (st.any && st.published > at_epoch)
            return {};
        if (auto r = snap.write_file(path); !r)
            return r;
        st.published = at_epoch;
        st.any = true;
        return {};
    } catch (const std::exception& error) {
        return fail<void>(Errc::io_error, std::string("save failed: ") + error.what());
    } catch (...) {
        return fail<void>(Errc::io_error, "save failed");
    }
}

// ─── Write-ahead log ────────────────────────────────────────────────────
Result<void> Corpus::open_wal(const std::string& path, store::SyncMode mode) {
    std::unique_lock lk(mu_);

    // REPLAY FIRST, then open for appending.
    //
    // Replay runs through the ordinary mutation paths rather than poking at
    // docs_/chunks_ directly, so recovery produces exactly the corpus the
    // original run had — same chunking, same ids, same index state. `replaying_`
    // stops those calls from logging what they are reading; without it every
    // restart would double the log.
    auto records = store::Wal::replay(path);
    if (!records)
        return unexpected(records.error());

    if (!records->empty()) {
        replaying_ = true;
        for (const auto& rec : *records) {
            if (rec.op == store::WalOp::add_document) {
                // Replayed as an UPSERT for the same reason index/add is one:
                // a log may contain several writes to the same uri, and the
                // last must win rather than accumulate duplicates.
                std::optional<DocId> existing;
                if (!rec.uri.empty())
                    existing = find_by_uri_locked(rec.uri);
                if (existing)
                    (void)remove_document_locked(*existing);
                if (auto r = add_document_locked(rec.uri, rec.text, rec.meta, rec.title); !r) {
                    replaying_ = false;
                    return unexpected(r.error());
                }
            } else {
                // A delete of a document the snapshot never had is not an
                // error: the log can outlive the doc it refers to.
                (void)remove_document_locked(DocId{rec.doc_id});
            }
        }
        replaying_ = false;
        if (auto b = build_locked(); !b)
            return b;
    }

    return wal_.open(path, mode);
}

bool Corpus::has_wal() const noexcept {
    std::shared_lock lk(mu_);
    return wal_.is_open();
}

std::uint64_t Corpus::wal_bytes() const noexcept {
    std::shared_lock lk(mu_);
    return wal_.size_bytes();
}

Result<void> Corpus::checkpoint(const std::string& path) {
    // save() takes its own lock and does the snapshot/publish dance, so it runs
    // OUTSIDE the write lock here — taking mu_ around it would deadlock (mu_ is
    // not recursive) and would also block writers for the whole serialization,
    // which is exactly what the snapshot/publish split was built to avoid.
    if (auto s = save(path); !s)
        return s;

    // Only now is it safe to drop the log. save() has fsync'd the snapshot and
    // renamed it into place, so every record in the log is represented on disk.
    // Truncating first would leave a window in which a crash loses mutations
    // that were already acknowledged.
    std::unique_lock lk(mu_);
    if (!wal_.is_open())
        return {};
    return wal_.truncate();
}

Corpus::PathState& Corpus::path_state(const std::string& path) {
    // The registry lock is held only long enough to find/insert the entry; the
    // per-path lock — which is what the slow file I/O runs under — is taken by
    // the caller. node-based map so references stay valid as it grows.
    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::unique_ptr<PathState>> registry;
    std::lock_guard lk(registry_mu);
    auto& slot = registry[path];
    if (!slot)
        slot = std::make_unique<PathState>();
    return *slot;
}

Result<store::Container> Corpus::snapshot_locked() const {
    ensure_linked();
    store::Container c;
    std::uint32_t flags = 0;

    // META — corpus config (round-trips the knobs that affect query behaviour).
    {
        json m;
        m["hnsw_threshold"] = cfg_.hnsw_threshold;
        m["embed_batch"] = cfg_.embed_batch;
        // Contextual Retrieval is an INGEST policy, not a query knob, but it
        // still has to round-trip: a corpus reopened for writing (serve
        // --write) must keep situating the documents it accepts, or the ones
        // added after the restart are indexed differently from the ones added
        // before it, and only half the store carries the disambiguating text.
        //
        // The same argument applies to the chunker's geometry, which was NOT
        // persisted before: reopening a corpus built with max_lines=3 and
        // adding a document chunked it at the default 40, so one store ended up
        // holding two incompatible chunk granularities. Both are ingest policy
        // and both round-trip.
        m["chunk"] = {{"max_lines", cfg_.chunk.max_lines},
                      {"max_chars", cfg_.chunk.max_chars},
                      {"overlap_lines", cfg_.chunk.overlap_lines},
                      {"heading_context", cfg_.chunk.heading_context},
                      {"fingerprint", text::chunking_fingerprint(cfg_.chunk)},
                      {"policy",
                       {{"model", cfg_.chunk.policy.model_identity},
                        {"tokenizer", cfg_.chunk.policy.tokenizer_identity},
                        {"dimension", cfg_.chunk.policy.dimension},
                        {"target_tokens", cfg_.chunk.policy.target_tokens},
                        {"max_tokens", cfg_.chunk.policy.max_tokens},
                        {"overlap_tokens", cfg_.chunk.policy.overlap_tokens},
                        {"reserved_tokens", cfg_.chunk.policy.reserved_tokens},
                        {"document_prefix", cfg_.chunk.policy.document_prefix},
                        {"query_prefix", cfg_.chunk.policy.query_prefix},
                        {"counting_mode", static_cast<int>(cfg_.chunk.policy.counting_mode)},
                        {"invalid_utf8", static_cast<int>(cfg_.chunk.policy.invalid_utf8)}}}};
        m["bm25"] = {{"k1", cfg_.bm25.k1}, {"b", cfg_.bm25.b}};
        c.put(store::Tag::meta, m.dump());
    }

    // DOCS.
    {
        store::Writer w;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(docs_.size()));
        for (const auto& d : docs_) {
            w.u<std::uint32_t>(d.id.get());
            w.str(d.uri);
            w.str(d.title);
            w.str(d.text);
            w.u<std::uint32_t>(static_cast<std::uint32_t>(d.meta.size()));
            for (const auto& [k, v] : d.meta) {
                w.str(k);
                w.str(v);
            }
        }
        c.put(store::Tag::docs, std::move(w.data()));
    }

    // CHNK + EMBD (parallel arrays).
    {
        store::Writer w, e;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(chunks_.size()));
        bool any_emb = false;
        for (const auto& ch : chunks_) {
            w.u<std::uint32_t>(ch.id.get());
            w.u<std::uint32_t>(ch.doc.get());
            w.str(ch.text);
            w.str(ch.context);
            w.u<std::uint32_t>(ch.start_line);
            w.u<std::uint32_t>(ch.end_line);
            e.u<std::uint32_t>(static_cast<std::uint32_t>(ch.embedding.size()));
            if (!ch.embedding.empty()) {
                any_emb = true;
                e.bytes(std::string_view(reinterpret_cast<const char*>(ch.embedding.data()),
                                         ch.embedding.size() * sizeof(float)));
            }
        }
        c.put(store::Tag::chunks, std::move(w.data()));
        if (any_emb) {
            c.put(store::Tag::embed, std::move(e.data()));
            flags |= store::kHasEmbeddings;
        }
    }

    // BM25 + HNSW blobs (already self-describing).
    c.put(store::Tag::bm25, bm25_.serialize());
    if (hnsw_) {
        c.put(store::Tag::hnsw, hnsw_->serialize());
        flags |= store::kHasHnsw;
    }

    // TOMB — soft-delete tombstones.
    //
    // These are NOT derivable from anything else in the file. A tombstoned
    // document keeps its row in DOCS and its rows in CHNK and its postings in
    // BM25 (that is what "soft" means — ids stay stable so no other structure
    // has to be rewritten); the ONLY thing that hides it from results is
    // membership in deleted_docs_. Omitting this section therefore resurrects
    // every deleted document on the next load, fully searchable.
    //
    // Written sorted so the file is byte-identical for identical corpora —
    // deleted_docs_ is an unordered_set, whose iteration order is not stable.
    // Skipped entirely when empty so the common case costs zero bytes.
    if (!deleted_docs_.empty()) {
        std::vector<std::uint32_t> ids(deleted_docs_.begin(), deleted_docs_.end());
        std::sort(ids.begin(), ids.end());
        store::Writer w;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(ids.size()));
        for (std::uint32_t id : ids)
            w.u<std::uint32_t>(id);
        c.put(store::Tag::tomb, std::move(w.data()));
    }

    c.set_flags(flags);
    return c;
}

Result<Corpus> Corpus::load(const std::string& path) {
    auto cont = store::Container::read_file(path);
    if (!cont)
        return unexpected(cont.error());
    try {

        Corpus c;

        const std::string* meta = cont->get(store::Tag::meta);
        const std::string* docs = cont->get(store::Tag::docs);
        const std::string* chunks = cont->get(store::Tag::chunks);
        const std::string* bm25 = cont->get(store::Tag::bm25);
        if (meta == nullptr || docs == nullptr || chunks == nullptr || bm25 == nullptr)
            return fail<Corpus>(Errc::corrupt_index, "missing required section");
        if (((cont->flags() & store::kHasEmbeddings) != 0) != cont->has(store::Tag::embed) ||
            ((cont->flags() & store::kHasHnsw) != 0) != cont->has(store::Tag::hnsw))
            return fail<Corpus>(Errc::corrupt_index, "section flags are inconsistent");

        {
            auto m = json::parse(*meta, nullptr, false);
            if (m.is_discarded() || !m.is_object()) {
                return fail<Corpus>(Errc::corrupt_index, "malformed metadata JSON");
            }
            c.cfg_.hnsw_threshold = m.value("hnsw_threshold", c.cfg_.hnsw_threshold);
            c.cfg_.embed_batch = m.value("embed_batch", c.cfg_.embed_batch);
            if (m.contains("chunk")) {
                const auto& ck = m["chunk"];
                c.cfg_.chunk.max_lines = ck.value("max_lines", c.cfg_.chunk.max_lines);
                c.cfg_.chunk.max_chars = ck.value("max_chars", c.cfg_.chunk.max_chars);
                c.cfg_.chunk.overlap_lines = ck.value("overlap_lines", c.cfg_.chunk.overlap_lines);
                c.cfg_.chunk.heading_context =
                    ck.value("heading_context", c.cfg_.chunk.heading_context);
                if (ck.contains("policy")) {
                    const auto& p = ck["policy"];
                    c.cfg_.chunk.policy.model_identity = p.value("model", std::string{});
                    c.cfg_.chunk.policy.tokenizer_identity = p.value("tokenizer", std::string{});
                    c.cfg_.chunk.policy.dimension = p.value("dimension", std::size_t{});
                    c.cfg_.chunk.policy.target_tokens = p.value("target_tokens", std::size_t{});
                    c.cfg_.chunk.policy.max_tokens = p.value("max_tokens", std::size_t{});
                    c.cfg_.chunk.policy.overlap_tokens = p.value("overlap_tokens", std::size_t{});
                    c.cfg_.chunk.policy.reserved_tokens = p.value("reserved_tokens", std::size_t{});
                    c.cfg_.chunk.policy.document_prefix = p.value("document_prefix", std::string{});
                    c.cfg_.chunk.policy.query_prefix = p.value("query_prefix", std::string{});
                    const int counting_mode = p.value("counting_mode", 1);
                    const int invalid_utf8 = p.value("invalid_utf8", 0);
                    if (counting_mode < 0 || counting_mode > 1 || invalid_utf8 < 0 ||
                        invalid_utf8 > 1)
                        return fail<Corpus>(Errc::corrupt_index, "invalid persisted chunking enum");
                    c.cfg_.chunk.policy.counting_mode =
                        static_cast<text::TokenCountingMode>(counting_mode);
                    c.cfg_.chunk.policy.invalid_utf8 =
                        static_cast<text::InvalidUtf8Policy>(invalid_utf8);
                }
            }
            if (m.contains("bm25")) {
                c.cfg_.bm25.k1 = m["bm25"].value("k1", c.cfg_.bm25.k1);
                c.cfg_.bm25.b = m["bm25"].value("b", c.cfg_.bm25.b);
            }
            if (c.cfg_.embed_batch == 0 || c.cfg_.embed_batch > store::kMaxChunks ||
                c.cfg_.hnsw_threshold > store::kMaxChunks ||
                c.cfg_.chunk.policy.dimension > store::kMaxVectorDimension ||
                c.cfg_.chunk.max_lines > store::kMaxChunks ||
                c.cfg_.chunk.max_chars > store::kMaxWalRecordBytes ||
                c.cfg_.chunk.overlap_lines > c.cfg_.chunk.max_lines ||
                c.cfg_.chunk.policy.overlap_tokens > c.cfg_.chunk.policy.max_tokens ||
                c.cfg_.chunk.policy.reserved_tokens > c.cfg_.chunk.policy.max_tokens ||
                !std::isfinite(c.cfg_.bm25.k1) || !std::isfinite(c.cfg_.bm25.b) ||
                c.cfg_.bm25.k1 < 0.0F || c.cfg_.bm25.b < 0.0F || c.cfg_.bm25.b > 1.0F)
                return fail<Corpus>(Errc::corrupt_index, "invalid persisted configuration");
        }

        // DOCS.
        {
            store::Reader r(*docs);
            std::uint32_t n;
            if (!r.u(n) || n > store::kMaxDocuments || n > (docs->size() - 4) / 20)
                return fail<Corpus>(Errc::corrupt_index, "docs count");
            c.docs_.reserve(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                Document d;
                std::uint32_t id;
                if (!r.u(id) || id != i || !r.str(d.uri) || !r.str(d.title) || !r.str(d.text))
                    return fail<Corpus>(Errc::corrupt_index, "doc");
                d.id = DocId{id};
                std::uint32_t mn;
                if (!r.u(mn) || mn > store::kMaxMetadataEntries || mn > r.remaining() / 8)
                    return fail<Corpus>(Errc::corrupt_index, "doc meta");
                for (std::uint32_t j = 0; j < mn; ++j) {
                    std::string k, v;
                    if (!r.str(k) || !r.str(v))
                        return fail<Corpus>(Errc::corrupt_index, "doc meta kv");
                    d.meta[k] = v;
                }
                c.docs_.push_back(std::move(d));
            }
            if (r.remaining() != 0)
                return fail<Corpus>(Errc::corrupt_index, "trailing document data");
        }

        // CHNK.
        {
            store::Reader r(*chunks);
            std::uint32_t n;
            if (!r.u(n) || n > store::kMaxChunks || n > (chunks->size() - 4) / 24)
                return fail<Corpus>(Errc::corrupt_index, "chunk count");
            c.chunks_.reserve(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                Chunk ch;
                std::uint32_t id, doc;
                if (!r.u(id) || id != i || !r.u(doc) || doc >= c.docs_.size() || !r.str(ch.text) ||
                    !r.str(ch.context) || !r.u(ch.start_line) || !r.u(ch.end_line) ||
                    ch.start_line > ch.end_line)
                    return fail<Corpus>(Errc::corrupt_index, "chunk");
                ch.id = ChunkId{id};
                ch.doc = DocId{doc};
                c.chunks_.push_back(std::move(ch));
            }
            if (r.remaining() != 0)
                return fail<Corpus>(Errc::corrupt_index, "trailing chunk data");
        }
        // Link all chunk meta pointers now that docs_ and chunks_ are populated.
        c.relink_meta();
        c.meta_stale_ = false;

        // EMBD (parallel to CHNK).
        if (const std::string* emb = cont->get(store::Tag::embed)) {
            store::Reader r(*emb);
            std::size_t expected_dimension = 0;
            for (auto& ch : c.chunks_) {
                std::uint32_t dim;
                if (!r.u(dim))
                    return fail<Corpus>(Errc::corrupt_index, "embedding count");
                if (dim == 0)
                    continue;
                if (dim > store::kMaxVectorDimension ||
                    (expected_dimension != 0 && dim != expected_dimension))
                    return fail<Corpus>(Errc::corrupt_index, "mixed embedding dimensions");
                expected_dimension = dim;
                std::string_view raw;
                if (dim > r.remaining() / sizeof(float) ||
                    !r.bytes(static_cast<std::size_t>(dim) * sizeof(float), raw))
                    return fail<Corpus>(Errc::corrupt_index, "embedding");
                ch.embedding.resize(dim);
                std::memcpy(ch.embedding.data(), raw.data(),
                            static_cast<std::size_t>(dim) * sizeof(float));
                double squared_norm = 0.0;
                for (const float value : ch.embedding) {
                    if (!std::isfinite(value))
                        return fail<Corpus>(Errc::corrupt_index, "non-finite embedding");
                    squared_norm += static_cast<double>(value) * value;
                }
                if (!std::isfinite(squared_norm) ||
                    squared_norm <= std::numeric_limits<double>::min() ||
                    std::abs(squared_norm - 1.0) > 0.01)
                    return fail<Corpus>(Errc::corrupt_index, "embedding is not normalized");
            }
            if (r.remaining() != 0)
                return fail<Corpus>(Errc::corrupt_index, "trailing embedding data");
        }

        // BM25 (required).
        {
            auto idx = lexical::Bm25Index::deserialize(*bm25);
            if (!idx)
                return unexpected(idx.error());
            c.bm25_ = std::move(*idx);
        }
        // HNSW (optional).
        if (const std::string* h = cont->get(store::Tag::hnsw)) {
            auto idx = HnswIndex::deserialize(*h);
            if (!idx)
                return unexpected(idx.error());
            c.hnsw_ = std::move(*idx);
        }
        // TOMB (optional; absent in format minor 0 and when nothing is deleted).
        if (const std::string* t = cont->get(store::Tag::tomb)) {
            store::Reader r(*t);
            std::uint32_t n;
            if (!r.u(n) || n > store::kMaxDocuments || n > r.remaining() / sizeof(std::uint32_t))
                return fail<Corpus>(Errc::corrupt_index, "tomb count");
            for (std::uint32_t i = 0; i < n; ++i) {
                std::uint32_t id;
                if (!r.u(id) || id >= c.docs_.size())
                    return fail<Corpus>(Errc::corrupt_index, "tomb id");
                if (!c.deleted_docs_.insert(id).second)
                    return fail<Corpus>(Errc::corrupt_index, "duplicate tomb id");
            }
            if (r.remaining() != 0)
                return fail<Corpus>(Errc::corrupt_index, "trailing tomb data");
        }
        // Epochs are process-local. When a file is reopened in the same process,
        // continue from the path's publication epoch so a subsequent tombstone or
        // upsert cannot be mistaken for an older concurrent snapshot and dropped.
        auto& publication = path_state(path);
        {
            std::lock_guard lock(publication.mu);
            if (publication.any)
                c.epoch_ = publication.published;
        }
        return c;
    } catch (const std::bad_alloc&) {
        return fail<Corpus>(Errc::corrupt_index, "index allocation failed");
    } catch (const std::exception& error) {
        return fail<Corpus>(Errc::corrupt_index,
                            std::string("index parse failed: ") + error.what());
    } catch (...) {
        return fail<Corpus>(Errc::corrupt_index, "index parse failed");
    }
}

} // namespace rag::index
