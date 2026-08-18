#pragma once
// rag/engine.hpp — the one-stop facade.
//
// Engine bundles a Corpus + a Pipeline behind a tiny API: add documents,
// build, search. Most users never touch anything below this; power users reach
// into corpus() / with_pipeline() to customize stages, fusion, and rerankers.

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/index/corpus.hpp"
#include "rag/ingestion/embedded_runtime.hpp"
#include "rag/ingestion/runtime.hpp"
#include "rag/pipeline/pipeline.hpp"
#include "rag/retrieval/profile.hpp"

namespace rag {

class Engine {
  public:
    Engine() : corpus_(index::CorpusConfig{}), pipeline_(pipeline::Pipeline::standard()) {}
    explicit Engine(index::CorpusConfig cfg)
        : corpus_(std::move(cfg)), pipeline_(pipeline::Pipeline::standard()) {}

    // Adopt an already-built Corpus (e.g. one just loaded from disk). The Engine
    // takes ownership; the standard pipeline is attached. This is the seam that
    // lets a saved index become a live, searchable/servable Engine with no
    // re-indexing.
    explicit Engine(index::Corpus corpus)
        : corpus_(std::move(corpus)), pipeline_(pipeline::Pipeline::standard()) {}

    // Open a persisted `.ragdb` container directly into an Engine. Total: the
    // load error propagates rather than throwing.
    //   auto eng = rag::Engine::open("docs.ragdb");
    //   if (eng) for (auto& r : *eng->search("...", 5)) { ... }
    [[nodiscard]] static Result<Engine> open(const std::string& path) {
        auto c = index::Corpus::load(path);
        if (!c)
            return unexpected(c.error());
        return Engine{std::move(*c)};
    }

    // Backend-neutral embedded runtime entry point. It restores a checkpoint,
    // repairs/replays the durable job tail, and owns ingestion workers.
    [[nodiscard]] static Result<Engine> open_runtime(ingestion::EmbeddedRuntimeConfig config) {
        auto runtime = ingestion::EmbeddedRuntime::open(std::move(config));
        if (!runtime)
            return unexpected(runtime.error());
        return Engine{std::move(*runtime)};
    }

    [[nodiscard]] static Result<Engine> open_runtime(std::unique_ptr<ingestion::Runtime> runtime) {
        if (!runtime)
            return fail<Engine>(Errc::invalid_argument, "ingestion runtime is required");
        return Engine{std::move(runtime)};
    }

    // Attach a dense embedder (enables hybrid). Fluent.
    Engine& with_embedder(dense::AnyEmbedder e) {
        corpus_.set_embedder(std::move(e));
        return *this;
    }
    Engine& with_pipeline(pipeline::Pipeline p) {
        pipeline_ = std::move(p);
        return *this;
    }

    Result<DocId> add(std::string uri, std::string text, Metadata meta = {},
                      std::string title = {}) {
        return corpus_.add_document(std::move(uri), std::move(text), std::move(meta),
                                    std::move(title));
    }
    Result<void> build() { return corpus_.build(); }

    [[nodiscard]] Result<ingestion::Submission> ingest(ingestion::IngestionInput input,
                                                       bool asynchronous = false) {
        if (!runtime_)
            return fail<ingestion::Submission>(Errc::unavailable,
                                               "engine is using the legacy Corpus path");
        return runtime_->submit(std::move(input), asynchronous);
    }

    [[nodiscard]] Result<ingestion::IngestionJob> erase(backend::DocumentKey document) {
        if (!runtime_)
            return fail<ingestion::IngestionJob>(Errc::unavailable,
                                                 "engine is using the legacy Corpus path");
        return runtime_->erase(std::move(document));
    }

    [[nodiscard]] Result<ingestion::JobInfo> job(const ingestion::JobId& id) const {
        if (!runtime_)
            return fail<ingestion::JobInfo>(Errc::unavailable,
                                            "engine is using the legacy Corpus path");
        return runtime_->job(id);
    }

    [[nodiscard]] Result<ingestion::IngestionJob> wait(const ingestion::JobId& id) {
        if (!runtime_)
            return fail<ingestion::IngestionJob>(Errc::unavailable,
                                                 "engine is using the legacy Corpus path");
        return runtime_->wait(id);
    }

    [[nodiscard]] Result<std::vector<SearchResult>> search(backend::SearchRequest request) const {
        if (!runtime_)
            return fail<std::vector<SearchResult>>(Errc::unavailable,
                                                   "engine is using the legacy Corpus path");
        return runtime_->search(std::move(request));
    }

    [[nodiscard]] Result<void> checkpoint() {
        if (!runtime_)
            return fail<void>(Errc::unavailable, "engine is using the legacy Corpus path");
        return runtime_->checkpoint();
    }

    // Search: returns resolved, ranked results.
    [[nodiscard]] Result<std::vector<SearchResult>>
    search(std::string_view query, std::size_t k = 10, index::MetaFilter filter = {},
           std::vector<std::string>* trace = nullptr) const {
        auto hits = pipeline_.run(corpus_, query, k, std::move(filter), trace);
        if (!hits)
            return unexpected(hits.error());
        std::vector<SearchResult> out;
        out.reserve(hits->size());
        for (const auto& h : *hits)
            out.push_back(corpus_.resolve(h));
        return out;
    }

    [[nodiscard]] Result<std::vector<SearchResult>> search(std::string_view query,
                                                           const retrieval::SearchOptions& options,
                                                           retrieval::Diagnostics* diagnostics,
                                                           index::MetaFilter filter = {}) const {
        const auto started = std::chrono::steady_clock::now();
        const std::size_t k = options.top_k ? options.top_k : 10;
        std::size_t pool = 0;
        std::size_t threshold = 0;
        switch (options.profile) {
            case retrieval::Profile::efficiency:
                pool = std::max<std::size_t>(3 * k, 24);
                threshold = 500;
                break;
            case retrieval::Profile::balanced:
                pool = std::max<std::size_t>(6 * k, 60);
                threshold = 2000;
                break;
            case retrieval::Profile::quality:
                pool = std::max<std::size_t>(20 * k, 200);
                threshold = 2000;
                break;
        }
        if (options.overrides.candidate_pool)
            pool = *options.overrides.candidate_pool;
        if (options.overrides.hnsw_threshold)
            threshold = *options.overrides.hnsw_threshold;

        pipeline::HybridRetrieveConfig hybrid;
        hybrid.candidate_k = std::max(pool, k);
        hybrid.fusion = pipeline::HybridRetrieveConfig::Fusion::rrf;
        pipeline::Pipeline selected;
        if (options.profile == retrieval::Profile::quality) {
            const float lambda = options.overrides.mmr_lambda.value_or(0.5f);
            const bool diversity = options.overrides.diversity.value_or(true);
            const bool stitching = options.overrides.context_stitching.value_or(true);
            if (diversity && stitching)
                selected = pipeline::Pipeline::quality_context_with(
                    hybrid, lambda, options.overrides.adjacent_line_gap.value_or(1));
            else if (diversity)
                selected = pipeline::Pipeline::quality_with(hybrid, lambda);
            else if (stitching)
                selected = pipeline::Pipeline::context_with(
                    hybrid, options.overrides.adjacent_line_gap.value_or(1));
            else
                selected = pipeline::Pipeline::standard_with(hybrid);
        } else {
            selected = pipeline::Pipeline::standard_with(hybrid);
        }

        std::vector<std::string> trace;
        auto hits = selected.run(corpus_, query, k, std::move(filter), &trace);
        if (!hits)
            return unexpected(hits.error());
        std::vector<SearchResult> out;
        out.reserve(hits->size());
        for (const auto& hit : *hits)
            out.push_back(corpus_.resolve(hit));
        if (diagnostics) {
            diagnostics->profile = options.profile;
            diagnostics->candidate_pool = pool;
            diagnostics->result_count = out.size();
            diagnostics->stages = std::move(trace);
            if (!corpus_.has_embedder())
                diagnostics->fallback_reasons.push_back("dense skipped: no embedder");
            if (threshold != corpus_.hnsw_threshold())
                diagnostics->fallback_reasons.push_back(
                    "profile HNSW threshold applies when the index is built");
            diagnostics->elapsed_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - started)
                                          .count();
        }
        return out;
    }

    // Deprecated compatibility accessor. It is meaningful only for engines
    // constructed through the legacy Corpus/open APIs.
    [[nodiscard]] index::Corpus& corpus() noexcept { return corpus_; }
    [[nodiscard]] const index::Corpus& corpus() const noexcept { return corpus_; }

    [[nodiscard]] bool uses_portable_runtime() const noexcept { return runtime_ != nullptr; }

    Result<void> save(const std::string& path) const { return corpus_.save(path); }

  private:
    explicit Engine(std::unique_ptr<ingestion::Runtime> runtime)
        : corpus_(index::CorpusConfig{}), pipeline_(pipeline::Pipeline::standard()),
          runtime_(std::move(runtime)) {}

    index::Corpus corpus_;
    pipeline::Pipeline pipeline_;
    std::unique_ptr<ingestion::Runtime> runtime_;
};

} // namespace rag
