#include "rag/migration/embedded_endpoint.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>

#include <nlohmann/json.hpp>

#include "rag/backend/embedded_checkpoint.hpp"
#include "rag/ingestion/job_store.hpp"
#include "rag/store/container.hpp"
#include "rag/store/container_view.hpp"
#include "rag/store/format.hpp"

namespace rag::migration {
namespace {

struct ActiveDocument {
    backend::PreparedDocument document;
    backend::DocumentRevision revision = 0;
};

bool same_document(const ActiveDocument& existing, const RevisionedDocument& incoming) {
    const auto& left = existing.document;
    const auto& right = incoming.document;
    if (existing.revision != incoming.revision || left.key != right.key ||
        left.title != right.title || left.content != right.content ||
        left.metadata != right.metadata || left.content_hash != right.content_hash ||
        left.chunking_fingerprint != right.chunking_fingerprint ||
        left.embedding_identity != right.embedding_identity ||
        left.chunks.size() != right.chunks.size())
        return false;
    for (std::size_t index = 0; index < left.chunks.size(); ++index) {
        const auto& a = left.chunks[index];
        const auto& b = right.chunks[index];
        if (a.key != b.key || a.ordinal != b.ordinal || a.text != b.text ||
            a.indexed_text != b.indexed_text || a.context != b.context ||
            a.start_line != b.start_line || a.end_line != b.end_line || a.embedding != b.embedding)
            return false;
    }
    return true;
}

Result<std::map<backend::DocumentKey, ActiveDocument>> load_active(const std::string& path,
                                                                   bool include_jobs) {
    std::map<backend::DocumentKey, ActiveDocument> active;
    if (std::filesystem::exists(path)) {
        auto checkpoint = backend::EmbeddedCheckpointStore::load(path);
        if (!checkpoint)
            return unexpected(checkpoint.error());
        for (const auto& entry : checkpoint->documents) {
            if (!entry.prepared)
                return fail<std::map<backend::DocumentKey, ActiveDocument>>(
                    Errc::corrupt_index, "embedded migration checkpoint document is missing");
            active.emplace(entry.prepared->key, ActiveDocument{*entry.prepared, entry.revision});
        }
    }
    if (!include_jobs)
        return active;
    auto jobs = ingestion::load_ingestion_jobs_read_only(path + ".jobs");
    if (!jobs)
        return unexpected(jobs.error());
    for (const auto& job : *jobs) {
        if (job.status != ingestion::JobStatus::ready)
            continue;
        const auto found = active.find(job.input.document);
        const auto current = found == active.end() ? 0 : found->second.revision;
        if (job.revision <= current)
            continue;
        if (job.operation == ingestion::JobOperation::erase) {
            active.erase(job.input.document);
            continue;
        }
        if (!job.prepared || job.prepared->key != job.input.document ||
            job.prepared->content_hash != job.content_hash)
            return fail<std::map<backend::DocumentKey, ActiveDocument>>(
                Errc::corrupt_index, "ready embedded migration job has invalid prepared data");
        active[job.input.document] = ActiveDocument{*job.prepared, job.revision};
    }
    return active;
}

std::string progress_path(const std::string& checkpoint) { return checkpoint + ".migration"; }

Result<std::optional<MigrationProgress>> load_progress(const std::string& path,
                                                       std::string_view run_id) {
    if (!std::filesystem::exists(path))
        return std::optional<MigrationProgress>{};
    auto container = store::ContainerView::open_file(path);
    if (!container)
        return unexpected(container.error());
    const auto payload = container->get(store::Tag::meta);
    if (!payload)
        return fail<std::optional<MigrationProgress>>(Errc::corrupt_index,
                                                      "embedded migration progress is missing");
    try {
        const auto value = nlohmann::json::parse(*payload);
        MigrationProgress progress;
        progress.run_id = value.at("run_id").get<std::string>();
        progress.source_fingerprint = value.at("source_fingerprint").get<std::string>();
        progress.last_document = value.at("last_document").get<std::string>();
        progress.documents = value.at("documents").get<std::size_t>();
        progress.chunks = value.at("chunks").get<std::size_t>();
        progress.complete = value.at("complete").get<bool>();
        if (progress.run_id != run_id)
            return fail<std::optional<MigrationProgress>>(
                Errc::already_exists, "embedded destination belongs to another migration run");
        return std::optional<MigrationProgress>{std::move(progress)};
    } catch (const std::exception& error) {
        return fail<std::optional<MigrationProgress>>(Errc::corrupt_index,
                                                      "embedded migration progress is invalid: " +
                                                          std::string(error.what()));
    }
}

Result<void> save_progress(const std::string& path, const MigrationProgress& progress) {
    store::Container container;
    container.put(store::Tag::meta,
                  nlohmann::json{{"version", 1},
                                 {"run_id", progress.run_id},
                                 {"source_fingerprint", progress.source_fingerprint},
                                 {"last_document", progress.last_document},
                                 {"documents", progress.documents},
                                 {"chunks", progress.chunks},
                                 {"complete", progress.complete}}
                      .dump());
    return container.write_file(path);
}

class EmbeddedEndpoint final : public Endpoint {
  public:
    EmbeddedEndpoint(std::string path, std::map<backend::DocumentKey, ActiveDocument> active,
                     bool writable)
        : path_(std::move(path)), active_(std::move(active)), writable_(writable) {}

    std::string description() const override { return "embedded:" + path_; }

    Result<DocumentBatch> read_batch(std::string_view after, std::size_t limit) const override {
        if (limit == 0)
            return fail<DocumentBatch>(Errc::invalid_argument, "migration batch limit is zero");
        DocumentBatch output;
        auto iterator = after.empty() ? active_.begin() : active_.upper_bound(std::string(after));
        for (; iterator != active_.end() && output.documents.size() < limit; ++iterator)
            output.documents.push_back({iterator->second.document, iterator->second.revision});
        if (!output.documents.empty())
            output.next_cursor = output.documents.back().document.key;
        else
            output.next_cursor = std::string(after);
        output.complete = iterator == active_.end();
        return output;
    }

    Result<std::optional<MigrationProgress>> progress(std::string_view run_id) const override {
        if (!writable_)
            return std::optional<MigrationProgress>{};
        return load_progress(progress_path(path_), run_id);
    }

    Result<void> write_batch(std::string_view run_id, const CorpusAudit& source,
                             const DocumentBatch& batch,
                             const MigrationProgress& progress_value) override {
        if (!writable_)
            return fail<void>(Errc::invalid_argument, "embedded migration source is read-only");
        if (progress_value.run_id != run_id ||
            progress_value.source_fingerprint != source.fingerprint)
            return fail<void>(Errc::invalid_argument,
                              "embedded migration progress is inconsistent");
        if (!active_.empty()) {
            const auto& first = active_.begin()->second.document;
            if (first.embedding_identity != source.embedding_identity ||
                first.chunking_fingerprint != source.chunking_fingerprint ||
                first.chunks.front().embedding.size() != source.dimension)
                return fail<void>(Errc::dimension_mismatch,
                                  "embedded destination corpus identity differs from source");
        }
        for (const auto& record : batch.documents) {
            const auto found = active_.find(record.document.key);
            if (found != active_.end()) {
                if (!same_document(found->second, record))
                    return fail<void>(Errc::already_exists,
                                      "embedded destination contains conflicting document");
                continue;
            }
            active_.emplace(record.document.key, ActiveDocument{record.document, record.revision});
        }
        backend::EmbeddedCheckpoint checkpoint;
        checkpoint.generation = progress_value.documents;
        checkpoint.documents.reserve(active_.size());
        for (const auto& [key, value] : active_) {
            auto prepared = std::make_shared<backend::PreparedDocument>(value.document);
            checkpoint.documents.push_back({std::move(prepared), value.revision});
            checkpoint.revisions.emplace(key, value.revision);
        }
        if (auto saved = backend::EmbeddedCheckpointStore::save(path_, checkpoint); !saved)
            return unexpected(saved.error());
        return save_progress(progress_path(path_), progress_value);
    }

    Result<void> finish(std::string_view run_id, const CorpusAudit& source,
                        const MigrationProgress& progress_value) override {
        if (!progress_value.complete || progress_value.run_id != run_id ||
            progress_value.source_fingerprint != source.fingerprint)
            return fail<void>(Errc::invalid_argument, "embedded migration cannot finish early");
        return save_progress(progress_path(path_), progress_value);
    }

    Result<backend::CandidateList> exact_candidates(VectorView query,
                                                    std::size_t k) const override {
        backend::CandidateList output;
        for (const auto& [key, value] : active_)
            for (const auto& chunk : value.document.chunks) {
                if (chunk.embedding.size() != query.size())
                    return fail<backend::CandidateList>(Errc::dimension_mismatch,
                                                        "migration query dimension differs");
                float score = 0.0F;
                for (std::size_t index = 0; index < query.size(); ++index)
                    score += query[index] * chunk.embedding[index];
                output.push_back({chunk.key, score, backend::ScoreType::cosine});
            }
        std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
            if (left.raw_score != right.raw_score)
                return left.raw_score > right.raw_score;
            return left.chunk < right.chunk;
        });
        if (output.size() > k)
            output.resize(k);
        return output;
    }

  private:
    std::string path_;
    std::map<backend::DocumentKey, ActiveDocument> active_;
    bool writable_ = false;
};

} // namespace

Result<std::unique_ptr<Endpoint>> open_embedded_source(std::string checkpoint_path) {
    if (checkpoint_path.empty() || !std::filesystem::exists(checkpoint_path))
        return fail<std::unique_ptr<Endpoint>>(Errc::not_found,
                                               "embedded migration source does not exist");
    auto active = load_active(checkpoint_path, true);
    if (!active)
        return unexpected(active.error());
    return std::unique_ptr<Endpoint>(
        new EmbeddedEndpoint(std::move(checkpoint_path), std::move(*active), false));
}

Result<std::unique_ptr<Endpoint>> open_embedded_destination(std::string checkpoint_path) {
    if (checkpoint_path.empty())
        return fail<std::unique_ptr<Endpoint>>(Errc::invalid_argument,
                                               "embedded migration destination is required");
    try {
        const std::filesystem::path path(checkpoint_path);
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
    } catch (const std::exception& error) {
        return fail<std::unique_ptr<Endpoint>>(Errc::io_error,
                                               "embedded migration destination directory failed: " +
                                                   std::string(error.what()));
    }
    auto active = load_active(checkpoint_path, false);
    if (!active)
        return unexpected(active.error());
    return std::unique_ptr<Endpoint>(
        new EmbeddedEndpoint(std::move(checkpoint_path), std::move(*active), true));
}

} // namespace rag::migration
