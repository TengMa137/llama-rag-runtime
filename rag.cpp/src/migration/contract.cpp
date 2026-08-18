#include "rag/migration/contract.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rag/core/keys.hpp"

namespace rag::migration {
namespace {

class Hash {
  public:
    void field(std::string_view value) {
        bytes(std::to_string(value.size()));
        bytes(":");
        bytes(value);
        bytes("\n");
    }
    void number(std::uint64_t value) { field(std::to_string(value)); }
    void real(float value) { number(std::bit_cast<std::uint32_t>(value)); }
    [[nodiscard]] std::string finish() const {
        std::ostringstream output;
        output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value_;
        return output.str();
    }

  private:
    void bytes(std::string_view value) {
        for (const unsigned char byte : value) {
            value_ ^= byte;
            value_ *= 1099511628211ULL;
        }
    }
    std::uint64_t value_ = 1469598103934665603ULL;
};

Result<void> validate_document(const RevisionedDocument& record, CorpusAudit& audit,
                               Hash& documents, Hash& vectors) {
    const auto& document = record.document;
    if (record.revision == 0 || document.key.empty() || document.chunks.empty() ||
        document.content_hash.empty() || document.embedding_identity.empty() ||
        document.chunking_fingerprint.empty())
        return fail<void>(Errc::corrupt_index, "migration document identity is incomplete");
    if (document.content != normalize_source_text(document.content) ||
        document.content_hash !=
            document_content_hash(document.title, document.content, document.metadata))
        return fail<void>(Errc::corrupt_index, "migration document content hash is invalid");
    if (audit.embedding_identity.empty()) {
        audit.embedding_identity = document.embedding_identity;
        audit.chunking_fingerprint = document.chunking_fingerprint;
        audit.dimension = document.chunks.front().embedding.size();
    }
    if (document.embedding_identity != audit.embedding_identity ||
        document.chunking_fingerprint != audit.chunking_fingerprint)
        return fail<void>(Errc::dimension_mismatch,
                          "migration source contains incompatible corpus identities");
    documents.field(document.key);
    documents.number(record.revision);
    documents.field(document.content_hash);
    documents.field(document.title);
    documents.field(document.content);
    documents.field(document.embedding_identity);
    documents.field(document.chunking_fingerprint);
    for (const auto& [key, value] : document.metadata) {
        documents.field(key);
        documents.field(value);
    }
    for (std::size_t ordinal = 0; ordinal < document.chunks.size(); ++ordinal) {
        const auto& chunk = document.chunks[ordinal];
        const std::string indexed =
            chunk.context.empty() ? chunk.text : chunk.context + "\n" + chunk.text;
        if (chunk.ordinal != ordinal || chunk.key.empty() ||
            chunk.embedding.size() != audit.dimension || chunk.indexed_text != indexed ||
            chunk.key !=
                stable_chunk_key(document.key, chunk.start_line, chunk.end_line, chunk.text))
            return fail<void>(Errc::dimension_mismatch,
                              "migration chunk ordering or dimension is invalid");
        double norm = 0.0;
        for (const float value : chunk.embedding) {
            if (!std::isfinite(value))
                return fail<void>(Errc::corrupt_index, "migration vector is non-finite");
            norm += static_cast<double>(value) * value;
        }
        if (std::abs(norm - 1.0) > 1.0e-3)
            return fail<void>(Errc::corrupt_index, "migration vector is not normalized");
        documents.field(chunk.key);
        documents.number(chunk.ordinal);
        documents.field(chunk.text);
        documents.field(chunk.indexed_text);
        documents.field(chunk.context);
        documents.number(chunk.start_line);
        documents.number(chunk.end_line);
        vectors.field(chunk.key);
        for (const float value : chunk.embedding)
            vectors.real(value);
        ++audit.chunks;
    }
    ++audit.documents;
    return {};
}

std::string run_id(std::string_view direction, std::string_view fingerprint) {
    Hash hash;
    hash.field(direction);
    hash.field(fingerprint);
    return "mig_" + hash.finish().substr(8);
}

Result<void> compare_audits(const CorpusAudit& source, const CorpusAudit& destination) {
    if (source != destination)
        return fail<void>(Errc::corrupt_index,
                          "migration destination audit does not match the source");
    return {};
}

bool same_record(const RevisionedDocument& left, const RevisionedDocument& right) {
    const auto& a = left.document;
    const auto& b = right.document;
    if (left.revision != right.revision || a.key != b.key || a.title != b.title ||
        a.content != b.content || a.metadata != b.metadata || a.content_hash != b.content_hash ||
        a.chunking_fingerprint != b.chunking_fingerprint ||
        a.embedding_identity != b.embedding_identity || a.chunks.size() != b.chunks.size())
        return false;
    for (std::size_t index = 0; index < a.chunks.size(); ++index) {
        const auto& x = a.chunks[index];
        const auto& y = b.chunks[index];
        if (x.key != y.key || x.ordinal != y.ordinal || x.text != y.text ||
            x.indexed_text != y.indexed_text || x.context != y.context ||
            x.start_line != y.start_line || x.end_line != y.end_line || x.embedding != y.embedding)
            return false;
    }
    return true;
}

Result<MigrationProgress> reconcile(const Endpoint& source, const Endpoint& destination,
                                    std::string run, std::string fingerprint,
                                    std::size_t batch_size) {
    MigrationProgress output;
    output.run_id = std::move(run);
    output.source_fingerprint = std::move(fingerprint);
    std::string cursor;
    while (true) {
        auto expected = source.read_batch(cursor, batch_size);
        auto actual = destination.read_batch(cursor, batch_size);
        if (!expected)
            return unexpected(expected.error());
        if (!actual)
            return unexpected(actual.error());
        if (actual->documents.size() > expected->documents.size())
            return fail<MigrationProgress>(Errc::already_exists,
                                           "migration destination is not a source prefix");
        for (std::size_t index = 0; index < actual->documents.size(); ++index)
            if (!same_record(expected->documents[index], actual->documents[index]))
                return fail<MigrationProgress>(Errc::already_exists,
                                               "migration destination conflicts with source");
        for (const auto& record : actual->documents) {
            ++output.documents;
            output.chunks += record.document.chunks.size();
        }
        if (!actual->documents.empty())
            cursor = actual->next_cursor;
        output.last_document = cursor;
        if (actual->complete) {
            output.complete =
                expected->complete && actual->documents.size() == expected->documents.size();
            return output;
        }
        if (actual->documents.empty() || expected->complete)
            return fail<MigrationProgress>(Errc::already_exists,
                                           "migration destination contains extra documents");
    }
}

} // namespace

Result<CorpusAudit> audit(const Endpoint& endpoint, std::size_t batch_size) {
    if (batch_size == 0)
        return fail<CorpusAudit>(Errc::invalid_argument, "migration batch size must be non-zero");
    CorpusAudit output;
    Hash document_hash;
    Hash vector_hash;
    std::string cursor;
    while (true) {
        auto batch = endpoint.read_batch(cursor, batch_size);
        if (!batch)
            return unexpected(batch.error());
        std::string previous = cursor;
        for (const auto& record : batch->documents) {
            if (!previous.empty() && record.document.key <= previous)
                return fail<CorpusAudit>(Errc::corrupt_index,
                                         "migration endpoint returned unordered documents");
            if (auto valid = validate_document(record, output, document_hash, vector_hash); !valid)
                return unexpected(valid.error());
            previous = record.document.key;
        }
        if (batch->next_cursor != previous || (!batch->complete && batch->documents.empty()))
            return fail<CorpusAudit>(Errc::corrupt_index,
                                     "migration endpoint returned an invalid cursor");
        cursor = batch->next_cursor;
        if (batch->complete)
            break;
    }
    output.document_checksum = document_hash.finish();
    output.vector_checksum = vector_hash.finish();
    Hash fingerprint;
    fingerprint.number(output.documents);
    fingerprint.number(output.chunks);
    fingerprint.number(output.dimension);
    fingerprint.field(output.embedding_identity);
    fingerprint.field(output.chunking_fingerprint);
    fingerprint.field(output.document_checksum);
    fingerprint.field(output.vector_checksum);
    output.fingerprint = fingerprint.finish();
    return output;
}

Result<MigrationReport> migrate(Endpoint& source, Endpoint& destination, std::string direction,
                                MigrationOptions options) {
    if (direction.empty() || options.batch_size == 0 || options.batch_size > 10'000 ||
        options.sample_searches > 1'000)
        return fail<MigrationReport>(Errc::invalid_argument, "migration options are invalid");
    auto source_audit = audit(source, options.batch_size);
    if (!source_audit)
        return unexpected(source_audit.error());
    MigrationReport report;
    report.direction = std::move(direction);
    report.source = source.description();
    report.destination = destination.description();
    report.source_audit = *source_audit;
    report.id = run_id(report.direction, source_audit->fingerprint);

    auto prior = destination.progress(report.id);
    if (!prior)
        return unexpected(prior.error());
    auto reconciled =
        reconcile(source, destination, report.id, source_audit->fingerprint, options.batch_size);
    if (!reconciled)
        return unexpected(reconciled.error());
    MigrationProgress progress = std::move(*reconciled);
    if (*prior) {
        if ((*prior)->source_fingerprint != source_audit->fingerprint)
            return fail<MigrationReport>(Errc::already_exists,
                                         "migration progress belongs to a different source");
        report.resumed = true;
    } else if (progress.documents != 0) {
        // The durable data batch may have reached its destination before the
        // progress record. A verified source prefix is safe to resume.
        report.resumed = true;
    }

    std::vector<Vector> samples;
    if (options.sample_searches != 0) {
        std::string sample_cursor;
        while (samples.size() < options.sample_searches) {
            auto batch = source.read_batch(sample_cursor, options.batch_size);
            if (!batch)
                return unexpected(batch.error());
            for (const auto& record : batch->documents)
                for (const auto& chunk : record.document.chunks) {
                    samples.push_back(chunk.embedding);
                    if (samples.size() == options.sample_searches)
                        break;
                }
            sample_cursor = batch->next_cursor;
            if (batch->complete)
                break;
        }
    }

    while (!progress.complete) {
        auto batch = source.read_batch(progress.last_document, options.batch_size);
        if (!batch)
            return unexpected(batch.error());
        MigrationProgress next = progress;
        next.last_document = batch->next_cursor;
        next.complete = batch->complete;
        for (const auto& record : batch->documents) {
            ++next.documents;
            next.chunks += record.document.chunks.size();
        }
        if (auto written = destination.write_batch(report.id, *source_audit, *batch, next);
            !written)
            return unexpected(written.error());
        progress = std::move(next);
        ++report.batches;
    }
    if (auto finished = destination.finish(report.id, *source_audit, progress); !finished)
        return unexpected(finished.error());

    auto final_source = audit(source, options.batch_size);
    if (!final_source)
        return unexpected(final_source.error());
    if (*final_source != *source_audit)
        return fail<MigrationReport>(Errc::already_exists,
                                     "migration source changed while it was being copied");
    auto destination_audit = audit(destination, options.batch_size);
    if (!destination_audit)
        return unexpected(destination_audit.error());
    if (auto equal = compare_audits(*source_audit, *destination_audit); !equal)
        return unexpected(equal.error());

    for (const auto& query : samples) {
        const std::size_t k = std::min<std::size_t>(10, source_audit->chunks);
        auto expected = source.exact_candidates(query, k);
        auto actual = destination.exact_candidates(query, k);
        if (!expected)
            return unexpected(expected.error());
        if (!actual)
            return unexpected(actual.error());
        if (expected->size() != actual->size())
            return fail<MigrationReport>(Errc::corrupt_index,
                                         "sampled exact search differs after migration");
        for (std::size_t index = 0; index < expected->size(); ++index)
            if ((*expected)[index].chunk != (*actual)[index].chunk ||
                std::abs((*expected)[index].raw_score - (*actual)[index].raw_score) > 1.0e-5F)
                return fail<MigrationReport>(Errc::corrupt_index,
                                             "sampled exact search differs after migration");
        ++report.sampled_searches;
    }
    report.destination_audit = std::move(*destination_audit);
    report.complete = true;
    return report;
}

std::string json_report(const MigrationReport& report) {
    const auto audit_json = [](const CorpusAudit& value) {
        return nlohmann::json{{"documents", value.documents},
                              {"chunks", value.chunks},
                              {"dimension", value.dimension},
                              {"embedding_identity", value.embedding_identity},
                              {"chunking_fingerprint", value.chunking_fingerprint},
                              {"document_checksum", value.document_checksum},
                              {"vector_checksum", value.vector_checksum},
                              {"fingerprint", value.fingerprint}};
    };
    return nlohmann::json{{"id", report.id},
                          {"object", "rag.migration_report"},
                          {"direction", report.direction},
                          {"source", report.source},
                          {"destination", report.destination},
                          {"resumed", report.resumed},
                          {"complete", report.complete},
                          {"batches", report.batches},
                          {"sampled_exact_searches", report.sampled_searches},
                          {"source_audit", audit_json(report.source_audit)},
                          {"destination_audit", audit_json(report.destination_audit)}}
        .dump(2);
}

} // namespace rag::migration
