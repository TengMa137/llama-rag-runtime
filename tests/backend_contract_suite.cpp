#include "backend_contract_suite.hpp"

#include <sstream>
#include <utility>
#include <vector>

#include <rag/dense/backends.hpp>
#include <rag/dense/simd.hpp>
#include <rag/preparation/document_preparer.hpp>
#include <rag/retrieval/runtime.hpp>

namespace lrs::tests {
namespace {

rag::Result<void> verify(bool condition, std::string message) {
    if (!condition)
        return rag::fail<void>(rag::Errc::corrupt_index,
                               "candidate backend contract: " + std::move(message));
    return {};
}

} // namespace

rag::Result<void> run_candidate_backend_contract(rag::backend::CandidateBackend& backend,
                                                 const CandidateBackendContractOptions& contract) {
    if (contract.key_prefix.empty() || contract.dimension == 0)
        return rag::fail<void>(rag::Errc::invalid_argument,
                               "candidate backend contract options are invalid");

    rag::preparation::PrepareOptions options;
    options.chunking.max_lines = 1;
    options.chunking.overlap_lines = 0;
    const rag::dense::AnyEmbedder embedder{rag::dense::HashEmbedder{contract.dimension}};
    const std::string north_key = contract.key_prefix + "/north";
    const std::string south_key = contract.key_prefix + "/south";
    auto north =
        rag::preparation::prepare_document(north_key, "alpha orchard\nshared guide",
                                           {{"tenant", "north"}}, "North", options, &embedder);
    auto south = rag::preparation::prepare_document(
        south_key, "alpha harbor", {{"tenant", "south"}}, "South", options, &embedder);
    if (!north)
        return rag::unexpected(north.error());
    if (!south)
        return rag::unexpected(south.error());
    if (auto valid = verify(north->chunks.size() == 2 && south->chunks.size() == 1,
                            "preparation did not preserve deterministic line chunks");
        !valid)
        return valid;
    const auto original_north = *north;
    if (auto activated = backend.activate(std::move(*north), 1); !activated)
        return activated;
    if (auto activated = backend.activate(std::move(*south), 1); !activated)
        return activated;

    if (contract.filler_chunks != 0) {
        std::ostringstream content;
        for (std::size_t row = 0; row < contract.filler_chunks; ++row)
            content << "contract filler " << row << '\n';
        auto filler = rag::preparation::prepare_document(contract.key_prefix + "/filler",
                                                         content.str(), {{"tenant", "filler"}},
                                                         "Filler", options, &embedder);
        if (!filler)
            return rag::unexpected(filler.error());
        if (auto valid = verify(filler->chunks.size() == contract.filler_chunks,
                                "filler chunk count is not reproducible");
            !valid)
            return valid;
        if (auto activated = backend.activate(std::move(*filler), 1); !activated)
            return activated;
    }
    if (contract.publish_base)
        if (auto published = contract.publish_base(); !published)
            return published;

    auto stats = backend.stats();
    if (!stats)
        return rag::unexpected(stats.error());
    if (auto valid = verify(stats->live_documents == 2 + (contract.filler_chunks == 0 ? 0 : 1),
                            "live document count differs");
        !valid)
        return valid;
    if (auto valid =
            verify(stats->live_chunks == 3 + contract.filler_chunks, "live chunk count differs");
        !valid)
        return valid;
    if (auto valid = verify(stats->capabilities.atomic_document_activation,
                            "atomic activation capability is absent");
        !valid)
        return valid;
    if (auto valid = verify(!contract.require_durable || stats->capabilities.durable,
                            "durable capability is absent");
        !valid)
        return valid;

    const rag::backend::MetadataFilter north_only{{{"tenant", "north"}}};
    auto lexical = backend.lexical_candidates({"alpha", 10, north_only});
    if (!lexical)
        return rag::unexpected(lexical.error());
    if (auto valid = verify(lexical->size() == 1, "lexical metadata filter leaked results"); !valid)
        return valid;

    auto query = embedder.embed_one("alpha");
    if (!query)
        return rag::unexpected(query.error());
    rag::dense::normalize(*query);
    auto dense = backend.dense_candidates({*query, 10, north_only});
    if (!dense)
        return rag::unexpected(dense.error());
    if (auto valid = verify(dense->size() == 2, "dense metadata filter is incomplete"); !valid)
        return valid;
    auto hybrid = backend.hybrid_candidates({{"alpha", 10, north_only}, {*query, 10, north_only}});
    if (!hybrid)
        return rag::unexpected(hybrid.error());
    if (auto valid = verify(hybrid->lexical.size() == 1 && hybrid->dense.size() == 2,
                            "hybrid snapshot differs from lexical or dense modes");
        !valid)
        return valid;

    std::vector<rag::backend::ChunkKey> dense_keys;
    for (const auto& candidate : *dense)
        dense_keys.push_back(candidate.chunk);
    auto fetched = backend.fetch(dense_keys, {true, true});
    if (!fetched)
        return rag::unexpected(fetched.error());
    if (auto valid = verify(fetched->size() == 2, "fetch did not resolve dense candidates"); !valid)
        return valid;
    for (const auto& chunk : *fetched)
        if (auto valid = verify(chunk.document == north_key && chunk.revision == 1 &&
                                    chunk.metadata.at("tenant") == "north" &&
                                    chunk.embedding.size() == contract.dimension,
                                "fetched chunk fields differ");
            !valid)
            return valid;

    rag::backend::SearchRequest request;
    request.query = "alpha";
    request.embedding = *query;
    request.top_k = 2;
    request.candidate_pool = 8;
    request.filter = north_only;
    auto first = rag::retrieval::search(backend, request);
    auto second = rag::retrieval::search(backend, request);
    if (!first)
        return rag::unexpected(first.error());
    if (!second)
        return rag::unexpected(second.error());
    bool deterministic = first->size() == second->size() && !first->empty();
    for (std::size_t index = 0; deterministic && index < first->size(); ++index)
        deterministic = (*first)[index].chunk_key == (*second)[index].chunk_key &&
                        (*first)[index].score == (*second)[index].score &&
                        (*first)[index].revision == (*second)[index].revision;
    if (auto valid = verify(deterministic, "resolved result ordering is not deterministic"); !valid)
        return valid;
    for (const auto& result : *first)
        if (auto valid = verify(result.document_key == north_key && result.revision == 1 &&
                                    result.metadata.at("tenant") == "north" &&
                                    result.chunk_key.rfind("chk_", 0) == 0,
                                "resolved result fields differ");
            !valid)
            return valid;

    auto replacement = rag::preparation::prepare_document(
        north_key, "replacement cedar", {{"tenant", "north"}}, "North v2", options, &embedder);
    if (!replacement)
        return rag::unexpected(replacement.error());
    if (auto activated = backend.activate(*replacement, 2); !activated)
        return activated;
    auto old = backend.lexical_candidates({"orchard", 10, {}});
    auto current = backend.lexical_candidates({"cedar", 10, {}});
    if (!old)
        return rag::unexpected(old.error());
    if (!current)
        return rag::unexpected(current.error());
    if (auto valid = verify(old->empty() && current->size() == 1,
                            "replacement was not atomically activated");
        !valid)
        return valid;

    if (auto stale = backend.activate(original_north, 1); stale)
        return rag::fail<void>(rag::Errc::corrupt_index,
                               "candidate backend contract accepted a stale revision");
    auto malformed = *replacement;
    malformed.chunks.front().embedding.clear();
    if (auto partial = backend.activate(std::move(malformed), 3); partial)
        return rag::fail<void>(rag::Errc::corrupt_index,
                               "candidate backend contract accepted malformed vectors");
    current = backend.lexical_candidates({"cedar", 10, {}});
    if (!current)
        return rag::unexpected(current.error());
    if (auto valid =
            verify(current->size() == 1, "failed activation changed the previous ready revision");
        !valid)
        return valid;

    auto erased = backend.erase(north_key, 3);
    if (!erased)
        return rag::unexpected(erased.error());
    if (auto valid = verify(*erased, "first deletion did not report a mutation"); !valid)
        return valid;
    auto erased_again = backend.erase(north_key, 3);
    if (!erased_again)
        return rag::unexpected(erased_again.error());
    if (auto valid = verify(!*erased_again, "idempotent deletion reported a second mutation");
        !valid)
        return valid;
    if (auto late = backend.activate(*replacement, 2); late)
        return rag::fail<void>(rag::Errc::corrupt_index,
                               "candidate backend contract resurrected a deleted revision");
    current = backend.lexical_candidates({"cedar", 10, {}});
    if (!current)
        return rag::unexpected(current.error());
    return verify(current->empty(), "deleted revision remains lexically visible");
}

} // namespace lrs::tests
