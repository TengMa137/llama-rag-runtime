#include "rag/retrieval/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::retrieval {
namespace {

backend::CandidateList rrf(const backend::CandidateList& lexical,
                           const backend::CandidateList& dense, const RuntimeConfig& config,
                           std::size_t limit) {
    std::map<backend::ChunkKey, float> scores;
    for (std::size_t rank = 0; rank < lexical.size(); ++rank)
        scores[lexical[rank].chunk] +=
            config.bm25_weight / static_cast<float>(config.rrf_rank_constant + rank + 1);
    for (std::size_t rank = 0; rank < dense.size(); ++rank)
        scores[dense[rank].chunk] +=
            config.dense_weight / static_cast<float>(config.rrf_rank_constant + rank + 1);
    backend::CandidateList output;
    output.reserve(scores.size());
    for (const auto& [key, score] : scores)
        output.push_back({key, score, backend::ScoreType::cosine});
    std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    });
    if (output.size() > limit)
        output.resize(limit);
    return output;
}

void feature_rerank(std::string_view query, backend::CandidateList& candidates,
                    const std::unordered_map<backend::ChunkKey, backend::StoredChunk>& chunks) {
    text::Tokenizer tokenizer;
    auto terms = tokenizer.tokenize(query);
    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
    if (terms.empty() || candidates.empty())
        return;
    float low = candidates.front().raw_score;
    float high = low;
    for (const auto& candidate : candidates) {
        low = std::min(low, candidate.raw_score);
        high = std::max(high, candidate.raw_score);
    }
    const float range = high - low;
    for (auto& candidate : candidates) {
        const auto found = chunks.find(candidate.chunk);
        if (found == chunks.end())
            continue;
        const auto chunk_terms =
            tokenizer.tokenize(found->second.context + "\n" + found->second.text);
        const std::unordered_set<std::string> present(chunk_terms.begin(), chunk_terms.end());
        std::size_t covered = 0;
        for (const auto& term : terms)
            covered += present.contains(term) ? 1U : 0U;
        const float coverage = static_cast<float>(covered) / static_cast<float>(terms.size());
        const float base = range > 1.0e-9F ? (candidate.raw_score - low) / range : 1.0F;
        candidate.raw_score = 0.6F * base + 0.4F * coverage;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.raw_score != right.raw_score)
            return left.raw_score > right.raw_score;
        return left.chunk < right.chunk;
    });
}

backend::CandidateList
mmr(const backend::CandidateList& candidates,
    const std::unordered_map<backend::ChunkKey, backend::StoredChunk>& chunks, std::size_t k,
    float lambda) {
    if (candidates.empty())
        return {};
    lambda = std::clamp(lambda, 0.0F, 1.0F);
    k = std::min(k, candidates.size());
    float low = candidates.front().raw_score;
    float high = low;
    for (const auto& candidate : candidates) {
        low = std::min(low, candidate.raw_score);
        high = std::max(high, candidate.raw_score);
    }
    const float range = high - low > 1.0e-9F ? high - low : 1.0F;
    auto similarity = [&](std::size_t left, std::size_t right) {
        const auto& a = chunks.at(candidates[left].chunk);
        const auto& b = chunks.at(candidates[right].chunk);
        if (!a.embedding.empty() && a.embedding.size() == b.embedding.size())
            return dense::dot(a.embedding, b.embedding);
        text::Tokenizer tokenizer;
        const auto left_terms = tokenizer.tokenize(a.text);
        const auto right_terms = tokenizer.tokenize(b.text);
        const std::unordered_set<std::string> x(left_terms.begin(), left_terms.end());
        const std::unordered_set<std::string> y(right_terms.begin(), right_terms.end());
        std::size_t intersection = 0;
        for (const auto& term : x)
            intersection += y.contains(term) ? 1U : 0U;
        const std::size_t union_size = x.size() + y.size() - intersection;
        return union_size == 0 ? 0.0F
                               : static_cast<float>(intersection) / static_cast<float>(union_size);
    };

    std::vector<std::size_t> selected;
    std::vector<bool> used(candidates.size(), false);
    selected.reserve(k);
    while (selected.size() < k) {
        std::size_t best = candidates.size();
        float best_score = -INFINITY;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (used[index])
                continue;
            float redundancy = 0.0F;
            for (const auto chosen : selected)
                redundancy = std::max(redundancy, similarity(index, chosen));
            const float relevance = (candidates[index].raw_score - low) / range;
            const float score = lambda * relevance - (1.0F - lambda) * redundancy;
            if (score > best_score) {
                best = index;
                best_score = score;
            }
        }
        if (best == candidates.size())
            break;
        used[best] = true;
        selected.push_back(best);
    }
    backend::CandidateList output;
    output.reserve(selected.size());
    for (const auto index : selected)
        output.push_back(candidates[index]);
    return output;
}

void stitch(backend::CandidateList& candidates,
            const std::unordered_map<backend::ChunkKey, backend::StoredChunk>& chunks,
            std::size_t max_gap) {
    std::vector<bool> dropped(candidates.size(), false);
    for (std::size_t left = 0; left < candidates.size(); ++left) {
        if (dropped[left])
            continue;
        const auto& a = chunks.at(candidates[left].chunk);
        for (std::size_t right = left + 1; right < candidates.size(); ++right) {
            if (dropped[right])
                continue;
            const auto& b = chunks.at(candidates[right].chunk);
            if (a.document != b.document || a.revision != b.revision)
                continue;
            const std::uint32_t gap =
                b.start_line > a.end_line
                    ? b.start_line - a.end_line
                    : (a.start_line > b.end_line ? a.start_line - b.end_line : 0);
            if (gap <= max_gap)
                dropped[right] = true;
        }
    }
    backend::CandidateList output;
    output.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index)
        if (!dropped[index])
            output.push_back(std::move(candidates[index]));
    candidates = std::move(output);
}

} // namespace

Result<std::vector<SearchResult>> search(const backend::CandidateBackend& backend,
                                         const backend::SearchRequest& request,
                                         RuntimeConfig config) {
    if (request.query.empty() || request.top_k == 0)
        return fail<std::vector<SearchResult>>(Errc::invalid_argument,
                                               "query and non-zero top-k are required");
    const std::size_t pool = std::max(request.top_k, request.candidate_pool);
    backend::CandidateList lexical;
    backend::CandidateList dense_candidates;
    Result<backend::CandidateList> lexical_result = backend::CandidateList{};
    Result<backend::CandidateList> dense_result = backend::CandidateList{};

    const bool run_lexical = request.mode != backend::SearchMode::dense;
    const bool run_dense = request.mode != backend::SearchMode::lexical;
    if (run_dense && !request.embedding)
        return fail<std::vector<SearchResult>>(Errc::invalid_argument,
                                               "dense retrieval requires a query embedding");

    if (run_lexical && run_dense) {
        auto batch = backend.hybrid_candidates(
            {{request.query, pool, request.filter}, {*request.embedding, pool, request.filter}});
        if (!batch)
            return unexpected(batch.error());
        lexical_result = std::move(batch->lexical);
        dense_result = std::move(batch->dense);
    } else if (run_lexical) {
        lexical_result = backend.lexical_candidates({request.query, pool, request.filter});
    } else {
        dense_result = backend.dense_candidates({*request.embedding, pool, request.filter});
    }
    if (run_lexical && !lexical_result)
        return unexpected(lexical_result.error());
    if (run_dense && !dense_result)
        return unexpected(dense_result.error());
    if (run_lexical)
        lexical = std::move(*lexical_result);
    if (run_dense)
        dense_candidates = std::move(*dense_result);

    backend::CandidateList candidates;
    if (request.mode == backend::SearchMode::hybrid)
        candidates = rrf(lexical, dense_candidates, config, pool);
    else
        candidates = request.mode == backend::SearchMode::lexical ? std::move(lexical)
                                                                  : std::move(dense_candidates);

    std::vector<backend::ChunkKey> keys;
    keys.reserve(candidates.size());
    for (const auto& candidate : candidates)
        keys.push_back(candidate.chunk);
    const bool quality = request.profile == "quality";
    auto fetched = backend.fetch(keys, {true, quality});
    if (!fetched)
        return unexpected(fetched.error());
    std::unordered_map<backend::ChunkKey, backend::StoredChunk> chunks;
    chunks.reserve(fetched->size());
    for (auto& chunk : *fetched)
        chunks.emplace(chunk.key, std::move(chunk));
    if (chunks.size() != candidates.size())
        return fail<std::vector<SearchResult>>(Errc::corrupt_index,
                                               "backend did not resolve every candidate");

    feature_rerank(request.query, candidates, chunks);
    if (quality) {
        candidates = mmr(candidates, chunks, request.top_k, config.mmr_lambda);
        stitch(candidates, chunks, config.adjacent_line_gap);
    }
    if (candidates.size() > request.top_k)
        candidates.resize(request.top_k);

    std::vector<SearchResult> output;
    output.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const auto& chunk = chunks.at(candidate.chunk);
        SearchResult result;
        result.chunk_key = chunk.key;
        result.document_key = chunk.document;
        result.revision = chunk.revision;
        result.score = Score{candidate.raw_score};
        result.text = chunk.text;
        result.context = chunk.context;
        result.uri = chunk.document;
        result.title = chunk.title;
        result.metadata = chunk.metadata;
        result.start_line = chunk.start_line;
        result.end_line = chunk.end_line;
        output.push_back(std::move(result));
    }
    return output;
}

} // namespace rag::retrieval
