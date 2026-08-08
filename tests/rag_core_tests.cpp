#include <rag/c/rag.h>
#include <rag/dense/backends.hpp>
#include <rag/engine.hpp>
#include <rag/fusion/fuse.hpp>
#include <rag/lexical/bm25.hpp>
#include <rag/store/container.hpp>
#include <rag/store/format.hpp>
#include <rag/store/wal.hpp>
#include <rag/text/chunker.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #condition << '\n';     \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("rag-owned-tests-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct MockTransport final : rag::dense::HttpTransport {
    rag::Result<rag::dense::HttpResponse> response = rag::dense::HttpResponse{};
    mutable rag::dense::HttpRequest request;
    rag::Result<rag::dense::HttpResponse>
    post(const rag::dense::HttpRequest& value) const override {
        request = value;
        return response;
    }
};

struct WrongCountEmbedder {
    std::size_t dimension() const { return 2; }
    std::string_view identity() const { return "wrong-count"; }
    rag::Result<std::vector<rag::Vector>> embed(std::span<const std::string>) const {
        return std::vector<rag::Vector>{};
    }
};

void rewrite_crc(std::string& blob) {
    const auto checksum = rag::store::crc32(std::string_view(blob).substr(0, blob.size() - 4));
    std::memcpy(blob.data() + blob.size() - 4, &checksum, sizeof(checksum));
}

void test_chunking() {
    rag::text::ChunkOptions options;
    options.max_lines = 20;
    options.overlap_lines = 1;
    options.heading_context = true;
    options.policy.target_tokens = 10;
    options.policy.max_tokens = 12;
    options.policy.overlap_tokens = 2;
    options.measure_tokens = [](std::string_view text) { return text.size(); };
    const std::string source = "# H\nalpha beta gamma\n\xF0\x9F\x8C\xB2 delta epsilon\nlast";
    const auto first = rag::text::chunk_document(rag::DocId{3}, source, options);
    const auto second = rag::text::chunk_document(rag::DocId{3}, source, options);
    CHECK(!first.empty());
    CHECK(first.size() == second.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        CHECK(first[index].text == second[index].text);
        CHECK(first[index].start_line <= first[index].end_line);
        CHECK(options.measure_tokens(first[index].text) <= options.policy.max_tokens);
    }
    CHECK(first.front().start_line == 0);
    CHECK(first.back().end_line == 3);
    CHECK(!rag::text::chunking_fingerprint(options).empty());
}

void test_retrieval_and_persistence() {
    TemporaryDirectory temporary;
    const auto database = temporary.path / "compat.ragdb";
    rag::index::CorpusConfig config;
    config.chunk.max_lines = 1;
    config.chunk.overlap_lines = 0;
    config.hnsw_threshold = 2;
    rag::Engine engine(config);
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{16}});
    CHECK(engine.add("a", "alpha orchard\nadjacent alpha", {{"kind", "fruit"}}));
    CHECK(engine.add("b", "beta harbor", {{"kind", "port"}}));
    CHECK(engine.build());
    CHECK(!engine.corpus().lexical_search("alpha", 5).empty());
    CHECK(engine.corpus().dense_search("alpha", 5));
    const auto filtered = engine.search("alpha", 5, [](const rag::Metadata& metadata) {
        const auto found = metadata.find("kind");
        return found != metadata.end() && found->second == "fruit";
    });
    CHECK(filtered && !filtered->empty());
    CHECK(filtered && filtered->front().uri == "a");
    for (const auto profile :
         {rag::retrieval::Profile::efficiency, rag::retrieval::Profile::balanced,
          rag::retrieval::Profile::quality}) {
        rag::retrieval::SearchOptions options;
        options.profile = profile;
        options.top_k = 2;
        rag::retrieval::Diagnostics diagnostics;
        const auto result = engine.search("alpha", options, &diagnostics);
        CHECK(result);
        CHECK(diagnostics.profile == profile);
        if (profile == rag::retrieval::Profile::quality) {
            bool saw_mmr = false;
            bool saw_stitch = false;
            for (const auto& stage : diagnostics.stages) {
                saw_mmr = saw_mmr || stage.find("mmr") != std::string::npos;
                saw_stitch = saw_stitch || stage.find("parent_stitch") != std::string::npos;
            }
            CHECK(saw_mmr);
            CHECK(saw_stitch);
        }
    }
    CHECK(engine.save(database.string()));
    CHECK(!engine.save(temporary.path.string()));
    CHECK(rag::Engine::open(database.string()));
    auto reopened = rag::Engine::open(database.string());
    CHECK(reopened);
    CHECK(reopened && reopened->corpus().chunks().size() == engine.corpus().chunks().size());
    if (reopened) {
        const auto id = reopened->corpus().find_by_uri("a");
        CHECK(id.has_value());
        if (id)
            CHECK(reopened->corpus().remove_document(*id));
        CHECK(reopened->save(database.string()));
    }
    auto deleted = rag::Engine::open(database.string());
    CHECK(deleted);
    CHECK(deleted && deleted->corpus().find_by_uri("a") == std::nullopt);

    std::ifstream input(database, std::ios::binary);
    std::string legacy((std::istreambuf_iterator<char>(input)), {});
    const std::uint16_t minor_zero = 0;
    std::memcpy(legacy.data() + 10, &minor_zero, sizeof(minor_zero));
    rewrite_crc(legacy);
    const auto legacy_path = temporary.path / "v1.0.ragdb";
    std::ofstream output(legacy_path, std::ios::binary);
    output.write(legacy.data(), static_cast<std::streamsize>(legacy.size()));
    output.close();
    CHECK(rag::Engine::open(legacy_path.string()));

    std::ifstream fixture_input(std::filesystem::path(LRS_SOURCE_DIR) /
                                "tests/fixtures/ragdb-v1.0-empty.hex");
    std::string fixture_hex;
    fixture_input >> fixture_hex;
    std::string fixture_bytes;
    fixture_bytes.reserve(fixture_hex.size() / 2);
    for (std::size_t index = 0; index + 1 < fixture_hex.size(); index += 2) {
        const auto byte = static_cast<char>(std::stoul(fixture_hex.substr(index, 2), nullptr, 16));
        fixture_bytes.push_back(byte);
    }
    const auto fixture_path = temporary.path / "checked-in-v1.0.ragdb";
    std::ofstream fixture_output(fixture_path, std::ios::binary);
    fixture_output.write(fixture_bytes.data(), static_cast<std::streamsize>(fixture_bytes.size()));
    fixture_output.close();
    CHECK(rag::Engine::open(fixture_path.string()));
}

void test_rrf_ordering() {
    const std::array<rag::fusion::RankedList, 2> lists{
        rag::fusion::RankedList{{rag::Hit{rag::ChunkId{2}, rag::Score{1.0F}},
                                 rag::Hit{rag::ChunkId{1}, rag::Score{0.5F}}},
                                1.0F},
        rag::fusion::RankedList{{rag::Hit{rag::ChunkId{1}, rag::Score{1.0F}},
                                 rag::Hit{rag::ChunkId{2}, rag::Score{0.5F}}},
                                1.0F}};
    const auto fused = rag::fusion::rrf(lists);
    CHECK(fused.size() == 2);
    CHECK(fused[0].chunk == rag::ChunkId{1});
}

void test_embedding_failures() {
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{WrongCountEmbedder{}});
    CHECK(engine.add("a", "alpha"));
    const auto built = engine.build();
    CHECK(!built);
    CHECK(built.error().code == rag::Errc::dimension_mismatch);
}

void test_container_parser() {
    rag::store::Container container;
    container.put(rag::store::Tag::meta, "{}");
    container.put(rag::store::Tag::docs, std::string(4, '\0'));
    auto valid = container.serialize();
    CHECK(rag::store::Container::parse(valid));
    CHECK(!rag::store::Container::parse(valid.substr(0, valid.size() - 1)));

    auto huge = valid;
    const std::uint32_t count = UINT32_MAX;
    std::memcpy(huge.data() + 16, &count, sizeof(count));
    rewrite_crc(huge);
    CHECK(!rag::store::Container::parse(huge));

    auto duplicate = valid;
    std::uint32_t first_tag = 0;
    std::memcpy(&first_tag, duplicate.data() + 28, sizeof(first_tag));
    std::memcpy(duplicate.data() + 48, &first_tag, sizeof(first_tag));
    rewrite_crc(duplicate);
    CHECK(!rag::store::Container::parse(duplicate));

    auto overlap = valid;
    std::uint64_t first_offset = 0;
    std::memcpy(&first_offset, overlap.data() + 32, sizeof(first_offset));
    std::memcpy(overlap.data() + 52, &first_offset, sizeof(first_offset));
    rewrite_crc(overlap);
    CHECK(!rag::store::Container::parse(overlap));

    TemporaryDirectory temporary;
    rag::store::Container malformed_json;
    malformed_json.put(rag::store::Tag::meta, "{");
    malformed_json.put(rag::store::Tag::docs, std::string(4, '\0'));
    malformed_json.put(rag::store::Tag::chunks, std::string(4, '\0'));
    malformed_json.put(rag::store::Tag::bm25, rag::lexical::Bm25Index{}.serialize());
    const auto malformed_path = temporary.path / "malformed-json.ragdb";
    CHECK(malformed_json.write_file(malformed_path.string()));
    CHECK(!rag::Engine::open(malformed_path.string()));

    rag::Engine vector_engine;
    vector_engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{2}});
    CHECK(vector_engine.add("vector", "alpha"));
    CHECK(vector_engine.build());
    const auto vector_path = temporary.path / "valid-vector.ragdb";
    CHECK(vector_engine.save(vector_path.string()));
    auto vector_container = rag::store::Container::read_file(vector_path.string());
    CHECK(vector_container);
    if (vector_container) {
        rag::store::Container invalid_vector;
        for (const auto tag : {rag::store::Tag::meta, rag::store::Tag::docs,
                               rag::store::Tag::chunks, rag::store::Tag::bm25})
            invalid_vector.put(tag, *vector_container->get(tag));
        std::string embedding = *vector_container->get(rag::store::Tag::embed);
        const float not_a_number = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(embedding.data() + sizeof(std::uint32_t), &not_a_number, sizeof(not_a_number));
        invalid_vector.put(rag::store::Tag::embed, std::move(embedding));
        invalid_vector.set_flags(rag::store::kHasEmbeddings);
        const auto invalid_path = temporary.path / "invalid-vector.ragdb";
        CHECK(invalid_vector.write_file(invalid_path.string()));
        CHECK(!rag::Engine::open(invalid_path.string()));
    }
}

void test_wal_recovery() {
    TemporaryDirectory temporary;
    const auto path = temporary.path / "index.wal";
    {
        rag::store::Wal wal;
        CHECK(wal.open(path.string(), rag::store::SyncMode::none));
        rag::store::WalRecord record;
        record.op = rag::store::WalOp::add_document;
        record.uri = "a";
        record.text = "alpha";
        CHECK(wal.append(record));
        record.uri = "b";
        record.text = "beta";
        CHECK(wal.append(record));
    }
    std::ifstream original_input(path, std::ios::binary);
    std::string original((std::istreambuf_iterator<char>(original_input)), {});
    auto corrupt = original;
    corrupt[12] ^= 1;
    const auto corrupt_path = temporary.path / "corrupt.wal";
    std::ofstream corrupt_output(corrupt_path, std::ios::binary);
    corrupt_output.write(corrupt.data(), static_cast<std::streamsize>(corrupt.size()));
    corrupt_output.close();
    CHECK(!rag::store::Wal::replay(corrupt_path.string()));
    std::ofstream tail(path, std::ios::binary | std::ios::app);
    tail.write("WAL", 3);
    tail.close();
    const auto replayed = rag::store::Wal::replay(path.string());
    CHECK(replayed && replayed->size() == 2);
}

void test_local_http_embedder() {
    auto transport = std::make_shared<MockTransport>();
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":0,"embedding":[3,4]},{"index":1,"embedding":[0,2]}]})"};
    rag::dense::LocalHttpEmbedderConfig config;
    config.host = "127.0.0.1";
    config.dimension = 2;
    config.max_response_bytes = 1024;
    auto embedder = rag::dense::LocalHttpEmbedder::create(config, transport);
    CHECK(embedder);
    const std::array<std::string, 2> text{"a", "b"};
    const auto vectors = embedder->embed(text);
    CHECK(vectors && vectors->size() == 2);
    CHECK(vectors && std::abs((*vectors)[0][0] - 0.6F) < 0.001F);
    CHECK(transport->request.path == "/v1/embeddings");
    CHECK(transport->request.max_response_bytes == 1024);

    transport->response = rag::dense::HttpResponse{200, "{"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":1,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]})"};
    CHECK(!embedder->embed(text));
    transport->response =
        rag::dense::HttpResponse{200, R"({"data":[{"index":0,"embedding":[1,0]}]})"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{
        200, R"({"data":[{"index":0,"embedding":[0,0]},{"index":1,"embedding":[0,1]}]})"};
    CHECK(!embedder->embed(text));
    transport->response = rag::dense::HttpResponse{200, std::string(2048, 'x')};
    CHECK(!embedder->embed(text));
    transport->response =
        rag::fail<rag::dense::HttpResponse>(rag::Errc::transport_error, "timeout");
    CHECK(!embedder->embed(text));

    config.host = "192.0.2.1";
    CHECK(!rag::dense::LocalHttpEmbedder::create(config, transport));
    config.host = "127.0.0.1";
    config.path = "/../escape";
    CHECK(!rag::dense::LocalHttpEmbedder::create(config, transport));
}

void test_c_abi() {
    rag_engine* engine = reinterpret_cast<rag_engine*>(1);
    rag_engine_options options{};
    options.abi_version = RAG_C_ABI_VERSION;
    options.struct_size = sizeof(options);
    CHECK(rag_engine_create(nullptr, &engine) == RAG_ERR_INVALID_ARGUMENT);
    CHECK(engine == nullptr);
    CHECK(rag_engine_create(&options, &engine) == RAG_OK);
    CHECK(engine != nullptr);
    const char* key = "kind";
    CHECK(rag_engine_add(engine, "a", "alpha", nullptr, &key, nullptr, 1, nullptr) ==
          RAG_ERR_INVALID_ARGUMENT);
    rag_results* results = reinterpret_cast<rag_results*>(1);
    CHECK(rag_engine_search(engine, "alpha", 2, &key, nullptr, 1, &results) ==
          RAG_ERR_INVALID_ARGUMENT);
    CHECK(results == nullptr);
    rag_engine_free(engine);
}
} // namespace

int main() {
    test_chunking();
    test_retrieval_and_persistence();
    test_embedding_failures();
    test_rrf_ordering();
    test_container_parser();
    test_wal_recovery();
    test_local_http_embedder();
    test_c_abi();
    if (failures != 0)
        std::cerr << failures << " owned-core checks failed\n";
    return failures == 0 ? 0 : 1;
}
