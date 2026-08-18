#include "lrs/bridge.h"
#include "lrs/config.hpp"
#include "lrs/mobile.h"
#include "lrs/service.hpp"
#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <rag/c/rag.h>
#include <rag/engine.hpp>
#include <rag/store/container.hpp>
#include <rag/text/chunker.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

namespace {
lrs::Config test_config(const std::filesystem::path& database) {
    lrs::Config config;
    config.spawn = false;
    config.deterministic_embeddings = true;
    config.embedding_dimension = 32;
    config.index_path = database.string();
    return config;
}

std::string document(
    std::string content = "# Local RAG\nThe index is persisted in a local retrieval database.") {
    return nlohmann::json{{"id", "demo/getting-started"},
                          {"content_type", "text/markdown"},
                          {"title", "Getting started"},
                          {"content", std::move(content)}}
        .dump();
}
} // namespace

TEST_CASE("token-aware chunks enforce hard limits and preserve source lines", "[unit][chunker]") {
    rag::text::ChunkOptions options;
    options.max_lines = 40;
    options.heading_context = false;
    options.policy.target_tokens = 9;
    options.policy.max_tokens = 12;
    options.policy.overlap_tokens = 2;
    options.measure_tokens = [](std::string_view text) { return text.size(); };

    const auto chunks = rag::text::chunk_document(
        rag::DocId{7}, "alpha beta gamma delta epsilon\r\nð²ð²ð²", options);
    REQUIRE(chunks.size() > 2);
    for (const auto& chunk : chunks) {
        REQUIRE(chunk.text.size() <= 12);
        REQUIRE(chunk.start_line <= chunk.end_line);
    }
    REQUIRE(chunks.front().start_line == 0);
    REQUIRE(chunks.back().end_line == 1);
}

TEST_CASE("chunking fingerprint covers embedding policy", "[unit][chunker]") {
    rag::text::ChunkOptions left;
    left.policy.max_tokens = 512;
    left.policy.model_identity = "model-a";
    auto right = left;
    right.policy.reserved_tokens = 8;
    REQUIRE(rag::text::chunking_fingerprint(left) != rag::text::chunking_fingerprint(right));
}

TEST_CASE("owned C ABI validates versioned option structs", "[unit][abi]") {
    rag_engine_options options{};
    options.abi_version = RAG_C_ABI_VERSION;
    options.struct_size = sizeof(options);
    options.writer_threads = 1;
    rag_engine* engine = nullptr;
    REQUIRE(rag_engine_create(&options, &engine) == RAG_OK);
    REQUIRE(engine != nullptr);
    rag_engine_free(engine);

    options.abi_version += 1;
    REQUIRE(rag_engine_create(&options, &engine) == RAG_ERR_INVALID_ARGUMENT);
}

TEST_CASE("retrieval profiles are parameter bundles with diagnostics", "[unit][retrieval]") {
    rag::Engine engine;
    REQUIRE(engine.add("doc/a", "alpha orchard"));
    REQUIRE(engine.add("doc/b", "beta harbor"));
    REQUIRE(engine.build());

    rag::retrieval::SearchOptions options;
    options.profile = rag::retrieval::Profile::quality;
    options.top_k = 1;
    options.overrides.candidate_pool = 7;
    rag::retrieval::Diagnostics diagnostics;
    const auto results = engine.search("alpha", options, &diagnostics);
    REQUIRE(results);
    REQUIRE(results->size() == 1);
    REQUIRE(diagnostics.profile == rag::retrieval::Profile::quality);
    REQUIRE(diagnostics.candidate_pool == 7);
    REQUIRE_FALSE(diagnostics.stages.empty());
}

TEST_CASE("retrieval configuration is typed and preserves the legacy index path",
          "[unit][config]") {
    auto legacy = test_config("legacy.ragdb");
    legacy.retrieval.embedded.path = legacy.index_path;
    REQUIRE_NOTHROW(lrs::validate_config(legacy));

    auto invalid = legacy;
    invalid.retrieval.embedded.dense.implementation = "native";
    invalid.retrieval.embedded.dense.algorithm = "ivf-pq";
    REQUIRE_THROWS(lrs::validate_config(invalid));
    invalid = legacy;
    invalid.retrieval.backend = "postgres";
#if LRS_ENABLE_POSTGRES
    REQUIRE_NOTHROW(lrs::validate_config(invalid));
#else
    REQUIRE_THROWS(lrs::validate_config(invalid));
#endif

    const auto path = std::filesystem::temp_directory_path() / "lrs-config-dense.json";
    std::ofstream output(path);
    output << R"({
      "retrieval": {
        "backend": "embedded",
        "embedded": {
          "path": "configured.ragdb",
          "dense": {
            "implementation": "native",
            "algorithm": "hnsw",
            "exact_threshold": 123,
            "hnsw": {"neighbors": 24, "ef_construction": 96, "ef_search": 48}
          }
        }
      },
      "inference": {
        "spawn": false,
        "embedding": {"port": 8081, "dimension": 384},
        "generation": {"port": 8082, "context_size": 16384}
      },
      "rag": {"context_tokens": 8192}
    })";
    output.close();
    const auto configured = lrs::load_config(path.string());
    std::filesystem::remove(path);
    REQUIRE(configured.index_path == "configured.ragdb");
    REQUIRE(configured.retrieval.embedded.path == configured.index_path);
    REQUIRE(configured.retrieval.embedded.dense.algorithm == "hnsw");
    REQUIRE(configured.retrieval.embedded.dense.exact_threshold == 123);
    REQUIRE(configured.retrieval.embedded.dense.hnsw.neighbors == 24);
}

#if LRS_ENABLE_POSTGRES
TEST_CASE("desktop service selects PostgreSQL through the runtime contract",
          "[component][postgres]") {
    const char* connection = std::getenv("LRS_TEST_POSTGRES_URL");
    if (!connection || *connection == '\0')
        return;
    auto config = test_config("unused-postgres.ragdb");
    config.retrieval.backend = "postgres";
    config.retrieval.postgres.connection_env = "LRS_TEST_POSTGRES_URL";
    config.retrieval.postgres.schema = "lrs_contract";
    config.retrieval.postgres.corpus =
        "service_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    config.retrieval.postgres.pool_size = 2;
    lrs::Service service(config);
    REQUIRE_NOTHROW(service.initialize());
    int status = 0;
    REQUIRE(nlohmann::json::parse(service.ingest(document("postgres service cedar"), status))
                .at("status") == "indexed");
    REQUIRE(status == 201);
    const auto found = nlohmann::json::parse(
        service.search(R"({"query":"cedar","mode":"hybrid","top_k":2})", status));
    REQUIRE(status == 200);
    REQUIRE(found.at("results").size() == 1);
    REQUIRE(found.at("results").front().at("document_id") == "demo/getting-started");
}
#endif

SCENARIO("the host and bridge preserve their language boundary", "[behavior][req:LRS-BUILD-001]") {
    REQUIRE(__cplusplus >= 201703L);
}

SCENARIO("private model listeners use loopback", "[unit][req:LRS-BUILD-002]") {
    auto config = test_config("unused.ragdb");
    config.embedding.host = "0.0.0.0";
    REQUIRE_THROWS(lrs::validate_config(config));
}

SCENARIO("public binds require an authenticated API key", "[unit][req:LRS-SEC-002]") {
    auto config = test_config("unused.ragdb");
    config.listen.host = "0.0.0.0";
    REQUIRE_THROWS(lrs::validate_config(config));
    config.api_key = "correct-horse-battery-staple";
    REQUIRE_NOTHROW(lrs::validate_config(config));
    REQUIRE(lrs::authenticate_api_key(config, "correct-horse-battery-staple"));
    REQUIRE_FALSE(lrs::authenticate_api_key(config, "correct-horse-battery-stapler"));
    REQUIRE_FALSE(lrs::authenticate_api_key(config, ""));
    REQUIRE(lrs::authorize_api_request(config, 1, "correct-horse-battery-staple"));
    REQUIRE_FALSE(lrs::authorize_api_request(config, 0, ""));
    REQUIRE_FALSE(lrs::authorize_api_request(config, 2, "correct-horse-battery-staple"));

    config.api_key = "too-short";
    REQUIRE_THROWS(lrs::validate_config(config));
}

SCENARIO("document ingestion persists and reopens", "[component][req:LRS-ING-001]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-ingest-persistence";
    std::filesystem::remove_all(root);
    const auto database = root / "knowledge.ragdb";
    {
        lrs::Service service(test_config(database));
        service.initialize();
        int status = 0;
        REQUIRE(nlohmann::json::parse(service.ingest(document(), status)).at("status") ==
                "indexed");
        REQUIRE(status == 201);
    }
    {
        lrs::Service service(test_config(database));
        service.initialize();
        int status = 0;
        const auto result = nlohmann::json::parse(
            service.search(R"({"query":"persisted database","mode":"hybrid","top_k":8})", status));
        REQUIRE(status == 200);
        REQUIRE_FALSE(result.at("results").empty());
    }
    std::filesystem::remove_all(root);
}

SCENARIO("identical ingestion is unchanged and replacement is atomic",
         "[component][req:LRS-ING-002]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-ingest-upsert";
    std::filesystem::remove_all(root);
    lrs::Service service(test_config(root / "knowledge.ragdb"));
    service.initialize();
    int status = 0;
    service.ingest(document(), status);
    REQUIRE(status == 201);
    REQUIRE(nlohmann::json::parse(service.ingest(document(), status)).at("status") == "unchanged");
    REQUIRE(status == 200);
    service.ingest(document("Replacement text has a narwhal marker."), status);
    REQUIRE(status == 201);
    const auto result = nlohmann::json::parse(
        service.search(R"({"query":"narwhal marker","mode":"lexical","top_k":8})", status));
    REQUIRE(result.at("results").size() == 1);
    std::filesystem::remove_all(root);
}

SCENARIO("asynchronous ingestion exposes content-free durable job status",
         "[component][req:LRS-ING-001]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-async-ingest";
    std::filesystem::remove_all(root);
    lrs::Service service(test_config(root / "knowledge.ragdb"));
    service.initialize();
    int status = 0;
    const auto accepted = nlohmann::json::parse(service.ingest(document(), status, true));
    REQUIRE(status == 202);
    REQUIRE(accepted.at("object") == "rag.index_job");
    REQUIRE(accepted.at("status") == "queued");
    const std::string id = accepted.at("id");

    nlohmann::json job;
    for (int attempt = 0; attempt < 200; ++attempt) {
        job = nlohmann::json::parse(service.get_job(id, status));
        REQUIRE(status == 200);
        if (job.at("status") == "ready")
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(job.at("status") == "ready");
    REQUIRE(job.at("document_id") == "demo/getting-started");
    REQUIRE_FALSE(job.contains("content"));
    REQUIRE_FALSE(job.contains("prepared"));
    const auto missing = nlohmann::json::parse(service.get_job("job_missing", status));
    REQUIRE(status == 404);
    REQUIRE(missing.at("error").at("code") == "job_not_found");
    std::filesystem::remove_all(root);
}

SCENARIO("document deletion is persistent and idempotent", "[component][req:LRS-ING-003]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-delete-document";
    std::filesystem::remove_all(root);
    const auto database = root / "knowledge.ragdb";
    {
        lrs::Service service(test_config(database));
        service.initialize();
        int status = 0;
        service.ingest(document("A semantic narwhal marker."), status);
        const auto removed =
            nlohmann::json::parse(service.delete_document("demo/getting-started", status));
        REQUIRE(status == 200);
        REQUIRE(removed.at("deleted") == true);
        for (const std::string mode : {"lexical", "dense", "hybrid"}) {
            const auto result = nlohmann::json::parse(service.search(
                nlohmann::json{{"query", "narwhal"}, {"mode", mode}, {"top_k", 8}}.dump(), status));
            REQUIRE(status == 200);
            REQUIRE(result.at("results").empty());
        }
        REQUIRE(nlohmann::json::parse(service.delete_document("demo/getting-started", status))
                    .at("deleted") == false);
    }
    {
        lrs::Service reopened(test_config(database));
        reopened.initialize();
        int status = 0;
        REQUIRE(nlohmann::json::parse(
                    reopened.search(R"({"query":"narwhal","mode":"hybrid","top_k":8})", status))
                    .at("results")
                    .empty());
    }
    std::filesystem::remove_all(root);
}

SCENARIO("failed embedding leaves the active index unchanged", "[fault][req:LRS-ING-004]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-failed-embedding";
    std::filesystem::remove_all(root);
    const std::string database = (root / "knowledge.ragdb").string();
    lrs_index_options local{database.c_str(), "127.0.0.1", 1, 32, 1, "default"};
    lrs_index* active = nullptr;
    char* error = nullptr;
    REQUIRE(lrs_index_open(&local, &active, &error) == 0);
    lrs_index* indexed = nullptr;
    int unchanged = 0;
    REQUIRE(lrs_index_stage_upsert(active, &local, "stable/doc", "Stable", "narwhal survives",
                                   &indexed, &unchanged, &error) == 0);
    lrs_index_destroy(active);
    active = indexed;

    lrs_index_options unavailable{database.c_str(), "127.0.0.1", 1, 32, 0, "default"};
    lrs_index* failed = nullptr;
    REQUIRE(lrs_index_stage_upsert(active, &unavailable, "stable/doc", "Stable", "replacement",
                                   &failed, &unchanged, &error) != 0);
    lrs_string_destroy(error);
    char* json = nullptr;
    error = nullptr;
    REQUIRE(lrs_index_search_json(active, "narwhal", "lexical", 8, &json, &error) == 0);
    REQUIRE(nlohmann::json::parse(json).at("results").size() == 1);
    lrs_string_destroy(json);
    lrs_index_destroy(active);
    std::filesystem::remove_all(root);
}

SCENARIO("metadata bridge arrays are validated before use", "[unit][req:LRS-SEARCH-003]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-metadata-boundary";
    std::filesystem::remove_all(root);
    const std::string database = (root / "knowledge.ragdb").string();
    lrs_index_options options{database.c_str(), "127.0.0.1", 1, 32, 1, "default"};
    lrs_index* active = nullptr;
    char* error = nullptr;
    REQUIRE(lrs_index_open(&options, &active, &error) == 0);

    lrs_index* candidate = active;
    int unchanged = 7;
    REQUIRE(lrs_index_stage_upsert_with_metadata(active, &options, "doc", "Title", "body", nullptr,
                                                 nullptr, 1, &candidate, &unchanged, &error) != 0);
    REQUIRE(candidate == nullptr);
    REQUIRE(unchanged == 0);
    lrs_string_destroy(error);
    error = nullptr;

    char marker = '\0';
    char* output = &marker;
    REQUIRE(lrs_index_search_filtered_json(active, "body", "lexical", 8, nullptr, nullptr, 1,
                                           &output, &error) != 0);
    REQUIRE(output == nullptr);
    lrs_string_destroy(error);
    lrs_index_destroy(active);
    std::filesystem::remove_all(root);
}

SCENARIO("search supports lexical dense and hybrid retrieval", "[component][req:LRS-SEARCH-001]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-search-modes";
    std::filesystem::remove_all(root);
    lrs::Service service(test_config(root / "knowledge.ragdb"));
    service.initialize();
    int status = 0;
    service.ingest(document(), status);
    for (const std::string mode : {"lexical", "dense", "hybrid"}) {
        const auto result = nlohmann::json::parse(service.search(
            nlohmann::json{{"query", "index persistence"}, {"mode", mode}, {"top_k", 8}}.dump(),
            status));
        REQUIRE(status == 200);
        REQUIRE_FALSE(result.at("results").empty());
    }
    std::filesystem::remove_all(root);
}

SCENARIO("metadata filters are exact AND predicates in every retrieval mode",
         "[component][req:LRS-SEARCH-003]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-search-metadata";
    std::filesystem::remove_all(root);
    const auto database = root / "knowledge.ragdb";
    {
        lrs::Service service(test_config(database));
        service.initialize();
        int status = 0;
        const auto notion = nlohmann::json{{"id", "notion/setup"},
                                           {"title", "Terminal Setup"},
                                           {"content", "shared permission marker terminal setup"},
                                           {"metadata",
                                            {{"audience", "all"},
                                             {"source", "notion"},
                                             {"url", "https://notion.example/setup"},
                                             {"lastSyncedAt", "2026-08-10T09:00:00Z"}}}}
                                .dump();
        const auto private_doc =
            nlohmann::json{{"id", "private/setup"},
                           {"title", "Private Setup"},
                           {"content", "shared permission marker terminal setup"},
                           {"metadata", {{"audience", "admins"}, {"source", "notion"}}}}
                .dump();
        service.ingest(notion, status);
        REQUIRE(status == 201);
        service.ingest(private_doc, status);
        REQUIRE(status == 201);

        for (const std::string mode : {"lexical", "dense", "hybrid"}) {
            const auto result = nlohmann::json::parse(service.search(
                nlohmann::json{{"query", "permission marker"},
                               {"mode", mode},
                               {"top_k", 8},
                               {"filter", {{"audience", "all"}, {"source", "notion"}}}}
                    .dump(),
                status));
            REQUIRE(status == 200);
            REQUIRE_FALSE(result.at("results").empty());
            for (const auto& hit : result.at("results")) {
                REQUIRE(hit.at("document_id") == "notion/setup");
                REQUIRE(hit.at("title") == "Terminal Setup");
                REQUIRE(hit.at("metadata").at("audience") == "all");
                REQUIRE(hit.at("metadata").at("source") == "notion");
                REQUIRE(hit.at("score").is_number());
            }
        }

        const auto excluded = nlohmann::json::parse(
            service.search(nlohmann::json{{"query", "permission marker"},
                                          {"mode", "hybrid"},
                                          {"filter", {{"audience", "all"}, {"source", "github"}}}}
                               .dump(),
                           status));
        REQUIRE(status == 200);
        REQUIRE(excluded.at("results").empty());

        service.search(nlohmann::json{{"query", "permission marker"},
                                      {"filter", {{"audience", nlohmann::json::array({"all"})}}}}
                           .dump(),
                       status);
        REQUIRE(status == 400);
    }
    {
        lrs::Service reopened(test_config(database));
        reopened.initialize();
        int status = 0;
        const auto result = nlohmann::json::parse(reopened.search(
            nlohmann::json{
                {"query", "terminal setup"}, {"mode", "hybrid"}, {"filter", {{"audience", "all"}}}}
                .dump(),
            status));
        REQUIRE(status == 200);
        REQUIRE_FALSE(result.at("results").empty());
        REQUIRE(result.at("results").front().at("metadata").at("source") == "notion");
    }
    std::filesystem::remove_all(root);
}

SCENARIO("search results expose stable citation fields", "[unit][req:LRS-SEARCH-002]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-search-fields";
    std::filesystem::remove_all(root);
    lrs::Service service(test_config(root / "knowledge.ragdb"));
    service.initialize();
    int status = 0;
    service.ingest(document(), status);
    const auto first =
        nlohmann::json::parse(
            service.search(R"({"query":"persisted","mode":"lexical","top_k":8})", status))
            .at("results")
            .at(0);
    const auto second =
        nlohmann::json::parse(
            service.search(R"({"query":"persisted","mode":"lexical","top_k":8})", status))
            .at("results")
            .at(0);
    REQUIRE(first.at("chunk_id") == second.at("chunk_id"));
    REQUIRE(first.contains("document_id"));
    REQUIRE(first.contains("start_line"));
    REQUIRE(first.contains("end_line"));
    REQUIRE(first.contains("score"));
    REQUIRE(first.contains("rank"));
    std::filesystem::remove_all(root);
}

SCENARIO("SSE event formatting is deterministic", "[unit][req:LRS-SSE-001]") {
    const std::string event = "event: rag.started\ndata: {\"status\":\"started\"}\n\n";
    REQUIRE(event.find("event: rag.started") == 0);
    REQUIRE(event.size() >= 2);
    REQUIRE(event.compare(event.size() - 2, 2, "\n\n") == 0);
}

SCENARIO("query emits sources before generation", "[behavior][req:LRS-SSE-002]") {
    std::unique_ptr<httplib::Server> model;
    int port = -1;
    for (int attempt = 0; attempt < 5 && port <= 0; ++attempt) {
        auto candidate = std::make_unique<httplib::Server>();
        candidate->Post("/tokenize", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"tokens":[1,2,3]})", "application/json");
        });
        candidate->Post(
            "/v1/chat/completions", [](const httplib::Request&, httplib::Response& response) {
                response.set_content("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\","
                                     "\"content\":null}}]}\n\n"
                                     "data: {\"choices\":[{\"delta\":{\"content\":\"answer "
                                     "[1]\"}}]}\n\ndata: [DONE]\n\n",
                                     "text/event-stream");
            });
        port = candidate->bind_to_any_port("127.0.0.1");
        if (port > 0)
            model = std::move(candidate);
    }
    REQUIRE(port > 0);
    REQUIRE(model);
    std::thread server([&] { model->listen_after_bind(); });
    model->wait_until_ready();

    const auto root = std::filesystem::temp_directory_path() / "lrs-query-sse";
    std::filesystem::remove_all(root);
    auto config = test_config(root / "knowledge.ragdb");
    config.generation.port = static_cast<std::uint16_t>(port);
    config.generation_api_model = "qwen-generation";
    lrs::Service service(config);
    service.initialize();
    int status = 0;
    service.ingest(document(), status);
    std::string stream;
    std::string error;
    const bool completed = service.query(
        R"({"messages":[{"role":"user","content":"How is it persisted?"}],"stream":true})",
        [&](const std::string& event) {
            stream += event;
            return true;
        },
        error);
    REQUIRE(completed);
    const auto started = stream.find("event: rag.started");
    const auto sources = stream.find("event: rag.retrieval.completed");
    const auto delta = stream.find("event: rag.generation.delta");
    const auto done = stream.find("event: rag.completed");
    REQUIRE(started < sources);
    REQUIRE(sources < delta);
    REQUIRE(delta < done);

    const std::string openai_request =
        R"({"model":"qwen-generation","messages":[{"role":"user","content":"How is it persisted?"}],"max_tokens":64,"stream":false})";
    const auto completion = nlohmann::json::parse(service.chat_completions(openai_request, status));
    REQUIRE(status == 200);
    REQUIRE(completion.at("object") == "chat.completion");
    REQUIRE(completion.at("model") == "qwen-generation");
    REQUIRE(completion.at("choices").at(0).at("message").at("role") == "assistant");
    REQUIRE(completion.at("choices").at(0).at("message").at("content") == "answer [1]");
    REQUIRE_FALSE(completion.at("rag_sources").empty());

    std::string openai_stream;
    REQUIRE(service.chat_completions_stream(
        R"({"model":"qwen-generation","messages":[{"role":"user","content":"How is it persisted?"}],"max_tokens":64,"stream":true})",
        [&](const std::string& event) {
            openai_stream += event;
            return true;
        },
        error));
    REQUIRE(openai_stream.find(R"("object":"chat.completion.chunk")") != std::string::npos);
    REQUIRE(openai_stream.find(R"("content":"answer [1]")") != std::string::npos);
    REQUIRE(openai_stream.find("data: [DONE]\n\n") != std::string::npos);

    model->stop();
    server.join();
    std::filesystem::remove_all(root);
}

SCENARIO("query rejects generation beyond the context budget", "[unit][req:LRS-CTX-001]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-context-budget";
    std::filesystem::remove_all(root);
    auto config = test_config(root / "knowledge.ragdb");
    config.context_tokens = 64;
    lrs::Service service(config);
    service.initialize();
    int status = 0;
    service.ingest(document(), status);
    std::string stream;
    std::string error;
    REQUIRE_FALSE(service.query(
        R"({"messages":[{"role":"user","content":"question"}],"generation":{"max_tokens":64}})",
        [&](const std::string& event) {
            stream += event;
            return true;
        },
        error));
    REQUIRE(stream.find("event: rag.generation.delta") == std::string::npos);
    REQUIRE(stream.find("event: rag.error") != std::string::npos);
    std::filesystem::remove_all(root);
}

SCENARIO("document ids reject control characters", "[unit][req:LRS-SEC-001]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-invalid-id";
    std::filesystem::remove_all(root);
    lrs::Service service(test_config(root / "knowledge.ragdb"));
    service.initialize();
    int status = 0;
    service.ingest(nlohmann::json{{"id", "bad\nid"}, {"content", "text"}}.dump(), status);
    REQUIRE(status == 400);
    std::filesystem::remove_all(root);
}

SCENARIO("health does not expose request content", "[unit][req:LRS-OPS-001]") {
    lrs::Service service(test_config("unused-health.ragdb"));
    REQUIRE(service.health_json() == R"({"ready":false,"status":"starting"})");
}

SCENARIO("mobile clients persist and search precomputed vectors",
         "[component][req:LRS-MOBILE-001][req:LRS-MOBILE-002]") {
    const auto root = std::filesystem::temp_directory_path() / "lrs-mobile-vectors";
    std::filesystem::remove_all(root);
    const std::string database = (root / "knowledge.ragdb").string();

    lrs_mobile_index* index = nullptr;
    char* error = nullptr;
    REQUIRE(lrs_mobile_open(database.c_str(), &index, &error) == 0);

    char* prepared = nullptr;
    REQUIRE(lrs_mobile_prepare_document_json(index, "mobile/alpha", "alpha narwhal fact", &prepared,
                                             &error) == 0);
    const auto chunk_count = nlohmann::json::parse(prepared).at("chunks").size();
    lrs_mobile_string_destroy(prepared);
    REQUIRE(chunk_count == 1);

    const float alpha[] = {1.0F, 0.0F};
    int unchanged = 0;
    REQUIRE(lrs_mobile_upsert_vectors(index, "mobile/alpha", "Alpha", "alpha narwhal fact", alpha,
                                      chunk_count, 2, &unchanged, &error) == 0);
    REQUIRE(unchanged == 0);

    const float beta[] = {0.0F, 1.0F};
    REQUIRE(lrs_mobile_upsert_vectors(index, "mobile/beta", "Beta", "beta orchard fact", beta, 1, 2,
                                      &unchanged, &error) == 0);
    lrs_mobile_destroy(index);

    const auto portable = rag::store::Container::read_file(database);
    REQUIRE(portable);
    REQUIRE(portable->has(rag::store::Tag::runtime));

    REQUIRE(lrs_mobile_open(database.c_str(), &index, &error) == 0);
    char* result = nullptr;
    REQUIRE(lrs_mobile_search_json(index, "semantic alpha", alpha, 2, "dense", 2, &result,
                                   &error) == 0);
    const auto hits = nlohmann::json::parse(result).at("results");
    lrs_mobile_string_destroy(result);
    REQUIRE(hits.size() == 2);
    REQUIRE(hits.at(0).at("document_id") == "mobile/alpha");

    result = nullptr;
    REQUIRE(lrs_mobile_search_json(index, "narwhal", alpha, 2, "hybrid", 2, &result, &error) == 0);
    REQUIRE(nlohmann::json::parse(result).at("results").at(0).at("document_id") == "mobile/alpha");
    lrs_mobile_string_destroy(result);
    REQUIRE(lrs_mobile_upsert_vectors(index, "mobile/alpha", "Alpha v2", "alpha sequoia fact",
                                      alpha, 1, 2, &unchanged, &error) == 0);
    REQUIRE(unchanged == 0);
    result = nullptr;
    REQUIRE(lrs_mobile_search_json(index, "narwhal", alpha, 2, "lexical", 2, &result, &error) == 0);
    REQUIRE(nlohmann::json::parse(result).at("results").empty());
    lrs_mobile_string_destroy(result);
    lrs_mobile_destroy(index);

    REQUIRE(lrs_mobile_open(database.c_str(), &index, &error) == 0);
    result = nullptr;
    REQUIRE(lrs_mobile_search_json(index, "sequoia", alpha, 2, "lexical", 2, &result, &error) == 0);
    REQUIRE(nlohmann::json::parse(result).at("results").size() == 1);
    lrs_mobile_string_destroy(result);
    lrs_mobile_destroy(index);
    std::filesystem::remove_all(root);
}
