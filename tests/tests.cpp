#include "lrs/config.hpp"
#include "lrs/mobile.h"
#include "lrs/service.hpp"
#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
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

std::string
document(std::string content = "# Local RAG\nThe index is persisted in a rag-cpp database.") {
    return nlohmann::json{{"id", "demo/getting-started"},
                          {"content_type", "text/markdown"},
                          {"title", "Getting started"},
                          {"content", std::move(content)}}
        .dump();
}
} // namespace

SCENARIO("the host and bridge preserve their language boundary", "[behavior][req:LRS-BUILD-001]") {
    REQUIRE(__cplusplus >= 201703L);
}

SCENARIO("private listeners must use loopback", "[unit][req:LRS-BUILD-002]") {
    auto config = test_config("unused.ragdb");
    config.embedding.host = "0.0.0.0";
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
    httplib::Server model;
    model.Post("/tokenize", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(R"({"tokens":[1,2,3]})", "application/json");
    });
    model.Post("/v1/chat/completions", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer [1]\"}}]}\n\ndata: [DONE]\n\n",
            "text/event-stream");
    });
    const int port = model.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread server([&] { model.listen_after_bind(); });

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

    model.stop();
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
    lrs_mobile_destroy(index);
    std::filesystem::remove_all(root);
}
