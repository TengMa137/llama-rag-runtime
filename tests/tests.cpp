#include "lrs/config.hpp"
#include "lrs/service.hpp"
#include <rag/text/chunker.hpp>
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

std::string document(std::string content = "# Local RAG\nThe index is persisted in a rag-cpp database.") {
  return nlohmann::json{{"id", "demo/getting-started"}, {"content_type", "text/markdown"},
                        {"title", "Getting started"}, {"content", std::move(content)}}.dump();
}
}

SCENARIO("the host and bridge preserve their language boundary", "[behavior][req:LRS-BUILD-001]") {
  REQUIRE(__cplusplus >= 201703L);
}

SCENARIO("long paper lines are split within the embedding budget", "[unit][req:LRS-ING-001]") {
  rag::text::ChunkOptions options;
  options.max_chars = 384;
  options.heading_context = false;
  const std::string long_line = std::string(900, 'a') + " λλλ";
  const auto chunks = rag::text::chunk_document(rag::DocId{0}, long_line, options);
  REQUIRE(chunks.size() == 3);
  for (const auto& chunk : chunks) {
    REQUIRE(chunk.text.size() <= 384);
    REQUIRE(chunk.start_line == 0);
    REQUIRE(chunk.end_line == 0);
  }
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
    REQUIRE(nlohmann::json::parse(service.ingest(document(), status)).at("status") == "indexed");
    REQUIRE(status == 201);
  }
  {
    lrs::Service service(test_config(database));
    service.initialize();
    int status = 0;
    const auto result = nlohmann::json::parse(service.search(
        R"({"query":"persisted database","mode":"hybrid","top_k":8})", status));
    REQUIRE(status == 200);
    REQUIRE_FALSE(result.at("results").empty());
  }
  std::filesystem::remove_all(root);
}

SCENARIO("identical ingestion is unchanged and replacement is atomic", "[component][req:LRS-ING-002]") {
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
  const auto result = nlohmann::json::parse(service.search(
      R"({"query":"narwhal marker","mode":"lexical","top_k":8})", status));
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
        nlohmann::json{{"query", "index persistence"}, {"mode", mode}, {"top_k", 8}}.dump(), status));
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
  const auto first = nlohmann::json::parse(service.search(
      R"({"query":"persisted","mode":"lexical","top_k":8})", status)).at("results").at(0);
  const auto second = nlohmann::json::parse(service.search(
      R"({"query":"persisted","mode":"lexical","top_k":8})", status)).at("results").at(0);
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
    response.set_content("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n"
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
  lrs::Service service(config);
  service.initialize();
  int status = 0;
  service.ingest(document(), status);
  std::string stream;
  std::string error;
  const bool completed = service.query(
      R"({"messages":[{"role":"user","content":"How is it persisted?"}],"stream":true})",
      [&](const std::string& event) { stream += event; return true; }, error);
  model.stop();
  server.join();
  REQUIRE(completed);
  const auto started = stream.find("event: rag.started");
  const auto sources = stream.find("event: rag.retrieval.completed");
  const auto delta = stream.find("event: rag.generation.delta");
  const auto done = stream.find("event: rag.completed");
  REQUIRE(started < sources);
  REQUIRE(sources < delta);
  REQUIRE(delta < done);
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
      [&](const std::string& event) { stream += event; return true; }, error));
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
