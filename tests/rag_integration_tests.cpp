#include "lrs/bridge.h"
#include "lrs/mobile.h"

#include <nlohmann/json.hpp>
#include <rag/engine.hpp>
#include <rag/text/chunker.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    int failures = 0;
    const auto root = std::filesystem::temp_directory_path() /
                      ("rag-integration-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const std::string database = (root / "shared.ragdb").string();
    lrs_index_options options{};
    options.database_path = database.c_str();
    options.embedding_host = "127.0.0.1";
    options.embedding_port = 8080;
    options.embedding_dimension = 8;
    options.embedding_model = "default";
    options.deterministic_embeddings = 1;

    char* error = nullptr;
    lrs_index* active = nullptr;
    if (lrs_index_open(&options, &active, &error) != 0) {
        std::cerr << (error ? error : "desktop open failed") << '\n';
        ++failures;
    }
    lrs_index* candidate = nullptr;
    int unchanged = -1;
    const std::string content = "# Shared\nalpha beta gamma\nsecond line";
    if (active != nullptr &&
        lrs_index_stage_upsert(active, &options, "shared/doc", "Shared", content.c_str(),
                               &candidate, &unchanged, &error) != 0) {
        std::cerr << (error ? error : "desktop upsert failed") << '\n';
        ++failures;
    }
    lrs_string_destroy(error);
    error = nullptr;
    lrs_index_destroy(active);
    active = candidate;

    std::ifstream manifest_input(database + ".manifest.json");
    if (!manifest_input) {
        std::cerr << "desktop manifest was not published\n";
        ++failures;
    } else {
        const auto manifest = nlohmann::json::parse(manifest_input, nullptr, false);
        if (manifest.is_discarded() || manifest.value("retrieval_core", "") != "v0.2.0") {
            std::cerr << "desktop manifest does not identify the retrieval core\n";
            ++failures;
        }
    }

    auto persisted = rag::Engine::open(database);
    if (!persisted) {
        std::cerr << "desktop persistence did not reopen\n";
        ++failures;
    }

    lrs_mobile_index* mobile = nullptr;
    if (lrs_mobile_open(database.c_str(), &mobile, &error) != 0) {
        std::cerr << (error ? error : "mobile open failed") << '\n';
        ++failures;
    }
    char* prepared = nullptr;
    if (mobile != nullptr && lrs_mobile_prepare_document_json(mobile, "shared/doc", content.c_str(),
                                                              &prepared, &error) != 0) {
        std::cerr << (error ? error : "mobile preparation failed") << '\n';
        ++failures;
    }
    if (persisted && prepared != nullptr) {
        const auto body = nlohmann::json::parse(prepared);
        if (body.at("chunks").size() != persisted->corpus().chunks().size())
            ++failures;
        rag::text::ChunkOptions mobile_policy;
        mobile_policy.max_lines = 40;
        mobile_policy.max_chars = 384;
        mobile_policy.overlap_lines = 4;
        mobile_policy.heading_context = false;
        mobile_policy.policy.model_identity = "default";
        mobile_policy.policy.tokenizer_identity = "conservative-utf8-bytes-v1";
        mobile_policy.policy.target_tokens = 320;
        mobile_policy.policy.max_tokens = 384;
        mobile_policy.policy.overlap_tokens = 32;
        if (rag::text::chunking_fingerprint(mobile_policy) !=
            rag::text::chunking_fingerprint(persisted->corpus().config().chunk))
            ++failures;
    }

    lrs_mobile_string_destroy(prepared);
    lrs_mobile_string_destroy(error);
    lrs_mobile_destroy(mobile);
    lrs_index_destroy(active);
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    if (failures != 0)
        std::cerr << failures << " desktop/mobile integration checks failed\n";
    return failures == 0 ? 0 : 1;
}
