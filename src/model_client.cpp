#include "lrs/model_client.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace lrs {
ModelClient::ModelClient(Endpoint endpoint, std::string model)
    : endpoint_(std::move(endpoint)), model_(std::move(model)) {}

bool ModelClient::healthy() const {
    httplib::Client client(endpoint_.host, endpoint_.port);
    client.set_connection_timeout(1);
    const auto response = client.Get("/health");
    return response && response->status == 200;
}

std::size_t ModelClient::tokenize(const std::string& text) const {
    httplib::Client client(endpoint_.host, endpoint_.port);
    const auto response =
        client.Post("/tokenize", nlohmann::json{{"content", text}}.dump(), "application/json");
    if (!response || response->status != 200)
        return (text.size() + 3) / 4;
    try {
        return nlohmann::json::parse(response->body).at("tokens").size();
    } catch (...) {
        return (text.size() + 3) / 4;
    }
}

bool ModelClient::generate(const std::string& prompt, std::size_t max_tokens, double temperature,
                           const std::function<bool(const std::string&)>& delta,
                           const std::function<bool()>& connected, std::string& error) const {
    httplib::Client client(endpoint_.host, endpoint_.port);
    nlohmann::json request = {{"messages", {{{"role", "user"}, {"content", prompt}}}},
                              {"model", model_},
                              {"max_tokens", max_tokens},
                              {"temperature", temperature},
                              {"stream", true},
                              {"chat_template_kwargs", {{"enable_thinking", false}}}};
    std::string buffered;
    bool callback_ok = true;
    bool emitted = false;
    auto response = client.Post(
        "/v1/chat/completions", httplib::Headers{}, request.dump(), "application/json",
        [&](const char* data, std::size_t length) {
            buffered.append(data, length);
            std::size_t newline = 0;
            while ((newline = buffered.find('\n')) != std::string::npos) {
                std::string line = buffered.substr(0, newline);
                buffered.erase(0, newline + 1);
                if (line.rfind("data: ", 0) != 0 || line == "data: [DONE]")
                    continue;
                if (!connected()) {
                    error = "generation cancelled";
                    callback_ok = false;
                    return false;
                }
                try {
                    const auto event = nlohmann::json::parse(line.substr(6));
                    const auto& item = event.at("choices").at(0).at("delta");
                    const std::string content =
                        item.contains("content") && item.at("content").is_string()
                            ? item.at("content").get<std::string>()
                            : "";
                    if (!content.empty()) {
                        emitted = true;
                        if (!delta(content)) {
                            callback_ok = false;
                            return false;
                        }
                    }
                } catch (...) {
                    error = "malformed generation stream";
                    callback_ok = false;
                    return false;
                }
            }
            return true;
        },
        nullptr);
    if (!callback_ok) {
        if (error.empty())
            error = "generation cancelled";
        return false;
    }
    if (!response) {
        error = "generation server unavailable";
        return false;
    }
    if (response->status != 200) {
        error = "generation server returned " + std::to_string(response->status);
        return false;
    }
    if (!emitted) {
        error = "generation ended before producing answer content";
        return false;
    }
    return true;
}
} // namespace lrs
