#include "rag/dense/backends.hpp"

#include "rag/dense/simd.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <nlohmann/json.hpp>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#endif

namespace rag::dense {
namespace {

bool is_loopback_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        const auto* in = reinterpret_cast<const sockaddr_in*>(address);
        return (ntohl(in->sin_addr.s_addr) >> 24U) == 127U;
    }
    if (address->sa_family == AF_INET6) {
        const auto* in6 = reinterpret_cast<const sockaddr_in6*>(address);
        static constexpr std::uint8_t loopback[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 1};
        return std::memcmp(in6->sin6_addr.s6_addr, loopback, sizeof(loopback)) == 0;
    }
    return false;
}

bool resolves_only_to_loopback(const std::string& host) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &addresses) != 0 || addresses == nullptr)
        return false;
    bool found = false;
    bool valid = true;
    for (const addrinfo* current = addresses; current != nullptr; current = current->ai_next) {
        found = true;
        if (!is_loopback_address(current->ai_addr)) {
            valid = false;
            break;
        }
    }
    ::freeaddrinfo(addresses);
    return found && valid;
}

bool valid_path(std::string_view path) {
    if (path.empty() || path.front() != '/' || path.find("..") != std::string_view::npos ||
        path.find('?') != std::string_view::npos || path.find('#') != std::string_view::npos ||
        path.find('\\') != std::string_view::npos)
        return false;
    for (const unsigned char value : path)
        if (value <= 0x20U || value == 0x7fU)
            return false;
    return true;
}

Result<Vector> parse_vector(const nlohmann::json& value, std::size_t expected_dimension) {
    if (!value.is_array() || value.size() != expected_dimension)
        return fail<Vector>(Errc::dimension_mismatch, "embedding dimension mismatch");
    Vector vector;
    vector.reserve(expected_dimension);
    double squared_norm = 0.0;
    for (const auto& component : value) {
        if (!component.is_number())
            return fail<Vector>(Errc::parse_error, "embedding component is not numeric");
        const float number = component.get<float>();
        if (!std::isfinite(number))
            return fail<Vector>(Errc::parse_error, "embedding contains a non-finite value");
        squared_norm += static_cast<double>(number) * number;
        vector.push_back(number);
    }
    if (!std::isfinite(squared_norm) || squared_norm <= std::numeric_limits<double>::min())
        return fail<Vector>(Errc::parse_error, "embedding must have a finite non-zero norm");
    normalize(vector);
    return vector;
}

std::uint64_t fnv1a(std::string_view text, std::uint64_t seed = 1469598103934665603ULL) {
    std::uint64_t hash = seed;
    for (const unsigned char value : text) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

Result<LocalHttpEmbedder> LocalHttpEmbedder::create(LocalHttpEmbedderConfig cfg,
                                                    std::shared_ptr<HttpTransport> transport) {
    if (!transport)
        return fail<LocalHttpEmbedder>(Errc::invalid_argument, "HTTP transport is required");
    if (!resolves_only_to_loopback(cfg.host))
        return fail<LocalHttpEmbedder>(Errc::invalid_argument,
                                       "embedding host must resolve only to loopback");
    if (cfg.port == 0 || cfg.dimension == 0 || cfg.concurrency == 0 ||
        cfg.max_response_bytes == 0 || cfg.connect_timeout.count() <= 0 ||
        cfg.read_timeout.count() <= 0 || !valid_path(cfg.path))
        return fail<LocalHttpEmbedder>(Errc::invalid_argument,
                                       "invalid local HTTP embedder configuration");
    return LocalHttpEmbedder(std::move(cfg), std::move(transport));
}

Result<std::vector<Vector>> LocalHttpEmbedder::embed(std::span<const std::string> texts) const {
    if (texts.empty())
        return std::vector<Vector>{};
    nlohmann::json request{{"model", cfg_.model}, {"input", nlohmann::json::array()}};
    for (const auto& text : texts)
        request["input"].push_back(text);
    const std::string body = request.dump();
    HttpRequest http_request;
    http_request.host = cfg_.host;
    http_request.port = cfg_.port;
    http_request.path = cfg_.path;
    http_request.body = body;
    http_request.connect_timeout = cfg_.connect_timeout;
    http_request.read_timeout = cfg_.read_timeout;
    http_request.max_response_bytes = cfg_.max_response_bytes;
    auto response = transport_->post(http_request);
    if (!response)
        return unexpected(response.error());
    if (response->body.size() > cfg_.max_response_bytes)
        return fail<std::vector<Vector>>(Errc::transport_error,
                                         "embedding response exceeds configured limit");
    if (response->status != 200)
        return fail<std::vector<Vector>>(
            Errc::transport_error, "embedding HTTP status " + std::to_string(response->status));
    auto parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("data") ||
        !parsed["data"].is_array())
        return fail<std::vector<Vector>>(Errc::parse_error, "malformed embedding response");
    const auto& data = parsed["data"];
    if (data.size() != texts.size())
        return fail<std::vector<Vector>>(Errc::parse_error, "embedding response count mismatch");
    std::vector<Vector> output;
    output.reserve(texts.size());
    for (std::size_t index = 0; index < data.size(); ++index) {
        const auto& item = data[index];
        if (!item.is_object() || !item.contains("index") || !item["index"].is_number_unsigned() ||
            item["index"].get<std::size_t>() != index || !item.contains("embedding"))
            return fail<std::vector<Vector>>(Errc::parse_error,
                                             "embedding response order is invalid");
        auto vector = parse_vector(item["embedding"], cfg_.dimension);
        if (!vector)
            return unexpected(vector.error());
        output.push_back(std::move(*vector));
    }
    return output;
}

Result<std::vector<Vector>> HashEmbedder::embed(std::span<const std::string> texts) const {
    if (dimension_ == 0)
        return fail<std::vector<Vector>>(Errc::invalid_argument,
                                         "hash embedding dimension must be non-zero");
    std::vector<Vector> output;
    output.reserve(texts.size());
    for (const auto& text : texts) {
        Vector vector(dimension_, 0.0F);
        std::string previous;
        std::string token;
        auto emit = [&](const std::string& current) {
            if (current.empty())
                return;
            const std::uint64_t hash = fnv1a(current);
            vector[hash % dimension_] += (hash & (1ULL << 63U)) ? -1.0F : 1.0F;
            if (!previous.empty()) {
                const std::uint64_t bigram = fnv1a(previous + "_" + current);
                vector[bigram % dimension_] += (bigram & (1ULL << 63U)) ? -0.5F : 0.5F;
            }
            previous = current;
        };
        for (const char character : text) {
            const unsigned char value = static_cast<unsigned char>(character);
            if (std::isalnum(value))
                token.push_back(static_cast<char>(std::tolower(value)));
            else {
                emit(token);
                token.clear();
            }
        }
        emit(token);
        double norm = 0.0;
        for (const float value : vector)
            norm += static_cast<double>(value) * value;
        if (norm == 0.0)
            vector[fnv1a(text) % dimension_] = 1.0F;
        normalize(vector);
        output.push_back(std::move(vector));
    }
    return output;
}

} // namespace rag::dense
