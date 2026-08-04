#include "lrs/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace lrs {
Config load_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open config: " + path);
  nlohmann::json j;
  input >> j;
  Config c;
  const auto& listen = j.value("listen", nlohmann::json::object());
  c.listen.host = listen.value("host", c.listen.host);
  c.listen.port = listen.value("port", c.listen.port);
  c.index_path = j.at("index").value("path", c.index_path);
  const auto& inference = j.at("inference");
  c.spawn = inference.value("spawn", c.spawn);
  c.llama_server = inference.value("llama_server", c.llama_server);
  const auto& embedding = inference.at("embedding");
  c.embedding.host = embedding.value("host", c.embedding.host);
  c.embedding.port = embedding.value("port", c.embedding.port);
  c.embedding_model = embedding.value("model", "");
  c.embedding_api_model = embedding.value("api_model", c.embedding_api_model);
  c.pooling = embedding.value("pooling", c.pooling);
  c.embedding_dimension = embedding.value("dimension", c.embedding_dimension);
  c.embedding_context_size = embedding.value("context_size", c.embedding_context_size);
  c.embedding_batch_size = embedding.value("batch_size", c.embedding_batch_size);
  c.deterministic_embeddings = embedding.value("deterministic", false);
  const auto& generation = inference.at("generation");
  c.generation.host = generation.value("host", c.generation.host);
  c.generation.port = generation.value("port", c.generation.port);
  c.generation_model = generation.value("model", "");
  c.generation_api_model = generation.value("api_model", c.generation_api_model);
  c.generation_context_size = generation.value("context_size", c.generation_context_size);
  c.context_tokens = j.value("rag", nlohmann::json::object()).value("context_tokens", c.context_tokens);
  validate_config(c);
  return c;
}

void validate_config(const Config& c) {
  const auto loopback = [](const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
  };
  if (!loopback(c.embedding.host) || !loopback(c.generation.host))
    throw std::runtime_error("private model listeners must bind to loopback");
  if (!loopback(c.listen.host)) throw std::runtime_error("v0.1 public listener must bind to loopback");
  if (c.embedding.port == c.generation.port || c.listen.port == c.embedding.port ||
      c.listen.port == c.generation.port) throw std::runtime_error("listener ports must be distinct");
  if (c.pooling != "mean") throw std::runtime_error("embedding pooling must be mean");
  if (c.embedding_dimension == 0 || c.embedding_context_size == 0 ||
      c.embedding_batch_size < c.embedding_context_size || c.context_tokens == 0 ||
      c.context_tokens >= c.generation_context_size)
    throw std::runtime_error("invalid model dimension or context budget");
  if (c.spawn && (c.embedding_model.empty() || c.generation_model.empty()))
    throw std::runtime_error("both GGUF model paths are required when process spawning is enabled");
}
}
