#pragma once

#include "lrs/config.hpp"
#include <functional>
#include <string>
#include <vector>

namespace lrs {
class ModelClient {
public:
  explicit ModelClient(Endpoint endpoint, std::string model = "default");
  bool healthy() const;
  std::size_t tokenize(const std::string& text) const;
  bool generate(const std::string& prompt, std::size_t max_tokens, double temperature,
                const std::function<bool(const std::string&)>& delta,
                const std::function<bool()>& connected,
                std::string& error) const;
private:
  Endpoint endpoint_;
  std::string model_;
};
}
