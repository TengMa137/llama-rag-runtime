#pragma once

#include "lrs/bridge.h"
#include "lrs/config.hpp"
#include "lrs/model_client.hpp"
#include <memory>
#include <mutex>
#include <string>

namespace lrs {
class Service {
  public:
    explicit Service(Config config);
    ~Service();
    void initialize();
    bool ready() const noexcept;
    std::string health_json() const;
    std::string ingest(const std::string& body, int& status);
    std::string delete_document(const std::string& id, int& status);
    std::string search(const std::string& body, int& status) const;
    std::string models_json() const;
    std::string chat_completions(const std::string& body, int& status) const;
    bool chat_completions_stream(const std::string& body,
                                 const std::function<bool(const std::string&)>& send,
                                 std::string& error) const;
    bool query(const std::string& body, const std::function<bool(const std::string&)>& send,
               std::string& error) const;

  private:
    lrs_index_options options() const;
    bool grounded_generate(const std::string& body,
                           const std::function<bool(const std::string&)>& delta,
                           std::string& sources, std::string& model, std::string& error) const;
    Config config_;
    ModelClient embedding_;
    ModelClient generation_;
    mutable std::mutex mutation_;
    std::shared_ptr<lrs_index> active_;
    bool ready_ = false;
};
} // namespace lrs
