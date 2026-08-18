#include "lrs/config.hpp"
#include "lrs/service.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
#ifndef _WIN32
class Children {
  public:
    ~Children() {
        for (const pid_t pid : pids_)
            if (pid > 0)
                kill(pid, SIGTERM);
        for (const pid_t pid : pids_)
            if (pid > 0)
                waitpid(pid, nullptr, 0);
    }
    void launch(const std::vector<std::string>& arguments) {
        const pid_t pid = fork();
        if (pid < 0)
            throw std::runtime_error("failed to fork llama-server");
        if (pid == 0) {
            std::vector<char*> argv;
            for (const auto& argument : arguments)
                argv.push_back(const_cast<char*>(argument.c_str()));
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            _exit(127);
        }
        pids_.push_back(pid);
    }

  private:
    std::vector<pid_t> pids_;
};
#endif

std::string require_config(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--config")
            return argv[i + 1];
    throw std::runtime_error("usage: llama-rag-server --config <path>");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const lrs::Config config = lrs::load_config(require_config(argc, argv));
#ifndef _WIN32
        Children children;
        if (config.spawn) {
            children.launch({config.llama_server, "--host", config.embedding.host, "--port",
                             std::to_string(config.embedding.port), "--model",
                             config.embedding_model, "--alias", config.embedding_api_model,
                             "--embeddings", "--pooling", config.pooling, "--ctx-size",
                             std::to_string(config.embedding_context_size), "--batch-size",
                             std::to_string(config.embedding_batch_size), "--ubatch-size",
                             std::to_string(config.embedding_batch_size)});
            children.launch({config.llama_server, "--host", config.generation.host, "--port",
                             std::to_string(config.generation.port), "--model",
                             config.generation_model, "--alias", config.generation_api_model,
                             "--ctx-size", std::to_string(config.generation_context_size)});
        }
#else
        if (config.spawn)
            throw std::runtime_error("child supervision is not implemented on Windows");
#endif
        lrs::Service service(config);
        std::exception_ptr last;
        for (int attempt = 0; attempt < 120; ++attempt) {
            try {
                service.initialize();
                last = nullptr;
                break;
            } catch (...) {
                last = std::current_exception();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        if (last)
            std::rethrow_exception(last);

        httplib::Server server;
        server.set_pre_routing_handler(
            [&config](const httplib::Request& request, httplib::Response& response) {
                const auto key_count = request.get_header_value_count("X-API-Key");
                if (lrs::authorize_api_request(config, key_count,
                                               request.get_header_value("X-API-Key")))
                    return httplib::Server::HandlerResponse::Unhandled;
                response.status = 401;
                response.set_header("WWW-Authenticate", "ApiKey header=\"X-API-Key\"");
                response.set_content(nlohmann::json{{"error",
                                                     {{"code", "invalid_api_key"},
                                                      {"message", "missing or invalid X-API-Key"}}}}
                                         .dump(),
                                     "application/json");
                return httplib::Server::HandlerResponse::Handled;
            });
        server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(service.health_json(), "application/json");
        });
        server.Get("/ready", [&](const httplib::Request&, httplib::Response& response) {
            response.status = service.ready() ? 200 : 503;
            response.set_content(service.health_json(), "application/json");
        });
        server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(service.models_json(), "application/json");
        });
        server.Post("/v1/chat/completions", [&](const httplib::Request& request,
                                                httplib::Response& response) {
            bool stream = false;
            try {
                stream = nlohmann::json::parse(request.body).value("stream", false);
            } catch (...) {
                int status = 0;
                response.set_content(service.chat_completions(request.body, status),
                                     "application/json");
                response.status = status;
                return;
            }
            if (!stream) {
                int status = 0;
                response.set_content(service.chat_completions(request.body, status),
                                     "application/json");
                response.status = status;
                return;
            }
            const std::string body = request.body;
            response.set_chunked_content_provider(
                "text/event-stream",
                [body, &service, started = false](std::size_t, httplib::DataSink& sink) mutable {
                    if (started)
                        return false;
                    started = true;
                    std::string error;
                    service.chat_completions_stream(
                        body,
                        [&](const std::string& event) {
                            return sink.is_writable() && sink.write(event.data(), event.size());
                        },
                        error);
                    sink.done();
                    return false;
                });
        });
        server.Post("/v1/rag/documents", [&](const httplib::Request& request,
                                             httplib::Response& response) {
            int status = 0;
            const bool asynchronous = request.get_header_value("Prefer") == "respond-async";
            const std::string body = service.ingest(request.body, status, asynchronous);
            response.set_content(body, "application/json");
            response.status = status;
            if (status == 202) {
                const auto job = nlohmann::json::parse(body);
                response.set_header("Location", "/v1/rag/jobs/" + job.at("id").get<std::string>());
            }
        });
        server.Get(R"(/v1/rag/jobs/(.+))", [&](const httplib::Request& request,
                                               httplib::Response& response) {
            int status = 0;
            response.set_content(service.get_job(request.matches[1], status), "application/json");
            response.status = status;
        });
        server.Delete(R"(/v1/rag/documents/(.+))",
                      [&](const httplib::Request& request, httplib::Response& response) {
                          int status = 0;
                          response.set_content(service.delete_document(request.matches[1], status),
                                               "application/json");
                          response.status = status;
                      });
        server.Post(
            "/v1/rag/search", [&](const httplib::Request& request, httplib::Response& response) {
                int status = 0;
                response.set_content(service.search(request.body, status), "application/json");
                response.status = status;
            });
        server.Post("/v1/rag/query", [&](const httplib::Request& request,
                                         httplib::Response& response) {
            const std::string body = request.body;
            response.set_chunked_content_provider(
                "text/event-stream",
                [body, &service, started = false](std::size_t, httplib::DataSink& sink) mutable {
                    if (started)
                        return false;
                    started = true;
                    std::string error;
                    service.query(
                        body,
                        [&](const std::string& event) {
                            return sink.is_writable() &&
                                   (event.empty() || sink.write(event.data(), event.size()));
                        },
                        error);
                    sink.done();
                    return false;
                });
        });
        std::clog << "llama-rag-server listening on " << config.listen.host << ':'
                  << config.listen.port << '\n';
        if (!server.listen(config.listen.host, config.listen.port))
            throw std::runtime_error("listen failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "llama-rag-server: " << e.what() << '\n';
        return 1;
    }
}
