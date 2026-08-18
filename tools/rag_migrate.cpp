#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include <rag/backend/postgres_config.hpp>
#include <rag/migration/contract.hpp>
#include <rag/migration/embedded_endpoint.hpp>
#include <rag/migration/postgres_endpoint.hpp>

namespace {

using rag::Result;

struct Arguments {
    std::string direction;
    std::string source;
    std::string destination;
    std::string connection_env = "LRS_POSTGRES_URL";
    std::string schema = "lrs_rag";
    std::string corpus = "default";
    std::string report;
    std::size_t pool_size = 4;
    std::size_t acquire_timeout_ms = 5'000;
    std::size_t statement_timeout_ms = 10'000;
    std::size_t batch_size = 256;
    std::size_t sample_searches = 16;
};

constexpr std::string_view usage =
    "Usage:\n"
    "  lrs-rag-migrate embedded-to-postgres --source PATH [options]\n"
    "  lrs-rag-migrate postgres-to-embedded --destination PATH [options]\n\n"
    "Options:\n"
    "  --connection-env NAME       PostgreSQL connection environment variable\n"
    "  --schema NAME               Project-owned PostgreSQL schema (default lrs_rag)\n"
    "  --corpus NAME               PostgreSQL corpus (default default)\n"
    "  --batch-size N              Documents per resumable batch (default 256)\n"
    "  --sample-searches N         Exact search validations (default 16)\n"
    "  --pool-size N               PostgreSQL connection pool size (default 4)\n"
    "  --acquire-timeout-ms N      Pool acquisition timeout\n"
    "  --statement-timeout-ms N    PostgreSQL statement timeout\n"
    "  --report PATH               Also atomically write the JSON report\n";

Result<std::size_t> number(std::string_view value, std::string_view option) {
    std::size_t output = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return rag::fail<std::size_t>(rag::Errc::invalid_argument,
                                      std::string(option) + " requires an unsigned integer");
    return output;
}

Result<Arguments> parse(int argc, char** argv) {
    if (argc < 2)
        return rag::fail<Arguments>(rag::Errc::invalid_argument, "migration direction is required");
    Arguments output;
    output.direction = argv[1];
    if (output.direction != "embedded-to-postgres" && output.direction != "postgres-to-embedded")
        return rag::fail<Arguments>(rag::Errc::invalid_argument, "migration direction is invalid");
    std::map<std::string, std::string*> strings{{"--source", &output.source},
                                                {"--destination", &output.destination},
                                                {"--connection-env", &output.connection_env},
                                                {"--schema", &output.schema},
                                                {"--corpus", &output.corpus},
                                                {"--report", &output.report}};
    std::map<std::string, std::size_t*> numbers{
        {"--pool-size", &output.pool_size},
        {"--acquire-timeout-ms", &output.acquire_timeout_ms},
        {"--statement-timeout-ms", &output.statement_timeout_ms},
        {"--batch-size", &output.batch_size},
        {"--sample-searches", &output.sample_searches}};
    for (int index = 2; index < argc; index += 2) {
        const std::string option = argv[index];
        if (index + 1 >= argc)
            return rag::fail<Arguments>(rag::Errc::invalid_argument, option + " requires a value");
        if (const auto found = strings.find(option); found != strings.end()) {
            *found->second = argv[index + 1];
            continue;
        }
        if (const auto found = numbers.find(option); found != numbers.end()) {
            auto parsed = number(argv[index + 1], option);
            if (!parsed)
                return rag::unexpected(parsed.error());
            *found->second = *parsed;
            continue;
        }
        return rag::fail<Arguments>(rag::Errc::invalid_argument,
                                    "unknown migration option: " + option);
    }
    if (output.direction == "embedded-to-postgres" && output.source.empty())
        return rag::fail<Arguments>(rag::Errc::invalid_argument, "--source is required");
    if (output.direction == "postgres-to-embedded" && output.destination.empty())
        return rag::fail<Arguments>(rag::Errc::invalid_argument, "--destination is required");
    return output;
}

Result<void> write_report(const std::string& path, std::string_view report) {
    if (path.empty())
        return {};
    try {
        const std::filesystem::path destination(path);
        if (destination.has_parent_path())
            std::filesystem::create_directories(destination.parent_path());
        const auto temporary = destination.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output)
                return rag::fail<void>(rag::Errc::io_error, "migration report could not be opened");
            output.write(report.data(), static_cast<std::streamsize>(report.size()));
            output.put('\n');
            output.flush();
            if (!output)
                return rag::fail<void>(rag::Errc::io_error,
                                       "migration report could not be written");
        }
        std::filesystem::rename(temporary, destination);
        return {};
    } catch (const std::exception& error) {
        return rag::fail<void>(rag::Errc::io_error,
                               "migration report publication failed: " + std::string(error.what()));
    }
}

int error(const rag::Error& failure) {
    std::cerr << nlohmann::json{{"object", "rag.migration_error"},
                                {"code", rag::to_string(failure.code)},
                                {"message", failure.message}}
                     .dump()
              << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    auto arguments = parse(argc, argv);
    if (!arguments) {
        std::cerr << usage;
        return error(arguments.error());
    }
    const char* connection = std::getenv(arguments->connection_env.c_str());
    if (!connection || *connection == '\0')
        return error(
            {rag::Errc::invalid_argument, "PostgreSQL connection environment variable is not set"});
    rag::backend::PostgresConfig postgres;
    postgres.connection_string = connection;
    postgres.schema = arguments->schema;
    postgres.corpus = arguments->corpus;
    postgres.pool_size = arguments->pool_size;
    postgres.acquire_timeout = std::chrono::milliseconds(arguments->acquire_timeout_ms);
    postgres.statement_timeout = std::chrono::milliseconds(arguments->statement_timeout_ms);
    postgres.vector_index = rag::backend::PostgresVectorIndex::exact;
    rag::Result<std::unique_ptr<rag::migration::Endpoint>> source =
        rag::fail<std::unique_ptr<rag::migration::Endpoint>>(rag::Errc::unavailable);
    rag::Result<std::unique_ptr<rag::migration::Endpoint>> destination =
        rag::fail<std::unique_ptr<rag::migration::Endpoint>>(rag::Errc::unavailable);
    if (arguments->direction == "embedded-to-postgres") {
        source = rag::migration::open_embedded_source(arguments->source);
        destination = rag::migration::open_postgres_destination(postgres);
    } else {
        source = rag::migration::open_postgres_source(postgres);
        destination = rag::migration::open_embedded_destination(arguments->destination);
    }
    if (!source)
        return error(source.error());
    if (!destination)
        return error(destination.error());
    auto report = rag::migration::migrate(**source, **destination, arguments->direction,
                                          {arguments->batch_size, arguments->sample_searches});
    if (!report)
        return error(report.error());
    const std::string json = rag::migration::json_report(*report);
    if (auto written = write_report(arguments->report, json); !written)
        return error(written.error());
    std::cout << json << '\n';
    return 0;
}
