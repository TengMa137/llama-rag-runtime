#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include <rag/dense/policy.hpp>

#include "eval/dense.hpp"
#include "eval/qrels.hpp"
#include "eval/runtime.hpp"

namespace {

constexpr std::string_view usage =
    "Usage:\n"
    "  lrs-rag-eval qrels [PATH]\n"
    "  lrs-rag-eval manifest [PATH]\n"
    "  lrs-rag-eval dense [--vectors N] [--dimension N] [--queries N] [--k N]\n"
    "                     [--implementation native|faiss]\n"
    "                     [--algorithm exact|hnsw|flat|ivf-sq8|ivf-pq]\n"
    "                     [--seed N] [--enforce-gate]\n"
    "  lrs-rag-eval runtime [the same sizing and native policy options]\n";

template <class Integer> bool parse_integer(std::string_view value, Integer& output) {
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

int failure(const rag::Error& error) {
    std::cerr << nlohmann::json{{"object", "rag.evaluation_error"},
                                {"code", rag::to_string(error.code)},
                                {"message", error.message}}
                     .dump()
              << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    const std::string command = argc > 1 ? argv[1] : "qrels";
    if (command == "--help") {
        std::cout << usage;
        return 0;
    }
    if (command == "manifest") {
        const std::string report = lrs::eval::corpus_manifest().dump(2) + "\n";
        if (argc > 2) {
            std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
            if (!output) {
                std::cerr << "cannot write corpus manifest\n";
                return 1;
            }
            output << report;
        }
        std::cout << report;
        return 0;
    }
    if (command == "qrels") {
        const std::string path =
            argc > 2 ? argv[2] : std::string(LRS_SOURCE_DIR) + "/tests/fixtures/qrels.json";
        auto report = lrs::eval::run_qrels(path);
        if (!report)
            return failure(report.error());
        std::cout << report->dump(2) << '\n';
        return 0;
    }
    if (command != "dense" && command != "runtime") {
        std::cerr << usage;
        return 1;
    }

    lrs::eval::DenseOptions options;
    std::string implementation = "native";
    std::string algorithm = "exact";
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--enforce-gate") {
            options.enforce_gate = true;
            continue;
        }
        if (++index >= argc) {
            std::cerr << usage;
            return 1;
        }
        const std::string_view value = argv[index];
        if (option == "--implementation")
            implementation = value;
        else if (option == "--algorithm")
            algorithm = value;
        else if (option == "--vectors") {
            if (!parse_integer(value, options.vectors))
                return 1;
        } else if (option == "--dimension") {
            if (!parse_integer(value, options.dimension))
                return 1;
        } else if (option == "--queries") {
            if (!parse_integer(value, options.queries))
                return 1;
        } else if (option == "--k") {
            if (!parse_integer(value, options.k))
                return 1;
        } else if (option == "--seed") {
            if (!parse_integer(value, options.seed))
                return 1;
        } else {
            std::cerr << usage;
            return 1;
        }
    }
    auto parsed_implementation = rag::dense::parse_dense_implementation(implementation);
    auto parsed_algorithm = rag::dense::parse_dense_algorithm(algorithm);
    if (!parsed_implementation)
        return failure(parsed_implementation.error());
    if (!parsed_algorithm)
        return failure(parsed_algorithm.error());
    options.policy.implementation = *parsed_implementation;
    options.policy.algorithm = *parsed_algorithm;
    if (options.policy.algorithm == rag::dense::DenseAlgorithm::hnsw) {
        options.policy.hnsw.ef_search = 160;
        options.policy.faiss.ef_search = 160;
    }
    if (options.policy.algorithm == rag::dense::DenseAlgorithm::ivf_sq8 ||
        options.policy.algorithm == rag::dense::DenseAlgorithm::ivf_pq) {
        options.policy.faiss.ivf_lists = 256;
        options.policy.faiss.ivf_probes = 256;
        options.policy.faiss.minimum_training_vectors_per_list = 39;
        options.policy.faiss.pq_subquantizers = 32;
        options.policy.faiss.pq_bits = 8;
    }
    rag::Result<nlohmann::json> report =
        command == "dense"
            ? lrs::eval::run_dense(options)
            : lrs::eval::run_runtime({options.vectors, options.dimension, options.queries,
                                      options.k, options.seed, options.policy});
    if (!report)
        return failure(report.error());
    std::cout << report->dump(2) << '\n';
    return 0;
}
