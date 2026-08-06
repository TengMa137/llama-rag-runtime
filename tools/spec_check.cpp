#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>

namespace {
std::string read(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string revision(const std::string& root, const std::string& dependency) {
    const std::string command =
        "git -C \"" + root + "/third_party/" + dependency + "\" rev-parse HEAD";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("cannot inspect dependency revision");
    char buffer[128]{};
    const std::string value = std::fgets(buffer, sizeof(buffer), pipe) ? buffer : "";
    const int status = pclose(pipe);
    if (status != 0)
        throw std::runtime_error("cannot inspect " + dependency + " revision");
    return value.substr(0, value.find_first_of("\r\n"));
}
} // namespace

int main() {
    try {
        const std::string root = LRS_SOURCE_DIR;
        const auto catalog = nlohmann::json::parse(read(root + "/requirements.json"));
        if (catalog.at("dependencies").at("llama.cpp") != LRS_LLAMA_PIN ||
            catalog.at("dependencies").at("rag-cpp") != LRS_RAGCPP_PIN) {
            std::cerr << "dependency pin differs from build metadata\n";
            return 1;
        }
        if (revision(root, "llama.cpp") != LRS_LLAMA_PIN ||
            revision(root, "rag-cpp") != LRS_RAGCPP_PIN) {
            std::cerr << "checked-out dependency differs from recorded build metadata\n";
            return 1;
        }
        const std::string tests = read(root + "/tests/tests.cpp");
        for (const auto& requirement : catalog.at("requirements")) {
            if (requirement.at("status") == "active") {
                const std::string tag = requirement.at("test");
                if (tag.empty() || tests.find(tag) == std::string::npos) {
                    std::cerr << requirement.at("id") << " has no referenced test\n";
                    return 1;
                }
            }
        }
        std::istringstream spec(read(root + "/docs/llama-rag-server-spec.md"));
        const std::regex normative("\\b(must|should)\\b", std::regex_constants::icase);
        const std::regex requirement_id("\\[(LRS-[A-Z]+-[0-9]{3})\\]");
        std::string line;
        std::size_t number = 0;
        bool acceptance = false;
        bool exit_criteria = false;
        std::unordered_set<std::string> ids;
        while (std::getline(spec, line)) {
            ++number;
            if (line.rfind("## 24.", 0) == 0)
                acceptance = true;
            if (line.rfind("## 25.", 0) == 0)
                acceptance = false;
            if (line == "Exit criteria:")
                exit_criteria = true;
            else if (exit_criteria && (line.rfind("### ", 0) == 0 || line.rfind("## ", 0) == 0))
                exit_criteria = false;
            const bool requires_id = std::regex_search(line, normative) ||
                                     ((acceptance || exit_criteria) && line.rfind("- ", 0) == 0);
            std::smatch match;
            if (requires_id && !std::regex_search(line, match, requirement_id)) {
                std::cerr << "normative statement lacks requirement ID at line " << number << '\n';
                return 1;
            }
            if (std::regex_search(line, match, requirement_id) && !ids.insert(match[1]).second) {
                std::cerr << "duplicate requirement ID " << match[1] << '\n';
                return 1;
            }
        }
        std::cout << "specification traceability verified\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "lrs-spec-check: " << e.what() << '\n';
        return 1;
    }
}
