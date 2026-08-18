#include "lrs/mobile.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string database = argc > 1 ? argv[1] : "mobile-smoke.ragdb";
    lrs_mobile_index* index = nullptr;
    char* error = nullptr;
    if (lrs_mobile_open(database.c_str(), &index, &error) != 0) {
        std::cerr << (error != nullptr ? error : "open failed") << '\n';
        lrs_mobile_string_destroy(error);
        return 1;
    }

    const float document_vector[] = {1.0F, 0.0F};
    int unchanged = 0;
    if (lrs_mobile_upsert_vectors(index, "device/smoke", "Device smoke",
                                  "PLQ110 native retrieval-core vector search", document_vector, 1,
                                  2, &unchanged, &error) != 0) {
        std::cerr << (error != nullptr ? error : "upsert failed") << '\n';
        lrs_mobile_string_destroy(error);
        lrs_mobile_destroy(index);
        return 1;
    }

    char* json = nullptr;
    if (lrs_mobile_search_json(index, "native vector", document_vector, 2, "hybrid", 4, &json,
                               &error) != 0) {
        std::cerr << (error != nullptr ? error : "search failed") << '\n';
        lrs_mobile_string_destroy(error);
        lrs_mobile_destroy(index);
        return 1;
    }
    std::cout << json << '\n';
    lrs_mobile_string_destroy(json);
    lrs_mobile_destroy(index);
    return 0;
}
