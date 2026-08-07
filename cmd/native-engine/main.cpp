#include <cstdio>
#include <cstring>

#include "observability/log.h"

namespace {

constexpr const char kVersion[] = "0.1.0-phase1";

void PrintUsage() {
    std::printf(
        "lykuro-native-engine %s\n"
        "\n"
        "Usage: native-engine [--version] [--help]\n"
        "\n"
        "Phase 1 build: model verification, tokenizer, and CPU reference\n"
        "inference are available through the test suites. The gRPC server\n"
        "and GPU backend arrive in Phase 2 (Linux + CUDA).\n",
        kVersion);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("%s\n", kVersion);
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0) {
            PrintUsage();
            return 0;
        }
        std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
        return 2;
    }
    PrintUsage();
    return 0;
}
