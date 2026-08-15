// `pull` must round-trip the names that `list` / /api/tags report:
// an already-local model directory name (e.g. "Qwen_Qwen2.5-0.5B-
// Instruct", not a valid HF repo id) succeeds as a no-op, and an
// unknown non-repo name fails fast without touching the network.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "cmd/native-engine/generator.h"

namespace lykuro::nie::cli {
namespace {
namespace fs = std::filesystem;

class PullLocalTest : public ::testing::Test {
protected:
    void SetUp() override {
        home_ = fs::temp_directory_path() / "lykuro_pull_test";
        fs::remove_all(home_);
        fs::create_directories(home_);
        old_home_ = std::getenv("HOME") ? std::getenv("HOME") : "";
        ::setenv("HOME", home_.c_str(), 1);
    }
    void TearDown() override {
        ::setenv("HOME", old_home_.c_str(), 1);
        fs::remove_all(home_);
    }
    fs::path home_;
    std::string old_home_;
};

TEST_F(PullLocalTest, LocalDirectoryNameIsANoOpSuccess) {
    const fs::path dir = home_ / ".lykuro/models/Some_Local-Model";
    fs::create_directories(dir);
    std::ofstream(dir / "manifest.json") << "{}";

    // The name `list` and /api/tags report (no '/', so not a repo id).
    EXPECT_EQ(PullModel("Some_Local-Model",
                        DefaultModelDir("Some_Local-Model")),
              0);
    // Repo-id form of an already-local model is also a no-op.
    EXPECT_EQ(
        PullModel("Some/Local-Model", DefaultModelDir("Some/Local-Model")),
        0);
}

TEST_F(PullLocalTest, UnknownNonRepoNameFailsFastWithoutNetwork) {
    EXPECT_EQ(PullModel("definitely_not_local",
                        DefaultModelDir("definitely_not_local")),
              2);
}

}  // namespace
}  // namespace lykuro::nie::cli
