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

TEST_F(PullLocalTest, DisplayNameMatchesPullInput) {
    const fs::path dir = home_ / ".lykuro/models/Qwen_Qwen2.5-0.5B-Instruct";
    fs::create_directories(dir);
    std::ofstream(dir / "manifest.json") << "{}";

    // Without a sidecar the repo id is derived (HF owner names cannot
    // contain '_', so the first '_' is the '/' pull replaced) ...
    const std::string shown = DisplayModelName(dir.string());
    EXPECT_EQ(shown, "Qwen/Qwen2.5-0.5B-Instruct");
    // ... and the displayed name must round-trip: it IS valid pull/run
    // input resolving to the same artifact.
    EXPECT_EQ(DefaultModelDir(shown), dir.string());
    EXPECT_EQ(PullModel(shown, DefaultModelDir(shown)), 0);

    // A recorded source repo takes precedence.
    std::ofstream(dir / "source_repo") << "Qwen/Qwen2.5-0.5B-Instruct\n";
    EXPECT_EQ(DisplayModelName(dir.string()),
              "Qwen/Qwen2.5-0.5B-Instruct");

    // A sidecar that does not resolve back to this directory is ignored
    // (never display a name that would load a different artifact).
    std::ofstream(dir / "source_repo", std::ios::trunc)
        << "Other/Model\n";
    EXPECT_EQ(DisplayModelName(dir.string()),
              "Qwen/Qwen2.5-0.5B-Instruct");
}

TEST_F(PullLocalTest, UnknownNonRepoNameFailsFastWithoutNetwork) {
    EXPECT_EQ(PullModel("definitely_not_local",
                        DefaultModelDir("definitely_not_local")),
              2);
}

}  // namespace
}  // namespace lykuro::nie::cli
