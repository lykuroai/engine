#include "core/engine/error.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace lykuro::nie {
namespace {

// Every code has a distinct, stable, content-free name (spec §21.2 /
// addendum §21). Names double as the wire-visible result_code label.
TEST(ErrorTest, CodeNamesAreUniqueAndStable) {
    std::set<std::string> names;
    for (int i = 0; i <= int(ErrorCode::kBackendAbiMismatch); ++i) {
        std::string_view name = ErrorCodeName(ErrorCode(i));
        EXPECT_FALSE(name.empty());
        EXPECT_TRUE(names.insert(std::string(name)).second)
            << "duplicate name: " << name;
    }
    EXPECT_EQ(ErrorCodeName(ErrorCode::kMetalMemoryPressure),
              "metal_memory_pressure");
    EXPECT_EQ(ErrorCodeName(ErrorCode::kMacProfileNotCertified),
              "mac_profile_not_certified");
}

TEST(ErrorTest, MetalRetryability) {
    EXPECT_TRUE(IsRetryable(ErrorCode::kMetalMemoryPressure));
    EXPECT_FALSE(IsRetryable(ErrorCode::kMetalDeviceUnsupported));
    EXPECT_FALSE(IsRetryable(ErrorCode::kMacProfileNotCertified));
    EXPECT_FALSE(IsRetryable(ErrorCode::kBackendAbiMismatch));
}

TEST(ErrorTest, StatusCarriesNoContent) {
    Status s(ErrorCode::kMetalOutOfMemory, "metal out of memory",
             "metal_backend");
    s.WithDetail("budget_bytes", 1024);
    EXPECT_EQ(s.code(), ErrorCode::kMetalOutOfMemory);
    EXPECT_EQ(s.details().at("budget_bytes"), 1024);
    EXPECT_FALSE(s.ok());
}

}  // namespace
}  // namespace lykuro::nie
