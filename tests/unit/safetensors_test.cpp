#include "model/loader/safetensors.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace lykuro::nie {
namespace {

// Builds a header JSON for two small F32 tensors covering `data_size` bytes.
std::string TwoTensorHeader() {
    return R"({
      "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
      "b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
      "__metadata__": {"format": "pt"}
    })";
}

TEST(SafetensorsHeaderTest, AcceptsValidHeader) {
    auto r = ParseSafetensorsHeader(TwoTensorHeader(), 32);
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    EXPECT_EQ(r.header.tensors.size(), 2u);
    const TensorInfo& a = r.header.tensors.at("a");
    EXPECT_EQ(a.dtype, Dtype::kF32);
    EXPECT_EQ(a.element_count, 4u);
    EXPECT_EQ(a.data_size, 16u);
    EXPECT_EQ(r.header.metadata.at("format"), "pt");
}

TEST(SafetensorsHeaderTest, RejectsDtypeOutsideAllowlist) {
    auto r = ParseSafetensorsHeader(
        R"({"a": {"dtype": "I64", "shape": [2], "data_offsets": [0, 16]}})",
        16);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsOutOfBoundsOffsets) {
    auto r = ParseSafetensorsHeader(
        R"({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}})",
        /*data_section_size=*/8);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsOverlap) {
    auto r = ParseSafetensorsHeader(
        R"({
          "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
          "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]}
        })",
        24);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsGaps) {
    auto r = ParseSafetensorsHeader(
        R"({
          "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
          "b": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}
        })",
        24);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsSizeMismatch) {
    // shape 3 * 4 bytes = 12, but offsets claim 16.
    auto r = ParseSafetensorsHeader(
        R"({"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 16]}})",
        16);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsShapeOverflow) {
    auto r = ParseSafetensorsHeader(
        R"({"a": {"dtype": "F32",
                  "shape": [4294967295, 4294967295, 4294967295],
                  "data_offsets": [0, 16]}})",
        16);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsExtraTensorFields) {
    auto r = ParseSafetensorsHeader(
        R"({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16],
                  "extra": 1}})",
        16);
    EXPECT_FALSE(r.status.ok());
}

TEST(SafetensorsHeaderTest, RejectsEmpty) {
    EXPECT_FALSE(ParseSafetensorsHeader("{}", 0).status.ok());
}

class SafetensorsFileTest : public ::testing::Test {
protected:
    std::string WriteFile(const std::string& header_json,
                          const std::vector<uint8_t>& data) {
        path_ = testing::TempDir() + "st_test.safetensors";
        std::ofstream f(path_, std::ios::binary | std::ios::trunc);
        uint64_t len = header_json.size();
        char len_le[8];
        for (int i = 0; i < 8; ++i) len_le[i] = char(len >> (8 * i));
        f.write(len_le, 8);
        f.write(header_json.data(), long(header_json.size()));
        f.write(reinterpret_cast<const char*>(data.data()),
                long(data.size()));
        f.close();
        return path_;
    }

    void TearDown() override {
        if (!path_.empty()) std::remove(path_.c_str());
    }

    std::string path_;
};

TEST_F(SafetensorsFileTest, OpensValidFile) {
    std::vector<uint8_t> data(32, 0);
    float v = 1.5f;
    std::memcpy(data.data(), &v, 4);
    auto path = WriteFile(TwoTensorHeader(), data);

    SafetensorsFile file;
    Status s = file.Open(path);
    ASSERT_TRUE(s.ok()) << s.message();
    const uint8_t* a = file.TensorData("a");
    ASSERT_NE(a, nullptr);
    float readback;
    std::memcpy(&readback, a, 4);
    EXPECT_FLOAT_EQ(readback, 1.5f);
    EXPECT_EQ(file.TensorData("missing"), nullptr);
}

TEST_F(SafetensorsFileTest, RejectsTruncatedData) {
    std::vector<uint8_t> data(16, 0);  // header claims 32
    auto path = WriteFile(TwoTensorHeader(), data);
    SafetensorsFile file;
    EXPECT_FALSE(file.Open(path).ok());
}

TEST_F(SafetensorsFileTest, RejectsHeaderLengthBeyondFile) {
    path_ = testing::TempDir() + "st_test.safetensors";
    std::ofstream f(path_, std::ios::binary | std::ios::trunc);
    uint64_t len = 1 << 20;  // claims 1MB header, file has none
    char len_le[8];
    for (int i = 0; i < 8; ++i) len_le[i] = char(len >> (8 * i));
    f.write(len_le, 8);
    f.close();
    SafetensorsFile file;
    EXPECT_FALSE(file.Open(path_).ok());
}

TEST_F(SafetensorsFileTest, RejectsMissingFile) {
    SafetensorsFile file;
    EXPECT_FALSE(file.Open(testing::TempDir() + "does_not_exist.st").ok());
}

TEST_F(SafetensorsFileTest, RejectsHeaderOverCap) {
    std::vector<uint8_t> data(32, 0);
    auto path = WriteFile(TwoTensorHeader(), data);
    SafetensorsFile file;
    EXPECT_FALSE(file.Open(path, /*max_header_bytes=*/8).ok());
}

}  // namespace
}  // namespace lykuro::nie
