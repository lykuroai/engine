#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Safetensors reader (spec §12.3). Treats the file as untrusted input:
//  - 8-byte little-endian header length with a hard cap
//  - strict JSON header validation (via the in-tree parser)
//  - dtype allowlist (BF16 / F16 / F32)
//  - offsets: in-bounds, non-overlapping, contiguous from 0, exact cover
//  - shape product must equal byte length / dtype size (overflow-checked)
//  - memory-mapped read-only access

enum class Dtype { kBf16, kF16, kF32 };

size_t DtypeSize(Dtype dtype);
std::string_view DtypeName(Dtype dtype);

struct TensorInfo {
    Dtype dtype = Dtype::kF32;
    std::vector<uint64_t> shape;   // may be empty (scalar)
    uint64_t data_offset = 0;      // relative to the data section
    uint64_t data_size = 0;        // bytes
    uint64_t element_count = 0;
};

struct SafetensorsHeader {
    std::map<std::string, TensorInfo> tensors;
    std::map<std::string, std::string> metadata;  // "__metadata__", optional
};

struct HeaderParseResult {
    Status status;
    SafetensorsHeader header;  // valid only when status.ok()
};

// Parses and validates a safetensors JSON header against the size of the
// data section that follows it. Exposed separately for negative testing.
HeaderParseResult ParseSafetensorsHeader(std::string_view header_json,
                                         uint64_t data_section_size);

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

    // Opens and fully validates the file. On failure the object stays empty.
    // max_header_bytes caps the declared JSON header size (default 100 MB
    // files still keep headers tiny; 16 MB is generous).
    Status Open(const std::string& path,
                uint64_t max_header_bytes = 16ull << 20);

    bool is_open() const { return data_ != nullptr; }
    const SafetensorsHeader& header() const { return header_; }

    // Returns the read-only bytes of a tensor, or nullptr when unknown.
    const uint8_t* TensorData(const std::string& name) const;
    const TensorInfo* FindTensor(const std::string& name) const;

    void Close();

private:
    SafetensorsHeader header_;
    void* map_base_ = nullptr;     // mmap base (whole file)
    size_t map_size_ = 0;
    const uint8_t* data_ = nullptr;  // start of data section
    uint64_t data_size_ = 0;
};

}  // namespace lykuro::nie
