#include "model/loader/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <limits>

#include "core/engine/json.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "safetensors_loader";
constexpr size_t kMaxTensorNameLen = 512;
constexpr size_t kMaxRank = 8;
constexpr size_t kMaxTensors = 100000;

Status Invalid(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

bool ParseDtype(std::string_view name, Dtype& out) {
    if (name == "BF16") { out = Dtype::kBf16; return true; }
    if (name == "F16") { out = Dtype::kF16; return true; }
    if (name == "F32") { out = Dtype::kF32; return true; }
    return false;
}

}  // namespace

size_t DtypeSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBf16:
        case Dtype::kF16:
            return 2;
        case Dtype::kF32:
            return 4;
    }
    return 4;
}

std::string_view DtypeName(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBf16: return "BF16";
        case Dtype::kF16: return "F16";
        case Dtype::kF32: return "F32";
    }
    return "F32";
}

HeaderParseResult ParseSafetensorsHeader(std::string_view header_json,
                                         uint64_t data_section_size) {
    HeaderParseResult result;

    json::ParseResult parsed = json::Parse(header_json);
    if (!parsed.ok()) {
        result.status = Invalid("safetensors header is not valid JSON: " +
                                parsed.error);
        return result;
    }
    if (!parsed.value->is_object()) {
        result.status = Invalid("safetensors header root must be an object");
        return result;
    }

    SafetensorsHeader& h = result.header;
    // (offset_begin, offset_end) pairs for overlap/coverage validation.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;

    for (const auto& [name, value] : parsed.value->as_object()) {
        if (name == "__metadata__") {
            if (!value->is_object()) {
                result.status = Invalid("safetensors __metadata__ invalid");
                return result;
            }
            for (const auto& [mk, mv] : value->as_object()) {
                if (!mv->is_string()) {
                    result.status =
                        Invalid("safetensors __metadata__ value invalid");
                    return result;
                }
                h.metadata.emplace(mk, mv->as_string());
            }
            continue;
        }
        if (name.empty() || name.size() > kMaxTensorNameLen) {
            result.status = Invalid("safetensors tensor name invalid");
            return result;
        }
        if (h.tensors.size() >= kMaxTensors) {
            result.status = Invalid("safetensors tensor count limit exceeded");
            return result;
        }
        if (!value->is_object()) {
            result.status = Invalid("safetensors tensor entry invalid");
            return result;
        }

        TensorInfo info;
        const json::Value* dtype = value->Find("dtype");
        const json::Value* shape = value->Find("shape");
        const json::Value* offsets = value->Find("data_offsets");
        if (dtype == nullptr || shape == nullptr || offsets == nullptr ||
            value->as_object().size() != 3) {
            result.status = Invalid("safetensors tensor entry fields invalid");
            return result;
        }
        if (!dtype->is_string() || !ParseDtype(dtype->as_string(), info.dtype)) {
            // dtype allowlist: anything outside BF16/F16/F32 is rejected,
            // never coerced.
            result.status = Invalid("safetensors dtype not allowed");
            return result;
        }
        if (!shape->is_array() || shape->as_array().size() > kMaxRank) {
            result.status = Invalid("safetensors shape invalid");
            return result;
        }
        uint64_t element_count = 1;
        for (const auto& dim : shape->as_array()) {
            if (!dim->is_int() || dim->as_int() < 0) {
                result.status = Invalid("safetensors shape dimension invalid");
                return result;
            }
            uint64_t d = uint64_t(dim->as_int());
            if (d != 0 &&
                element_count > std::numeric_limits<uint64_t>::max() / d) {
                result.status = Invalid("safetensors shape overflows");
                return result;
            }
            element_count *= d;
            info.shape.push_back(d);
        }
        info.element_count = element_count;

        if (!offsets->is_array() || offsets->as_array().size() != 2 ||
            !offsets->as_array()[0]->is_int() ||
            !offsets->as_array()[1]->is_int()) {
            result.status = Invalid("safetensors data_offsets invalid");
            return result;
        }
        int64_t begin_s = offsets->as_array()[0]->as_int();
        int64_t end_s = offsets->as_array()[1]->as_int();
        if (begin_s < 0 || end_s < begin_s) {
            result.status = Invalid("safetensors data_offsets out of order");
            return result;
        }
        uint64_t begin = uint64_t(begin_s);
        uint64_t end = uint64_t(end_s);
        if (end > data_section_size) {
            result.status = Invalid("safetensors data_offsets out of bounds");
            return result;
        }
        info.data_offset = begin;
        info.data_size = end - begin;

        const uint64_t expected =
            element_count * uint64_t(DtypeSize(info.dtype));
        if (expected != info.data_size) {
            result.status = Invalid("safetensors tensor size mismatch");
            return result;
        }

        ranges.emplace_back(begin, end);
        h.tensors.emplace(name, std::move(info));
    }

    if (h.tensors.empty()) {
        result.status = Invalid("safetensors file contains no tensors");
        return result;
    }

    // Offsets must tile the data section exactly: contiguous from 0 with
    // no overlap and no trailing slack.
    std::sort(ranges.begin(), ranges.end());
    uint64_t cursor = 0;
    for (const auto& [begin, end] : ranges) {
        if (begin != cursor) {
            result.status =
                Invalid("safetensors data_offsets overlap or leave gaps");
            return result;
        }
        cursor = end;
    }
    if (cursor != data_section_size) {
        result.status = Invalid("safetensors data section size mismatch");
        return result;
    }

    result.status = Status::Ok();
    return result;
}

SafetensorsFile::~SafetensorsFile() { Close(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept {
    *this = std::move(other);
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
        Close();
        header_ = std::move(other.header_);
        map_base_ = other.map_base_;
        map_size_ = other.map_size_;
        data_ = other.data_;
        data_size_ = other.data_size_;
        other.map_base_ = nullptr;
        other.map_size_ = 0;
        other.data_ = nullptr;
        other.data_size_ = 0;
    }
    return *this;
}

void SafetensorsFile::Close() {
    if (map_base_ != nullptr) {
        ::munmap(map_base_, map_size_);
        map_base_ = nullptr;
    }
    map_size_ = 0;
    data_ = nullptr;
    data_size_ = 0;
    header_ = SafetensorsHeader{};
}

Status SafetensorsFile::Open(const std::string& path,
                             uint64_t max_header_bytes) {
    Close();

    // O_NOFOLLOW: symlinked artifacts are rejected (spec §10.4).
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        return Invalid("safetensors file cannot be opened");
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return Invalid("safetensors file is not a regular file");
    }
    const uint64_t file_size = uint64_t(st.st_size);
    if (file_size < 8) {
        ::close(fd);
        return Invalid("safetensors file truncated");
    }

    void* base = ::mmap(nullptr, size_t(file_size), PROT_READ, MAP_PRIVATE,
                        fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        return Invalid("safetensors file cannot be mapped");
    }

    const uint8_t* bytes = static_cast<const uint8_t*>(base);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= uint64_t(bytes[i]) << (8 * i);
    }
    if (header_len > max_header_bytes || header_len > file_size - 8) {
        ::munmap(base, size_t(file_size));
        return Invalid("safetensors header size invalid");
    }

    const uint64_t data_size = file_size - 8 - header_len;
    std::string_view header_json(
        reinterpret_cast<const char*>(bytes + 8), size_t(header_len));

    HeaderParseResult parsed = ParseSafetensorsHeader(header_json, data_size);
    if (!parsed.status.ok()) {
        ::munmap(base, size_t(file_size));
        return parsed.status;
    }

    header_ = std::move(parsed.header);
    map_base_ = base;
    map_size_ = size_t(file_size);
    data_ = bytes + 8 + header_len;
    data_size_ = data_size;
    return Status::Ok();
}

const TensorInfo* SafetensorsFile::FindTensor(const std::string& name) const {
    auto it = header_.tensors.find(name);
    return it == header_.tensors.end() ? nullptr : &it->second;
}

const uint8_t* SafetensorsFile::TensorData(const std::string& name) const {
    const TensorInfo* info = FindTensor(name);
    if (info == nullptr || data_ == nullptr) return nullptr;
    return data_ + info->data_offset;
}

}  // namespace lykuro::nie
