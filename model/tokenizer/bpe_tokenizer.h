#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Byte-level BPE tokenizer (approved_qwen_tokenizer_v1).
//
// Config format (strict JSON, digest-pinned by the manifest):
// {
//   "tokenizer_type": "approved_qwen_tokenizer_v1",
//   "vocab": { "<byte-level token>": id, ... },
//   "merges": [ "left right", ... ],
//   "special_tokens": { "<|im_start|>": id, ... }
// }
//
// Security property: Encode() treats its input purely as text. Special
// token strings appearing inside user content are byte-level encoded like
// any other text and can never produce a special token ID; only the
// prompt template layer emits special IDs (spec §13.3).
class BpeTokenizer {
public:
    struct LoadResult;

    static LoadResult FromConfig(std::string_view config_json,
                                 size_t max_bytes = 64ull << 20);

    // Encodes plain text (no special token recognition).
    Status Encode(std::string_view text, std::vector<uint32_t>& out) const;

    // Decodes token IDs to raw bytes. Output may end mid-UTF-8-sequence;
    // streaming callers must hold back incomplete sequences.
    Status DecodeBytes(const std::vector<uint32_t>& ids,
                       std::string& out) const;

    // Decodes and validates/replaces invalid UTF-8 (U+FFFD policy).
    Status DecodeText(const std::vector<uint32_t>& ids,
                      std::string& out) const;

    // Special token ID by exact name; nullopt when not declared.
    std::optional<uint32_t> SpecialTokenId(std::string_view name) const;
    bool IsSpecialToken(uint32_t id) const;

    uint32_t vocab_size() const { return vocab_size_; }

private:
    // token string (byte-level alphabet) -> id
    std::map<std::string, uint32_t> vocab_;
    // id -> raw bytes the token decodes to
    std::vector<std::string> id_to_bytes_;
    // merge pair -> rank
    std::map<std::pair<std::string, std::string>, uint32_t> merge_ranks_;
    std::map<std::string, uint32_t> special_tokens_;
    std::vector<bool> is_special_;
    uint32_t vocab_size_ = 0;
};

struct BpeTokenizer::LoadResult {
    Status status;
    BpeTokenizer tokenizer;  // valid only when status.ok()
};

// Splits `out` so that it only contains complete UTF-8 sequences, moving
// any trailing incomplete sequence into `carry`. Used by streaming to
// avoid emitting broken code points (spec §19.1).
void SplitUtf8Boundary(std::string& out, std::string& carry);

}  // namespace lykuro::nie
