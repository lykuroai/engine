#include "model/tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <array>
#include <limits>

#include "core/engine/json.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "tokenizer";

Status Invalid(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

// GPT-2 style byte<->unicode table: maps each byte to a printable unicode
// codepoint so vocab entries are valid UTF-8 strings.
struct ByteUnicodeTable {
    std::array<std::string, 256> byte_to_str;
    std::map<std::string, uint8_t> str_to_byte;

    ByteUnicodeTable() {
        std::array<uint32_t, 256> cp{};
        uint32_t n = 0;
        auto printable = [](uint32_t b) {
            return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                   (b >= 0xAE && b <= 0xFF);
        };
        for (uint32_t b = 0; b < 256; ++b) {
            if (printable(b)) {
                cp[b] = b;
            } else {
                cp[b] = 256 + n;
                ++n;
            }
        }
        for (uint32_t b = 0; b < 256; ++b) {
            std::string s;
            uint32_t c = cp[b];
            if (c < 0x80) {
                s.push_back(char(c));
            } else if (c < 0x800) {
                s.push_back(char(0xC0 | (c >> 6)));
                s.push_back(char(0x80 | (c & 0x3F)));
            } else {
                s.push_back(char(0xE0 | (c >> 12)));
                s.push_back(char(0x80 | ((c >> 6) & 0x3F)));
                s.push_back(char(0x80 | (c & 0x3F)));
            }
            byte_to_str[b] = s;
            str_to_byte[s] = uint8_t(b);
        }
    }
};

const ByteUnicodeTable& ByteTable() {
    static ByteUnicodeTable table;
    return table;
}

// Converts a byte-level vocab token string back to the raw bytes it
// represents. Returns false when the string is not in the byte alphabet.
bool TokenStringToBytes(const std::string& token, std::string& out) {
    const auto& table = ByteTable();
    size_t i = 0;
    while (i < token.size()) {
        unsigned char c = static_cast<unsigned char>(token[i]);
        size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2
                   : (c & 0xF0) == 0xE0 ? 3 : 4;
        if (i + len > token.size()) return false;
        auto it = table.str_to_byte.find(token.substr(i, len));
        if (it == table.str_to_byte.end()) return false;
        out.push_back(char(it->second));
        i += len;
    }
    return true;
}

}  // namespace

BpeTokenizer::LoadResult BpeTokenizer::FromConfig(std::string_view config_json,
                                                  size_t max_bytes) {
    LoadResult result;
    BpeTokenizer& t = result.tokenizer;

    if (config_json.size() > max_bytes) {
        result.status = Invalid("tokenizer config exceeds size limit");
        return result;
    }
    json::ParseResult parsed = json::Parse(config_json);
    if (!parsed.ok()) {
        result.status = Invalid("tokenizer config is not valid JSON: " +
                                parsed.error);
        return result;
    }
    if (!parsed.value->is_object()) {
        result.status = Invalid("tokenizer config root must be an object");
        return result;
    }
    const json::Value& root = *parsed.value;
    for (const auto& [key, value] : root.as_object()) {
        (void)value;
        if (key != "tokenizer_type" && key != "vocab" && key != "merges" &&
            key != "special_tokens") {
            result.status = Invalid("tokenizer config unknown field: " + key);
            return result;
        }
    }

    const json::Value* type = root.Find("tokenizer_type");
    if (type == nullptr || !type->is_string() ||
        type->as_string() != "approved_qwen_tokenizer_v1") {
        result.status = Invalid("tokenizer_type is not approved");
        return result;
    }

    const json::Value* vocab = root.Find("vocab");
    const json::Value* merges = root.Find("merges");
    const json::Value* specials = root.Find("special_tokens");
    if (vocab == nullptr || !vocab->is_object() || merges == nullptr ||
        !merges->is_array()) {
        result.status = Invalid("tokenizer config vocab/merges invalid");
        return result;
    }

    uint32_t max_id = 0;
    for (const auto& [token, id_value] : vocab->as_object()) {
        if (token.empty() || token.size() > 512 || !id_value->is_int() ||
            id_value->as_int() < 0 ||
            id_value->as_int() > int64_t(UINT32_MAX)) {
            result.status = Invalid("tokenizer vocab entry invalid");
            return result;
        }
        uint32_t id = uint32_t(id_value->as_int());
        if (!t.vocab_.emplace(token, id).second) {
            result.status = Invalid("tokenizer vocab token duplicated");
            return result;
        }
        max_id = std::max(max_id, id);
    }
    if (t.vocab_.empty()) {
        result.status = Invalid("tokenizer vocab empty");
        return result;
    }

    if (specials != nullptr) {
        if (!specials->is_object()) {
            result.status = Invalid("tokenizer special_tokens invalid");
            return result;
        }
        for (const auto& [name, id_value] : specials->as_object()) {
            if (name.empty() || name.size() > 128 || !id_value->is_int() ||
                id_value->as_int() < 0 ||
                id_value->as_int() > int64_t(UINT32_MAX)) {
                result.status = Invalid("tokenizer special token invalid");
                return result;
            }
            uint32_t id = uint32_t(id_value->as_int());
            if (!t.special_tokens_.emplace(name, id).second) {
                result.status = Invalid("tokenizer special token duplicated");
                return result;
            }
            max_id = std::max(max_id, id);
        }
    }

    t.vocab_size_ = max_id + 1;
    t.id_to_bytes_.assign(t.vocab_size_, std::string());
    t.is_special_.assign(t.vocab_size_, false);

    std::vector<bool> id_used(t.vocab_size_, false);
    for (const auto& [token, id] : t.vocab_) {
        if (id_used[id]) {
            result.status = Invalid("tokenizer vocab id duplicated");
            return result;
        }
        id_used[id] = true;
        std::string bytes;
        if (!TokenStringToBytes(token, bytes)) {
            result.status = Invalid("tokenizer vocab token not byte-level");
            return result;
        }
        t.id_to_bytes_[id] = std::move(bytes);
    }
    for (const auto& [name, id] : t.special_tokens_) {
        if (id_used[id]) {
            result.status = Invalid("tokenizer special token id collides");
            return result;
        }
        id_used[id] = true;
        t.is_special_[id] = true;
        // Specials decode to their literal name (visible marker text).
        t.id_to_bytes_[id] = name;
    }

    // Base byte coverage: every single byte must be encodable.
    for (uint32_t b = 0; b < 256; ++b) {
        if (!t.vocab_.count(ByteTable().byte_to_str[b])) {
            result.status = Invalid("tokenizer vocab missing base byte token");
            return result;
        }
    }

    uint32_t rank = 0;
    for (const auto& merge : merges->as_array()) {
        if (!merge->is_string()) {
            result.status = Invalid("tokenizer merge entry invalid");
            return result;
        }
        const std::string& line = merge->as_string();
        size_t space = line.find(' ');
        if (space == std::string::npos || space == 0 ||
            space + 1 >= line.size() ||
            line.find(' ', space + 1) != std::string::npos) {
            result.status = Invalid("tokenizer merge format invalid");
            return result;
        }
        std::string left = line.substr(0, space);
        std::string right = line.substr(space + 1);
        // Merged result must exist in the vocab, otherwise encoding could
        // produce an unknown token.
        if (!t.vocab_.count(left) || !t.vocab_.count(right) ||
            !t.vocab_.count(left + right)) {
            result.status = Invalid("tokenizer merge references unknown token");
            return result;
        }
        auto key = std::make_pair(std::move(left), std::move(right));
        if (!t.merge_ranks_.emplace(std::move(key), rank).second) {
            result.status = Invalid("tokenizer merge duplicated");
            return result;
        }
        ++rank;
    }

    result.status = Status::Ok();
    return result;
}

Status BpeTokenizer::Encode(std::string_view text,
                            std::vector<uint32_t>& out) const {
    // Byte-level split: every input byte becomes one initial symbol.
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (unsigned char c : text) {
        symbols.push_back(ByteTable().byte_to_str[c]);
    }

    // Greedy BPE: repeatedly apply the lowest-rank adjacent merge.
    while (symbols.size() >= 2) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_index = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_ranks_.find({symbols[i], symbols[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_rank == std::numeric_limits<uint32_t>::max()) break;
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + long(best_index) + 1);
    }

    for (const auto& symbol : symbols) {
        auto it = vocab_.find(symbol);
        if (it == vocab_.end()) {
            return Status(ErrorCode::kInferenceFailed,
                          "tokenizer produced unknown symbol", kComponent);
        }
        out.push_back(it->second);
    }
    return Status::Ok();
}

Status BpeTokenizer::DecodeBytes(const std::vector<uint32_t>& ids,
                                 std::string& out) const {
    for (uint32_t id : ids) {
        if (id >= vocab_size_) {
            return Status(ErrorCode::kInferenceFailed,
                          "token id out of range", kComponent);
        }
        out += id_to_bytes_[id];
    }
    return Status::Ok();
}

Status BpeTokenizer::DecodeText(const std::vector<uint32_t>& ids,
                                std::string& out) const {
    std::string bytes;
    Status s = DecodeBytes(ids, bytes);
    if (!s.ok()) return s;

    // Replace invalid UTF-8 with U+FFFD (defined invalid-UTF-8 policy,
    // spec §13.1).
    size_t i = 0;
    while (i < bytes.size()) {
        unsigned char c = static_cast<unsigned char>(bytes[i]);
        size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2
                   : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        bool valid = len != 0 && i + len <= bytes.size();
        if (valid) {
            uint32_t cp = len == 1 ? c
                        : uint32_t(c & (0xFFu >> (len + 1)));
            for (size_t k = 1; k < len && valid; ++k) {
                unsigned char cc = static_cast<unsigned char>(bytes[i + k]);
                if ((cc & 0xC0) != 0x80) valid = false;
                cp = (cp << 6) | (cc & 0x3Fu);
            }
            if (valid) {
                if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
                    (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
                    (cp >= 0xD800 && cp <= 0xDFFF)) {
                    valid = false;
                }
            }
        }
        if (valid) {
            out.append(bytes, i, len);
            i += len;
        } else {
            out += "\xEF\xBF\xBD";  // U+FFFD
            i += 1;
        }
    }
    return Status::Ok();
}

std::optional<uint32_t> BpeTokenizer::SpecialTokenId(
    std::string_view name) const {
    auto it = special_tokens_.find(std::string(name));
    if (it == special_tokens_.end()) return std::nullopt;
    return it->second;
}

bool BpeTokenizer::IsSpecialToken(uint32_t id) const {
    return id < is_special_.size() && is_special_[id];
}

void SplitUtf8Boundary(std::string& out, std::string& carry) {
    carry.clear();
    if (out.empty()) return;
    // Walk back over at most 3 trailing continuation bytes.
    size_t i = out.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = static_cast<unsigned char>(out[i - 1]);
        if ((c & 0xC0) == 0x80) {
            --i;
            ++back;
            continue;
        }
        size_t expect = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2
                      : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (expect > back + 1) {
            // Incomplete sequence: hold it back.
            carry = out.substr(i - 1);
            out.resize(i - 1);
        }
        return;
    }
}

}  // namespace lykuro::nie
