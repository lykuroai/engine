#include "tests/unit/tokenizer_fixture.h"

#include <array>
#include <cstdint>

namespace lykuro::nie::testfixture {

namespace {

// Mirrors the GPT-2 byte-to-unicode mapping used by the tokenizer.
std::array<std::string, 256> ByteAlphabet() {
    auto printable = [](uint32_t b) {
        return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
               (b >= 0xAE && b <= 0xFF);
    };
    std::array<std::string, 256> out;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        uint32_t c = printable(b) ? b : 256 + n++;
        std::string s;
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
        out[b] = s;
    }
    return out;
}

void AppendJsonString(std::string& out, const std::string& value) {
    out.push_back('"');
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (static_cast<unsigned char>(c) < 0x20) {
            static const char kHex[] = "0123456789abcdef";
            out += "\\u00";
            out.push_back(kHex[(c >> 4) & 0xF]);
            out.push_back(kHex[c & 0xF]);
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
}

}  // namespace

std::string SmallTokenizerConfig() {
    auto alphabet = ByteAlphabet();
    std::string vocab;
    for (uint32_t b = 0; b < 256; ++b) {
        if (!vocab.empty()) vocab += ",";
        AppendJsonString(vocab, alphabet[b]);
        vocab += ":" + std::to_string(b);
    }
    for (const auto& [token, id] :
         {std::pair<std::string, int>{"he", 260},
          {"ll", 261},
          {"hell", 262},
          {"hello", 263}}) {
        vocab += ",";
        AppendJsonString(vocab, token);
        vocab += ":" + std::to_string(id);
    }

    std::string config = "{\"tokenizer_type\":\"approved_qwen_tokenizer_v1\",";
    config += "\"vocab\":{" + vocab + "},";
    config +=
        "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"],";
    config +=
        "\"special_tokens\":{\"<|im_start|>\":300,\"<|im_end|>\":301,"
        "\"<|endoftext|>\":302}}";
    return config;
}

}  // namespace lykuro::nie::testfixture
