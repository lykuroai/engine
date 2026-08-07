#include "core/engine/json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace lykuro::nie::json {

const Value* Value::Find(std::string_view key) const {
    if (type_ != Type::kObject) return nullptr;
    auto it = object_.find(std::string(key));
    return it == object_.end() ? nullptr : it->second.get();
}

ValuePtr Value::MakeNull() {
    return std::make_shared<Value>();
}
ValuePtr Value::MakeBool(bool v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kBool;
    p->bool_ = v;
    return p;
}
ValuePtr Value::MakeInt(int64_t v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kInt;
    p->int_ = v;
    return p;
}
ValuePtr Value::MakeDouble(double v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kDouble;
    p->double_ = v;
    return p;
}
ValuePtr Value::MakeString(std::string v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kString;
    p->string_ = std::move(v);
    return p;
}
ValuePtr Value::MakeArray(std::vector<ValuePtr> v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kArray;
    p->array_ = std::move(v);
    return p;
}
ValuePtr Value::MakeObject(std::map<std::string, ValuePtr> v) {
    auto p = std::make_shared<Value>();
    p->type_ = Type::kObject;
    p->object_ = std::move(v);
    return p;
}

namespace {

class Parser {
public:
    Parser(std::string_view input, size_t max_depth)
        : input_(input), max_depth_(max_depth) {}

    ParseResult Run() {
        SkipWhitespace();
        ValuePtr v = ParseValue(0);
        if (!v) return Fail();
        SkipWhitespace();
        if (pos_ != input_.size()) {
            return FailAt("trailing data after JSON value", pos_);
        }
        ParseResult r;
        r.value = std::move(v);
        return r;
    }

private:
    ParseResult Fail() {
        ParseResult r;
        r.error = error_;
        r.error_offset = error_offset_;
        return r;
    }
    ParseResult FailAt(const char* msg, size_t off) {
        SetError(msg, off);
        return Fail();
    }
    void SetError(const char* msg, size_t off) {
        if (error_.empty()) {
            error_ = msg;
            error_offset_ = off;
        }
    }

    void SkipWhitespace() {
        while (pos_ < input_.size()) {
            char c = input_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool Consume(char expected) {
        if (pos_ < input_.size() && input_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    ValuePtr ParseValue(size_t depth) {
        if (depth > max_depth_) {
            SetError("nesting depth limit exceeded", pos_);
            return nullptr;
        }
        if (pos_ >= input_.size()) {
            SetError("unexpected end of input", pos_);
            return nullptr;
        }
        char c = input_[pos_];
        switch (c) {
            case '{': return ParseObject(depth);
            case '[': return ParseArray(depth);
            case '"': return ParseString();
            case 't':
                if (input_.substr(pos_, 4) == "true") {
                    pos_ += 4;
                    return Value::MakeBool(true);
                }
                break;
            case 'f':
                if (input_.substr(pos_, 5) == "false") {
                    pos_ += 5;
                    return Value::MakeBool(false);
                }
                break;
            case 'n':
                if (input_.substr(pos_, 4) == "null") {
                    pos_ += 4;
                    return Value::MakeNull();
                }
                break;
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    return ParseNumber();
                }
        }
        SetError("invalid JSON token", pos_);
        return nullptr;
    }

    ValuePtr ParseObject(size_t depth) {
        ++pos_;  // '{'
        std::map<std::string, ValuePtr> members;
        SkipWhitespace();
        if (Consume('}')) return Value::MakeObject(std::move(members));
        while (true) {
            SkipWhitespace();
            if (pos_ >= input_.size() || input_[pos_] != '"') {
                SetError("expected object key string", pos_);
                return nullptr;
            }
            ValuePtr key = ParseString();
            if (!key) return nullptr;
            if (members.count(key->as_string())) {
                SetError("duplicate object key", pos_);
                return nullptr;
            }
            SkipWhitespace();
            if (!Consume(':')) {
                SetError("expected ':' in object", pos_);
                return nullptr;
            }
            SkipWhitespace();
            ValuePtr val = ParseValue(depth + 1);
            if (!val) return nullptr;
            members.emplace(key->as_string(), std::move(val));
            SkipWhitespace();
            if (Consume(',')) continue;
            if (Consume('}')) return Value::MakeObject(std::move(members));
            SetError("expected ',' or '}' in object", pos_);
            return nullptr;
        }
    }

    ValuePtr ParseArray(size_t depth) {
        ++pos_;  // '['
        std::vector<ValuePtr> items;
        SkipWhitespace();
        if (Consume(']')) return Value::MakeArray(std::move(items));
        while (true) {
            SkipWhitespace();
            ValuePtr val = ParseValue(depth + 1);
            if (!val) return nullptr;
            items.push_back(std::move(val));
            SkipWhitespace();
            if (Consume(',')) continue;
            if (Consume(']')) return Value::MakeArray(std::move(items));
            SetError("expected ',' or ']' in array", pos_);
            return nullptr;
        }
    }

    // Appends the UTF-8 encoding of `cp` to `out`.
    static void AppendCodepoint(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(char(cp));
        } else if (cp < 0x800) {
            out.push_back(char(0xC0 | (cp >> 6)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(char(0xE0 | (cp >> 12)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(char(0xF0 | (cp >> 18)));
            out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        }
    }

    bool ParseHex4(uint32_t& out) {
        if (pos_ + 4 > input_.size()) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = input_[pos_ + i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else return false;
        }
        pos_ += 4;
        out = v;
        return true;
    }

    // Validates one UTF-8 sequence starting at pos_; appends it to out.
    bool ConsumeUtf8(std::string& out) {
        unsigned char c0 = static_cast<unsigned char>(input_[pos_]);
        size_t len;
        uint32_t cp;
        if (c0 < 0x80) { len = 1; cp = c0; }
        else if ((c0 & 0xE0) == 0xC0) { len = 2; cp = c0 & 0x1Fu; }
        else if ((c0 & 0xF0) == 0xE0) { len = 3; cp = c0 & 0x0Fu; }
        else if ((c0 & 0xF8) == 0xF0) { len = 4; cp = c0 & 0x07u; }
        else return false;
        if (pos_ + len > input_.size()) return false;
        for (size_t i = 1; i < len; ++i) {
            unsigned char cc = static_cast<unsigned char>(input_[pos_ + i]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        // Reject overlong encodings, surrogates, and out-of-range values.
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
            (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
            (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        out.append(input_.substr(pos_, len));
        pos_ += len;
        return true;
    }

    ValuePtr ParseString() {
        ++pos_;  // '"'
        std::string out;
        while (true) {
            if (pos_ >= input_.size()) {
                SetError("unterminated string", pos_);
                return nullptr;
            }
            char c = input_[pos_];
            if (c == '"') {
                ++pos_;
                return Value::MakeString(std::move(out));
            }
            if (c == '\\') {
                ++pos_;
                if (pos_ >= input_.size()) {
                    SetError("unterminated escape", pos_);
                    return nullptr;
                }
                char e = input_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp;
                        if (!ParseHex4(cp)) {
                            SetError("invalid \\u escape", pos_);
                            return nullptr;
                        }
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // Expect a low surrogate.
                            if (pos_ + 2 > input_.size() ||
                                input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                                SetError("unpaired surrogate", pos_);
                                return nullptr;
                            }
                            pos_ += 2;
                            uint32_t lo;
                            if (!ParseHex4(lo) || lo < 0xDC00 || lo > 0xDFFF) {
                                SetError("invalid low surrogate", pos_);
                                return nullptr;
                            }
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            SetError("unpaired surrogate", pos_);
                            return nullptr;
                        }
                        AppendCodepoint(out, cp);
                        break;
                    }
                    default:
                        SetError("invalid escape character", pos_);
                        return nullptr;
                }
                continue;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                SetError("unescaped control character in string", pos_);
                return nullptr;
            }
            if (!ConsumeUtf8(out)) {
                SetError("invalid UTF-8 in string", pos_);
                return nullptr;
            }
        }
    }

    ValuePtr ParseNumber() {
        size_t start = pos_;
        if (Consume('-')) {}
        if (pos_ >= input_.size()) {
            SetError("invalid number", pos_);
            return nullptr;
        }
        if (input_[pos_] == '0') {
            ++pos_;
        } else if (input_[pos_] >= '1' && input_[pos_] <= '9') {
            while (pos_ < input_.size() && input_[pos_] >= '0' &&
                   input_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            SetError("invalid number", pos_);
            return nullptr;
        }
        bool is_integer = true;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            is_integer = false;
            ++pos_;
            if (pos_ >= input_.size() || input_[pos_] < '0' ||
                input_[pos_] > '9') {
                SetError("invalid number fraction", pos_);
                return nullptr;
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' &&
                   input_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < input_.size() &&
            (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            is_integer = false;
            ++pos_;
            if (pos_ < input_.size() &&
                (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= input_.size() || input_[pos_] < '0' ||
                input_[pos_] > '9') {
                SetError("invalid number exponent", pos_);
                return nullptr;
            }
            while (pos_ < input_.size() && input_[pos_] >= '0' &&
                   input_[pos_] <= '9') {
                ++pos_;
            }
        }

        std::string text(input_.substr(start, pos_ - start));
        if (is_integer) {
            errno = 0;
            char* end = nullptr;
            long long v = std::strtoll(text.c_str(), &end, 10);
            if (errno == 0 && end == text.c_str() + text.size()) {
                return Value::MakeInt(int64_t(v));
            }
            // Falls through to double for out-of-range integers.
        }
        errno = 0;
        char* end = nullptr;
        double d = std::strtod(text.c_str(), &end);
        if (errno != 0 || end != text.c_str() + text.size() ||
            !std::isfinite(d)) {
            SetError("number out of range", start);
            return nullptr;
        }
        return Value::MakeDouble(d);
    }

    std::string_view input_;
    size_t pos_ = 0;
    size_t max_depth_;
    std::string error_;
    size_t error_offset_ = 0;
};

}  // namespace

ParseResult Parse(std::string_view input, size_t max_depth) {
    return Parser(input, max_depth).Run();
}

}  // namespace lykuro::nie::json
