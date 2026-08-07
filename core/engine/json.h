#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lykuro::nie::json {

// Strict, self-contained JSON parser used for untrusted artifact input
// (manifest, safetensors header, tokenizer config). Design constraints:
//  - no recursion beyond a fixed depth limit (stack-safety on hostile input)
//  - no duplicate object keys (rejected, never last-wins)
//  - full UTF-8 validation of string values
//  - numbers preserved as int64 when exactly representable
// Input size limits are enforced by callers before parsing.

class Value;
using ValuePtr = std::shared_ptr<Value>;

enum class Type { kNull, kBool, kInt, kDouble, kString, kArray, kObject };

class Value {
public:
    Type type() const { return type_; }

    bool is_null() const { return type_ == Type::kNull; }
    bool is_bool() const { return type_ == Type::kBool; }
    bool is_int() const { return type_ == Type::kInt; }
    bool is_number() const { return type_ == Type::kInt || type_ == Type::kDouble; }
    bool is_string() const { return type_ == Type::kString; }
    bool is_array() const { return type_ == Type::kArray; }
    bool is_object() const { return type_ == Type::kObject; }

    bool as_bool() const { return bool_; }
    int64_t as_int() const { return int_; }
    double as_double() const { return type_ == Type::kInt ? double(int_) : double_; }
    const std::string& as_string() const { return string_; }
    const std::vector<ValuePtr>& as_array() const { return array_; }
    // Ordered map: preserves no insertion order, but iteration is
    // deterministic, which keeps validation output stable.
    const std::map<std::string, ValuePtr>& as_object() const { return object_; }

    // Returns nullptr when missing (object type only).
    const Value* Find(std::string_view key) const;

    static ValuePtr MakeNull();
    static ValuePtr MakeBool(bool v);
    static ValuePtr MakeInt(int64_t v);
    static ValuePtr MakeDouble(double v);
    static ValuePtr MakeString(std::string v);
    static ValuePtr MakeArray(std::vector<ValuePtr> v);
    static ValuePtr MakeObject(std::map<std::string, ValuePtr> v);

private:
    Type type_ = Type::kNull;
    bool bool_ = false;
    int64_t int_ = 0;
    double double_ = 0.0;
    std::string string_;
    std::vector<ValuePtr> array_;
    std::map<std::string, ValuePtr> object_;
};

struct ParseResult {
    ValuePtr value;       // null on failure
    std::string error;    // static-ish description, no input echoed
    size_t error_offset = 0;

    bool ok() const { return value != nullptr; }
};

// max_depth bounds nesting of arrays/objects.
ParseResult Parse(std::string_view input, size_t max_depth = 64);

}  // namespace lykuro::nie::json
