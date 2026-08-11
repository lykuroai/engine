#include "model/convert/hf_convert.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/engine/json.h"
#include "security/sha256.h"

namespace lykuro::nie {
namespace {

namespace fs = std::filesystem;
constexpr const char kComponent[] = "hf_convert";

Status Err(const std::string& msg) {
    return Status(ErrorCode::kInvalidRequest, msg, kComponent);
}

Status ReadFile(const fs::path& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return Err("cannot open " + p.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return Status::Ok();
}

Status CopyFile(const fs::path& src, const fs::path& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) return Err("cannot open " + src.string());
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) return Err("cannot write " + dst.string());
    char buf[1 << 20];
    while (in) {
        in.read(buf, sizeof(buf));
        out.write(buf, in.gcount());
    }
    if (!out) return Err("write failed: " + dst.string());
    return Status::Ok();
}

// Minimal JSON string escaper (input is already valid UTF-8; emit multibyte
// bytes verbatim, escape the mandatory characters + control codes).
void EscapeInto(const std::string& s, std::string& out) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char u[8];
                    std::snprintf(u, sizeof(u), "\\u%04x", c);
                    out += u;
                } else {
                    out.push_back(char(c));
                }
        }
    }
    out.push_back('"');
}

const json::Value* Req(const json::Value* obj, const char* key, Status& st) {
    if (!st.ok()) return nullptr;
    const json::Value* v = obj ? obj->Find(key) : nullptr;
    if (v == nullptr) st = Err(std::string("missing key: ") + key);
    return v;
}

void CollectEos(const json::Value* v, std::set<int64_t>& eos) {
    if (v == nullptr) return;
    if (v->is_int()) {
        eos.insert(v->as_int());
    } else if (v->is_array()) {
        for (const auto& e : v->as_array())
            if (e->is_int()) eos.insert(e->as_int());
    }
}

}  // namespace

Status ConvertHfQwen(const std::string& hf_dir_s,
                     const std::string& out_dir_s) {
    fs::path hf = hf_dir_s, out = out_dir_s;

    // --- config.json
    std::string config_text;
    Status s = ReadFile(hf / "config.json", config_text);
    if (!s.ok()) return s;
    json::ParseResult cfg = json::Parse(config_text);
    if (!cfg.ok()) return Err("config.json parse: " + cfg.error);
    const json::Value* c = cfg.value.get();
    if (!c->is_object()) return Err("config.json not an object");

    const json::Value* mt = c->Find("model_type");
    if (mt == nullptr || !mt->is_string() || mt->as_string() != "qwen2")
        return Err("unsupported model_type (only qwen2)");

    Status st = Status::Ok();
    auto geti = [&](const char* k) -> int64_t {
        const json::Value* v = Req(c, k, st);
        return (v && v->is_number()) ? v->as_int() : 0;
    };
    auto getd = [&](const char* k, double dflt) -> double {
        const json::Value* v = c->Find(k);
        return (v && v->is_number()) ? v->as_double() : dflt;
    };
    const int64_t hidden = geti("hidden_size");
    const int64_t heads = geti("num_attention_heads");
    const int64_t layers = geti("num_hidden_layers");
    const int64_t kv_heads = geti("num_key_value_heads");
    const int64_t vocab_size = geti("vocab_size");
    const int64_t max_ctx = geti("max_position_embeddings");
    const int64_t inter = geti("intermediate_size");
    const double eps = getd("rms_norm_eps", 1e-6);
    const double rope = getd("rope_theta", 10000.0);
    if (!st.ok()) return st;
    const json::Value* tie_v = c->Find("tie_word_embeddings");
    const bool tie = tie_v && tie_v->is_bool() && tie_v->as_bool();

    std::set<int64_t> eos;
    CollectEos(c->Find("eos_token_id"), eos);

    // --- generation_config.json (optional eos)
    std::string gen_text;
    if (ReadFile(hf / "generation_config.json", gen_text).ok()) {
        json::ParseResult g = json::Parse(gen_text);
        if (g.ok() && g.value->is_object())
            CollectEos(g.value->Find("eos_token_id"), eos);
    }

    // --- tokenizer.json -> Lykuro tokenizer format
    std::string tok_text;
    s = ReadFile(hf / "tokenizer.json", tok_text);
    if (!s.ok()) return s;
    json::ParseResult tk = json::Parse(tok_text, 128);
    if (!tk.ok()) return Err("tokenizer.json parse: " + tk.error);
    const json::Value* model = tk.value->Find("model");
    if (model == nullptr || !model->is_object())
        return Err("tokenizer.json: no model");
    const json::Value* mtype = model->Find("type");
    if (mtype == nullptr || !mtype->is_string() || mtype->as_string() != "BPE")
        return Err("tokenizer.json: expected byte-level BPE");
    const json::Value* vocab = model->Find("vocab");
    const json::Value* merges = model->Find("merges");
    if (vocab == nullptr || !vocab->is_object() || merges == nullptr ||
        !merges->is_array())
        return Err("tokenizer.json: missing vocab/merges");

    std::string tok_out = "{\"tokenizer_type\":\"approved_qwen_tokenizer_v1\",";
    // vocab
    tok_out += "\"vocab\":{";
    bool first = true;
    for (const auto& [k, v] : vocab->as_object()) {
        if (!v->is_int()) continue;
        if (!first) tok_out.push_back(',');
        first = false;
        EscapeInto(k, tok_out);
        tok_out.push_back(':');
        tok_out += std::to_string(v->as_int());
    }
    tok_out += "},";
    // merges: array of "a b" strings, or array of [a,b] pairs
    tok_out += "\"merges\":[";
    first = true;
    for (const auto& m : merges->as_array()) {
        std::string merge_str;
        if (m->is_string()) {
            merge_str = m->as_string();
        } else if (m->is_array() && m->as_array().size() == 2 &&
                   m->as_array()[0]->is_string() &&
                   m->as_array()[1]->is_string()) {
            merge_str = m->as_array()[0]->as_string() + " " +
                        m->as_array()[1]->as_string();
        } else {
            return Err("tokenizer.json: bad merges entry");
        }
        if (!first) tok_out.push_back(',');
        first = false;
        EscapeInto(merge_str, tok_out);
    }
    tok_out += "],";
    // special_tokens from added_tokens[].special
    tok_out += "\"special_tokens\":{";
    first = true;
    const json::Value* added = tk.value->Find("added_tokens");
    if (added && added->is_array()) {
        for (const auto& a : added->as_array()) {
            if (!a->is_object()) continue;
            const json::Value* sp = a->Find("special");
            const json::Value* content = a->Find("content");
            const json::Value* id = a->Find("id");
            if (sp && sp->is_bool() && sp->as_bool() && content &&
                content->is_string() && id && id->is_int()) {
                if (!first) tok_out.push_back(',');
                first = false;
                EscapeInto(content->as_string(), tok_out);
                tok_out.push_back(':');
                tok_out += std::to_string(id->as_int());
            }
        }
    }
    tok_out += "}}";

    // --- write outputs
    std::error_code ec;
    fs::create_directories(out / "weights", ec);
    fs::create_directories(out / "config", ec);

    s = CopyFile(hf / "model.safetensors", out / "weights" / "model.safetensors");
    if (!s.ok()) return s;
    {
        std::ofstream tf(out / "config" / "tokenizer.json", std::ios::binary);
        if (!tf) return Err("cannot write tokenizer.json");
        tf << tok_out;
    }

    // checksums
    auto sha_of = [](const fs::path& p, std::string& hex) -> Status {
        std::string data;
        Status r = ReadFile(p, data);
        if (!r.ok()) return r;
        hex = Sha256::HexDigest(data.data(), data.size());
        return Status::Ok();
    };
    std::string w_sha, t_sha;
    s = sha_of(out / "weights" / "model.safetensors", w_sha);
    if (!s.ok()) return s;
    s = sha_of(out / "config" / "tokenizer.json", t_sha);
    if (!s.ok()) return s;
    const auto w_size =
        fs::file_size(out / "weights" / "model.safetensors", ec);
    const auto t_size = fs::file_size(out / "config" / "tokenizer.json", ec);

    // --- manifest.json
    std::string eos_arr;
    {
        bool f = true;
        for (int64_t e : eos) {
            if (!f) eos_arr.push_back(',');
            f = false;
            eos_arr += std::to_string(e);
        }
    }
    const int64_t head_dim = heads > 0 ? hidden / heads : 0;
    std::ostringstream mf;
    mf << "{\n"
       << " \"schema_version\": \"1\",\n"
       << " \"artifact_id\": \"ma_qwen2_converted\",\n"
       << " \"model_family\": \"qwen\",\n"
       << " \"architecture\": \"approved_qwen_decoder_v1\",\n"
       << " \"model_version\": \"qwen2\",\n"
       << " \"weight_format\": \"safetensors\",\n"
       << " \"precision\": \"bf16\",\n"
       << " \"vocab_size\": " << vocab_size << ",\n"
       << " \"hidden_size\": " << hidden << ",\n"
       << " \"num_layers\": " << layers << ",\n"
       << " \"num_attention_heads\": " << heads << ",\n"
       << " \"num_key_value_heads\": " << kv_heads << ",\n"
       << " \"head_dim\": " << head_dim << ",\n"
       << " \"max_context_tokens\": " << max_ctx << ",\n"
       << " \"intermediate_size\": " << inter << ",\n"
       << " \"rms_norm_eps\": " << eps << ",\n"
       << " \"rope_theta\": " << rope << ",\n"
       << " \"tie_word_embeddings\": " << (tie ? "true" : "false") << ",\n"
       << " \"eos_token_ids\": [" << eos_arr << "],\n"
       << " \"tokenizer_type\": \"approved_qwen_tokenizer_v1\",\n"
       << " \"chat_template_id\": \"qwen_chat_v1\",\n"
       << " \"files\": [\n"
       << "  { \"path\": \"weights/model.safetensors\", \"sha256\": \""
       << w_sha << "\", \"size_bytes\": " << w_size << " },\n"
       << "  { \"path\": \"config/tokenizer.json\", \"sha256\": \"" << t_sha
       << "\", \"size_bytes\": " << t_size << " }\n"
       << " ],\n"
       << " \"license_review_id\": \"lic_dev_converted\",\n"
       << " \"certified_profiles\": [],\n"
       << " \"created_at\": \"1970-01-01T00:00:00Z\"\n"
       << "}\n";
    {
        std::ofstream m(out / "manifest.json", std::ios::binary);
        if (!m) return Err("cannot write manifest.json");
        m << mf.str();
    }
    return Status::Ok();
}

}  // namespace lykuro::nie
