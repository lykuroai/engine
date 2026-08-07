#include "tests/testutil/tiny_model.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <map>
#include <vector>

namespace lykuro::nie::testutil {

namespace {

class WeightGen {
public:
    explicit WeightGen(uint64_t seed) : state_(seed) {}
    float Next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float(state_ >> 40) / float(1 << 24) - 0.5f) * 0.2f;
    }

private:
    uint64_t state_;
};

}  // namespace

ModelManifest MakeTinyManifest(const TinyModelSpec& spec) {
    ModelManifest m;
    m.schema_version = "1";
    m.artifact_id = "ma_qwen_tiny";
    m.model_family = "qwen";
    m.architecture = "approved_qwen_decoder_v1";
    m.model_version = "tiny-test";
    m.weight_format = "safetensors";
    m.precision = "fp32";
    m.vocab_size = spec.vocab_size;
    m.hidden_size = spec.hidden_size;
    m.num_layers = spec.num_layers;
    m.num_attention_heads = spec.num_heads;
    m.num_key_value_heads = spec.num_kv_heads;
    m.head_dim = spec.head_dim;
    m.max_context_tokens = spec.max_context_tokens;
    m.intermediate_size = spec.intermediate_size;
    m.rms_norm_eps = 1e-6;
    m.rope_theta = 10000.0;
    m.tie_word_embeddings = true;
    m.eos_token_ids = {spec.eos_token_id};
    return m;
}

std::string WriteTinyWeights(const TinyModelSpec& spec,
                             const std::string& filename) {
    std::map<std::string, std::vector<uint64_t>> shapes;
    shapes["model.embed_tokens.weight"] = {spec.vocab_size,
                                           spec.hidden_size};
    for (uint32_t l = 0; l < spec.num_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        uint64_t q = uint64_t(spec.num_heads) * spec.head_dim;
        uint64_t kv = uint64_t(spec.num_kv_heads) * spec.head_dim;
        shapes[p + "input_layernorm.weight"] = {spec.hidden_size};
        shapes[p + "self_attn.q_proj.weight"] = {q, spec.hidden_size};
        shapes[p + "self_attn.q_proj.bias"] = {q};
        shapes[p + "self_attn.k_proj.weight"] = {kv, spec.hidden_size};
        shapes[p + "self_attn.k_proj.bias"] = {kv};
        shapes[p + "self_attn.v_proj.weight"] = {kv, spec.hidden_size};
        shapes[p + "self_attn.v_proj.bias"] = {kv};
        shapes[p + "self_attn.o_proj.weight"] = {spec.hidden_size, q};
        shapes[p + "post_attention_layernorm.weight"] = {spec.hidden_size};
        shapes[p + "mlp.gate_proj.weight"] = {spec.intermediate_size,
                                              spec.hidden_size};
        shapes[p + "mlp.up_proj.weight"] = {spec.intermediate_size,
                                            spec.hidden_size};
        shapes[p + "mlp.down_proj.weight"] = {spec.hidden_size,
                                              spec.intermediate_size};
    }
    shapes["model.norm.weight"] = {spec.hidden_size};

    WeightGen gen(spec.seed);
    std::string header = "{";
    std::string data;
    uint64_t offset = 0;
    for (const auto& [name, shape] : shapes) {
        uint64_t count = 1;
        std::string shape_json = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            count *= shape[i];
            shape_json += (i ? "," : "") + std::to_string(shape[i]);
        }
        shape_json += "]";
        const bool is_norm = name.find("norm") != std::string::npos;
        for (uint64_t i = 0; i < count; ++i) {
            float v = is_norm ? 1.0f : gen.Next();
            char bytes[4];
            std::memcpy(bytes, &v, 4);
            data.append(bytes, 4);
        }
        if (header.size() > 1) header += ",";
        header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":" +
                  shape_json + ",\"data_offsets\":[" +
                  std::to_string(offset) + "," +
                  std::to_string(offset + count * 4) + "]}";
        offset += count * 4;
    }
    header += "}";

    std::string path = testing::TempDir() + filename;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    uint64_t len = header.size();
    char len_le[8];
    for (int i = 0; i < 8; ++i) len_le[i] = char(len >> (8 * i));
    f.write(len_le, 8);
    f.write(header.data(), long(header.size()));
    f.write(data.data(), long(data.size()));
    return path;
}

}  // namespace lykuro::nie::testutil
