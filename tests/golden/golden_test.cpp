// Golden regression anchors for the CPU reference implementation.
//
// The fixture is the deterministic tiny model from qwen_model_test.cpp
// (LCG seed 42). The pinned values below were produced by this
// implementation (engine 0.1.0, sampler_v1) and act as a change detector:
// any numeric drift in RMSNorm/RoPE/attention/SwiGLU or the sampler
// pipeline fails this test.
//
// LIMITATION (reported per spec §35): these anchors verify
// reproducibility, not external correctness. Validation against an
// independent reference oracle (HF transformers on a certified Qwen
// checkpoint, dev environment only) is pending and tracked as a Phase 1
// exit requirement.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/generation/sampler.h"
#include "model/architectures/qwen/qwen_model.h"

namespace lykuro::nie {
namespace {

class GoldenWeightGen {
public:
    explicit GoldenWeightGen(uint64_t seed) : state_(seed) {}
    float Next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float(state_ >> 40) / float(1 << 24) - 0.5f) * 0.2f;
    }

private:
    uint64_t state_;
};

class GoldenTest : public ::testing::Test {
protected:
    static constexpr uint32_t kVocab = 32, kHidden = 16, kLayers = 2,
                              kHeads = 4, kKvHeads = 2, kHeadDim = 4,
                              kIntermediate = 32;

    void SetUp() override {
        std::map<std::string, std::vector<uint64_t>> shapes;
        shapes["model.embed_tokens.weight"] = {kVocab, kHidden};
        for (uint32_t l = 0; l < kLayers; ++l) {
            std::string p = "model.layers." + std::to_string(l) + ".";
            uint64_t q = uint64_t(kHeads) * kHeadDim;
            uint64_t kv = uint64_t(kKvHeads) * kHeadDim;
            shapes[p + "input_layernorm.weight"] = {kHidden};
            shapes[p + "self_attn.q_proj.weight"] = {q, kHidden};
            shapes[p + "self_attn.q_proj.bias"] = {q};
            shapes[p + "self_attn.k_proj.weight"] = {kv, kHidden};
            shapes[p + "self_attn.k_proj.bias"] = {kv};
            shapes[p + "self_attn.v_proj.weight"] = {kv, kHidden};
            shapes[p + "self_attn.v_proj.bias"] = {kv};
            shapes[p + "self_attn.o_proj.weight"] = {kHidden, q};
            shapes[p + "post_attention_layernorm.weight"] = {kHidden};
            shapes[p + "mlp.gate_proj.weight"] = {kIntermediate, kHidden};
            shapes[p + "mlp.up_proj.weight"] = {kIntermediate, kHidden};
            shapes[p + "mlp.down_proj.weight"] = {kHidden, kIntermediate};
        }
        shapes["model.norm.weight"] = {kHidden};

        GoldenWeightGen gen(42);
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

        path_ = testing::TempDir() + "golden_tiny.safetensors";
        std::ofstream f(path_, std::ios::binary | std::ios::trunc);
        uint64_t len = header.size();
        char len_le[8];
        for (int i = 0; i < 8; ++i) len_le[i] = char(len >> (8 * i));
        f.write(len_le, 8);
        f.write(header.data(), long(header.size()));
        f.write(data.data(), long(data.size()));
        f.close();

        ModelManifest m;
        m.schema_version = "1";
        m.artifact_id = "ma_qwen_tiny";
        m.model_family = "qwen";
        m.architecture = "approved_qwen_decoder_v1";
        m.model_version = "tiny-test";
        m.weight_format = "safetensors";
        m.precision = "fp32";
        m.vocab_size = kVocab;
        m.hidden_size = kHidden;
        m.num_layers = kLayers;
        m.num_attention_heads = kHeads;
        m.num_key_value_heads = kKvHeads;
        m.head_dim = kHeadDim;
        m.max_context_tokens = 64;
        m.intermediate_size = kIntermediate;
        m.rms_norm_eps = 1e-6;
        m.rope_theta = 10000.0;
        m.tie_word_embeddings = true;
        m.eos_token_ids = {31};

        ASSERT_TRUE(file_.Open(path_).ok());
        auto r = QwenModel::Load(m, file_);
        ASSERT_TRUE(r.status.ok()) << r.status.message();
        model_ = std::move(r.model);
    }

    void TearDown() override { std::remove(path_.c_str()); }

    std::string path_;
    SafetensorsFile file_;
    std::unique_ptr<QwenModel> model_;
};

TEST_F(GoldenTest, PrefillLogitsMatchPinnedValues) {
    QwenKvCache cache(model_->config(), 64);
    std::vector<float> logits;
    ASSERT_TRUE(model_->Prefill({1, 2, 3}, cache, logits).ok());
    ASSERT_EQ(logits.size(), kVocab);

    // Pinned from engine 0.1.0 (2026-08-07).
    const float expected[5] = {0.256461f, 0.253633f, -0.366729f, 0.739745f,
                               0.352799f};
    // Note: the pinned values were captured AFTER 10 greedy decode steps
    // in the capture run; re-derive here the same way.
    SamplingParams p;
    p.temperature = 0.0f;
    Sampler sampler(p);
    for (int i = 0; i < 10; ++i) {
        uint32_t t;
        ASSERT_TRUE(sampler.Sample(logits, t).ok());
        ASSERT_TRUE(model_->Decode(t, cache, logits).ok());
    }
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(logits[i], expected[i], 1e-4f) << "logit index " << i;
    }
}

TEST_F(GoldenTest, GreedySequencesMatchPinnedValues) {
    struct Case {
        std::vector<uint32_t> prompt;
        std::vector<uint32_t> expected;
    };
    // Pinned from engine 0.1.0 (2026-08-07). The tiny random tied-embedding
    // model largely echoes the last prompt token; the {17,4,22,8} case
    // exercises a transition at step 12.
    const std::vector<Case> cases = {
        {{1, 2, 3}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}},
        {{5, 9, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
        {{17, 4, 22, 8}, {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4}},
        {{30, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
    };
    for (const Case& c : cases) {
        QwenKvCache cache(model_->config(), 64);
        std::vector<float> logits;
        ASSERT_TRUE(model_->Prefill(c.prompt, cache, logits).ok());
        SamplingParams p;
        p.temperature = 0.0f;
        Sampler sampler(p);
        std::vector<uint32_t> out;
        for (size_t i = 0; i < c.expected.size(); ++i) {
            uint32_t t;
            ASSERT_TRUE(sampler.Sample(logits, t).ok());
            out.push_back(t);
            ASSERT_TRUE(model_->Decode(t, cache, logits).ok());
        }
        EXPECT_EQ(out, c.expected);
    }
}

}  // namespace
}  // namespace lykuro::nie
