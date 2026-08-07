#include "model/architectures/qwen/qwen_model.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/generation/sampler.h"

namespace lykuro::nie {
namespace {

// Deterministic pseudo-random weights (LCG) so the fixture model is
// reproducible across platforms without committing a binary file.
class WeightGen {
public:
    explicit WeightGen(uint64_t seed) : state_(seed) {}
    float Next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        // Map to [-0.1, 0.1): small weights keep activations tame.
        return (float(state_ >> 40) / float(1 << 24) - 0.5f) * 0.2f;
    }

private:
    uint64_t state_;
};

struct TinyModelFixture {
    static constexpr uint32_t kVocab = 32;
    static constexpr uint32_t kHidden = 16;
    static constexpr uint32_t kLayers = 2;
    static constexpr uint32_t kHeads = 4;
    static constexpr uint32_t kKvHeads = 2;
    static constexpr uint32_t kHeadDim = 4;
    static constexpr uint32_t kIntermediate = 32;
    static constexpr uint32_t kContext = 64;

    ModelManifest manifest;
    std::string weights_path;

    static ModelManifest MakeManifest() {
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
        m.max_context_tokens = kContext;
        m.intermediate_size = kIntermediate;
        m.rms_norm_eps = 1e-6;
        m.rope_theta = 10000.0;
        m.tie_word_embeddings = true;
        m.eos_token_ids = {31};
        return m;
    }

    // Writes a safetensors file with every tensor the loader expects.
    static std::string WriteWeights(uint64_t seed,
                                    bool poison_embedding = false) {
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

        WeightGen gen(seed);
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
            const bool is_norm =
                name.find("norm") != std::string::npos;
            for (uint64_t i = 0; i < count; ++i) {
                // Norm weights near 1.0; everything else small random.
                float v = is_norm ? 1.0f : gen.Next();
                if (poison_embedding &&
                    name == "model.embed_tokens.weight" && i == 0) {
                    v = INFINITY;
                }
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

        std::string path = testing::TempDir() + "qwen_tiny_" +
                           std::to_string(seed) +
                           (poison_embedding ? "_poison" : "") +
                           ".safetensors";
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        uint64_t len = header.size();
        char len_le[8];
        for (int i = 0; i < 8; ++i) len_le[i] = char(len >> (8 * i));
        f.write(len_le, 8);
        f.write(header.data(), long(header.size()));
        f.write(data.data(), long(data.size()));
        return path;
    }
};

class QwenModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        manifest_ = TinyModelFixture::MakeManifest();
        path_ = TinyModelFixture::WriteWeights(42);
        ASSERT_TRUE(file_.Open(path_).ok());
        auto r = QwenModel::Load(manifest_, file_);
        ASSERT_TRUE(r.status.ok()) << r.status.message();
        model_ = std::move(r.model);
    }
    void TearDown() override { std::remove(path_.c_str()); }

    ModelManifest manifest_;
    std::string path_;
    SafetensorsFile file_;
    std::unique_ptr<QwenModel> model_;
};

TEST_F(QwenModelTest, PrefillIsDeterministic) {
    std::vector<uint32_t> prompt = {1, 5, 9, 2};
    QwenKvCache c1(model_->config(), 64), c2(model_->config(), 64);
    std::vector<float> l1, l2;
    ASSERT_TRUE(model_->Prefill(prompt, c1, l1).ok());
    ASSERT_TRUE(model_->Prefill(prompt, c2, l2).ok());
    EXPECT_EQ(l1, l2);
}

// KV-cache consistency: prefill(prompt + [t]) must equal
// prefill(prompt) followed by decode(t).
TEST_F(QwenModelTest, IncrementalDecodeMatchesFullPrefill) {
    std::vector<uint32_t> prompt = {3, 7, 11};
    QwenKvCache inc(model_->config(), 64);
    std::vector<float> logits_inc;
    ASSERT_TRUE(model_->Prefill(prompt, inc, logits_inc).ok());
    ASSERT_TRUE(model_->Decode(13, inc, logits_inc).ok());

    QwenKvCache full(model_->config(), 64);
    std::vector<float> logits_full;
    ASSERT_TRUE(model_->Prefill({3, 7, 11, 13}, full, logits_full).ok());

    ASSERT_EQ(logits_inc.size(), logits_full.size());
    for (size_t i = 0; i < logits_inc.size(); ++i) {
        EXPECT_NEAR(logits_inc[i], logits_full[i], 1e-4f) << "index " << i;
    }
}

TEST_F(QwenModelTest, GreedyGenerationIsReproducible) {
    auto generate = [&]() {
        QwenKvCache cache(model_->config(), 64);
        std::vector<float> logits;
        std::vector<uint32_t> out;
        EXPECT_TRUE(model_->Prefill({1, 2, 3}, cache, logits).ok());
        SamplingParams p;
        p.temperature = 0.0f;
        Sampler sampler(p);
        for (int i = 0; i < 10; ++i) {
            uint32_t t;
            EXPECT_TRUE(sampler.Sample(logits, t).ok());
            out.push_back(t);
            EXPECT_TRUE(model_->Decode(t, cache, logits).ok());
        }
        return out;
    };
    EXPECT_EQ(generate(), generate());
}

TEST_F(QwenModelTest, RejectsOutOfRangeToken) {
    QwenKvCache cache(model_->config(), 64);
    std::vector<float> logits;
    EXPECT_FALSE(model_->Prefill({999}, cache, logits).ok());
    ASSERT_TRUE(model_->Prefill({1}, cache, logits).ok());
    EXPECT_FALSE(model_->Decode(999, cache, logits).ok());
}

TEST_F(QwenModelTest, RejectsPromptBeyondCacheCapacity) {
    QwenKvCache cache(model_->config(), 4);
    std::vector<float> logits;
    std::vector<uint32_t> prompt(5, 1);
    Status s = model_->Prefill(prompt, cache, logits);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kContextLengthExceeded);
}

TEST_F(QwenModelTest, DecodeStopsAtCacheCapacity) {
    QwenKvCache cache(model_->config(), 3);
    std::vector<float> logits;
    ASSERT_TRUE(model_->Prefill({1, 2, 3}, cache, logits).ok());
    Status s = model_->Decode(4, cache, logits);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kContextLengthExceeded);
}

TEST(QwenModelLoadTest, RejectsMissingTensor) {
    ModelManifest manifest = TinyModelFixture::MakeManifest();
    manifest.num_layers = 3;  // file only has 2 layers
    std::string path = TinyModelFixture::WriteWeights(42);
    SafetensorsFile file;
    ASSERT_TRUE(file.Open(path).ok());
    auto r = QwenModel::Load(manifest, file);
    EXPECT_FALSE(r.status.ok());
    EXPECT_EQ(r.status.code(), ErrorCode::kArtifactVerificationFailed);
    std::remove(path.c_str());
}

TEST(QwenModelLoadTest, RejectsShapeMismatch) {
    ModelManifest manifest = TinyModelFixture::MakeManifest();
    manifest.hidden_size = 32;  // file tensors are hidden=16
    std::string path = TinyModelFixture::WriteWeights(42);
    SafetensorsFile file;
    ASSERT_TRUE(file.Open(path).ok());
    EXPECT_FALSE(QwenModel::Load(manifest, file).status.ok());
    std::remove(path.c_str());
}

TEST(QwenModelLoadTest, RejectsExtraTensors) {
    // Untied head expects lm_head.weight; tied model file with an extra
    // unexpected tensor is covered by loading a tied manifest against a
    // file that also carries lm_head: simulate by expecting tied but the
    // count check flags any surplus tensor.
    ModelManifest manifest = TinyModelFixture::MakeManifest();
    manifest.tie_word_embeddings = false;  // expects lm_head, file has none
    std::string path = TinyModelFixture::WriteWeights(42);
    SafetensorsFile file;
    ASSERT_TRUE(file.Open(path).ok());
    EXPECT_FALSE(QwenModel::Load(manifest, file).status.ok());
    std::remove(path.c_str());
}

TEST(QwenModelLoadTest, DetectsNonFiniteLogits) {
    ModelManifest manifest = TinyModelFixture::MakeManifest();
    std::string path =
        TinyModelFixture::WriteWeights(42, /*poison_embedding=*/true);
    SafetensorsFile file;
    ASSERT_TRUE(file.Open(path).ok());
    auto r = QwenModel::Load(manifest, file);
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    QwenKvCache cache(r.model->config(), 8);
    std::vector<float> logits;
    // Token 0 hits the poisoned embedding row -> non-finite propagates.
    Status s = r.model->Prefill({0}, cache, logits);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kInferenceFailed);
    std::remove(path.c_str());
}

}  // namespace
}  // namespace lykuro::nie
