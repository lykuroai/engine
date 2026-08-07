#include "model/architectures/qwen/qwen_model.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_model";

Status VerifyFailed(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

// out[o] = bias[o] + sum_i weight[o, i] * x[i]   (weight row-major [out, in])
void MatVec(const std::vector<float>& weight, const float* bias,
            const float* x, size_t in_dim, size_t out_dim, float* out) {
    for (size_t o = 0; o < out_dim; ++o) {
        const float* row = weight.data() + o * in_dim;
        float acc = bias != nullptr ? bias[o] : 0.0f;
        for (size_t i = 0; i < in_dim; ++i) {
            acc += row[i] * x[i];
        }
        out[o] = acc;
    }
}

void RmsNorm(const float* x, const std::vector<float>& weight, float eps,
             size_t dim, float* out) {
    float ss = 0.0f;
    for (size_t i = 0; i < dim; ++i) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / float(dim) + eps);
    for (size_t i = 0; i < dim; ++i) out[i] = x[i] * scale * weight[i];
}

// Rotate-half RoPE, applied in place to one head vector of size head_dim.
void ApplyRope(float* vec, uint32_t head_dim, uint32_t pos, float theta) {
    const uint32_t half = head_dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        float freq = std::pow(theta, -2.0f * float(i) / float(head_dim));
        float angle = float(pos) * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float a = vec[i];
        float b = vec[i + half];
        vec[i] = a * c - b * s;
        vec[i + half] = b * c + a * s;
    }
}

// Loads a tensor into FP32, verifying dtype (matches manifest precision)
// and exact shape.
Status LoadTensor(const SafetensorsFile& file, const std::string& name,
                  const std::vector<uint64_t>& expected_shape,
                  std::vector<float>& out) {
    const TensorInfo* info = file.FindTensor(name);
    if (info == nullptr) {
        return VerifyFailed("expected weight tensor missing");
    }
    if (info->shape != expected_shape) {
        return VerifyFailed("weight tensor shape mismatch");
    }
    const uint8_t* data = file.TensorData(name);
    out.resize(info->element_count);
    switch (info->dtype) {
        case Dtype::kF32:
            std::memcpy(out.data(), data, info->data_size);
            break;
        case Dtype::kBf16:
            Bf16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             out.data(), info->element_count);
            break;
        case Dtype::kF16:
            Fp16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             out.data(), info->element_count);
            break;
    }
    return Status::Ok();
}

}  // namespace

Status QwenConfig::FromManifest(const ModelManifest& manifest,
                                QwenConfig& out) {
    out.vocab_size = manifest.vocab_size;
    out.hidden_size = manifest.hidden_size;
    out.num_layers = manifest.num_layers;
    out.num_heads = manifest.num_attention_heads;
    out.num_kv_heads = manifest.num_key_value_heads;
    out.head_dim = manifest.head_dim;
    out.intermediate_size = manifest.intermediate_size;
    out.max_context_tokens = manifest.max_context_tokens;
    out.rms_norm_eps = manifest.rms_norm_eps > 0
                           ? float(manifest.rms_norm_eps)
                           : 1e-6f;
    out.rope_theta =
        manifest.rope_theta > 0 ? float(manifest.rope_theta) : 10000.0f;
    out.tie_word_embeddings = manifest.tie_word_embeddings;
    out.eos_token_ids = manifest.eos_token_ids;

    if (out.intermediate_size == 0) {
        return VerifyFailed("manifest missing intermediate_size");
    }
    if (out.head_dim % 2 != 0) {
        return VerifyFailed("head_dim must be even for RoPE");
    }
    if (out.num_heads % out.num_kv_heads != 0) {
        return VerifyFailed("attention head configuration inconsistent");
    }
    if (out.eos_token_ids.empty()) {
        return VerifyFailed("manifest missing eos_token_ids");
    }
    for (uint32_t id : out.eos_token_ids) {
        if (id >= out.vocab_size) {
            return VerifyFailed("eos token id out of vocab range");
        }
    }
    return Status::Ok();
}

QwenKvCache::QwenKvCache(const QwenConfig& config, uint32_t max_tokens)
    : kv_stride_(config.num_kv_heads * config.head_dim),
      max_tokens_(max_tokens),
      keys_(config.num_layers),
      values_(config.num_layers) {
    for (uint32_t l = 0; l < config.num_layers; ++l) {
        keys_[l].assign(size_t(max_tokens_) * kv_stride_, 0.0f);
        values_[l].assign(size_t(max_tokens_) * kv_stride_, 0.0f);
    }
}

float* QwenKvCache::KeyAt(uint32_t layer, uint32_t token_index) {
    return keys_[layer].data() + size_t(token_index) * kv_stride_;
}

float* QwenKvCache::ValueAt(uint32_t layer, uint32_t token_index) {
    return values_[layer].data() + size_t(token_index) * kv_stride_;
}

QwenModel::LoadResult QwenModel::Load(const ModelManifest& manifest,
                                      const SafetensorsFile& weights) {
    LoadResult result;
    auto model = std::make_unique<QwenModel>();

    result.status = QwenConfig::FromManifest(manifest, model->config_);
    if (!result.status.ok()) return result;
    const QwenConfig& c = model->config_;
    model->limits_.vocab_size = c.vocab_size;
    model->limits_.max_context_tokens = c.max_context_tokens;
    model->limits_.eos_token_ids = c.eos_token_ids;

    const uint64_t hidden = c.hidden_size;
    const uint64_t q_dim = uint64_t(c.num_heads) * c.head_dim;
    const uint64_t kv_dim = uint64_t(c.num_kv_heads) * c.head_dim;

    size_t expected_tensor_count = 0;
    auto load = [&](const std::string& name,
                    std::vector<uint64_t> shape,
                    std::vector<float>& out) {
        if (!result.status.ok()) return;
        ++expected_tensor_count;
        result.status = LoadTensor(weights, name, shape, out);
    };

    load("model.embed_tokens.weight", {c.vocab_size, hidden},
         model->embed_tokens_);

    model->layers_.resize(c.num_layers);
    for (uint32_t l = 0; l < c.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        Layer& layer = model->layers_[l];
        load(p + "input_layernorm.weight", {hidden}, layer.input_norm);
        load(p + "self_attn.q_proj.weight", {q_dim, hidden}, layer.q_proj_w);
        load(p + "self_attn.q_proj.bias", {q_dim}, layer.q_proj_b);
        load(p + "self_attn.k_proj.weight", {kv_dim, hidden}, layer.k_proj_w);
        load(p + "self_attn.k_proj.bias", {kv_dim}, layer.k_proj_b);
        load(p + "self_attn.v_proj.weight", {kv_dim, hidden}, layer.v_proj_w);
        load(p + "self_attn.v_proj.bias", {kv_dim}, layer.v_proj_b);
        load(p + "self_attn.o_proj.weight", {hidden, q_dim}, layer.o_proj_w);
        load(p + "post_attention_layernorm.weight", {hidden},
             layer.post_attn_norm);
        load(p + "mlp.gate_proj.weight", {c.intermediate_size, hidden},
             layer.gate_proj_w);
        load(p + "mlp.up_proj.weight", {c.intermediate_size, hidden},
             layer.up_proj_w);
        load(p + "mlp.down_proj.weight", {hidden, c.intermediate_size},
             layer.down_proj_w);
    }

    load("model.norm.weight", {hidden}, model->final_norm_);
    if (!c.tie_word_embeddings) {
        load("lm_head.weight", {c.vocab_size, hidden}, model->lm_head_);
    }
    if (!result.status.ok()) return result;

    // Every tensor in the file must have been consumed: unexpected extra
    // tensors fail verification.
    if (weights.header().tensors.size() != expected_tensor_count) {
        result.status = VerifyFailed("weight file contains unexpected tensors");
        return result;
    }

    result.model = std::move(model);
    return result;
}

void QwenModel::ForwardToken(uint32_t token, uint32_t pos,
                             QwenKvCache& cache,
                             std::vector<float>& hidden) const {
    const QwenConfig& c = config_;
    const size_t h = c.hidden_size;
    const uint32_t group = c.num_heads / c.num_kv_heads;
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));

    hidden.assign(embed_tokens_.begin() + size_t(token) * h,
                  embed_tokens_.begin() + size_t(token) * h + h);

    std::vector<float> normed(h);
    std::vector<float> q(size_t(c.num_heads) * c.head_dim);
    std::vector<float> attn_out(size_t(c.num_heads) * c.head_dim);
    std::vector<float> proj(h);
    std::vector<float> gate(c.intermediate_size);
    std::vector<float> up(c.intermediate_size);
    std::vector<float> scores;

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        const Layer& layer = layers_[l];

        // Self-attention block.
        RmsNorm(hidden.data(), layer.input_norm, c.rms_norm_eps, h,
                normed.data());
        MatVec(layer.q_proj_w, layer.q_proj_b.data(), normed.data(), h,
               q.size(), q.data());
        float* k = cache.KeyAt(l, pos);
        float* v = cache.ValueAt(l, pos);
        MatVec(layer.k_proj_w, layer.k_proj_b.data(), normed.data(), h,
               size_t(c.num_kv_heads) * c.head_dim, k);
        MatVec(layer.v_proj_w, layer.v_proj_b.data(), normed.data(), h,
               size_t(c.num_kv_heads) * c.head_dim, v);

        for (uint32_t head = 0; head < c.num_heads; ++head) {
            ApplyRope(q.data() + size_t(head) * c.head_dim, c.head_dim, pos,
                      c.rope_theta);
        }
        for (uint32_t head = 0; head < c.num_kv_heads; ++head) {
            ApplyRope(k + size_t(head) * c.head_dim, c.head_dim, pos,
                      c.rope_theta);
        }

        const uint32_t context = pos + 1;  // causal: attend to <= pos
        scores.resize(context);
        for (uint32_t head = 0; head < c.num_heads; ++head) {
            const uint32_t kv_head = head / group;
            const float* qh = q.data() + size_t(head) * c.head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (uint32_t t = 0; t < context; ++t) {
                const float* kt =
                    cache.KeyAt(l, t) + size_t(kv_head) * c.head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < c.head_dim; ++d) {
                    dot += qh[d] * kt[d];
                }
                scores[t] = dot * attn_scale;
                max_score = std::max(max_score, scores[t]);
            }
            float denom = 0.0f;
            for (uint32_t t = 0; t < context; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                denom += scores[t];
            }
            float* out = attn_out.data() + size_t(head) * c.head_dim;
            std::fill(out, out + c.head_dim, 0.0f);
            for (uint32_t t = 0; t < context; ++t) {
                const float w = scores[t] / denom;
                const float* vt =
                    cache.ValueAt(l, t) + size_t(kv_head) * c.head_dim;
                for (uint32_t d = 0; d < c.head_dim; ++d) {
                    out[d] += w * vt[d];
                }
            }
        }
        MatVec(layer.o_proj_w, nullptr, attn_out.data(), attn_out.size(), h,
               proj.data());
        for (size_t i = 0; i < h; ++i) hidden[i] += proj[i];

        // MLP block (SwiGLU).
        RmsNorm(hidden.data(), layer.post_attn_norm, c.rms_norm_eps, h,
                normed.data());
        MatVec(layer.gate_proj_w, nullptr, normed.data(), h,
               c.intermediate_size, gate.data());
        MatVec(layer.up_proj_w, nullptr, normed.data(), h,
               c.intermediate_size, up.data());
        for (uint32_t i = 0; i < c.intermediate_size; ++i) {
            const float x = gate[i];
            const float silu = x / (1.0f + std::exp(-x));
            gate[i] = silu * up[i];
        }
        MatVec(layer.down_proj_w, nullptr, gate.data(), c.intermediate_size,
               h, proj.data());
        for (size_t i = 0; i < h; ++i) hidden[i] += proj[i];
    }
}

Status QwenModel::LogitsFromHidden(const std::vector<float>& hidden,
                                   std::vector<float>& logits) const {
    const QwenConfig& c = config_;
    std::vector<float> normed(c.hidden_size);
    RmsNorm(hidden.data(), final_norm_, c.rms_norm_eps, c.hidden_size,
            normed.data());
    logits.resize(c.vocab_size);
    const std::vector<float>& head =
        c.tie_word_embeddings ? embed_tokens_ : lm_head_;
    MatVec(head, nullptr, normed.data(), c.hidden_size, c.vocab_size,
           logits.data());

    for (float v : logits) {
        if (!std::isfinite(v)) {
            // Spec §14.3 / §29: invalid logits fail the request and flag
            // the model for degraded evaluation.
            return Status(ErrorCode::kInferenceFailed,
                          "logits contain non-finite values", kComponent);
        }
    }
    return Status::Ok();
}

Status QwenModel::Prefill(const std::vector<uint32_t>& tokens,
                          QwenKvCache& cache,
                          std::vector<float>& logits) const {
    if (tokens.empty()) {
        return Status(ErrorCode::kInvalidRequest, "empty prompt", kComponent);
    }
    if (cache.length() != 0) {
        return Status(ErrorCode::kInternalError,
                      "prefill requires an empty cache", kComponent);
    }
    if (tokens.size() > cache.capacity()) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "prompt exceeds cache capacity", kComponent);
    }
    for (uint32_t t : tokens) {
        if (t >= config_.vocab_size) {
            return Status(ErrorCode::kInvalidRequest,
                          "token id out of vocab range", kComponent);
        }
    }

    std::vector<float> hidden;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ForwardToken(tokens[i], uint32_t(i), cache, hidden);
        cache.Advance(1);
    }
    return LogitsFromHidden(hidden, logits);
}

namespace {

// SequenceState wrapper for the host-side KV cache.
class CpuSequenceState final : public SequenceState {
public:
    CpuSequenceState(const QwenConfig& config, uint32_t max_tokens)
        : cache_(config, max_tokens) {}
    uint32_t length() const override { return cache_.length(); }
    uint32_t capacity() const override { return cache_.capacity(); }
    QwenKvCache& cache() { return cache_; }

private:
    QwenKvCache cache_;
};

}  // namespace

Status QwenModel::CreateSequence(uint32_t max_tokens,
                                 std::unique_ptr<SequenceState>& out) {
    out = std::make_unique<CpuSequenceState>(config_, max_tokens);
    return Status::Ok();
}

Status QwenModel::Prefill(SequenceState& state,
                          const std::vector<uint32_t>& tokens,
                          std::vector<float>& logits) {
    return Prefill(tokens, static_cast<CpuSequenceState&>(state).cache(),
                   logits);
}

Status QwenModel::Decode(SequenceState& state, uint32_t token,
                         std::vector<float>& logits) {
    return Decode(token, static_cast<CpuSequenceState&>(state).cache(),
                  logits);
}

Status QwenModel::Decode(uint32_t token, QwenKvCache& cache,
                         std::vector<float>& logits) const {
    if (token >= config_.vocab_size) {
        return Status(ErrorCode::kInvalidRequest,
                      "token id out of vocab range", kComponent);
    }
    if (cache.length() >= cache.capacity()) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "kv cache capacity exhausted", kComponent);
    }
    std::vector<float> hidden;
    ForwardToken(token, cache.length(), cache, hidden);
    cache.Advance(1);
    return LogitsFromHidden(hidden, logits);
}

}  // namespace lykuro::nie
