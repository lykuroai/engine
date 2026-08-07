#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/engine/error.h"
#include "model/loader/safetensors.h"
#include "model/manifest/manifest.h"

namespace lykuro::nie {

// CPU reference implementation of approved_qwen_decoder_v1
// (Qwen2-style dense decoder: RMSNorm, RoPE (rotate-half), GQA attention
// with Q/K/V bias, SwiGLU MLP). All math in FP32.
//
// Purpose (spec Phase 1): correctness oracle and golden-test anchor.
// The CUDA backend must reproduce these outputs within certified
// tolerances. Not a performance path.

struct QwenConfig {
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t intermediate_size = 0;
    uint32_t max_context_tokens = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;
    std::vector<uint32_t> eos_token_ids;

    // Derives and validates the config from a verified manifest.
    static Status FromManifest(const ModelManifest& manifest, QwenConfig& out);
};

// Per-sequence KV cache (spec §16.2 MVP: contiguous per sequence, fixed
// maximum context, released with the request).
class QwenKvCache {
public:
    QwenKvCache(const QwenConfig& config, uint32_t max_tokens);

    uint32_t length() const { return length_; }
    uint32_t capacity() const { return max_tokens_; }

    // Layout per layer: [token][kv_head][head_dim]
    float* KeyAt(uint32_t layer, uint32_t token_index);
    float* ValueAt(uint32_t layer, uint32_t token_index);

    void Advance(uint32_t n) { length_ += n; }
    void Reset() { length_ = 0; }

private:
    uint32_t kv_stride_;  // kv_heads * head_dim
    uint32_t max_tokens_;
    uint32_t length_ = 0;
    std::vector<std::vector<float>> keys_;    // per layer
    std::vector<std::vector<float>> values_;  // per layer
};

class QwenModel {
public:
    struct LoadResult;

    // Loads and shape-checks every expected tensor. Unknown, missing, or
    // wrongly-shaped tensors fail the load (spec §0.2: no silent guessing).
    static LoadResult Load(const ModelManifest& manifest,
                           const SafetensorsFile& weights);

    const QwenConfig& config() const { return config_; }

    // Runs the prompt through the model, filling `cache` and returning
    // logits for the last position. Detects NaN/Inf in logits.
    Status Prefill(const std::vector<uint32_t>& tokens, QwenKvCache& cache,
                   std::vector<float>& logits) const;

    // Generates logits for one new token appended after the cache.
    Status Decode(uint32_t token, QwenKvCache& cache,
                  std::vector<float>& logits) const;

private:
    struct Layer {
        std::vector<float> input_norm;       // [hidden]
        std::vector<float> q_proj_w;         // [heads*head_dim, hidden]
        std::vector<float> q_proj_b;         // [heads*head_dim]
        std::vector<float> k_proj_w;         // [kv_heads*head_dim, hidden]
        std::vector<float> k_proj_b;
        std::vector<float> v_proj_w;
        std::vector<float> v_proj_b;
        std::vector<float> o_proj_w;         // [hidden, heads*head_dim]
        std::vector<float> post_attn_norm;   // [hidden]
        std::vector<float> gate_proj_w;      // [intermediate, hidden]
        std::vector<float> up_proj_w;        // [intermediate, hidden]
        std::vector<float> down_proj_w;      // [hidden, intermediate]
    };

    // Computes hidden state for one token at position `pos` and appends
    // its K/V to the cache. Returns the final hidden state.
    void ForwardToken(uint32_t token, uint32_t pos, QwenKvCache& cache,
                      std::vector<float>& hidden) const;
    Status LogitsFromHidden(const std::vector<float>& hidden,
                            std::vector<float>& logits) const;

    QwenConfig config_;
    std::vector<float> embed_tokens_;  // [vocab, hidden]
    std::vector<Layer> layers_;
    std::vector<float> final_norm_;    // [hidden]
    std::vector<float> lm_head_;       // [vocab, hidden]; empty when tied
};

struct QwenModel::LoadResult {
    Status status;
    std::unique_ptr<QwenModel> model;  // set only when status.ok()
};

}  // namespace lykuro::nie
