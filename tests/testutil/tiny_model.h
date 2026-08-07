#pragma once

#include <string>

#include "model/manifest/manifest.h"

namespace lykuro::nie::testutil {

struct TinyModelSpec {
    uint32_t vocab_size = 32;
    uint32_t hidden_size = 16;
    uint32_t num_layers = 2;
    uint32_t num_heads = 4;
    uint32_t num_kv_heads = 2;
    uint32_t head_dim = 4;
    uint32_t intermediate_size = 32;
    uint32_t max_context_tokens = 64;
    uint32_t eos_token_id = 31;
    uint64_t seed = 42;
};

ModelManifest MakeTinyManifest(const TinyModelSpec& spec);

// Writes a deterministic random-weight safetensors file matching the spec
// and returns its path (under the gtest temp dir).
std::string WriteTinyWeights(const TinyModelSpec& spec,
                             const std::string& filename);

}  // namespace lykuro::nie::testutil
