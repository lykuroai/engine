#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Sampling parameters (spec §18). Validated against engine limits before
// use; out-of-range values are rejected, never clamped silently.
struct SamplingParams {
    float temperature = 1.0f;  // 0 => greedy
    float top_p = 1.0f;        // (0, 1]
    uint32_t top_k = 0;        // 0 => disabled
    uint64_t seed = 0;
};

struct SamplingLimits {
    float max_temperature = 2.0f;
    uint32_t max_top_k = 1024;
};

Status ValidateSamplingParams(const SamplingParams& params,
                              const SamplingLimits& limits);

// Versioned logits pipeline (spec §18.4), v1:
//   validate logits -> temperature -> top-k -> top-p -> sample
// Greedy (temperature == 0) is deterministic argmax with lowest-index
// tie-breaking. Sampling uses a seeded engine-owned PRNG so that equal
// (seed, logits) sequences reproduce equal outputs on this engine version.
class Sampler {
public:
    explicit Sampler(const SamplingParams& params);

    // Returns the chosen token id, or an error for non-finite logits.
    Status Sample(const std::vector<float>& logits, uint32_t& token_out);

    static constexpr const char kPipelineVersion[] = "sampler_v1";

private:
    SamplingParams params_;
    std::mt19937_64 rng_;
};

}  // namespace lykuro::nie
