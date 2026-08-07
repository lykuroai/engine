#include "core/generation/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lykuro::nie {

namespace {
constexpr const char kComponent[] = "sampler";
}

Status ValidateSamplingParams(const SamplingParams& params,
                              const SamplingLimits& limits) {
    if (!std::isfinite(params.temperature) || params.temperature < 0.0f ||
        params.temperature > limits.max_temperature) {
        return Status(ErrorCode::kInvalidRequest,
                      "temperature out of range", kComponent);
    }
    if (!std::isfinite(params.top_p) || params.top_p <= 0.0f ||
        params.top_p > 1.0f) {
        return Status(ErrorCode::kInvalidRequest, "top_p out of range",
                      kComponent);
    }
    if (params.top_k > limits.max_top_k) {
        return Status(ErrorCode::kInvalidRequest, "top_k out of range",
                      kComponent);
    }
    return Status::Ok();
}

Sampler::Sampler(const SamplingParams& params)
    : params_(params), rng_(params.seed) {}

Status Sampler::Sample(const std::vector<float>& logits,
                       uint32_t& token_out) {
    if (logits.empty()) {
        return Status(ErrorCode::kInferenceFailed, "empty logits",
                      kComponent);
    }
    for (float v : logits) {
        if (!std::isfinite(v)) {
            return Status(ErrorCode::kInferenceFailed,
                          "logits contain non-finite values", kComponent);
        }
    }

    // Greedy: deterministic argmax, ties resolved to the lowest index.
    if (params_.temperature == 0.0f) {
        token_out = uint32_t(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        return Status::Ok();
    }

    // Candidate set with temperature applied.
    std::vector<uint32_t> index(logits.size());
    std::iota(index.begin(), index.end(), 0u);
    // Sort descending by logit; stable order for reproducibility.
    std::stable_sort(index.begin(), index.end(),
                     [&](uint32_t a, uint32_t b) {
                         return logits[a] > logits[b];
                     });

    size_t candidates = index.size();
    if (params_.top_k > 0) {
        candidates = std::min<size_t>(candidates, params_.top_k);
    }

    // Softmax over the top-k candidates (max-shifted for stability).
    const float inv_temp = 1.0f / params_.temperature;
    const float max_logit = logits[index[0]];
    std::vector<double> probs(candidates);
    double denom = 0.0;
    for (size_t i = 0; i < candidates; ++i) {
        probs[i] = std::exp(double(logits[index[i]] - max_logit) * inv_temp);
        denom += probs[i];
    }

    // Top-p: keep the smallest prefix with cumulative probability >= top_p.
    size_t nucleus = candidates;
    if (params_.top_p < 1.0f) {
        double cumulative = 0.0;
        for (size_t i = 0; i < candidates; ++i) {
            cumulative += probs[i] / denom;
            if (cumulative >= double(params_.top_p)) {
                nucleus = i + 1;
                break;
            }
        }
    }

    double nucleus_mass = 0.0;
    for (size_t i = 0; i < nucleus; ++i) nucleus_mass += probs[i];

    std::uniform_real_distribution<double> dist(0.0, nucleus_mass);
    double r = dist(rng_);
    double acc = 0.0;
    for (size_t i = 0; i < nucleus; ++i) {
        acc += probs[i];
        if (r <= acc) {
            token_out = index[i];
            return Status::Ok();
        }
    }
    token_out = index[nucleus - 1];
    return Status::Ok();
}

}  // namespace lykuro::nie
