#pragma once

#include <memory>
#include <vector>

#include "core/engine/error.h"
#include "model/architectures/generative_model.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/loader/safetensors.h"
#include "model/manifest/manifest.h"

namespace lykuro::nie {

// Apple Silicon Metal backend (LYK-NIE-ADD-METAL-001), MPSGraph-based
// correctness PoC.
//
// Design:
//  - FP32 compute (parity anchor against the CPU reference; FP16 is the
//    spec's target precision and follows with per-operator tolerances).
//  - Weights live in unified-memory MTLBuffers and are fed to the graph
//    as placeholders — no copies on either side of the fence.
//  - One whole-model single-token graph, specialized per context bucket
//    (multiples of 128) with an additive mask neutralizing the padding;
//    executables are cached per bucket (spec §5.2 MPSGraph cache).
//  - RoPE cos/sin for the current position are computed host-side and
//    fed per step.
//  - KV cache: bounded contiguous unified-memory buffers per sequence
//    (spec MVP §16); the new K/V row is written back by the host after
//    each step (unified memory, no transfer).
class QwenMetalModel : public GenerativeModel {
public:
    struct LoadResult {
        Status status;
        std::unique_ptr<QwenMetalModel> model;
    };

    static LoadResult Load(const ModelManifest& manifest,
                           const SafetensorsFile& weights);

    ~QwenMetalModel() override;
    QwenMetalModel(const QwenMetalModel&) = delete;
    QwenMetalModel& operator=(const QwenMetalModel&) = delete;

    const ModelLimits& limits() const override { return limits_; }
    Status CreateSequence(uint32_t max_tokens,
                          std::unique_ptr<SequenceState>& out) override;
    Status Prefill(SequenceState& state, const std::vector<uint32_t>& tokens,
                   std::vector<float>& logits) override;
    Status Decode(SequenceState& state, uint32_t token,
                  std::vector<float>& logits) override;

private:
    QwenMetalModel() = default;

    struct Impl;
    Status ForwardToken(uint32_t token, uint32_t pos, void* sequence_state,
                        std::vector<float>& logits_out, bool want_logits);

    QwenConfig config_;
    ModelLimits limits_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lykuro::nie
