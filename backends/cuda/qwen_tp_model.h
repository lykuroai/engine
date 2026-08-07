#pragma once

#include <memory>
#include <vector>

#include "core/engine/error.h"
#include "model/architectures/generative_model.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/loader/safetensors.h"
#include "model/manifest/manifest.h"

namespace lykuro::nie {

// Two-way tensor-parallel CUDA implementation (spec §17.5, Phase 5).
//
// Correctness-first PoC over heterogeneous GPUs:
//  - Q/K/V head-parallel (KV heads split evenly; each device owns the KV
//    cache for its heads only), o_proj/down_proj column-parallel with a
//    fixed-order host-side all-reduce (deterministic), gate/up
//    row-parallel.
//  - embed / final norm / lm_head live on the first device (no vocab
//    split), FP32 weights on device.
//  - Head/intermediate dimensions that do not divide evenly are refused
//    (unsupported_model) — never silently padded.
//
// This path exists to validate the TP architecture and to host models
// larger than a single GPU's VRAM. On a PCIe pair it is NOT faster than
// single-GPU for small models; measurements are reported honestly.
class QwenTpModel : public GenerativeModel {
public:
    struct Options {
        std::vector<int> device_ids;  // exactly 2 for the PoC
    };

    struct LoadResult {
        Status status;
        std::unique_ptr<QwenTpModel> model;
    };

    static LoadResult Load(const ModelManifest& manifest,
                           const SafetensorsFile& weights,
                           const Options& options);

    ~QwenTpModel() override;
    QwenTpModel(const QwenTpModel&) = delete;
    QwenTpModel& operator=(const QwenTpModel&) = delete;

    const ModelLimits& limits() const override { return limits_; }
    Status CreateSequence(uint32_t max_tokens,
                          std::unique_ptr<SequenceState>& out) override;
    Status Prefill(SequenceState& state, const std::vector<uint32_t>& tokens,
                   std::vector<float>& logits) override;
    Status Decode(SequenceState& state, uint32_t token,
                  std::vector<float>& logits) override;

private:
    QwenTpModel() = default;

    struct Impl;
    Status ForwardToken(uint32_t token, uint32_t pos, void* sequence_state,
                        std::vector<float>& logits_out, bool want_logits);

    QwenConfig config_;
    ModelLimits limits_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lykuro::nie
