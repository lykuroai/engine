#pragma once

#include <memory>
#include <vector>

#include "core/engine/error.h"
#include "model/architectures/generative_model.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/loader/safetensors.h"
#include "model/manifest/manifest.h"

namespace lykuro::nie {

// CUDA implementation of approved_qwen_decoder_v1.
//
// Correctness-first design (spec §0.3, §2.3: no Flash-Attention-class
// optimization in this phase): token-by-token forward using cuBLAS SGEMV
// for projections and small custom kernels for RMSNorm / RoPE / attention
// / SwiGLU. All math in FP32 (BF16/FP16 weights are widened at load).
// Outputs must match the CPU reference within certified tolerance; the
// parity test enforces this on real hardware.
class QwenCudaModel : public GenerativeModel {
public:
    struct LoadResult {
        Status status;
        std::unique_ptr<QwenCudaModel> model;
    };

    // Uploads verified weights to `device_id`. Fails (gpu_oom /
    // gpu_unhealthy) without partial residency.
    static LoadResult Load(const ModelManifest& manifest,
                           const SafetensorsFile& weights, int device_id);

    ~QwenCudaModel() override;
    QwenCudaModel(const QwenCudaModel&) = delete;
    QwenCudaModel& operator=(const QwenCudaModel&) = delete;

    const ModelLimits& limits() const override { return limits_; }
    Status CreateSequence(uint32_t max_tokens,
                          std::unique_ptr<SequenceState>& out) override;
    Status Prefill(SequenceState& state, const std::vector<uint32_t>& tokens,
                   std::vector<float>& logits) override;
    Status Decode(SequenceState& state, uint32_t token,
                  std::vector<float>& logits) override;

    const QwenConfig& config() const { return config_; }

private:
    QwenCudaModel() = default;

    struct Impl;
    Status ForwardToken(uint32_t token, uint32_t pos, void* sequence_state,
                        std::vector<float>& logits_out, bool want_logits);

    QwenConfig config_;
    ModelLimits limits_;
    int device_id_ = 0;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lykuro::nie
