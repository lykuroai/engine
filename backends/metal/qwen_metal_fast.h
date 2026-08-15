#pragma once

#include <memory>
#include <vector>

#include "core/engine/error.h"
#include "model/architectures/generative_model.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/loader/safetensors.h"
#include "model/manifest/manifest.h"

namespace lykuro::nie {

// Apple Silicon Metal backend, hand-written compute kernels (addendum
// §2.2 performance path). Unlike the MPSGraph model (the parity anchor),
// this path targets decode throughput:
//  - the whole per-token forward is encoded into ONE command buffer
//    (fused MSL kernels: RMSNorm, GEMV, RoPE, online-softmax attention,
//    SwiGLU) — no per-token graph dispatch or feed-dictionary cost;
//  - weights are resident FP16 or weight-only quantized INT8/INT4
//    (CUDA-backend scheme: INT8 per-output-row absmax, INT4 per-row
//    per-128-group absmax, nibble packed); activations stay FP16 with
//    FP32 accumulation everywhere, logits are emitted FP32;
//  - K/V rows are written by the GPU directly into the sequence's cache
//    (no staging copies). All reductions have a fixed order, so output
//    is bit-exact run-to-run (spec §14.2).
struct MetalFastOptions {
    enum class Quant {
        kFp16,  // FP16 weights (embed/head included)
        kInt8,  // INT8 projections + INT8 head
        kInt4,  // INT4 projections + INT8 head (logits precision)
    };
    Quant quant = Quant::kFp16;
};

class QwenMetalFastModel : public GenerativeModel {
public:
    struct LoadResult {
        Status status;
        std::unique_ptr<QwenMetalFastModel> model;
    };

    static LoadResult Load(const ModelManifest& manifest,
                           const SafetensorsFile& weights,
                           const MetalFastOptions& options);

    ~QwenMetalFastModel() override;
    QwenMetalFastModel(const QwenMetalFastModel&) = delete;
    QwenMetalFastModel& operator=(const QwenMetalFastModel&) = delete;

    const ModelLimits& limits() const override { return limits_; }
    Status CreateSequence(uint32_t max_tokens,
                          std::unique_ptr<SequenceState>& out) override;
    Status Prefill(SequenceState& state, const std::vector<uint32_t>& tokens,
                   std::vector<float>& logits) override;
    Status Decode(SequenceState& state, uint32_t token,
                  std::vector<float>& logits) override;

private:
    QwenMetalFastModel() = default;

    struct Impl;
    Status ForwardToken(uint32_t token, uint32_t pos, void* sequence_state,
                        std::vector<float>& logits_out, bool want_logits);

    QwenConfig config_;
    ModelLimits limits_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lykuro::nie
