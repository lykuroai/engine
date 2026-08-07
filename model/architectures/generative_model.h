#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Backend-agnostic decoder-model interface. The engine schedules against
// this; concrete implementations are the CPU reference (correctness
// oracle) and the CUDA backend. Logits are always returned on the host in
// FP32 so sampling stays backend-independent and deterministic.

// Opaque per-sequence state (KV cache and whatever the backend needs).
// Released by destruction; the engine drops it when the request finishes
// (spec §16.1 ownership).
class SequenceState {
public:
    virtual ~SequenceState() = default;
    virtual uint32_t length() const = 0;
    virtual uint32_t capacity() const = 0;
};

struct ModelLimits {
    uint32_t vocab_size = 0;
    uint32_t max_context_tokens = 0;
    std::vector<uint32_t> eos_token_ids;
};

class GenerativeModel {
public:
    virtual ~GenerativeModel() = default;

    virtual const ModelLimits& limits() const = 0;

    // Allocates a sequence with room for `max_tokens` positions.
    virtual Status CreateSequence(uint32_t max_tokens,
                                  std::unique_ptr<SequenceState>& out) = 0;

    // Runs the prompt, filling the sequence state; returns last-position
    // logits (host FP32). Non-finite logits fail with inference_failed.
    virtual Status Prefill(SequenceState& state,
                           const std::vector<uint32_t>& tokens,
                           std::vector<float>& logits) = 0;

    // Appends one token and returns logits for the next position.
    virtual Status Decode(SequenceState& state, uint32_t token,
                          std::vector<float>& logits) = 0;

    struct DecodeBatchItem {
        SequenceState* state = nullptr;
        uint32_t token = 0;
        std::vector<float>* logits = nullptr;  // output
    };

    // Decodes one token for each sequence in a single pass when the
    // backend supports it (weight reads amortized across sequences).
    // Contract: an item's sequence advances only when its per_item status
    // is ok; on a non-ok overall return, NO sequence advanced, so the
    // caller may retry items individually (idempotent by construction).
    virtual Status DecodeBatch(std::vector<DecodeBatchItem>& items,
                               std::vector<Status>& per_item) {
        per_item.resize(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            per_item[i] =
                Decode(*items[i].state, items[i].token, *items[i].logits);
        }
        return Status::Ok();
    }
};

}  // namespace lykuro::nie
