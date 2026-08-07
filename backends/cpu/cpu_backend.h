#pragma once

#include <cstdint>
#include <cstddef>

namespace lykuro::nie {

// Scalar dtype conversion helpers for the CPU reference backend.
// All CPU reference math runs in FP32; BF16/FP16 weights are widened on
// load and results are compared against golden fixtures in FP32.

float Bf16ToFloat(uint16_t v);
uint16_t FloatToBf16(float v);  // round-to-nearest-even

float Fp16ToFloat(uint16_t v);
uint16_t FloatToFp16(float v);

void Bf16ToFloatArray(const uint16_t* in, float* out, size_t n);
void Fp16ToFloatArray(const uint16_t* in, float* out, size_t n);

}  // namespace lykuro::nie
