#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backends/metal/qwen_metal_fast.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_metal_fast";
constexpr uint32_t kCtxBucket = 128;   // KV growth granularity
constexpr uint32_t kQ4Group = 128;     // INT4 scale group along K
constexpr uint32_t kAttnTpt = 128;     // attention threads per group
constexpr uint32_t kGreedyRun = 16;    // speculative decode steps per CB
constexpr uint32_t kPrefillBatch = 32;  // prompt tokens per CB
constexpr uint32_t kTokBuf = 32;       // token slots (>= both above)
constexpr uint32_t kAttnMaxSplit = 32;  // max row-splits per head

Status MetalFailed(const char* what) {
    return Status(ErrorCode::kGpuUnhealthy, what, kComponent);
}

// MSL kernel library. All reductions are fixed-order (threadgroup tree /
// hardware simd_sum), so outputs are bit-exact run-to-run. Activations
// are FP16 in memory; every dot product and softmax accumulates in FP32.
//
// Dispatch count dominates single-token latency on Apple GPUs, so the
// per-layer pass is fused into 9 dispatches: rmsnorm, fused-QKV GEMV
// (split outputs, K/V written straight into the cache rows), fused RoPE
// (q + cached k in one grid), attention, o-proj GEMV with fused residual
// accumulate, rmsnorm, fused gate/up GEMV, SiLU-mul, down-proj GEMV with
// fused residual accumulate.
constexpr const char* kMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint FLAG_BIAS = 1;
constant uint FLAG_RESIDUAL = 2;  // y[row] += acc instead of y[row] = acc

kernel void rmsnorm(device const half* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device half* out [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    constant float& eps [[buffer(4)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tpt [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += tpt) {
        float v = float(x[i]);
        acc += v * v;
    }
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(red[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tpt) {
        out[i] = half(float(x[i]) * scale * w[i]);
    }
}

// ---- dot-product cores (one simdgroup per output row) ----

// Two output rows per simdgroup: the x loads are shared and the two
// accumulator chains double the in-flight memory parallelism, which is
// what the short (~0.5-2KB) rows need to approach peak bandwidth.
inline float2 dot2_f16(device const half* wr0, device const half* wr1,
                       device const half4* x4, uint k4, uint lane) {
    device const half4* a4 = (device const half4*)wr0;
    device const half4* b4 = (device const half4*)wr1;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xv = float4(x4[i]);
        acc.x += dot(float4(a4[i]), xv);
        acc.y += dot(float4(b4[i]), xv);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2_q8(device const char* wr0, device const char* wr1,
                      device const half4* x4, uint k4, uint lane) {
    device const uint2* a8 = (device const uint2*)wr0;
    device const uint2* b8 = (device const uint2*)wr1;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]);
        float4 xb = float4(x4[2 * i + 1]);
        uint2 u = a8[i];
        char4 c0 = as_type<char4>(u.x);
        char4 c1 = as_type<char4>(u.y);
        acc.x += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
        u = b8[i];
        c0 = as_type<char4>(u.x);
        c1 = as_type<char4>(u.y);
        acc.y += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float4 q4_nibbles_lo(uchar4 b) {
    return float4(float(int(b.x & 0xF) - 8), float(int(b.x >> 4) - 8),
                  float(int(b.y & 0xF) - 8), float(int(b.y >> 4) - 8));
}
inline float4 q4_nibbles_hi(uchar4 b) {
    return float4(float(int(b.z & 0xF) - 8), float(int(b.z >> 4) - 8),
                  float(int(b.w & 0xF) - 8), float(int(b.w >> 4) - 8));
}

inline float2 dot2_q4(device const uchar* wr0, device const uchar* wr1,
                      device const float* sr0, device const float* sr1,
                      device const half4* x4, uint k4, uint lane) {
    const uint GROUP = 128;
    device const uchar4* a8 = (device const uchar4*)wr0;
    device const uchar4* b8 = (device const uchar4*)wr1;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]);
        float4 xb = float4(x4[2 * i + 1]);
        uchar4 b = a8[i];
        acc.x += sr0[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
        b = b8[i];
        acc.y += sr1[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float dot_f16_tg(device const half* wr, threadgroup const half* xt,
                        uint k4, uint lane) {
    device const half4* w4 = (device const half4*)wr;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    float acc = 0.0f;
    for (uint i = lane; i < k4; i += 32) {
        acc += dot(float4(w4[i]), float4(x4[i]));
    }
    return simd_sum(acc);
}

inline float dot_q8_tg(device const char* wr, threadgroup const half* xt,
                       uint k4, uint lane) {
    device const uint2* w8 = (device const uint2*)wr;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    const uint k8 = k4 / 2;
    float acc = 0.0f;
    for (uint i = lane; i < k8; i += 32) {
        uint2 u = w8[i];
        char4 a = as_type<char4>(u.x);
        char4 b = as_type<char4>(u.y);
        acc += dot(float4(a.x, a.y, a.z, a.w), float4(x4[2 * i])) +
               dot(float4(b.x, b.y, b.z, b.w), float4(x4[2 * i + 1]));
    }
    return simd_sum(acc);
}

inline float dot_q4_tg(device const uchar* wr, device const float* sr,
                       threadgroup const half* xt, uint k4, uint lane) {
    const uint GROUP = 128;
    device const uchar4* w8 = (device const uchar4*)wr;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    const uint k8 = k4 / 2;
    float acc = 0.0f;
    for (uint i = lane; i < k8; i += 32) {
        uchar4 b = w8[i];
        float4 lo = float4(float(int(b.x & 0xF) - 8),
                           float(int(b.x >> 4) - 8),
                           float(int(b.y & 0xF) - 8),
                           float(int(b.y >> 4) - 8));
        float4 hi = float4(float(int(b.z & 0xF) - 8),
                           float(int(b.z >> 4) - 8),
                           float(int(b.w & 0xF) - 8),
                           float(int(b.w >> 4) - 8));
        acc += sr[8 * i / GROUP] * (dot(lo, float4(x4[2 * i])) +
                                    dot(hi, float4(x4[2 * i + 1])));
    }
    return simd_sum(acc);
}

// ---- generic GEMV: y[row] (+)= dot(W[row,:], x) (+ bias) ----

kernel void gemv_f16(device const half* W [[buffer(0)]],
                     device const half* x [[buffer(1)]],
                     device const half* bias [[buffer(2)]],
                     device half* y [[buffer(3)]],
                     constant uint& K [[buffer(4)]],
                     constant uint& N [[buffer(5)]],
                     constant uint& flags [[buffer(6)]],
                     uint tgpig [[threadgroup_position_in_grid]],
                     uint sgitg [[simdgroup_index_in_threadgroup]],
                     uint tiisg [[thread_index_in_simdgroup]],
                     uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_f16(W + ulong(row) * K, W + ulong(rowb) * K,
                          (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        if (flags & FLAG_BIAS) acc += float2(bias[row], bias[rowb]);
        if (flags & FLAG_RESIDUAL) acc += float2(y[row], y[rowb]);
        y[row] = half(acc.x);
        if (rowb != row) y[rowb] = half(acc.y);
    }
}

kernel void gemv_q8(device const char* W [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const half* x [[buffer(2)]],
                    device const half* bias [[buffer(3)]],
                    device half* y [[buffer(4)]],
                    constant uint& K [[buffer(5)]],
                    constant uint& N [[buffer(6)]],
                    constant uint& flags [[buffer(7)]],
                    uint tgpig [[threadgroup_position_in_grid]],
                    uint sgitg [[simdgroup_index_in_threadgroup]],
                    uint tiisg [[thread_index_in_simdgroup]],
                    uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q8(W + ulong(row) * K, W + ulong(rowb) * K,
                         (device const half4*)x, K / 4, tiisg);
    acc *= float2(scales[row], scales[rowb]);
    if (tiisg == 0) {
        if (flags & FLAG_BIAS) acc += float2(bias[row], bias[rowb]);
        if (flags & FLAG_RESIDUAL) acc += float2(y[row], y[rowb]);
        y[row] = half(acc.x);
        if (rowb != row) y[rowb] = half(acc.y);
    }
}

kernel void gemv_q4(device const uchar* W [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const half* x [[buffer(2)]],
                    device const half* bias [[buffer(3)]],
                    device half* y [[buffer(4)]],
                    constant uint& K [[buffer(5)]],
                    constant uint& N [[buffer(6)]],
                    constant uint& flags [[buffer(7)]],
                    constant uint& groups [[buffer(8)]],
                    uint tgpig [[threadgroup_position_in_grid]],
                    uint sgitg [[simdgroup_index_in_threadgroup]],
                    uint tiisg [[thread_index_in_simdgroup]],
                    uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q4(W + ulong(row) * (K / 2), W + ulong(rowb) * (K / 2),
                         scales + ulong(row) * groups,
                         scales + ulong(rowb) * groups,
                         (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        if (flags & FLAG_BIAS) acc += float2(bias[row], bias[rowb]);
        if (flags & FLAG_RESIDUAL) acc += float2(y[row], y[rowb]);
        y[row] = half(acc.x);
        if (rowb != row) y[rowb] = half(acc.y);
    }
}

// ---- split-output GEMV over a row-concatenated matrix ----
// rows [0, n0) -> y0, [n0, n0+n1) -> y1, rest -> y2. Used for the fused
// QKV projection (y1/y2 are the K/V cache rows) and fused gate/up.

inline void split_store(float acc, uint row, device const half* bias,
                        device half* y0, device half* y1, device half* y2,
                        uint n0, uint n1, uint flags) {
    if (flags & FLAG_BIAS) acc += float(bias[row]);
    if (row < n0) {
        y0[row] = half(acc);
    } else if (row < n0 + n1) {
        y1[row - n0] = half(acc);
    } else {
        y2[row - n0 - n1] = half(acc);
    }
}

kernel void gemv_split_f16(device const half* W [[buffer(0)]],
                           device const half* x [[buffer(1)]],
                           device const half* bias [[buffer(2)]],
                           device half* y0 [[buffer(3)]],
                           device half* y1 [[buffer(4)]],
                           device half* y2 [[buffer(5)]],
                           constant uint& K [[buffer(6)]],
                           constant uint& N [[buffer(7)]],
                           constant uint& n0 [[buffer(8)]],
                           constant uint& n1 [[buffer(9)]],
                           constant uint& flags [[buffer(10)]],
                           uint tgpig [[threadgroup_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint tiisg [[thread_index_in_simdgroup]],
                           uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_f16(W + ulong(row) * K, W + ulong(rowb) * K,
                          (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

kernel void gemv_split_q8(device const char* W [[buffer(0)]],
                          device const float* scales [[buffer(1)]],
                          device const half* x [[buffer(2)]],
                          device const half* bias [[buffer(3)]],
                          device half* y0 [[buffer(4)]],
                          device half* y1 [[buffer(5)]],
                          device half* y2 [[buffer(6)]],
                          constant uint& K [[buffer(7)]],
                          constant uint& N [[buffer(8)]],
                          constant uint& n0 [[buffer(9)]],
                          constant uint& n1 [[buffer(10)]],
                          constant uint& flags [[buffer(11)]],
                          uint tgpig [[threadgroup_position_in_grid]],
                          uint sgitg [[simdgroup_index_in_threadgroup]],
                          uint tiisg [[thread_index_in_simdgroup]],
                          uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q8(W + ulong(row) * K, W + ulong(rowb) * K,
                         (device const half4*)x, K / 4, tiisg);
    acc *= float2(scales[row], scales[rowb]);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

kernel void gemv_split_q4(device const uchar* W [[buffer(0)]],
                          device const float* scales [[buffer(1)]],
                          device const half* x [[buffer(2)]],
                          device const half* bias [[buffer(3)]],
                          device half* y0 [[buffer(4)]],
                          device half* y1 [[buffer(5)]],
                          device half* y2 [[buffer(6)]],
                          constant uint& K [[buffer(7)]],
                          constant uint& N [[buffer(8)]],
                          constant uint& n0 [[buffer(9)]],
                          constant uint& n1 [[buffer(10)]],
                          constant uint& flags [[buffer(11)]],
                          constant uint& groups [[buffer(12)]],
                          uint tgpig [[threadgroup_position_in_grid]],
                          uint sgitg [[simdgroup_index_in_threadgroup]],
                          uint tiisg [[thread_index_in_simdgroup]],
                          uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q4(W + ulong(row) * (K / 2), W + ulong(rowb) * (K / 2),
                         scales + ulong(row) * groups,
                         scales + ulong(rowb) * groups,
                         (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

// ---- wide-row GEMVs: one 128-thread threadgroup per output row ----
// The layer projections have only ~1-10k rows; one simdgroup per row
// leaves the GPU underoccupied and memory-latency bound. A threadgroup
// per row puts 128 threads on each dot product (threadgroup-tree
// reduction, fixed order).

inline float tg_row_reduce(float acc, uint tid, uint tpt,
                           threadgroup float* red) {
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return red[0];
}

kernel void gemv_wide_f16(device const half* W [[buffer(0)]],
                          device const half* x [[buffer(1)]],
                          device const half* bias [[buffer(2)]],
                          device half* y [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          constant uint& flags [[buffer(6)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tpt [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device const half4* w4 = (device const half4*)(W + ulong(row) * K);
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        acc += dot(float4(w4[i]), float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red);
    if (tid == 0) {
        if (flags & FLAG_BIAS) acc += float(bias[row]);
        if (flags & FLAG_RESIDUAL) acc += float(y[row]);
        y[row] = half(acc);
    }
}

kernel void gemv_wide_q8(device const char* W [[buffer(0)]],
                         device const float* scales [[buffer(1)]],
                         device const half* x [[buffer(2)]],
                         device const half* bias [[buffer(3)]],
                         device half* y [[buffer(4)]],
                         constant uint& K [[buffer(5)]],
                         constant uint& N [[buffer(6)]],
                         constant uint& flags [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tpt [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device const char4* w4 = (device const char4*)(W + ulong(row) * K);
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        char4 c = w4[i];
        acc += dot(float4(c.x, c.y, c.z, c.w), float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red) * scales[row];
    if (tid == 0) {
        if (flags & FLAG_BIAS) acc += float(bias[row]);
        if (flags & FLAG_RESIDUAL) acc += float(y[row]);
        y[row] = half(acc);
    }
}

kernel void gemv_wide_q4(device const uchar* W [[buffer(0)]],
                         device const float* scales [[buffer(1)]],
                         device const half* x [[buffer(2)]],
                         device const half* bias [[buffer(3)]],
                         device half* y [[buffer(4)]],
                         constant uint& K [[buffer(5)]],
                         constant uint& N [[buffer(6)]],
                         constant uint& flags [[buffer(7)]],
                         constant uint& groups [[buffer(8)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tpt [[threads_per_threadgroup]]) {
    const uint GROUP = 128;
    threadgroup float red[128];
    device const uchar* wr = W + ulong(row) * (K / 2);
    device const float* sr = scales + ulong(row) * groups;
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        uint e = 4 * i;
        uchar b0 = wr[e / 2];
        uchar b1 = wr[e / 2 + 1];
        float4 wv = float4(float(int(b0 & 0xF) - 8),
                           float(int(b0 >> 4) - 8),
                           float(int(b1 & 0xF) - 8),
                           float(int(b1 >> 4) - 8));
        acc += sr[e / GROUP] * dot(wv, float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red);
    if (tid == 0) {
        if (flags & FLAG_BIAS) acc += float(bias[row]);
        if (flags & FLAG_RESIDUAL) acc += float(y[row]);
        y[row] = half(acc);
    }
}

kernel void gemv_split_wide_f16(device const half* W [[buffer(0)]],
                                device const half* x [[buffer(1)]],
                                device const half* bias [[buffer(2)]],
                                device half* y0 [[buffer(3)]],
                                device half* y1 [[buffer(4)]],
                                device half* y2 [[buffer(5)]],
                                constant uint& K [[buffer(6)]],
                                constant uint& N [[buffer(7)]],
                                constant uint& n0 [[buffer(8)]],
                                constant uint& n1 [[buffer(9)]],
                                constant uint& flags [[buffer(10)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tpt [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device const half4* w4 = (device const half4*)(W + ulong(row) * K);
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        acc += dot(float4(w4[i]), float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red);
    if (tid == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
}

kernel void gemv_split_wide_q8(device const char* W [[buffer(0)]],
                               device const float* scales [[buffer(1)]],
                               device const half* x [[buffer(2)]],
                               device const half* bias [[buffer(3)]],
                               device half* y0 [[buffer(4)]],
                               device half* y1 [[buffer(5)]],
                               device half* y2 [[buffer(6)]],
                               constant uint& K [[buffer(7)]],
                               constant uint& N [[buffer(8)]],
                               constant uint& n0 [[buffer(9)]],
                               constant uint& n1 [[buffer(10)]],
                               constant uint& flags [[buffer(11)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]]) {
    threadgroup float red[128];
    device const char4* w4 = (device const char4*)(W + ulong(row) * K);
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        char4 c = w4[i];
        acc += dot(float4(c.x, c.y, c.z, c.w), float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red) * scales[row];
    if (tid == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
}

kernel void gemv_split_wide_q4(device const uchar* W [[buffer(0)]],
                               device const float* scales [[buffer(1)]],
                               device const half* x [[buffer(2)]],
                               device const half* bias [[buffer(3)]],
                               device half* y0 [[buffer(4)]],
                               device half* y1 [[buffer(5)]],
                               device half* y2 [[buffer(6)]],
                               constant uint& K [[buffer(7)]],
                               constant uint& N [[buffer(8)]],
                               constant uint& n0 [[buffer(9)]],
                               constant uint& n1 [[buffer(10)]],
                               constant uint& flags [[buffer(11)]],
                               constant uint& groups [[buffer(12)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]]) {
    const uint GROUP = 128;
    threadgroup float red[128];
    device const uchar* wr = W + ulong(row) * (K / 2);
    device const float* sr = scales + ulong(row) * groups;
    device const half4* x4 = (device const half4*)x;
    const uint k4 = K / 4;
    float acc = 0.0f;
    for (uint i = tid; i < k4; i += tpt) {
        uint e = 4 * i;
        uchar b0 = wr[e / 2];
        uchar b1 = wr[e / 2 + 1];
        float4 wv = float4(float(int(b0 & 0xF) - 8),
                           float(int(b0 >> 4) - 8),
                           float(int(b1 & 0xF) - 8),
                           float(int(b1 >> 4) - 8));
        acc += sr[e / GROUP] * dot(wv, float4(x4[i]));
    }
    acc = tg_row_reduce(acc, tid, tpt, red);
    if (tid == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
}

// Wide-K plain GEMV (q4): 16 elements per lane iteration — pays off
// only on long reduction dims (the down projection), where the extra
// register pressure is amortized.
kernel void gemv_q4_k16(device const uchar* W [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const half* x [[buffer(2)]],
                        device const half* bias [[buffer(3)]],
                        device half* y [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        constant uint& N [[buffer(6)]],
                        constant uint& flags [[buffer(7)]],
                        constant uint& groups [[buffer(8)]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgpt [[simdgroups_per_threadgroup]]) {
    const uint GROUP = 128;
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    device const uint2* a16 = (device const uint2*)(W + ulong(row) * (K / 2));
    device const uint2* b16 = (device const uint2*)(W + ulong(rowb) * (K / 2));
    device const float* s0 = scales + ulong(row) * groups;
    device const float* s1 = scales + ulong(rowb) * groups;
    device const half4* x4 = (device const half4*)x;
    const uint k16 = K / 16;
    float2 acc = float2(0.0f);
    for (uint i = tiisg; i < k16; i += 32) {
        float4 xa = float4(x4[4 * i]);
        float4 xb = float4(x4[4 * i + 1]);
        float4 xc = float4(x4[4 * i + 2]);
        float4 xd = float4(x4[4 * i + 3]);
        uint2 u = a16[i];
        uchar4 p = as_type<uchar4>(u.x);
        uchar4 q2 = as_type<uchar4>(u.y);
        acc.x += s0[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
        u = b16[i];
        p = as_type<uchar4>(u.x);
        q2 = as_type<uchar4>(u.y);
        acc.y += s1[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
    }
    acc = float2(simd_sum(acc.x), simd_sum(acc.y));
    if (tiisg == 0) {
        if (flags & FLAG_BIAS) acc += float2(bias[row], bias[rowb]);
        if (flags & FLAG_RESIDUAL) acc += float2(y[row], y[rowb]);
        y[row] = half(acc.x);
        if (rowb != row) y[rowb] = half(acc.y);
    }
}

// ---- barrier-free fused RMSNorm ----
// Each simdgroup redundantly computes the norm scale with a simd_sum
// (fixed order, no threadgroup barriers) and normalizes x on the fly
// inside the dot product; the norm weight vector is tiny and L1-cached.

inline float rms_scale_sg(device const half* x, uint K, float eps,
                          uint lane) {
    float ss = 0.0f;
    for (uint i = lane; i < K; i += 32) {
        float v = float(x[i]);
        ss += v * v;
    }
    ss = simd_sum(ss);
    return rsqrt(ss / float(K) + eps);
}

inline float2 dot2n_f16(device const half* wr0, device const half* wr1,
                        device const half4* x4, device const float4* nw4,
                        float ns, uint k4, uint lane) {
    device const half4* a4 = (device const half4*)wr0;
    device const half4* b4 = (device const half4*)wr1;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xv = float4(x4[i]) * ns * nw4[i];
        acc.x += dot(float4(a4[i]), xv);
        acc.y += dot(float4(b4[i]), xv);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2n_q8(device const char* wr0, device const char* wr1,
                       device const half4* x4, device const float4* nw4,
                       float ns, uint k4, uint lane) {
    device const uint2* a8 = (device const uint2*)wr0;
    device const uint2* b8 = (device const uint2*)wr1;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]) * ns * nw4[2 * i];
        float4 xb = float4(x4[2 * i + 1]) * ns * nw4[2 * i + 1];
        uint2 u = a8[i];
        char4 c0 = as_type<char4>(u.x);
        char4 c1 = as_type<char4>(u.y);
        acc.x += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
        u = b8[i];
        c0 = as_type<char4>(u.x);
        c1 = as_type<char4>(u.y);
        acc.y += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2n_q4(device const uchar* wr0, device const uchar* wr1,
                       device const float* sr0, device const float* sr1,
                       device const half4* x4, device const float4* nw4,
                       float ns, uint k4, uint lane) {
    const uint GROUP = 128;
    device const uchar4* a8 = (device const uchar4*)wr0;
    device const uchar4* b8 = (device const uchar4*)wr1;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]) * ns * nw4[2 * i];
        float4 xb = float4(x4[2 * i + 1]) * ns * nw4[2 * i + 1];
        uchar4 b = a8[i];
        acc.x += sr0[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
        b = b8[i];
        acc.y += sr1[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2n_q4_k16(device const uchar* wr0,
                           device const uchar* wr1,
                           device const float* sr0,
                           device const float* sr1,
                           device const half4* x4,
                           device const float4* nw4, float ns, uint k4,
                           uint lane) {
    const uint GROUP = 128;
    device const uint2* a16 = (device const uint2*)wr0;
    device const uint2* b16 = (device const uint2*)wr1;
    const uint k16 = k4 / 4;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k16; i += 32) {
        float4 xa = float4(x4[4 * i]) * ns * nw4[4 * i];
        float4 xb = float4(x4[4 * i + 1]) * ns * nw4[4 * i + 1];
        float4 xc = float4(x4[4 * i + 2]) * ns * nw4[4 * i + 2];
        float4 xd = float4(x4[4 * i + 3]) * ns * nw4[4 * i + 3];
        uint2 u = a16[i];
        uchar4 p = as_type<uchar4>(u.x);
        uchar4 q2 = as_type<uchar4>(u.y);
        acc.x += sr0[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
        u = b16[i];
        p = as_type<uchar4>(u.x);
        q2 = as_type<uchar4>(u.y);
        acc.y += sr1[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

kernel void gemv_gateup_q4_k16(device const uchar* W [[buffer(0)]],
                               device const float* scales [[buffer(1)]],
                               device const half* x [[buffer(2)]],
                               device half* act [[buffer(3)]],
                               constant uint& K [[buffer(4)]],
                               constant uint& N [[buffer(5)]],
                               constant uint& groups [[buffer(6)]],
                               device const float* nw [[buffer(7)]],
                               constant float& eps [[buffer(8)]],
                               uint tgpig [[threadgroup_position_in_grid]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_q4_k16(W + ulong(row) * (K / 2),
                              W + ulong(row + 1) * (K / 2),
                              scales + ulong(row) * groups,
                              scales + ulong(row + 1) * groups,
                              (device const half4*)x,
                              (device const float4*)nw, ns, K / 4, tiisg);
    if (tiisg == 0) {
        float g = acc.x;
        act[row / 2] = half(g / (1.0f + exp(-g)) * acc.y);
    }
}

// ---- fused RMSNorm + QKV + bias + RoPE ----
// The fused matrix is laid out in ROTATION PAIRS: q rows come as
// (head*hd+i, head*hd+i+hd/2) pairs, then k rows likewise, then v rows
// unchanged — a dual-row simdgroup therefore owns exactly one rotation
// pair and applies RoPE in the epilogue. K/V rows are written straight
// into the cache (buffer offsets select the position's row).

struct QkvDims {
    uint q_dim;
    uint kv_dim;
    uint hd;
    uint pos;
    float theta;
};

inline void qkv_store(float2 acc, uint row, device const half* bias,
                      device half* qo, device half* ko, device half* vo,
                      constant QkvDims& d) {
    acc += float2(bias[row], bias[row + 1]);
    if (row < d.q_dim + d.kv_dim) {
        // rotation pair (q or k section)
        const bool is_q = row < d.q_dim;
        const uint j = (is_q ? row : row - d.q_dim) / 2;
        const uint half_hd = d.hd / 2;
        const uint head = j / half_hd;
        const uint i = j % half_hd;
        float freq = pow(d.theta, -2.0f * float(i) / float(d.hd));
        float cs = cos(float(d.pos) * freq);
        float sn = sin(float(d.pos) * freq);
        float a = acc.x * cs - acc.y * sn;
        float b = acc.y * cs + acc.x * sn;
        device half* o = is_q ? qo : ko;
        o[head * d.hd + i] = half(a);
        o[head * d.hd + i + half_hd] = half(b);
    } else {
        const uint rv = row - d.q_dim - d.kv_dim;
        vo[rv] = half(acc.x);
        vo[rv + 1] = half(acc.y);
    }
}

kernel void gemv_qkv_f16(device const half* W [[buffer(0)]],
                         device const half* x [[buffer(1)]],
                         device const half* bias [[buffer(2)]],
                         device half* qo [[buffer(3)]],
                         device half* ko [[buffer(4)]],
                         device half* vo [[buffer(5)]],
                         constant uint& K [[buffer(6)]],
                         constant uint& N [[buffer(7)]],
                         constant QkvDims& d [[buffer(8)]],
                         device const float* nw [[buffer(9)]],
                         constant float& eps [[buffer(10)]],
                         uint tgpig [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiisg [[thread_index_in_simdgroup]],
                         uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_f16(W + ulong(row) * K, W + ulong(row + 1) * K,
                           (device const half4*)x,
                           (device const float4*)nw, ns, K / 4, tiisg);
    if (tiisg == 0) qkv_store(acc, row, bias, qo, ko, vo, d);
}

kernel void gemv_qkv_q8(device const char* W [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const half* x [[buffer(2)]],
                        device const half* bias [[buffer(3)]],
                        device half* qo [[buffer(4)]],
                        device half* ko [[buffer(5)]],
                        device half* vo [[buffer(6)]],
                        constant uint& K [[buffer(7)]],
                        constant uint& N [[buffer(8)]],
                        constant QkvDims& d [[buffer(9)]],
                        device const float* nw [[buffer(10)]],
                        constant float& eps [[buffer(11)]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_q8(W + ulong(row) * K, W + ulong(row + 1) * K,
                          (device const half4*)x,
                          (device const float4*)nw, ns, K / 4, tiisg);
    acc *= float2(scales[row], scales[row + 1]);
    if (tiisg == 0) qkv_store(acc, row, bias, qo, ko, vo, d);
}

kernel void gemv_qkv_q4(device const uchar* W [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const half* x [[buffer(2)]],
                        device const half* bias [[buffer(3)]],
                        device half* qo [[buffer(4)]],
                        device half* ko [[buffer(5)]],
                        device half* vo [[buffer(6)]],
                        constant uint& K [[buffer(7)]],
                        constant uint& N [[buffer(8)]],
                        constant QkvDims& d [[buffer(9)]],
                        device const float* nw [[buffer(10)]],
                        constant float& eps [[buffer(11)]],
                        constant uint& groups [[buffer(12)]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_q4(W + ulong(row) * (K / 2),
                          W + ulong(row + 1) * (K / 2),
                          scales + ulong(row) * groups,
                          scales + ulong(row + 1) * groups,
                          (device const half4*)x,
                          (device const float4*)nw, ns, K / 4, tiisg);
    if (tiisg == 0) qkv_store(acc, row, bias, qo, ko, vo, d);
}

// ---- fused gate/up GEMV + SiLU ----
// The gate and up matrices are interleaved at load (row 2j = gate_j,
// row 2j+1 = up_j) so the dual-row simdgroup computes the matching pair
// and writes act[j] = silu(gate_j . x) * (up_j . x) directly — the
// separate silu_mul dispatch and the gate/up staging buffers disappear.

kernel void gemv_gateup_f16(device const half* W [[buffer(0)]],
                            device const half* x [[buffer(1)]],
                            device half* act [[buffer(2)]],
                            constant uint& K [[buffer(3)]],
                            constant uint& N [[buffer(4)]],
                            device const float* nw [[buffer(5)]],
                            constant float& eps [[buffer(6)]],
                            uint tgpig [[threadgroup_position_in_grid]],
                            uint sgitg [[simdgroup_index_in_threadgroup]],
                            uint tiisg [[thread_index_in_simdgroup]],
                            uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_f16(W + ulong(row) * K, W + ulong(row + 1) * K,
                           (device const half4*)x,
                           (device const float4*)nw, ns, K / 4, tiisg);
    if (tiisg == 0) {
        float g = acc.x;
        act[row / 2] = half(g / (1.0f + exp(-g)) * acc.y);
    }
}

kernel void gemv_gateup_q8(device const char* W [[buffer(0)]],
                           device const float* scales [[buffer(1)]],
                           device const half* x [[buffer(2)]],
                           device half* act [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           device const float* nw [[buffer(6)]],
                           constant float& eps [[buffer(7)]],
                           uint tgpig [[threadgroup_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint tiisg [[thread_index_in_simdgroup]],
                           uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_q8(W + ulong(row) * K, W + ulong(row + 1) * K,
                          (device const half4*)x,
                          (device const float4*)nw, ns, K / 4, tiisg);
    acc *= float2(scales[row], scales[row + 1]);
    if (tiisg == 0) {
        float g = acc.x;
        act[row / 2] = half(g / (1.0f + exp(-g)) * acc.y);
    }
}

kernel void gemv_gateup_q4(device const uchar* W [[buffer(0)]],
                           device const float* scales [[buffer(1)]],
                           device const half* x [[buffer(2)]],
                           device half* act [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           constant uint& groups [[buffer(6)]],
                           device const float* nw [[buffer(7)]],
                           constant float& eps [[buffer(8)]],
                           uint tgpig [[threadgroup_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint tiisg [[thread_index_in_simdgroup]],
                           uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    float ns = rms_scale_sg(x, K, eps, tiisg);
    float2 acc = dot2n_q4(W + ulong(row) * (K / 2),
                          W + ulong(row + 1) * (K / 2),
                          scales + ulong(row) * groups,
                          scales + ulong(row + 1) * groups,
                          (device const half4*)x,
                          (device const float4*)nw, ns, K / 4, tiisg);
    if (tiisg == 0) {
        float g = acc.x;
        act[row / 2] = half(g / (1.0f + exp(-g)) * acc.y);
    }
}

// ---- RMSNorm-fused GEMVs ----
// Each threadgroup recomputes the (cheap) norm reduction, materializes
// the normalized activation once in DYNAMIC threadgroup memory (K*2
// bytes — a static worst-case array would cap occupancy), and runs its
// rows against it: one dispatch replaces rmsnorm + GEMV and the dot
// products read threadgroup memory. Two rows per simdgroup, as in the
// unfused kernels.

#define NORM_PROLOGUE                                                     \
    threadgroup float red[256];                                           \
    {                                                                     \
        float ss = 0.0f;                                                  \
        for (uint i = tid; i < K; i += tpt) {                             \
            float v = float(x[i]);                                        \
            ss += v * v;                                                  \
        }                                                                 \
        red[tid] = ss;                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint s = tpt / 2; s > 0; s >>= 1) {                          \
            if (tid < s) red[tid] += red[tid + s];                        \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
        float nscale = rsqrt(red[0] / float(K) + eps);                    \
        for (uint i = tid; i < K; i += tpt) {                             \
            xn[i] = half(float(x[i]) * nscale * nw[i]);                   \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
    }

inline float2 dot2_f16_tg(device const half* wr0, device const half* wr1,
                          threadgroup const half* xt, uint k4, uint lane) {
    device const half4* a4 = (device const half4*)wr0;
    device const half4* b4 = (device const half4*)wr1;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xv = float4(x4[i]);
        acc.x += dot(float4(a4[i]), xv);
        acc.y += dot(float4(b4[i]), xv);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2_q8_tg(device const char* wr0, device const char* wr1,
                         threadgroup const half* xt, uint k4, uint lane) {
    device const uint2* a8 = (device const uint2*)wr0;
    device const uint2* b8 = (device const uint2*)wr1;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]);
        float4 xb = float4(x4[2 * i + 1]);
        uint2 u = a8[i];
        char4 c0 = as_type<char4>(u.x);
        char4 c1 = as_type<char4>(u.y);
        acc.x += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
        u = b8[i];
        c0 = as_type<char4>(u.x);
        c1 = as_type<char4>(u.y);
        acc.y += dot(float4(c0.x, c0.y, c0.z, c0.w), xa) +
                 dot(float4(c1.x, c1.y, c1.z, c1.w), xb);
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

inline float2 dot2_q4_tg(device const uchar* wr0, device const uchar* wr1,
                         device const float* sr0, device const float* sr1,
                         threadgroup const half* xt, uint k4, uint lane) {
    const uint GROUP = 128;
    device const uchar4* a8 = (device const uchar4*)wr0;
    device const uchar4* b8 = (device const uchar4*)wr1;
    threadgroup const half4* x4 = (threadgroup const half4*)xt;
    const uint k8 = k4 / 2;
    float2 acc = float2(0.0f);
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = float4(x4[2 * i]);
        float4 xb = float4(x4[2 * i + 1]);
        uchar4 b = a8[i];
        acc.x += sr0[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
        b = b8[i];
        acc.y += sr1[8 * i / GROUP] *
                 (dot(q4_nibbles_lo(b), xa) + dot(q4_nibbles_hi(b), xb));
    }
    return float2(simd_sum(acc.x), simd_sum(acc.y));
}

kernel void gemv_split_norm_f16(device const half* W [[buffer(0)]],
                                device const half* x [[buffer(1)]],
                                device const half* bias [[buffer(2)]],
                                device half* y0 [[buffer(3)]],
                                device half* y1 [[buffer(4)]],
                                device half* y2 [[buffer(5)]],
                                constant uint& K [[buffer(6)]],
                                constant uint& N [[buffer(7)]],
                                constant uint& n0 [[buffer(8)]],
                                constant uint& n1 [[buffer(9)]],
                                constant uint& flags [[buffer(10)]],
                                device const float* nw [[buffer(11)]],
                                constant float& eps [[buffer(12)]],
                                threadgroup half* xn [[threadgroup(0)]],
                                uint tgpig [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tpt [[threads_per_threadgroup]],
                                uint sgitg [[simdgroup_index_in_threadgroup]],
                                uint tiisg [[thread_index_in_simdgroup]],
                                uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_f16_tg(W + ulong(row) * K, W + ulong(rowb) * K, xn,
                             K / 4, tiisg);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

kernel void gemv_split_norm_q8(device const char* W [[buffer(0)]],
                               device const float* scales [[buffer(1)]],
                               device const half* x [[buffer(2)]],
                               device const half* bias [[buffer(3)]],
                               device half* y0 [[buffer(4)]],
                               device half* y1 [[buffer(5)]],
                               device half* y2 [[buffer(6)]],
                               constant uint& K [[buffer(7)]],
                               constant uint& N [[buffer(8)]],
                               constant uint& n0 [[buffer(9)]],
                               constant uint& n1 [[buffer(10)]],
                               constant uint& flags [[buffer(11)]],
                               device const float* nw [[buffer(12)]],
                               constant float& eps [[buffer(13)]],
                               threadgroup half* xn [[threadgroup(0)]],
                               uint tgpig [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q8_tg(W + ulong(row) * K, W + ulong(rowb) * K, xn,
                            K / 4, tiisg);
    acc *= float2(scales[row], scales[rowb]);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

kernel void gemv_split_norm_q4(device const uchar* W [[buffer(0)]],
                               device const float* scales [[buffer(1)]],
                               device const half* x [[buffer(2)]],
                               device const half* bias [[buffer(3)]],
                               device half* y0 [[buffer(4)]],
                               device half* y1 [[buffer(5)]],
                               device half* y2 [[buffer(6)]],
                               constant uint& K [[buffer(7)]],
                               constant uint& N [[buffer(8)]],
                               constant uint& n0 [[buffer(9)]],
                               constant uint& n1 [[buffer(10)]],
                               constant uint& flags [[buffer(11)]],
                               constant uint& groups [[buffer(12)]],
                               device const float* nw [[buffer(13)]],
                               constant float& eps [[buffer(14)]],
                               threadgroup half* xn [[threadgroup(0)]],
                               uint tgpig [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q4_tg(W + ulong(row) * (K / 2),
                            W + ulong(rowb) * (K / 2),
                            scales + ulong(row) * groups,
                            scales + ulong(rowb) * groups, xn, K / 4, tiisg);
    if (tiisg == 0) {
        split_store(acc.x, row, bias, y0, y1, y2, n0, n1, flags);
        if (rowb != row) {
            split_store(acc.y, rowb, bias, y0, y1, y2, n0, n1, flags);
        }
    }
}

// FP32-output GEMV (logits head).
kernel void gemv_f16_f32(device const half* W [[buffer(0)]],
                         device const half* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint& K [[buffer(3)]],
                         constant uint& N [[buffer(4)]],
                         uint tgpig [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiisg [[thread_index_in_simdgroup]],
                         uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_f16(W + ulong(row) * K, W + ulong(rowb) * K,
                          (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        y[row] = acc.x;
        if (rowb != row) y[rowb] = acc.y;
    }
}

kernel void gemv_q4_f32(device const uchar* W [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const half* x [[buffer(2)]],
                        device float* y [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        constant uint& N [[buffer(5)]],
                        constant uint& groups [[buffer(6)]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q4(W + ulong(row) * (K / 2), W + ulong(rowb) * (K / 2),
                         scales + ulong(row) * groups,
                         scales + ulong(rowb) * groups,
                         (device const half4*)x, K / 4, tiisg);
    if (tiisg == 0) {
        y[row] = acc.x;
        if (rowb != row) y[rowb] = acc.y;
    }
}

kernel void gemv_q8_f32(device const char* W [[buffer(0)]],
                        device const float* scales [[buffer(1)]],
                        device const half* x [[buffer(2)]],
                        device float* y [[buffer(3)]],
                        constant uint& K [[buffer(4)]],
                        constant uint& N [[buffer(5)]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgpt [[simdgroups_per_threadgroup]]) {
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    float2 acc = dot2_q8(W + ulong(row) * K, W + ulong(rowb) * K,
                         (device const half4*)x, K / 4, tiisg);
    acc *= float2(scales[row], scales[rowb]);
    if (tiisg == 0) {
        y[row] = acc.x;
        if (rowb != row) y[rowb] = acc.y;
    }
}

// Wide-K q4 head (FP32 logits).
kernel void gemv_q4_f32_k16(device const uchar* W [[buffer(0)]],
                            device const float* scales [[buffer(1)]],
                            device const half* x [[buffer(2)]],
                            device float* y [[buffer(3)]],
                            constant uint& K [[buffer(4)]],
                            constant uint& N [[buffer(5)]],
                            constant uint& groups [[buffer(6)]],
                            uint tgpig [[threadgroup_position_in_grid]],
                            uint sgitg [[simdgroup_index_in_threadgroup]],
                            uint tiisg [[thread_index_in_simdgroup]],
                            uint sgpt [[simdgroups_per_threadgroup]]) {
    const uint GROUP = 128;
    uint row = (tgpig * sgpt + sgitg) * 2;
    if (row >= N) return;
    const uint rowb = min(row + 1, N - 1);
    device const uint2* a16 = (device const uint2*)(W + ulong(row) * (K / 2));
    device const uint2* b16 = (device const uint2*)(W + ulong(rowb) * (K / 2));
    device const float* s0 = scales + ulong(row) * groups;
    device const float* s1 = scales + ulong(rowb) * groups;
    device const half4* x4 = (device const half4*)x;
    const uint k16 = K / 16;
    float2 acc = float2(0.0f);
    for (uint i = tiisg; i < k16; i += 32) {
        float4 xa = float4(x4[4 * i]);
        float4 xb = float4(x4[4 * i + 1]);
        float4 xc = float4(x4[4 * i + 2]);
        float4 xd = float4(x4[4 * i + 3]);
        uint2 u = a16[i];
        uchar4 p = as_type<uchar4>(u.x);
        uchar4 q2 = as_type<uchar4>(u.y);
        acc.x += s0[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
        u = b16[i];
        p = as_type<uchar4>(u.x);
        q2 = as_type<uchar4>(u.y);
        acc.y += s1[16 * i / GROUP] *
                 (dot(q4_nibbles_lo(p), xa) + dot(q4_nibbles_hi(p), xb) +
                  dot(q4_nibbles_lo(q2), xc) + dot(q4_nibbles_hi(q2), xd));
    }
    acc = float2(simd_sum(acc.x), simd_sum(acc.y));
    if (tiisg == 0) {
        y[row] = acc.x;
        if (rowb != row) y[rowb] = acc.y;
    }
}

// Rotate-half RoPE at position pos over q (heads_q) and the cached k row
// (heads_k) in one grid: lanes [0, heads_q*hd/2) rotate q, the rest k.
kernel void rope_qk(device half* q [[buffer(0)]],
                    device half* k [[buffer(1)]],
                    constant uint& hd [[buffer(2)]],
                    constant uint& heads_q [[buffer(3)]],
                    constant float& theta [[buffer(4)]],
                    constant uint& pos [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint half_hd = hd / 2;
    const uint qlanes = heads_q * half_hd;
    device half* p;
    uint i;
    if (gid < qlanes) {
        p = q + (gid / half_hd) * hd;
        i = gid % half_hd;
    } else {
        uint g = gid - qlanes;
        p = k + (g / half_hd) * hd;
        i = g % half_hd;
    }
    float freq = pow(theta, -2.0f * float(i) / float(hd));
    float c = cos(float(pos) * freq);
    float s = sin(float(pos) * freq);
    float a = float(p[i]);
    float b = float(p[i + half_hd]);
    p[i] = half(a * c - b * s);
    p[i + half_hd] = half(b * c + a * s);
}

// Single-token attention with online softmax over T cached rows.
// One threadgroup (kAttnTpt threads) per query head: each thread scores
// one row per chunk directly from the cache (T stays small enough that
// staging tiles cost more than the strided reads), and the softmax
// statistics reduce via simd ops + a fixed-order cross-simdgroup
// combine. Deterministic: chunks are consumed in order, all reductions
// have a fixed order.
kernel void attn_decode(device const half* q [[buffer(0)]],
                        device const half* K [[buffer(1)]],
                        device const half* V [[buffer(2)]],
                        device half* out [[buffer(3)]],
                        constant uint& T [[buffer(4)]],
                        constant uint& hd [[buffer(5)]],
                        constant uint& kv_dim [[buffer(6)]],
                        constant uint& q_per_kv [[buffer(7)]],
                        constant float& scale [[buffer(8)]],
                        uint head [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiisg [[thread_index_in_simdgroup]]) {
    const uint TPT = 128;
    threadgroup float qh[128];
    threadgroup float sc[128];
    threadgroup float part[4];
    const uint koff = (head / q_per_kv) * hd;
    if (tid < hd) qh[tid] = float(q[head * hd + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;  // ctx element owned by tid < hd
    for (uint base = 0; base < T; base += TPT) {
        const uint t = base + tid;
        float s = -INFINITY;
        if (t < T) {
            device const half4* kr =
                (device const half4*)(K + ulong(t) * kv_dim + koff);
            float d = 0.0f;
            for (uint i = 0; i < hd / 4; ++i) {
                float4 kv = float4(kr[i]);
                d += qh[4 * i] * kv.x + qh[4 * i + 1] * kv.y +
                     qh[4 * i + 2] * kv.z + qh[4 * i + 3] * kv.w;
            }
            s = d * scale;
        }
        float lmax = simd_max(s);
        if (tiisg == 0) part[sgitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float cmax =
            max(max(part[0], part[1]), max(part[2], part[3]));
        const float mnew = max(m, cmax);
        const float e = (t < T) ? exp(s - mnew) : 0.0f;
        sc[tid] = e;
        float lsum = simd_sum(e);
        if (tiisg == 0) part[sgitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float esum = ((part[0] + part[1]) + (part[2] + part[3]));
        const float corr = (m == -INFINITY) ? 0.0f : exp(m - mnew);
        l = l * corr + esum;
        if (tid < hd) {
            acc *= corr;
            const uint lim = min(TPT, T - base);
            for (uint r = 0; r < lim; ++r) {
                acc += sc[r] *
                       float(V[ulong(base + r) * kv_dim + koff + tid]);
            }
        }
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < hd) out[head * hd + tid] = half(acc / l);
}

// Long-context variant: same online softmax, but the context
// accumulation is restructured for coalescing — each simdgroup owns a
// quarter of the chunk rows and its lanes read consecutive V elements
// (the per-thread-owns-a-dim layout above strides by kv_dim and is
// memory-latency bound once T grows). Fixed-order combine of the four
// simdgroup partials keeps determinism.
kernel void attn_decode_l(device const half* q [[buffer(0)]],
                          device const half* K [[buffer(1)]],
                          device const half* V [[buffer(2)]],
                          device half* out [[buffer(3)]],
                          constant uint& T [[buffer(4)]],
                          constant uint& hd [[buffer(5)]],
                          constant uint& kv_dim [[buffer(6)]],
                          constant uint& q_per_kv [[buffer(7)]],
                          constant float& scale [[buffer(8)]],
                          uint head [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint sgitg [[simdgroup_index_in_threadgroup]],
                          uint tiisg [[thread_index_in_simdgroup]]) {
    const uint TPT = 128;
    threadgroup float qh[128];
    threadgroup float sc[128];
    threadgroup float part[4];
    threadgroup float ctx[4][128];
    const uint koff = (head / q_per_kv) * hd;
    if (tid < hd) qh[tid] = float(q[head * hd + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // dims tiisg + 32k
    const uint kdim = hd / 32;
    for (uint base = 0; base < T; base += TPT) {
        const uint t = base + tid;
        float s = -INFINITY;
        if (t < T) {
            device const half4* kr =
                (device const half4*)(K + ulong(t) * kv_dim + koff);
            float d = 0.0f;
            for (uint i = 0; i < hd / 4; ++i) {
                float4 kv = float4(kr[i]);
                d += qh[4 * i] * kv.x + qh[4 * i + 1] * kv.y +
                     qh[4 * i + 2] * kv.z + qh[4 * i + 3] * kv.w;
            }
            s = d * scale;
        }
        float lmax = simd_max(s);
        if (tiisg == 0) part[sgitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float cmax =
            max(max(part[0], part[1]), max(part[2], part[3]));
        const float mnew = max(m, cmax);
        const float e = (t < T) ? exp(s - mnew) : 0.0f;
        sc[tid] = e;
        float lsum = simd_sum(e);
        if (tiisg == 0) part[sgitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float esum = ((part[0] + part[1]) + (part[2] + part[3]));
        const float corr = (m == -INFINITY) ? 0.0f : exp(m - mnew);
        l = l * corr + esum;
        const uint lim = min(TPT, T - base);
        for (uint k = 0; k < kdim; ++k) acc[k] *= corr;
        for (uint r = sgitg; r < lim; r += 4) {
            const float e_r = sc[r];
            device const half* vr = V + ulong(base + r) * kv_dim + koff;
            for (uint k = 0; k < kdim; ++k) {
                acc[k] += e_r * float(vr[tiisg + 32 * k]);
            }
        }
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = 0; k < kdim; ++k) ctx[sgitg][tiisg + 32 * k] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hd) {
        float v = ((ctx[0][tid] + ctx[1][tid]) + (ctx[2][tid] + ctx[3][tid]));
        out[head * hd + tid] = half(v / l);
    }
}

// Split-row attention for long contexts: grid = heads x S threadgroups,
// each running the same online-softmax pass over its contiguous row
// slice and writing a partial (m, l, ctx[hd]); a tiny second dispatch
// merges the S partials per head in fixed order (deterministic).
kernel void attn_part(device const half* q [[buffer(0)]],
                      device const half* K [[buffer(1)]],
                      device const half* V [[buffer(2)]],
                      device float* pm [[buffer(3)]],
                      device float* pl [[buffer(4)]],
                      device float* pa [[buffer(5)]],
                      constant uint& T [[buffer(6)]],
                      constant uint& hd [[buffer(7)]],
                      constant uint& kv_dim [[buffer(8)]],
                      constant uint& q_per_kv [[buffer(9)]],
                      constant float& scale [[buffer(10)]],
                      constant uint& S [[buffer(11)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint3 tid3 [[thread_position_in_threadgroup]],
                      uint sgitg [[simdgroup_index_in_threadgroup]],
                      uint tiisg [[thread_index_in_simdgroup]]) {
    const uint TPT = 128;
    const uint tid = tid3.x;
    const uint head = tg.x;
    const uint sidx = tg.y;
    const uint pid = head * S + sidx;
    const uint seg = (T + S - 1) / S;
    const uint start = sidx * seg;
    const uint end = min(start + seg, T);
    threadgroup float qh[128];
    threadgroup float sc[128];
    threadgroup float part[4];
    threadgroup float ctx[4][128];
    const uint koff = (head / q_per_kv) * hd;
    if (tid < hd) qh[tid] = float(q[head * hd + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint kdim = hd / 32;
    for (uint base = start; base < end; base += TPT) {
        const uint t = base + tid;
        float s = -INFINITY;
        if (t < end) {
            device const half4* kr =
                (device const half4*)(K + ulong(t) * kv_dim + koff);
            float d = 0.0f;
            for (uint i = 0; i < hd / 4; ++i) {
                float4 kv = float4(kr[i]);
                d += qh[4 * i] * kv.x + qh[4 * i + 1] * kv.y +
                     qh[4 * i + 2] * kv.z + qh[4 * i + 3] * kv.w;
            }
            s = d * scale;
        }
        float lmax = simd_max(s);
        if (tiisg == 0) part[sgitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float cmax =
            max(max(part[0], part[1]), max(part[2], part[3]));
        const float mnew = max(m, cmax);
        const float e = (t < end) ? exp(s - mnew) : 0.0f;
        sc[tid] = e;
        float lsum = simd_sum(e);
        if (tiisg == 0) part[sgitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float esum = ((part[0] + part[1]) + (part[2] + part[3]));
        const float corr = (m == -INFINITY) ? 0.0f : exp(m - mnew);
        l = l * corr + esum;
        const uint lim = min(TPT, end - base);
        for (uint k = 0; k < kdim; ++k) acc[k] *= corr;
        for (uint r = sgitg; r < lim; r += 4) {
            const float e_r = sc[r];
            device const half* vr = V + ulong(base + r) * kv_dim + koff;
            for (uint k = 0; k < kdim; ++k) {
                acc[k] += e_r * float(vr[tiisg + 32 * k]);
            }
        }
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = 0; k < kdim; ++k) ctx[sgitg][tiisg + 32 * k] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        pm[pid] = m;
        pl[pid] = l;
    }
    if (tid < hd) {
        pa[ulong(pid) * hd + tid] =
            ((ctx[0][tid] + ctx[1][tid]) + (ctx[2][tid] + ctx[3][tid]));
    }
}

kernel void attn_merge(device const float* pm [[buffer(0)]],
                       device const float* pl [[buffer(1)]],
                       device const float* pa [[buffer(2)]],
                       device half* out [[buffer(3)]],
                       constant uint& hd [[buffer(4)]],
                       constant uint& S [[buffer(5)]],
                       uint head [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]]) {
    float m = -INFINITY;
    for (uint s = 0; s < S; ++s) m = max(m, pm[head * S + s]);
    float l = 0.0f;
    float acc = 0.0f;
    for (uint s = 0; s < S; ++s) {
        const float pms = pm[head * S + s];
        const float w = (pms == -INFINITY) ? 0.0f : exp(pms - m);
        l += w * pl[head * S + s];
        if (tid < hd) acc += w * pa[ulong(head * S + s) * hd + tid];
    }
    if (tid < hd) out[head * hd + tid] = half(acc / l);
}

kernel void silu_mul(device const half* gate [[buffer(0)]],
                     device const half* up [[buffer(1)]],
                     device half* out [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    out[gid] = half(g / (1.0f + exp(-g)) * u);
}

// ---- greedy-run support: on-GPU argmax + embedding gather -----------
// Argmax matches Sampler's greedy rule exactly: maximum value, ties to
// the lowest index. NaNs never win a (v > best) compare, so a poisoned
// logits vector yields index 0 and the run-final finite scan reports
// the failure. Two stages over 256 contiguous chunks (deterministic).

kernel void argmax_stage1(device const float* logits [[buffer(0)]],
                          constant uint& n [[buffer(1)]],
                          device float* pmax [[buffer(2)]],
                          device uint* pidx [[buffer(3)]],
                          uint tgpig [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tpt [[threads_per_threadgroup]]) {
    threadgroup float bv[256];
    threadgroup uint bi[256];
    const uint chunk = (n + 255) / 256;
    const uint start = tgpig * chunk;
    const uint end = min(start + chunk, n);
    float best = -INFINITY;
    uint besti = 0xFFFFFFFFu;
    for (uint i = start + tid; i < end; i += tpt) {
        float v = logits[i];
        if (v > best || (v == best && i < besti)) {
            best = v;
            besti = i;
        }
    }
    bv[tid] = best;
    bi[tid] = besti;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpt / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool take = bv[tid + s] > bv[tid] ||
                        (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]);
            if (take) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        pmax[tgpig] = bv[0];
        pidx[tgpig] = bi[0];
    }
}

kernel void argmax_stage2(device const float* pmax [[buffer(0)]],
                          device const uint* pidx [[buffer(1)]],
                          constant uint& slot [[buffer(2)]],
                          device uint* tokens [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tpt [[threads_per_threadgroup]]) {
    threadgroup float bv[256];
    threadgroup uint bi[256];
    bv[tid] = pmax[tid];
    bi[tid] = pidx[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpt / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool take = bv[tid + s] > bv[tid] ||
                        (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]);
            if (take) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tokens[slot] = bi[0];
}

// Stage-2 reduce and next-step embedding gather in one dispatch: after
// lane 0 publishes the winner, all 256 threads copy its embedding row.
kernel void argmax2_gather(device const float* pmax [[buffer(0)]],
                           device const uint* pidx [[buffer(1)]],
                           constant uint& slot [[buffer(2)]],
                           device uint* tokens [[buffer(3)]],
                           device const half* embed [[buffer(4)]],
                           constant uint& h [[buffer(5)]],
                           device half* x [[buffer(6)]],
                           constant uint& do_gather [[buffer(7)]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpt [[threads_per_threadgroup]]) {
    threadgroup float bv[256];
    threadgroup uint bi[256];
    bv[tid] = pmax[tid];
    bi[tid] = pidx[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpt / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool take = bv[tid + s] > bv[tid] ||
                        (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid]);
            if (take) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tokens[slot] = bi[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (do_gather != 0) {
        const ulong roff = ulong(bi[0]) * h;
        for (uint i = tid; i < h; i += tpt) x[i] = embed[roff + i];
    }
}

kernel void embed_gather(device const half* embed [[buffer(0)]],
                         device const uint* tokens [[buffer(1)]],
                         constant uint& slot [[buffer(2)]],
                         constant uint& h [[buffer(3)]],
                         device half* x [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= h) return;
    x[gid] = embed[ulong(tokens[slot]) * h + gid];
}
)MSL";

// Reads a verified tensor as host FP32.
Status TensorToHost(const SafetensorsFile& file, const std::string& name,
                    std::vector<float>& out) {
    const TensorInfo* info = file.FindTensor(name);
    const uint8_t* data = file.TensorData(name);
    if (info == nullptr || data == nullptr) {
        return Status(ErrorCode::kArtifactVerificationFailed,
                      "expected weight tensor missing", kComponent);
    }
    out.resize(info->element_count);
    switch (info->dtype) {
        case Dtype::kF32:
            std::memcpy(out.data(), data, info->data_size);
            break;
        case Dtype::kBf16:
            Bf16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             out.data(), info->element_count);
            break;
        case Dtype::kF16:
            Fp16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             out.data(), info->element_count);
            break;
    }
    return Status::Ok();
}

// FP16 KV cache with lazy bucket growth (same policy as the MPSGraph
// backend's sequence state, minus the graph-side view cache).
class FastSequenceState final : public SequenceState {
public:
    static Status Create(id<MTLDevice> device, const QwenConfig& config,
                         uint32_t max_tokens,
                         std::unique_ptr<FastSequenceState>& out) {
        auto st = std::unique_ptr<FastSequenceState>(new FastSequenceState());
        st->device_ = device;
        st->kv_stride_ = config.num_kv_heads * config.head_dim;
        st->max_tokens_ = max_tokens;
        st->keys_.resize(config.num_layers);
        st->values_.resize(config.num_layers);
        Status s = st->Reserve(std::min<uint32_t>(max_tokens, 2 * kCtxBucket));
        if (!s.ok()) return s;
        out = std::move(st);
        return Status::Ok();
    }

    Status Reserve(uint32_t tokens) {
        const uint32_t want =
            ((tokens + kCtxBucket - 1) / kCtxBucket) * kCtxBucket;
        if (want <= alloc_tokens_) return Status::Ok();
        uint32_t grown = alloc_tokens_ == 0 ? want : alloc_tokens_;
        while (grown < want) grown *= 2;
        grown = std::min(grown, ((max_tokens_ + kCtxBucket - 1) / kCtxBucket) *
                                    kCtxBucket);
        const size_t new_bytes = size_t(grown) * kv_stride_ * 2;
        const size_t old_bytes = size_t(alloc_tokens_) * kv_stride_ * 2;
        bool oom = false;
        @autoreleasepool {
            for (uint32_t l = 0; l < keys_.size(); ++l) {
                id<MTLBuffer> nk =
                    [device_ newBufferWithLength:new_bytes
                                         options:MTLResourceStorageModeShared];
                id<MTLBuffer> nv =
                    [device_ newBufferWithLength:new_bytes
                                         options:MTLResourceStorageModeShared];
                if (nk == nil || nv == nil) {
                    oom = true;
                    break;
                }
                std::memset(nk.contents, 0, new_bytes);
                std::memset(nv.contents, 0, new_bytes);
                if (keys_[l] != nil && old_bytes > 0) {
                    std::memcpy(nk.contents, keys_[l].contents, old_bytes);
                    std::memcpy(nv.contents, values_[l].contents, old_bytes);
                }
                keys_[l] = nk;
                values_[l] = nv;
            }
        }
        if (oom) {
            return Status(ErrorCode::kGpuOom, "kv cache allocation failed",
                          kComponent);
        }
        alloc_tokens_ = grown;
        return Status::Ok();
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }
    uint32_t kv_stride() const { return kv_stride_; }
    id<MTLBuffer> key_buffer(uint32_t l) { return keys_[l]; }
    id<MTLBuffer> value_buffer(uint32_t l) { return values_[l]; }
    void Advance() { ++length_; }

    // Never-reused identity, used to validate a model-level speculative
    // batch against the sequence it was issued for.
    uint64_t seq_id() const { return id_; }
    // Invoked on destruction so the model can drain a speculative batch
    // issued for this sequence (otherwise it would keep the GPU busy
    // into the next request's prefill).
    void set_on_destroy(std::function<void(uint64_t)> cb) {
        on_destroy_ = std::move(cb);
    }

    ~FastSequenceState() {
        if (on_destroy_) on_destroy_(id_);
        @autoreleasepool {
            for (auto& b : keys_) b = nil;
            for (auto& b : values_) b = nil;
        }
    }

private:
    FastSequenceState() {
        static std::atomic<uint64_t> next_id{1};
        id_ = next_id.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t id_ = 0;
    std::function<void(uint64_t)> on_destroy_;
    id<MTLDevice> device_ = nil;
    uint32_t kv_stride_ = 0;
    uint32_t max_tokens_ = 0;
    uint32_t alloc_tokens_ = 0;
    uint32_t length_ = 0;
    std::vector<id<MTLBuffer>> keys_;
    std::vector<id<MTLBuffer>> values_;
};

// Load-time quantization, CUDA scheme (qwen_cuda_model.h).
void QuantizeInt8(const std::vector<float>& w, uint32_t rows, uint32_t cols,
                  std::vector<int8_t>& out, std::vector<float>& scales) {
    out.resize(w.size());
    scales.resize(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* src = w.data() + size_t(r) * cols;
        float amax = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            amax = std::max(amax, std::abs(src[c]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        scales[r] = scale;
        int8_t* dst = out.data() + size_t(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            float q = std::nearbyint(src[c] / scale);
            dst[c] = int8_t(std::max(-127.0f, std::min(127.0f, q)));
        }
    }
}

void QuantizeInt4(const std::vector<float>& w, uint32_t rows, uint32_t cols,
                  std::vector<uint8_t>& out, std::vector<float>& scales,
                  uint32_t& groups_out) {
    const uint32_t groups = (cols + kQ4Group - 1) / kQ4Group;
    groups_out = groups;
    out.assign(size_t(rows) * cols / 2, 0);
    scales.resize(size_t(rows) * groups);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* src = w.data() + size_t(r) * cols;
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t c0 = g * kQ4Group;
            const uint32_t c1 = std::min(cols, c0 + kQ4Group);
            float amax = 0.0f;
            for (uint32_t c = c0; c < c1; ++c) {
                amax = std::max(amax, std::abs(src[c]));
            }
            const float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            scales[size_t(r) * groups + g] = scale;
            for (uint32_t c = c0; c < c1; ++c) {
                float q = std::nearbyint(src[c] / scale);
                int nib = int(std::max(-7.0f, std::min(7.0f, q))) + 8;
                uint8_t& b = out[(size_t(r) * cols + c) / 2];
                if ((c & 1) == 0) {
                    b = uint8_t((b & 0xF0) | nib);
                } else {
                    b = uint8_t((b & 0x0F) | (nib << 4));
                }
            }
        }
    }
}

}  // namespace

struct QwenMetalFastModel::Impl {
    MetalFastOptions::Quant quant = MetalFastOptions::Quant::kFp16;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> ps_rmsnorm = nil;
    id<MTLComputePipelineState> ps_gemv = nil;        // mode-selected
    id<MTLComputePipelineState> ps_gemv_split = nil;  // mode-selected
    // Wide-row variants (threadgroup per row) for the small layer
    // projections, where simdgroup-per-row underoccupies the GPU.
    id<MTLComputePipelineState> ps_gemv_wide = nil;
    id<MTLComputePipelineState> ps_gemv_split_wide = nil;
    id<MTLComputePipelineState> ps_gemv_f32 = nil;    // head (mode-selected)
    // RMSNorm-fused split GEMV (dynamic threadgroup memory), used when
    // hidden_size fits threadgroup memory.
    id<MTLComputePipelineState> ps_gemv_split_norm = nil;
    bool norm_fused = false;
    id<MTLComputePipelineState> ps_rope_qk = nil;
    id<MTLComputePipelineState> ps_attn = nil;
    id<MTLComputePipelineState> ps_attn_l = nil;  // long-context variant
    id<MTLComputePipelineState> ps_attn_part = nil;   // split-row partial
    id<MTLComputePipelineState> ps_attn_merge = nil;  // fixed-order merge
    id<MTLComputePipelineState> ps_silu_mul = nil;
    id<MTLComputePipelineState> ps_gemv_k16 = nil;  // wide-K q4 variant
    id<MTLComputePipelineState> ps_gemv_f32_k16 = nil;
    id<MTLComputePipelineState> ps_qkv = nil;      // norm+QKV+bias+RoPE
    id<MTLComputePipelineState> ps_gateup = nil;   // norm+gate/up+SiLU
    id<MTLComputePipelineState> ps_gateup_k16 = nil;  // wide-K q4 variant
    id<MTLComputePipelineState> ps_am2_gather = nil;
    // Greedy-run support (on-GPU sampling between speculative steps).
    id<MTLComputePipelineState> ps_argmax1 = nil;
    id<MTLComputePipelineState> ps_argmax2 = nil;
    id<MTLComputePipelineState> ps_gather = nil;
    id<MTLBuffer> embed_gpu = nil;  // half [vocab, h]
    id<MTLBuffer> tok_buf = nil;    // uint32 [kTokBuf]
    // Ping-pong argmax scratch + logits so a speculative batch can run
    // while the CPU still reads the previous batch's results.
    id<MTLBuffer> am_val[2] = {nil, nil};   // float [256] each
    id<MTLBuffer> am_idx[2] = {nil, nil};   // uint32 [256] each
    id<MTLBuffer> logits2 = nil;            // float [vocab] (parity 1)

    // The one in-flight speculative greedy batch. It shares the resident
    // activation/result buffers with every other operation on this
    // model, so ANY other encode must drain (and invalidate) it first;
    // the sequence id (never reused) guards against stale collection.
    id<MTLCommandBuffer> spec_cb = nil;
    uint64_t spec_seq_id = 0;
    uint32_t spec_nsteps = 0;
    uint32_t spec_parity = 0;
    uint32_t spec_expected = 0;
    uint32_t spec_base_len = 0;

    void DrainSpec() {
        if (spec_cb != nil) {
            [spec_cb waitUntilCompleted];
            spec_cb = nil;
            spec_seq_id = 0;
        }
    }
    // Split-row attention partials: [heads * kAttnMaxSplit] each.
    id<MTLBuffer> at_m = nil;
    id<MTLBuffer> at_l = nil;
    id<MTLBuffer> at_a = nil;  // float [heads * kAttnMaxSplit * hd]

    // One projection matrix: FP16 data, or quantized data + scales.
    struct Mat {
        id<MTLBuffer> data = nil;    // half[N*K] | int8[N*K] | u8[N*K/2]
        id<MTLBuffer> scales = nil;  // float[N] (q8) | float[N*groups] (q4)
        uint32_t rows = 0;           // N
        uint32_t cols = 0;           // K
        uint32_t groups = 0;         // q4 only
        MetalFastOptions::Quant q = MetalFastOptions::Quant::kFp16;
    };
    struct Layer {
        id<MTLBuffer> input_norm = nil, post_norm = nil;  // float[h]
        Mat qkv;      // [q_dim + 2*kv_dim, h], bias fused
        Mat ow;       // [h, q_dim]
        Mat gate_up;  // [2*inter, h]
        Mat down;     // [h, inter]
        id<MTLBuffer> qkv_bias = nil;  // half[q_dim + 2*kv_dim]
    };
    std::vector<Layer> layers;
    id<MTLBuffer> final_norm = nil;    // float[h]
    Mat head;                          // [vocab, h]
    std::vector<uint16_t> embed_fp16;  // host [vocab, h]

    // Persistent activation buffers (generation is serialized per model).
    id<MTLBuffer> x = nil;         // half[h]
    id<MTLBuffer> xn = nil;        // half[h]
    id<MTLBuffer> q = nil;         // half[q_dim]
    id<MTLBuffer> attn_out = nil;  // half[q_dim]
    id<MTLBuffer> gate_a = nil;    // half[inter]
    id<MTLBuffer> up_a = nil;      // half[inter]
    id<MTLBuffer> act = nil;       // half[inter]
    id<MTLBuffer> logits = nil;    // float[vocab]

    // Uploads one projection from host FP32 in the mode-selected format.
    // `head_min_q8` upgrades INT4 to INT8 (logits/quality floor for the
    // lm head, mirroring the CUDA policy of protecting the head).
    bool LoadMat(const std::vector<float>& host, uint32_t rows,
                 uint32_t cols, Mat& mat, bool head_min_q8);

    ~Impl() = default;
};

QwenMetalFastModel::~QwenMetalFastModel() = default;

bool QwenMetalFastModel::Impl::LoadMat(const std::vector<float>& host,
                                       uint32_t rows, uint32_t cols,
                                       Mat& mat, bool head_min_q8) {
    mat.rows = rows;
    mat.cols = cols;
    MetalFastOptions::Quant q = quant;
    if (head_min_q8 && q == MetalFastOptions::Quant::kInt4 &&
        std::getenv("LYKURO_METAL_FAST_HEAD_Q8") != nullptr) {
        // Optional quality floor for the lm head (CUDA-policy style);
        // INT4+group scales measured close enough to ship as default.
        q = MetalFastOptions::Quant::kInt8;
    }
    mat.q = q;
    switch (q) {
        case MetalFastOptions::Quant::kFp16: {
            std::vector<uint16_t> hh(host.size());
            for (size_t i = 0; i < host.size(); ++i) {
                hh[i] = FloatToFp16(host[i]);
            }
            mat.data =
                [device newBufferWithBytes:hh.data()
                                    length:hh.size() * 2
                                   options:MTLResourceStorageModeShared];
            return mat.data != nil;
        }
        case MetalFastOptions::Quant::kInt8: {
            std::vector<int8_t> qd;
            std::vector<float> sc;
            QuantizeInt8(host, rows, cols, qd, sc);
            mat.data =
                [device newBufferWithBytes:qd.data()
                                    length:qd.size()
                                   options:MTLResourceStorageModeShared];
            mat.scales =
                [device newBufferWithBytes:sc.data()
                                    length:sc.size() * sizeof(float)
                                   options:MTLResourceStorageModeShared];
            return mat.data != nil && mat.scales != nil;
        }
        case MetalFastOptions::Quant::kInt4: {
            std::vector<uint8_t> qd;
            std::vector<float> sc;
            QuantizeInt4(host, rows, cols, qd, sc, mat.groups);
            mat.data =
                [device newBufferWithBytes:qd.data()
                                    length:qd.size()
                                   options:MTLResourceStorageModeShared];
            mat.scales =
                [device newBufferWithBytes:sc.data()
                                    length:sc.size() * sizeof(float)
                                   options:MTLResourceStorageModeShared];
            return mat.data != nil && mat.scales != nil;
        }
    }
    return false;
}

QwenMetalFastModel::LoadResult QwenMetalFastModel::Load(
    const ModelManifest& manifest, const SafetensorsFile& weights,
    const MetalFastOptions& options) {
    LoadResult result;
    @autoreleasepool {
        auto model =
            std::unique_ptr<QwenMetalFastModel>(new QwenMetalFastModel());
        result.status = QwenConfig::FromManifest(manifest, model->config_);
        if (!result.status.ok()) return result;
        const QwenConfig& c = model->config_;
        model->limits_.vocab_size = c.vocab_size;
        model->limits_.max_context_tokens = c.max_context_tokens;
        model->limits_.eos_token_ids = c.eos_token_ids;

        // The vectorized GEMVs load 8 elements at a time (uchar4 in the
        // INT4 path), so every reduction dimension must be 8-aligned.
        if (c.hidden_size % 8 != 0 || c.intermediate_size % 8 != 0 ||
            (c.num_kv_heads * c.head_dim) % 8 != 0 || c.head_dim % 4 != 0) {
            result.status = Status(ErrorCode::kUnsupportedModel,
                                   "dimensions not vector-aligned",
                                   kComponent);
            return result;
        }

        model->impl_ = std::make_unique<Impl>();
        Impl& impl = *model->impl_;
        impl.quant = options.quant;
        impl.device = MTLCreateSystemDefaultDevice();
        if (impl.device == nil || !impl.device.hasUnifiedMemory) {
            result.status = MetalFailed("metal device unavailable");
            return result;
        }
        impl.queue = [impl.device newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [impl.device
            newLibraryWithSource:[NSString stringWithUTF8String:kMsl]
                         options:nil
                           error:&err];
        if (lib == nil) {
            std::fprintf(stderr, "metal library compile: %s\n",
                         err != nil ? err.localizedDescription.UTF8String
                                    : "(no error info)");
            result.status = MetalFailed("kernel library compile failed");
            return result;
        }
        auto pso = [&](const char* name,
                       id<MTLComputePipelineState> __strong& out) -> bool {
            id<MTLFunction> f = [lib
                newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (f == nil) return false;
            out = [impl.device newComputePipelineStateWithFunction:f
                                                             error:&err];
            return out != nil;
        };
        const bool q8 = options.quant == MetalFastOptions::Quant::kInt8;
        const bool q4 = options.quant == MetalFastOptions::Quant::kInt4;
        bool ok =
            pso("rmsnorm", impl.ps_rmsnorm) &&
            pso(q4 ? "gemv_q4" : (q8 ? "gemv_q8" : "gemv_f16"),
                impl.ps_gemv) &&
            pso(q4 ? "gemv_split_q4"
                   : (q8 ? "gemv_split_q8" : "gemv_split_f16"),
                impl.ps_gemv_split) &&
            pso(q4 ? (std::getenv("LYKURO_METAL_FAST_HEAD_Q8") != nullptr
                          ? "gemv_q8_f32"
                          : "gemv_q4_f32")
                   : (q8 ? "gemv_q8_f32" : "gemv_f16_f32"),
                impl.ps_gemv_f32) &&
            pso(q4 ? "gemv_wide_q4" : (q8 ? "gemv_wide_q8" : "gemv_wide_f16"),
                impl.ps_gemv_wide) &&
            pso(q4 ? "gemv_split_wide_q4"
                   : (q8 ? "gemv_split_wide_q8" : "gemv_split_wide_f16"),
                impl.ps_gemv_split_wide) &&
            pso("rope_qk", impl.ps_rope_qk) &&
            pso("attn_decode", impl.ps_attn) &&
            pso("attn_decode_l", impl.ps_attn_l) &&
            pso("attn_part", impl.ps_attn_part) &&
            pso("attn_merge", impl.ps_attn_merge) &&
            pso("silu_mul", impl.ps_silu_mul) &&
            (!q4 || (pso("gemv_q4_k16", impl.ps_gemv_k16) &&
                     pso("gemv_gateup_q4_k16", impl.ps_gateup_k16) &&
                     pso("gemv_q4_f32_k16", impl.ps_gemv_f32_k16))) &&
            pso(q4 ? "gemv_qkv_q4" : (q8 ? "gemv_qkv_q8" : "gemv_qkv_f16"),
                impl.ps_qkv) &&
            pso(q4 ? "gemv_gateup_q4"
                   : (q8 ? "gemv_gateup_q8" : "gemv_gateup_f16"),
                impl.ps_gateup) &&
            pso("argmax2_gather", impl.ps_am2_gather) &&
            pso("argmax_stage1", impl.ps_argmax1) &&
            pso("argmax_stage2", impl.ps_argmax2) &&
            pso("embed_gather", impl.ps_gather);
        // RMSNorm fusion measured slower on M4 Pro even with dynamic
        // threadgroup memory (the per-threadgroup norm barriers cost more
        // than the one small dispatch they replace); opt-in for other
        // chips.
        impl.norm_fused =
            c.hidden_size <= 8192 &&
            std::getenv("LYKURO_METAL_FAST_NORM_FUSED") != nullptr;
        if (ok && impl.norm_fused) {
            ok = pso(q4 ? "gemv_split_norm_q4"
                        : (q8 ? "gemv_split_norm_q8" : "gemv_split_norm_f16"),
                     impl.ps_gemv_split_norm);
        }
        if (!ok) {
            result.status = MetalFailed("pipeline state creation failed");
            return result;
        }

        const uint32_t h = c.hidden_size;
        const uint32_t q_dim = c.num_heads * c.head_dim;
        const uint32_t kv_dim = c.num_kv_heads * c.head_dim;

        std::vector<float> host;
        auto upload_f32 = [&](const std::vector<float>& v,
                              id<MTLBuffer> __strong& buf) -> bool {
            buf = [impl.device newBufferWithBytes:v.data()
                                           length:v.size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
            return buf != nil;
        };
        auto upload_f16 = [&](const std::vector<float>& v,
                              id<MTLBuffer> __strong& buf) -> bool {
            std::vector<uint16_t> hh(v.size());
            for (size_t i = 0; i < v.size(); ++i) hh[i] = FloatToFp16(v[i]);
            buf = [impl.device newBufferWithBytes:hh.data()
                                           length:hh.size() * 2
                                          options:MTLResourceStorageModeShared];
            return buf != nil;
        };
        auto oom = [&]() {
            return Status(ErrorCode::kGpuOom, "weight upload failed",
                          kComponent);
        };

        impl.layers.resize(c.num_layers);
        Status s = Status::Ok();
        std::vector<float> tmp;
        for (uint32_t l = 0; l < c.num_layers && s.ok(); ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            Impl::Layer& L = impl.layers[l];
            s = TensorToHost(weights, p + "input_layernorm.weight", host);
            if (s.ok() && !upload_f32(host, L.input_norm)) s = oom();
            if (s.ok()) {
                s = TensorToHost(weights, p + "post_attention_layernorm.weight",
                                 host);
            }
            if (s.ok() && !upload_f32(host, L.post_norm)) s = oom();

            // Fused QKV in ROTATION-PAIR order: the q section emits rows
            // as (head*hd+i, head*hd+i+hd/2) pairs, k likewise, v rows
            // unchanged — the dual-row GEMV simdgroup then owns exactly
            // one RoPE pair and rotates in its epilogue. The fused bias
            // is permuted identically.
            if (s.ok()) {
                const uint32_t half_hd = c.head_dim / 2;
                std::vector<float> fused;
                std::vector<float> fused_bias;
                fused.reserve(size_t(q_dim + 2 * kv_dim) * h);
                fused_bias.reserve(q_dim + 2 * kv_dim);
                auto append_paired = [&](const std::vector<float>& w,
                                         const std::vector<float>& b,
                                         uint32_t heads) {
                    for (uint32_t hh = 0; hh < heads; ++hh) {
                        for (uint32_t i = 0; i < half_hd; ++i) {
                            const uint32_t ra = hh * c.head_dim + i;
                            const uint32_t rb = ra + half_hd;
                            fused.insert(fused.end(),
                                         w.begin() + size_t(ra) * h,
                                         w.begin() + size_t(ra + 1) * h);
                            fused.insert(fused.end(),
                                         w.begin() + size_t(rb) * h,
                                         w.begin() + size_t(rb + 1) * h);
                            fused_bias.push_back(b[ra]);
                            fused_bias.push_back(b[rb]);
                        }
                    }
                };
                std::vector<float> wq, wk, wv, bq, bk, bv;
                s = TensorToHost(weights, p + "self_attn.q_proj.weight", wq);
                if (s.ok())
                    s = TensorToHost(weights, p + "self_attn.k_proj.weight",
                                     wk);
                if (s.ok())
                    s = TensorToHost(weights, p + "self_attn.v_proj.weight",
                                     wv);
                if (s.ok())
                    s = TensorToHost(weights, p + "self_attn.q_proj.bias",
                                     bq);
                if (s.ok())
                    s = TensorToHost(weights, p + "self_attn.k_proj.bias",
                                     bk);
                if (s.ok())
                    s = TensorToHost(weights, p + "self_attn.v_proj.bias",
                                     bv);
                if (s.ok()) {
                    append_paired(wq, bq, c.num_heads);
                    append_paired(wk, bk, c.num_kv_heads);
                    fused.insert(fused.end(), wv.begin(), wv.end());
                    fused_bias.insert(fused_bias.end(), bv.begin(),
                                      bv.end());
                    if (!impl.LoadMat(fused, q_dim + 2 * kv_dim, h, L.qkv,
                                      false) ||
                        !upload_f16(fused_bias, L.qkv_bias)) {
                        s = oom();
                    }
                }
            }
            if (s.ok()) {
                s = TensorToHost(weights, p + "self_attn.o_proj.weight", tmp);
                if (s.ok() && !impl.LoadMat(tmp, h, q_dim, L.ow, false)) {
                    s = oom();
                }
            }
            // Fused gate/up, ROW-INTERLEAVED (2j = gate_j, 2j+1 = up_j)
            // so one dual-row simdgroup computes the SiLU pair.
            if (s.ok()) {
                std::vector<float> gate_h;
                s = TensorToHost(weights, p + "mlp.gate_proj.weight", gate_h);
                if (s.ok()) {
                    s = TensorToHost(weights, p + "mlp.up_proj.weight", tmp);
                }
                if (s.ok()) {
                    std::vector<float> fused(size_t(2 * c.intermediate_size) *
                                             h);
                    for (uint32_t j = 0; j < c.intermediate_size; ++j) {
                        std::memcpy(fused.data() + size_t(2 * j) * h,
                                    gate_h.data() + size_t(j) * h,
                                    h * sizeof(float));
                        std::memcpy(fused.data() + size_t(2 * j + 1) * h,
                                    tmp.data() + size_t(j) * h,
                                    h * sizeof(float));
                    }
                    if (!impl.LoadMat(fused, 2 * c.intermediate_size, h,
                                      L.gate_up, false)) {
                        s = oom();
                    }
                }
            }
            if (s.ok()) {
                s = TensorToHost(weights, p + "mlp.down_proj.weight", tmp);
                if (s.ok() && !impl.LoadMat(tmp, h, c.intermediate_size,
                                            L.down, false)) {
                    s = oom();
                }
            }
        }
        if (s.ok()) s = TensorToHost(weights, "model.norm.weight", host);
        if (s.ok() && !upload_f32(host, impl.final_norm)) s = oom();
        if (s.ok()) {
            std::vector<float> embed;
            s = TensorToHost(weights, "model.embed_tokens.weight", embed);
            if (s.ok()) {
                impl.embed_fp16.resize(embed.size());
                for (size_t i = 0; i < embed.size(); ++i) {
                    impl.embed_fp16[i] = FloatToFp16(embed[i]);
                }
                if (c.tie_word_embeddings) {
                    if (!impl.LoadMat(embed, c.vocab_size, h, impl.head,
                                      true)) {
                        s = oom();
                    }
                } else {
                    std::vector<float> head;
                    s = TensorToHost(weights, "lm_head.weight", head);
                    if (s.ok() && !impl.LoadMat(head, c.vocab_size, h,
                                                impl.head, true)) {
                        s = oom();
                    }
                }
            }
        }

        // Activation buffers.
        if (s.ok()) {
            auto alloc = [&](size_t bytes) {
                return [impl.device
                    newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];
            };
            impl.x = alloc(h * 2);
            impl.xn = alloc(h * 2);
            impl.q = alloc(size_t(q_dim) * 2);
            impl.attn_out = alloc(size_t(q_dim) * 2);
            impl.gate_a = alloc(size_t(c.intermediate_size) * 2);
            impl.up_a = alloc(size_t(c.intermediate_size) * 2);
            impl.act = alloc(size_t(c.intermediate_size) * 2);
            impl.logits = alloc(size_t(c.vocab_size) * sizeof(float));
            impl.tok_buf = alloc(kTokBuf * sizeof(uint32_t));
            impl.am_val[0] = alloc(256 * sizeof(float));
            impl.am_val[1] = alloc(256 * sizeof(float));
            impl.am_idx[0] = alloc(256 * sizeof(uint32_t));
            impl.am_idx[1] = alloc(256 * sizeof(uint32_t));
            impl.logits2 = alloc(size_t(c.vocab_size) * sizeof(float));
            impl.at_m = alloc(size_t(c.num_heads) * kAttnMaxSplit *
                              sizeof(float));
            impl.at_l = alloc(size_t(c.num_heads) * kAttnMaxSplit *
                              sizeof(float));
            impl.at_a = alloc(size_t(c.num_heads) * kAttnMaxSplit *
                              c.head_dim * sizeof(float));
            impl.embed_gpu = [impl.device
                newBufferWithBytes:impl.embed_fp16.data()
                            length:impl.embed_fp16.size() * 2
                           options:MTLResourceStorageModeShared];
            if (impl.x == nil || impl.xn == nil || impl.q == nil ||
                impl.attn_out == nil || impl.gate_a == nil ||
                impl.up_a == nil || impl.act == nil || impl.logits == nil ||
                impl.tok_buf == nil || impl.am_val[0] == nil ||
                impl.am_val[1] == nil || impl.am_idx[0] == nil ||
                impl.am_idx[1] == nil || impl.logits2 == nil ||
                impl.at_m == nil || impl.at_l == nil || impl.at_a == nil ||
                impl.embed_gpu == nil) {
                s = Status(ErrorCode::kGpuOom, "buffer allocation failed",
                           kComponent);
            }
        }
        if (!s.ok()) {
            result.status = s;
            return result;
        }

        // Pre-warm: one decode against a throwaway sequence faults pages
        // and validates the pass end to end before the first request.
        {
            std::unique_ptr<SequenceState> warm;
            s = model->CreateSequence(4, warm);
            if (s.ok()) {
                std::vector<float> lg;
                s = model->Prefill(*warm, {0}, lg);
            }
            if (!s.ok()) {
                result.status = s;
                return result;
            }
        }

        result.model = std::move(model);
        return result;
    }
}

Status QwenMetalFastModel::CreateSequence(
    uint32_t max_tokens, std::unique_ptr<SequenceState>& out) {
    std::unique_ptr<FastSequenceState> st;
    Status s =
        FastSequenceState::Create(impl_->device, config_, max_tokens, st);
    if (!s.ok()) return s;
    st->set_on_destroy([impl = impl_.get()](uint64_t sid) {
        if (impl->spec_seq_id == sid) impl->DrainSpec();
    });
    out = std::move(st);
    return Status::Ok();
}

void QwenMetalFastModel::EncodeStep(void* enc_v, void* sequence_state,
                                    uint32_t pos, bool want_logits,
                                    uint32_t parity) {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)enc_v;
    {
        auto& state = *static_cast<FastSequenceState*>(sequence_state);
        const QwenConfig& c = config_;
        Impl& impl = *impl_;
        const uint32_t h = c.hidden_size;
        const uint32_t q_dim = c.num_heads * c.head_dim;
        const uint32_t kv_dim = c.num_kv_heads * c.head_dim;
        const uint32_t T = pos + 1;
        const bool is_q4 = impl.quant == MetalFastOptions::Quant::kInt4;
        const bool is_quant = impl.quant != MetalFastOptions::Quant::kFp16;
        (void)h;

        auto set_u32 = [&](uint32_t v, int idx) {
            [enc setBytes:&v length:sizeof(v) atIndex:idx];
        };
        auto set_f32 = [&](float v, int idx) {
            [enc setBytes:&v length:sizeof(v) atIndex:idx];
        };
        auto gemv_grid = [&](uint32_t rows, uint32_t k) {
            (void)k;
            const uint32_t sgpt = 4;
            const uint32_t per_tg = sgpt * 2;  // two rows per simdgroup
            [enc dispatchThreadgroups:MTLSizeMake(
                                          (rows + per_tg - 1) / per_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * sgpt, 1, 1)];
        };

        auto rmsnorm = [&](id<MTLBuffer> in, id<MTLBuffer> w,
                           id<MTLBuffer> out) {
            [enc setComputePipelineState:impl.ps_rmsnorm];
            [enc setBuffer:in offset:0 atIndex:0];
            [enc setBuffer:w offset:0 atIndex:1];
            [enc setBuffer:out offset:0 atIndex:2];
            set_u32(h, 3);
            set_f32(c.rms_norm_eps, 4);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        // Wide-row kernels for the (small) layer projections; the head
        // has enough rows to occupy the GPU one simdgroup per row.
        // Measured slower on M4 Pro (tree-reduce barriers cost more
        // than the occupancy gained); kept for experiments.
        const uint32_t kWideRowMax =
            std::getenv("LYKURO_METAL_FAST_WIDE") != nullptr ? 16384u
                                                             : 0u;
        // y (+bias) (+residual) = M · xn.
        auto gemv = [&](const Impl::Mat& m, id<MTLBuffer> x_in,
                        id<MTLBuffer> y, uint32_t flags) {
            const bool wide = m.rows <= kWideRowMax;
            const bool k16 = !wide && is_q4 && m.cols >= 1024 &&
                             m.cols % 16 == 0;
            [enc setComputePipelineState:wide ? impl.ps_gemv_wide
                                              : (k16 ? impl.ps_gemv_k16
                                                     : impl.ps_gemv)];
            int i = 0;
            [enc setBuffer:m.data offset:0 atIndex:i++];
            if (is_quant) [enc setBuffer:m.scales offset:0 atIndex:i++];
            [enc setBuffer:x_in offset:0 atIndex:i++];
            [enc setBuffer:x_in offset:0 atIndex:i++];  // bias slot (unused)
            [enc setBuffer:y offset:0 atIndex:i++];
            set_u32(m.cols, i++);
            set_u32(m.rows, i++);
            set_u32(flags, i++);
            if (is_q4) set_u32(m.groups, i++);
            if (wide) {
                [enc dispatchThreadgroups:MTLSizeMake(m.rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                gemv_grid(m.rows, m.cols);
            }
        };
        // Fused QKV / gate-up: split outputs (y1/y2 may carry offsets).
        // When norm_fused, RMSNorm folds into the same dispatch: x_in is
        // the RAW residual stream and `nw` its norm weight.
        auto gemv_split = [&](const Impl::Mat& m, id<MTLBuffer> x_in,
                              id<MTLBuffer> nw, id<MTLBuffer> bias,
                              id<MTLBuffer> y0, id<MTLBuffer> y1,
                              size_t y1_off, id<MTLBuffer> y2, size_t y2_off,
                              uint32_t n0, uint32_t n1, uint32_t flags) {
            const bool fused = impl.norm_fused;
            const bool wide = !fused && m.rows <= kWideRowMax;
            [enc setComputePipelineState:fused ? impl.ps_gemv_split_norm
                                               : (wide ? impl.ps_gemv_split_wide
                                                       : impl.ps_gemv_split)];
            int i = 0;
            [enc setBuffer:m.data offset:0 atIndex:i++];
            if (is_quant) [enc setBuffer:m.scales offset:0 atIndex:i++];
            [enc setBuffer:x_in offset:0 atIndex:i++];
            [enc setBuffer:(bias != nil ? bias : x_in) offset:0 atIndex:i++];
            [enc setBuffer:y0 offset:0 atIndex:i++];
            [enc setBuffer:y1 offset:y1_off atIndex:i++];
            [enc setBuffer:y2 offset:y2_off atIndex:i++];
            set_u32(m.cols, i++);
            set_u32(m.rows, i++);
            set_u32(n0, i++);
            set_u32(n1, i++);
            set_u32(flags, i++);
            if (is_q4) set_u32(m.groups, i++);
            if (fused) {
                [enc setBuffer:nw offset:0 atIndex:i++];
                set_f32(c.rms_norm_eps, i++);
                [enc setThreadgroupMemoryLength:((size_t(m.cols) * 2 + 15) /
                                                 16) *
                                                16
                                        atIndex:0];
                const uint32_t sgpt = 4;
                const uint32_t per_tg = sgpt * 2;  // two rows per simdgroup
                [enc dispatchThreadgroups:MTLSizeMake(
                                              (m.rows + per_tg - 1) / per_tg,
                                              1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * sgpt, 1, 1)];
            } else if (wide) {
                [enc dispatchThreadgroups:MTLSizeMake(m.rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                gemv_grid(m.rows, m.cols);
            }
        };

        const size_t kv_row = size_t(pos) * kv_dim * 2;
        const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            Impl::Layer& L = impl.layers[l];
            id<MTLBuffer> kc = state.key_buffer(l);
            id<MTLBuffer> vc = state.value_buffer(l);

            // Fused RMSNorm + QKV + bias + RoPE in a single dispatch;
            // roped K and raw V land straight in this position's cache
            // rows.
            {
                struct {
                    uint32_t q_dim, kv_dim, hd, pos;
                    float theta;
                } qd = {q_dim, kv_dim, c.head_dim, pos, c.rope_theta};
                [enc setComputePipelineState:impl.ps_qkv];
                int i = 0;
                [enc setBuffer:L.qkv.data offset:0 atIndex:i++];
                if (is_quant) {
                    [enc setBuffer:L.qkv.scales offset:0 atIndex:i++];
                }
                [enc setBuffer:impl.x offset:0 atIndex:i++];
                [enc setBuffer:L.qkv_bias offset:0 atIndex:i++];
                [enc setBuffer:impl.q offset:0 atIndex:i++];
                [enc setBuffer:kc offset:kv_row atIndex:i++];
                [enc setBuffer:vc offset:kv_row atIndex:i++];
                set_u32(L.qkv.cols, i++);
                set_u32(L.qkv.rows, i++);
                [enc setBytes:&qd length:sizeof(qd) atIndex:i++];
                [enc setBuffer:L.input_norm offset:0 atIndex:i++];
                set_f32(c.rms_norm_eps, i++);
                if (is_q4) set_u32(L.qkv.groups, i++);
                const uint32_t sgpt = 4;
                const uint32_t per_tg = sgpt * 2;
                [enc dispatchThreadgroups:MTLSizeMake((L.qkv.rows + per_tg -
                                                       1) /
                                                          per_tg,
                                                      1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * sgpt, 1, 1)];
            }
            {
                // Three regimes: tiny T -> per-dim kernel; medium ->
                // coalesced single-pass; large -> split-row partials +
                // fixed-order merge (14 threadgroups cannot occupy the
                // GPU once the K/V walk dominates). hd >= 32 required by
                // the coalesced layouts.
                // Empirical split rule: segment size ~1024/hd rows keeps
                // heads*S threadgroups in the GPU's sweet spot for both
                // hd=64 and hd=128 (measured on M4 Pro).
                const uint32_t S = std::min(
                    kAttnMaxSplit,
                    std::max(1u, (T * c.head_dim) / 768));
                if (c.head_dim >= 32 && S > 1) {
                    [enc setComputePipelineState:impl.ps_attn_part];
                    [enc setBuffer:impl.q offset:0 atIndex:0];
                    [enc setBuffer:kc offset:0 atIndex:1];
                    [enc setBuffer:vc offset:0 atIndex:2];
                    [enc setBuffer:impl.at_m offset:0 atIndex:3];
                    [enc setBuffer:impl.at_l offset:0 atIndex:4];
                    [enc setBuffer:impl.at_a offset:0 atIndex:5];
                    set_u32(T, 6);
                    set_u32(c.head_dim, 7);
                    set_u32(kv_dim, 8);
                    set_u32(c.num_heads / c.num_kv_heads, 9);
                    set_f32(attn_scale, 10);
                    set_u32(S, 11);
                    [enc dispatchThreadgroups:MTLSizeMake(c.num_heads, S, 1)
                        threadsPerThreadgroup:MTLSizeMake(kAttnTpt, 1, 1)];
                    [enc setComputePipelineState:impl.ps_attn_merge];
                    [enc setBuffer:impl.at_m offset:0 atIndex:0];
                    [enc setBuffer:impl.at_l offset:0 atIndex:1];
                    [enc setBuffer:impl.at_a offset:0 atIndex:2];
                    [enc setBuffer:impl.attn_out offset:0 atIndex:3];
                    set_u32(c.head_dim, 4);
                    set_u32(S, 5);
                    [enc dispatchThreadgroups:MTLSizeMake(c.num_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(
                                                  std::max(c.head_dim, 32u),
                                                  1, 1)];
                } else {
                    const bool long_ctx = T >= 96 && c.head_dim >= 32;
                    [enc setComputePipelineState:long_ctx ? impl.ps_attn_l
                                                          : impl.ps_attn];
                    [enc setBuffer:impl.q offset:0 atIndex:0];
                    [enc setBuffer:kc offset:0 atIndex:1];
                    [enc setBuffer:vc offset:0 atIndex:2];
                    [enc setBuffer:impl.attn_out offset:0 atIndex:3];
                    set_u32(T, 4);
                    set_u32(c.head_dim, 5);
                    set_u32(kv_dim, 6);
                    set_u32(c.num_heads / c.num_kv_heads, 7);
                    set_f32(attn_scale, 8);
                    [enc dispatchThreadgroups:MTLSizeMake(c.num_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kAttnTpt, 1, 1)];
                }
            }
            gemv(L.ow, impl.attn_out, impl.x, /*flags=residual*/ 2);

            {
                const bool gk16 = is_q4 && L.gate_up.cols % 16 == 0 &&
                                  L.gate_up.cols >= 1024;
                [enc setComputePipelineState:gk16 ? impl.ps_gateup_k16
                                                  : impl.ps_gateup];
                int i = 0;
                [enc setBuffer:L.gate_up.data offset:0 atIndex:i++];
                if (is_quant) {
                    [enc setBuffer:L.gate_up.scales offset:0 atIndex:i++];
                }
                [enc setBuffer:impl.x offset:0 atIndex:i++];
                [enc setBuffer:impl.act offset:0 atIndex:i++];
                set_u32(L.gate_up.cols, i++);
                set_u32(L.gate_up.rows, i++);
                if (is_q4) set_u32(L.gate_up.groups, i++);
                [enc setBuffer:L.post_norm offset:0 atIndex:i++];
                set_f32(c.rms_norm_eps, i++);
                const uint32_t sgpt = 4;
                const uint32_t per_tg = sgpt * 2;
                [enc dispatchThreadgroups:MTLSizeMake((L.gate_up.rows +
                                                       per_tg - 1) /
                                                          per_tg,
                                                      1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * sgpt, 1, 1)];
            }
            gemv(L.down, impl.act, impl.x, /*flags=residual*/ 2);
        }
        if (want_logits) {
            // The head GEMV stays norm-unfused: the extra dispatch is one
            // tiny rmsnorm per token and the head is already at its
            // bandwidth floor.

                rmsnorm(impl.x, impl.final_norm, impl.xn);
                const bool hk16 =
                    impl.head.q == MetalFastOptions::Quant::kInt4 &&
                    impl.head.cols % 16 == 0 && impl.head.cols >= 1024;
                [enc setComputePipelineState:hk16 ? impl.ps_gemv_f32_k16
                                                  : impl.ps_gemv_f32];
                int i = 0;
                [enc setBuffer:impl.head.data offset:0 atIndex:i++];
                if (is_quant) {
                    [enc setBuffer:impl.head.scales offset:0 atIndex:i++];
                }
                [enc setBuffer:impl.xn offset:0 atIndex:i++];
                [enc setBuffer:(parity ? impl.logits2 : impl.logits)
                        offset:0
                       atIndex:i++];
                set_u32(impl.head.cols, i++);
                set_u32(impl.head.rows, i++);
                if (impl.head.q == MetalFastOptions::Quant::kInt4) {
                    set_u32(impl.head.groups, i++);
                }
                gemv_grid(impl.head.rows, impl.head.cols);
            
        }
    }
}

void QwenMetalFastModel::EncodeArgmax(void* enc_v, uint32_t slot,
                                      bool gather_next, uint32_t parity) {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)enc_v;
    Impl& impl = *impl_;
    uint32_t n = config_.vocab_size;
    const uint32_t h = config_.hidden_size;
    [enc setComputePipelineState:impl.ps_argmax1];
    [enc setBuffer:(parity ? impl.logits2 : impl.logits)
            offset:0
           atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    [enc setBuffer:impl.am_val[parity] offset:0 atIndex:2];
    [enc setBuffer:impl.am_idx[parity] offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    const uint32_t do_gather = gather_next ? 1 : 0;
    [enc setComputePipelineState:impl.ps_am2_gather];
    [enc setBuffer:impl.am_val[parity] offset:0 atIndex:0];
    [enc setBuffer:impl.am_idx[parity] offset:0 atIndex:1];
    [enc setBytes:&slot length:sizeof(slot) atIndex:2];
    [enc setBuffer:impl.tok_buf offset:0 atIndex:3];
    [enc setBuffer:impl.embed_gpu offset:0 atIndex:4];
    [enc setBytes:&h length:sizeof(h) atIndex:5];
    [enc setBuffer:impl.x offset:0 atIndex:6];
    [enc setBytes:&do_gather length:sizeof(do_gather) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void QwenMetalFastModel::EncodeGather(void* enc_v, uint32_t slot) {
    id<MTLComputeCommandEncoder> enc =
        (__bridge id<MTLComputeCommandEncoder>)enc_v;
    Impl& impl = *impl_;
    const uint32_t h = config_.hidden_size;
    [enc setComputePipelineState:impl.ps_gather];
    [enc setBuffer:impl.embed_gpu offset:0 atIndex:0];
    [enc setBuffer:impl.tok_buf offset:0 atIndex:1];
    [enc setBytes:&slot length:sizeof(slot) atIndex:2];
    [enc setBytes:&h length:sizeof(h) atIndex:3];
    [enc setBuffer:impl.x offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(h, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(h, 256u), 1, 1)];
}

Status QwenMetalFastModel::ForwardToken(uint32_t token, uint32_t pos,
                                        void* sequence_state,
                                        std::vector<float>& logits_out,
                                        bool want_logits) {
    @autoreleasepool {
        auto& state = *static_cast<FastSequenceState*>(sequence_state);
        const QwenConfig& c = config_;
        Impl& impl = *impl_;
        impl.DrainSpec();  // shares x/act/logits with any speculation
        Status reserve = state.Reserve(pos + 1);
        if (!reserve.ok()) return reserve;

        // Embedding row into x (host write; unified memory).
        std::memcpy(impl.x.contents,
                    impl.embed_fp16.data() + size_t(token) * c.hidden_size,
                    size_t(c.hidden_size) * 2);

        id<MTLCommandBuffer> cb = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        EncodeStep((__bridge void*)enc, sequence_state, pos, want_logits);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            return MetalFailed("command buffer failed");
        }
        if (std::getenv("LYKURO_METAL_FAST_PROF") != nullptr) {
            static double gpu_s = 0, sched_s = 0;
            static int n = 0;
            gpu_s += cb.GPUEndTime - cb.GPUStartTime;
            sched_s += cb.GPUStartTime - cb.kernelStartTime;
            if (++n % 128 == 0) {
                std::fprintf(stderr,
                             "[prof] %d tokens: gpu %.2fms/tok sched+kern "
                             "%.2fms/tok\n",
                             n, gpu_s / n * 1e3, sched_s / n * 1e3);
            }
        }

        if (want_logits) {
            const float* lp = static_cast<const float*>(impl.logits.contents);
            logits_out.assign(lp, lp + c.vocab_size);
            // Branch-free finiteness scan (exponent all-ones => inf/nan);
            // the reduction vectorizes where the per-element test cannot.
            const uint32_t* bits =
                reinterpret_cast<const uint32_t*>(logits_out.data());
            uint32_t bad = 0;
            for (uint32_t i = 0; i < c.vocab_size; ++i) {
                bad |= uint32_t((bits[i] & 0x7f800000u) == 0x7f800000u);
            }
            if (bad != 0) {
                return Status(ErrorCode::kInferenceFailed,
                              "logits contain non-finite values",
                              kComponent);
            }
        }
        return Status::Ok();
    }
}

Status QwenMetalFastModel::Prefill(SequenceState& state,
                                   const std::vector<uint32_t>& tokens,
                                   std::vector<float>& logits) {
    auto& seq = static_cast<FastSequenceState&>(state);
    if (tokens.empty()) {
        return Status(ErrorCode::kInvalidRequest, "empty prompt", kComponent);
    }
    if (seq.length() != 0) {
        return Status(ErrorCode::kInternalError,
                      "prefill requires an empty cache", kComponent);
    }
    if (tokens.size() > seq.capacity()) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "prompt exceeds cache capacity", kComponent);
    }
    for (uint32_t t : tokens) {
        if (t >= config_.vocab_size) {
            return Status(ErrorCode::kInvalidRequest,
                          "token id out of vocab range", kComponent);
        }
    }
    // Prompt tokens are known upfront, so whole chunks are encoded into
    // one command buffer (embedding rows gathered on-GPU from the token
    // slot buffer) — one CPU/GPU round trip per kPrefillBatch tokens.
    const uint32_t n = uint32_t(tokens.size());
    Impl& impl = *impl_;
    impl.DrainSpec();  // shares tok_buf/x/logits with any speculation
    uint32_t done = 0;
    while (done < n) {
        const uint32_t m = std::min(kPrefillBatch, n - done);
        Status reserve = seq.Reserve(done + m);
        if (!reserve.ok()) return reserve;
        uint32_t* tb = static_cast<uint32_t*>(impl.tok_buf.contents);
        for (uint32_t i = 0; i < m; ++i) tb[i] = tokens[done + i];
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [impl.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (uint32_t i = 0; i < m; ++i) {
                EncodeGather((__bridge void*)enc, i);
                EncodeStep((__bridge void*)enc, &seq, done + i,
                           done + i + 1 == n);
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                return MetalFailed("command buffer failed");
            }
        }
        for (uint32_t i = 0; i < m; ++i) seq.Advance();
        done += m;
    }
    {
        const float* lp = static_cast<const float*>(impl.logits.contents);
        logits.assign(lp, lp + config_.vocab_size);
        const uint32_t* bits =
            reinterpret_cast<const uint32_t*>(logits.data());
        uint32_t bad = 0;
        for (uint32_t i = 0; i < config_.vocab_size; ++i) {
            bad |= uint32_t((bits[i] & 0x7f800000u) == 0x7f800000u);
        }
        if (bad != 0) {
            return Status(ErrorCode::kInferenceFailed,
                          "logits contain non-finite values", kComponent);
        }
    }
    return Status::Ok();
}

Status QwenMetalFastModel::GreedyRun(SequenceState& state, uint32_t token,
                                     uint32_t max_new,
                                     std::vector<uint32_t>& out_tokens) {
    auto& seq = static_cast<FastSequenceState&>(state);
    out_tokens.clear();
    if (token >= config_.vocab_size) {
        return Status(ErrorCode::kInvalidRequest,
                      "token id out of vocab range", kComponent);
    }
    if (max_new == 0) {
        return Status(ErrorCode::kInvalidRequest, "max_new must be > 0",
                      kComponent);
    }
    if (seq.length() >= seq.capacity()) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "kv cache capacity exhausted", kComponent);
    }
    Impl& impl = *impl_;

    // Encodes and commits one speculative batch: `nsteps` forwards from
    // the CURRENT x buffer, argmax after each, the winner's embedding
    // gathered back into x (also after the last step, feeding the next
    // batch). Token ids land in slots [parity*kGreedyRun, ...).
    auto commit_batch = [&](uint32_t base_pos, uint32_t nsteps,
                            uint32_t parity) -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cb = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t s = 0; s < nsteps; ++s) {
            EncodeStep((__bridge void*)enc, &seq, base_pos + s, true,
                       parity);
            EncodeArgmax((__bridge void*)enc, parity * kGreedyRun + s, true,
                         parity);
        }
        [enc endEncoding];
        [cb commit];
        return cb;
    };
    auto batch_steps = [&](uint32_t from_len) -> uint32_t {
        if (from_len >= seq.capacity()) return 0;
        return std::min(kGreedyRun, seq.capacity() - from_len);
    };
    auto arm_spec = [&](id<MTLCommandBuffer> cb, uint32_t nsteps,
                        uint32_t parity, uint32_t expected) {
        impl.spec_cb = cb;
        impl.spec_seq_id = seq.seq_id();
        impl.spec_nsteps = nsteps;
        impl.spec_parity = parity;
        impl.spec_expected = expected;
        impl.spec_base_len = seq.length();
    };
    auto collect = [&](id<MTLCommandBuffer> cb, uint32_t parity,
                       uint32_t nsteps) -> Status {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            return MetalFailed("command buffer failed");
        }
        if (std::getenv("LYKURO_METAL_FAST_PROF") != nullptr) {
            static double gpu_s = 0;
            static int tok = 0;
            gpu_s += cb.GPUEndTime - cb.GPUStartTime;
            tok += int(nsteps);
            if (tok >= 128) {
                std::fprintf(stderr, "[prof] greedy gpu %.2fms/tok\n",
                             gpu_s / tok * 1e3);
                gpu_s = 0;
                tok = 0;
            }
        }
        id<MTLBuffer> lg = parity ? impl.logits2 : impl.logits;
        const uint32_t* bits = static_cast<const uint32_t*>(lg.contents);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < config_.vocab_size; ++i) {
            bad |= uint32_t((bits[i] & 0x7f800000u) == 0x7f800000u);
        }
        if (bad != 0) {
            return Status(ErrorCode::kInferenceFailed,
                          "logits contain non-finite values", kComponent);
        }
        const uint32_t* tb =
            static_cast<const uint32_t*>(impl.tok_buf.contents) +
            parity * kGreedyRun;
        out_tokens.assign(tb, tb + nsteps);
        for (uint32_t s = 0; s < nsteps; ++s) seq.Advance();
        return Status::Ok();
    };

    @autoreleasepool {
        if (impl.spec_cb != nil) {
            const bool match = impl.spec_seq_id == seq.seq_id() &&
                               impl.spec_expected == token &&
                               impl.spec_base_len == seq.length();
            id<MTLCommandBuffer> cb = impl.spec_cb;
            const uint32_t parity = impl.spec_parity;
            const uint32_t nsteps = impl.spec_nsteps;
            impl.spec_cb = nil;
            impl.spec_seq_id = 0;
            if (match) {
                // Keep the GPU fed: commit the NEXT speculative batch
                // before reading this one's results (its input embedding
                // is already in x, gathered on-GPU by this batch's last
                // argmax).
                const uint32_t next_len = seq.length() + nsteps;
                const uint32_t next_n = batch_steps(next_len);
                id<MTLCommandBuffer> next_cb = nil;
                if (next_n > 0) {
                    next_cb = commit_batch(next_len, next_n, 1 - parity);
                }
                Status s = collect(cb, parity, nsteps);
                if (!s.ok()) {
                    if (next_cb != nil) [next_cb waitUntilCompleted];
                    return s;
                }
                if (next_cb != nil) {
                    arm_spec(next_cb, next_n, 1 - parity,
                             out_tokens.back());
                }
                return Status::Ok();
            }
            // Another sequence, a diverged trajectory, or a stale length:
            // drain and drop. The speculated KV rows sit beyond the
            // owner's accepted length and are rewritten before any read.
            [cb waitUntilCompleted];
        }

        // Cold start. Reserve the whole logical capacity up front: with a
        // speculative batch permanently in flight there is no later safe
        // point to reallocate the KV buffers.
        Status reserve = seq.Reserve(seq.capacity());
        if (!reserve.ok()) return reserve;
        const uint32_t len = seq.length();
        const uint32_t n1 = std::min(
            {max_new, kGreedyRun, seq.capacity() - len});
        std::memcpy(impl.x.contents,
                    impl.embed_fp16.data() +
                        size_t(token) * config_.hidden_size,
                    size_t(config_.hidden_size) * 2);
        id<MTLCommandBuffer> cb1 = commit_batch(len, n1, 0);
        const uint32_t n2 = batch_steps(len + n1);
        id<MTLCommandBuffer> cb2 =
            n2 > 0 ? commit_batch(len + n1, n2, 1) : nil;
        Status s = collect(cb1, 0, n1);
        if (!s.ok()) {
            if (cb2 != nil) [cb2 waitUntilCompleted];
            return s;
        }
        if (cb2 != nil) arm_spec(cb2, n2, 1, out_tokens.back());
        return Status::Ok();
    }
}

Status QwenMetalFastModel::Decode(SequenceState& state, uint32_t token,
                                  std::vector<float>& logits) {
    auto& seq = static_cast<FastSequenceState&>(state);
    if (token >= config_.vocab_size) {
        return Status(ErrorCode::kInvalidRequest,
                      "token id out of vocab range", kComponent);
    }
    if (seq.length() >= seq.capacity()) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "kv cache capacity exhausted", kComponent);
    }
    Status s = ForwardToken(token, seq.length(), &seq, logits, true);
    if (!s.ok()) return s;
    seq.Advance();
    return Status::Ok();
}

}  // namespace lykuro::nie
