#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backends/metal/qwen_metal_fast.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_metal_fast";
constexpr uint32_t kCtxBucket = 128;   // KV growth granularity
constexpr uint32_t kQ4Group = 128;     // INT4 scale group along K
constexpr uint32_t kAttnTpt = 128;     // attention threads per group

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

// ---- RMSNorm-fused GEMVs ----
// Each threadgroup recomputes the (cheap) norm reduction, materializes
// the normalized activation once in threadgroup memory, and runs its
// rows against it — one dispatch replaces rmsnorm + GEMV, and the dot
// products read threadgroup memory instead of device memory. K <= 6144.

#define NORM_PROLOGUE                                                     \
    threadgroup half xn[6144];                                            \
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
                                uint tgpig [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tpt [[threads_per_threadgroup]],
                                uint sgitg [[simdgroup_index_in_threadgroup]],
                                uint tiisg [[thread_index_in_simdgroup]],
                                uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = tgpig * sgpt + sgitg;
    if (row >= N) return;
    float acc = dot_f16_tg(W + ulong(row) * K, xn, K / 4, tiisg);
    if (tiisg == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
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
                               uint tgpig [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = tgpig * sgpt + sgitg;
    if (row >= N) return;
    float acc =
        dot_q8_tg(W + ulong(row) * K, xn, K / 4, tiisg) * scales[row];
    if (tiisg == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
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
                               uint tgpig [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tpt [[threads_per_threadgroup]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = tgpig * sgpt + sgitg;
    if (row >= N) return;
    float acc = dot_q4_tg(W + ulong(row) * (K / 2),
                          scales + ulong(row) * groups, xn, K / 4, tiisg);
    if (tiisg == 0) split_store(acc, row, bias, y0, y1, y2, n0, n1, flags);
}

// RMSNorm-fused FP32-output GEMV (final norm + logits head).
kernel void gemv_f32_norm_f16(device const half* W [[buffer(0)]],
                              device const half* x [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant uint& K [[buffer(3)]],
                              constant uint& N [[buffer(4)]],
                              device const float* nw [[buffer(5)]],
                              constant float& eps [[buffer(6)]],
                              uint tgpig [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tpt [[threads_per_threadgroup]],
                              uint sgitg [[simdgroup_index_in_threadgroup]],
                              uint tiisg [[thread_index_in_simdgroup]],
                              uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = tgpig * sgpt + sgitg;
    if (row >= N) return;
    float acc = dot_f16_tg(W + ulong(row) * K, xn, K / 4, tiisg);
    if (tiisg == 0) y[row] = acc;
}

kernel void gemv_f32_norm_q8(device const char* W [[buffer(0)]],
                             device const float* scales [[buffer(1)]],
                             device const half* x [[buffer(2)]],
                             device float* y [[buffer(3)]],
                             constant uint& K [[buffer(4)]],
                             constant uint& N [[buffer(5)]],
                             device const float* nw [[buffer(6)]],
                             constant float& eps [[buffer(7)]],
                             uint tgpig [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tpt [[threads_per_threadgroup]],
                             uint sgitg [[simdgroup_index_in_threadgroup]],
                             uint tiisg [[thread_index_in_simdgroup]],
                             uint sgpt [[simdgroups_per_threadgroup]]) {
    NORM_PROLOGUE
    uint row = tgpig * sgpt + sgitg;
    if (row >= N) return;
    float acc =
        dot_q8_tg(W + ulong(row) * K, xn, K / 4, tiisg) * scales[row];
    if (tiisg == 0) y[row] = acc;
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

    ~FastSequenceState() {
        @autoreleasepool {
            for (auto& b : keys_) b = nil;
            for (auto& b : values_) b = nil;
        }
    }

private:
    FastSequenceState() = default;
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
    // RMSNorm-fused variants, used when hidden_size fits threadgroup
    // memory (h <= 6144; covers the supported Qwen sizes).
    id<MTLComputePipelineState> ps_gemv_split_norm = nil;
    id<MTLComputePipelineState> ps_gemv_f32_norm = nil;
    bool norm_fused = false;
    id<MTLComputePipelineState> ps_rope_qk = nil;
    id<MTLComputePipelineState> ps_attn = nil;
    id<MTLComputePipelineState> ps_silu_mul = nil;

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
            pso("silu_mul", impl.ps_silu_mul);
        // Norm-fused GEMVs measured slower on M4 Pro (the 13KB
        // threadgroup footprint costs more occupancy than the saved
        // dispatch): keep them available for experiments, default off.
        impl.norm_fused =
            c.hidden_size <= 6144 &&
            std::getenv("LYKURO_METAL_FAST_NORM_FUSED") != nullptr;
        if (ok && impl.norm_fused) {
            ok = pso(q4 ? "gemv_split_norm_q4"
                        : (q8 ? "gemv_split_norm_q8" : "gemv_split_norm_f16"),
                     impl.ps_gemv_split_norm) &&
                 pso((q8 || q4) ? "gemv_f32_norm_q8" : "gemv_f32_norm_f16",
                     impl.ps_gemv_f32_norm);
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

            // Fused QKV: rows [q; k; v], plus the fused bias vector.
            if (s.ok()) {
                std::vector<float> fused;
                fused.reserve(size_t(q_dim + 2 * kv_dim) * h);
                s = TensorToHost(weights, p + "self_attn.q_proj.weight", tmp);
                if (s.ok()) {
                    fused.insert(fused.end(), tmp.begin(), tmp.end());
                    s = TensorToHost(weights, p + "self_attn.k_proj.weight",
                                     tmp);
                }
                if (s.ok()) {
                    fused.insert(fused.end(), tmp.begin(), tmp.end());
                    s = TensorToHost(weights, p + "self_attn.v_proj.weight",
                                     tmp);
                }
                if (s.ok()) {
                    fused.insert(fused.end(), tmp.begin(), tmp.end());
                    if (!impl.LoadMat(fused, q_dim + 2 * kv_dim, h, L.qkv,
                                      false)) {
                        s = oom();
                    }
                }
                std::vector<float> fused_bias;
                if (s.ok()) {
                    s = TensorToHost(weights, p + "self_attn.q_proj.bias",
                                     tmp);
                }
                if (s.ok()) {
                    fused_bias.insert(fused_bias.end(), tmp.begin(),
                                      tmp.end());
                    s = TensorToHost(weights, p + "self_attn.k_proj.bias",
                                     tmp);
                }
                if (s.ok()) {
                    fused_bias.insert(fused_bias.end(), tmp.begin(),
                                      tmp.end());
                    s = TensorToHost(weights, p + "self_attn.v_proj.bias",
                                     tmp);
                }
                if (s.ok()) {
                    fused_bias.insert(fused_bias.end(), tmp.begin(),
                                      tmp.end());
                    if (!upload_f16(fused_bias, L.qkv_bias)) s = oom();
                }
            }
            if (s.ok()) {
                s = TensorToHost(weights, p + "self_attn.o_proj.weight", tmp);
                if (s.ok() && !impl.LoadMat(tmp, h, q_dim, L.ow, false)) {
                    s = oom();
                }
            }
            // Fused gate/up: rows [gate; up].
            if (s.ok()) {
                std::vector<float> fused;
                fused.reserve(size_t(2 * c.intermediate_size) * h);
                s = TensorToHost(weights, p + "mlp.gate_proj.weight", tmp);
                if (s.ok()) {
                    fused.insert(fused.end(), tmp.begin(), tmp.end());
                    s = TensorToHost(weights, p + "mlp.up_proj.weight", tmp);
                }
                if (s.ok()) {
                    fused.insert(fused.end(), tmp.begin(), tmp.end());
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
            if (impl.x == nil || impl.xn == nil || impl.q == nil ||
                impl.attn_out == nil || impl.gate_a == nil ||
                impl.up_a == nil || impl.act == nil || impl.logits == nil) {
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
    out = std::move(st);
    return Status::Ok();
}

Status QwenMetalFastModel::ForwardToken(uint32_t token, uint32_t pos,
                                        void* sequence_state,
                                        std::vector<float>& logits_out,
                                        bool want_logits) {
    @autoreleasepool {
        auto& state = *static_cast<FastSequenceState*>(sequence_state);
        const QwenConfig& c = config_;
        Impl& impl = *impl_;
        const uint32_t h = c.hidden_size;
        const uint32_t q_dim = c.num_heads * c.head_dim;
        const uint32_t kv_dim = c.num_kv_heads * c.head_dim;
        const uint32_t T = pos + 1;
        const bool is_q4 = impl.quant == MetalFastOptions::Quant::kInt4;
        const bool is_quant = impl.quant != MetalFastOptions::Quant::kFp16;

        Status reserve = state.Reserve(T);
        if (!reserve.ok()) return reserve;

        // Embedding row into x (host write; unified memory).
        std::memcpy(impl.x.contents,
                    impl.embed_fp16.data() + size_t(token) * h, size_t(h) * 2);

        id<MTLCommandBuffer> cb = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

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
            [enc setComputePipelineState:wide ? impl.ps_gemv_wide
                                              : impl.ps_gemv];
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
                const uint32_t sgpt = 8;
                [enc dispatchThreadgroups:MTLSizeMake(
                                              (m.rows + sgpt - 1) / sgpt, 1,
                                              1)
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

            // QKV in one pass; K/V land in this position's cache rows.
            if (impl.norm_fused) {
                gemv_split(L.qkv, impl.x, L.input_norm, L.qkv_bias, impl.q,
                           kc, kv_row, vc, kv_row, q_dim, kv_dim,
                           /*flags=*/1);
            } else {
                rmsnorm(impl.x, L.input_norm, impl.xn);
                gemv_split(L.qkv, impl.xn, nil, L.qkv_bias, impl.q, kc,
                           kv_row, vc, kv_row, q_dim, kv_dim, /*flags=*/1);
            }
            // RoPE over q and the cached k row in one grid.
            {
                [enc setComputePipelineState:impl.ps_rope_qk];
                [enc setBuffer:impl.q offset:0 atIndex:0];
                [enc setBuffer:kc offset:kv_row atIndex:1];
                set_u32(c.head_dim, 2);
                set_u32(c.num_heads, 3);
                set_f32(c.rope_theta, 4);
                set_u32(pos, 5);
                const uint32_t n =
                    (c.num_heads + c.num_kv_heads) * c.head_dim / 2;
                [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min(n, 64u), 1,
                                                      1)];
            }
            {
                [enc setComputePipelineState:impl.ps_attn];
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
            gemv(L.ow, impl.attn_out, impl.x, /*flags=residual*/ 2);

            if (impl.norm_fused) {
                gemv_split(L.gate_up, impl.x, L.post_norm, nil, impl.gate_a,
                           impl.up_a, 0, impl.up_a, 0, c.intermediate_size,
                           c.intermediate_size, /*flags=*/0);
            } else {
                rmsnorm(impl.x, L.post_norm, impl.xn);
                gemv_split(L.gate_up, impl.xn, nil, nil, impl.gate_a,
                           impl.up_a, 0, impl.up_a, 0, c.intermediate_size,
                           c.intermediate_size, /*flags=*/0);
            }
            {
                [enc setComputePipelineState:impl.ps_silu_mul];
                [enc setBuffer:impl.gate_a offset:0 atIndex:0];
                [enc setBuffer:impl.up_a offset:0 atIndex:1];
                [enc setBuffer:impl.act offset:0 atIndex:2];
                set_u32(c.intermediate_size, 3);
                [enc dispatchThreads:MTLSizeMake(c.intermediate_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(
                                              std::min(c.intermediate_size,
                                                       256u),
                                              1, 1)];
            }
            gemv(L.down, impl.act, impl.x, /*flags=residual*/ 2);
        }
        if (want_logits) {
            // The head GEMV stays norm-unfused: its 13KB threadgroup
            // footprint would cap occupancy on the one op that actually
            // saturates bandwidth (vocab x h). The extra dispatch is one
            // tiny rmsnorm per token.
            if (false) {
                [enc setComputePipelineState:impl.ps_gemv_f32_norm];
                int i = 0;
                [enc setBuffer:impl.head.data offset:0 atIndex:i++];
                if (is_quant) {
                    [enc setBuffer:impl.head.scales offset:0 atIndex:i++];
                }
                [enc setBuffer:impl.x offset:0 atIndex:i++];
                [enc setBuffer:impl.logits offset:0 atIndex:i++];
                set_u32(impl.head.cols, i++);
                set_u32(impl.head.rows, i++);
                [enc setBuffer:impl.final_norm offset:0 atIndex:i++];
                set_f32(c.rms_norm_eps, i++);
                const uint32_t sgpt = 8;
                [enc dispatchThreadgroups:MTLSizeMake(
                                              (impl.head.rows + sgpt - 1) /
                                                  sgpt,
                                              1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * sgpt, 1, 1)];
            } else {
                rmsnorm(impl.x, impl.final_norm, impl.xn);
                [enc setComputePipelineState:impl.ps_gemv_f32];
                int i = 0;
                [enc setBuffer:impl.head.data offset:0 atIndex:i++];
                if (is_quant) {
                    [enc setBuffer:impl.head.scales offset:0 atIndex:i++];
                }
                [enc setBuffer:impl.xn offset:0 atIndex:i++];
                [enc setBuffer:impl.logits offset:0 atIndex:i++];
                set_u32(impl.head.cols, i++);
                set_u32(impl.head.rows, i++);
                if (impl.head.q == MetalFastOptions::Quant::kInt4) {
                    set_u32(impl.head.groups, i++);
                }
                gemv_grid(impl.head.rows, impl.head.cols);
            }
        }
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
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool last = i + 1 == tokens.size();
        Status s = ForwardToken(tokens[i], uint32_t(i), &seq, logits, last);
        if (!s.ok()) return s;
        seq.Advance();
    }
    return Status::Ok();
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
