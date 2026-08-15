#include "backends/cuda/qwen_cuda_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
#include <typeinfo>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_cuda";

Status CudaError(cudaError_t err, const char* what) {
    if (err == cudaErrorMemoryAllocation) {
        return Status(ErrorCode::kGpuOom, what, kComponent);
    }
    return Status(ErrorCode::kGpuUnhealthy, what, kComponent);
}

#define LYKURO_CUDA_CHECK(expr, what)                       \
    do {                                                    \
        cudaError_t lykuro_err_ = (expr);                   \
        if (lykuro_err_ != cudaSuccess) {                   \
            return CudaError(lykuro_err_, what);            \
        }                                                   \
    } while (0)

// ------------------------------------------------------------- kernels
//
// Weights stay in checkpoint dtype (BF16 never widened); accumulation is
// FP32 with fixed reduction order everywhere, so results are
// deterministic and match the CPU reference semantics.
//
// KV storage is paged (spec §16.3): fixed-size blocks shared across
// layers via a per-sequence block table. Token t of a sequence lives at
// pool[table[t / block_tokens] * block_tokens + t % block_tokens].

__device__ inline float LoadWeight(const float* w, size_t i) { return w[i]; }
__device__ inline float LoadWeight(const __nv_bfloat16* w, size_t i) {
    return __bfloat162float(w[i]);
}

// Weight accessors: uniform Load(row, i) view over f32 / bf16 / int8 /
// int4 storage. Quantized loads dequantize in registers; accumulation
// stays FP32 with fixed order, so quantized runs are still deterministic.
// Load8(row, i0, out): eight consecutive elements per call with one
// wide (coalesced) memory transaction. Per-element scalar loads were
// the decode bottleneck for the quantized formats: a warp of byte-wise
// nibble reads touches 16 useful bytes per 32-lane transaction, an 8x
// DRAM amplification. i0 must be 8-aligned; every reduction dim the
// engine feeds through here is a multiple of 8.
struct F32Weight {
    const float* w;
    int in_dim;
    __device__ float Load(size_t row, int i) const {
        return w[row * in_dim + i];
    }
    __device__ void Load8(size_t row, int i0, float* o) const {
        const float4* p =
            reinterpret_cast<const float4*>(w + row * in_dim + i0);
        const float4 a = p[0];
        const float4 b = p[1];
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
        o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    }
};
struct Bf16Weight {
    const __nv_bfloat16* w;
    int in_dim;
    __device__ float Load(size_t row, int i) const {
        return __bfloat162float(w[row * in_dim + i]);
    }
    __device__ void Load8(size_t row, int i0, float* o) const {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(
            w + row * in_dim + i0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            o[2 * j] = f.x;
            o[2 * j + 1] = f.y;
        }
    }
};
struct Int8Weight {
    const int8_t* w;
    const float* scales;  // [out]
    int in_dim;
    __device__ float Load(size_t row, int i) const {
        return float(w[row * in_dim + i]) * scales[row];
    }
    __device__ void Load8(size_t row, int i0, float* o) const {
        const uint2 u = *reinterpret_cast<const uint2*>(
            w + row * in_dim + i0);
        const float s = scales[row];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = float(int(int8_t((u.x >> (8 * j)) & 0xFFu))) * s;
            o[4 + j] = float(int(int8_t((u.y >> (8 * j)) & 0xFFu))) * s;
        }
    }
};
constexpr int kQuantGroup = 128;
struct Int4Weight {
    const uint8_t* packed;   // two nibbles per byte, row-major
    const float* scales;     // [out, groups_per_row]
    int in_dim;
    int groups_per_row;
    __device__ float Load(size_t row, int i) const {
        const size_t idx = row * in_dim + i;
        const uint8_t byte = packed[idx >> 1];
        const int nib = (idx & 1) ? (byte >> 4) : (byte & 0xF);
        return float(nib - 8) *
               scales[row * groups_per_row + i / kQuantGroup];
    }
    __device__ void Load8(size_t row, int i0, float* o) const {
        const uint32_t u = *reinterpret_cast<const uint32_t*>(
            packed + ((row * in_dim + i0) >> 1));
        const float s = scales[row * groups_per_row + i0 / kQuantGroup];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j] = float(int((u >> (4 * j)) & 0xFu) - 8) * s;
        }
    }
};

__device__ inline const float* KvRow(const float* pool,
                                     const int* __restrict__ table,
                                     int block_tokens, int t,
                                     int kv_stride) {
    return pool + (size_t(table[t / block_tokens]) * block_tokens +
                   (t % block_tokens)) *
                      kv_stride;
}

template <typename W>
__global__ void MatVecKernel(W weight, const float* __restrict__ x,
                             const float* __restrict__ bias, int in_dim,
                             float* __restrict__ y) {
    __shared__ float partial[128];
    const size_t row = blockIdx.x;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += weight.Load(row, i) * x[i];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = partial[0] + (bias != nullptr ? bias[row] : 0.0f);
    }
}

template <typename WT>
__global__ void GatherEmbedKernel(const WT* __restrict__ embed,
                                  uint32_t token, int hidden,
                                  float* __restrict__ out) {
    const WT* row = embed + size_t(token) * hidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += gridDim.x * blockDim.x) {
        out[i] = LoadWeight(row, i);
    }
}

__global__ void RmsNormKernel(const float* x, const float* weight,
                              float eps, int dim, float* out) {
    __shared__ float partial[256];
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local += x[i] * x[i];
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(dim) + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = x[i] * scale * weight[i];
    }
}

__global__ void RopeKernel(float* vec, int head_dim, uint32_t pos,
                           float theta) {
    const int head = blockIdx.x;
    const int half = head_dim / 2;
    float* h = vec + head * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(theta, -2.0f * float(i) / float(head_dim));
        float angle = float(pos) * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = h[i];
        float b = h[i + half];
        h[i] = a * c - b * s;
        h[i + half] = b * c + a * s;
    }
}

// ---- split-K decode attention over paged KV ----

constexpr int kMaxSplits = 32;
constexpr int kMaxSegment = 1024;
constexpr int kAttnThreads = 128;

__global__ void SplitAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ k_pool,
    const float* __restrict__ v_pool, const int* __restrict__ table,
    int block_tokens, float* __restrict__ partials, int context,
    int head_dim, int kv_stride, int group, float scale, int splits) {
    __shared__ float scores[kMaxSegment];
    __shared__ float red[kAttnThreads];
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int kv_head = head / group;
    const float* qh = q + head * head_dim;
    float* part =
        partials + (size_t(head) * kMaxSplits + split) * (head_dim + 2);

    const int seg = (context + splits - 1) / splits;
    const int t0 = split * seg;
    const int t1 = min(context, t0 + seg);
    const int len = t1 - t0;
    if (len <= 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            part[d] = 0.0f;
        }
        if (threadIdx.x == 0) {
            part[head_dim] = -1e30f;
            part[head_dim + 1] = 0.0f;
        }
        return;
    }

    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float* kt = KvRow(k_pool, table, block_tokens, t0 + j,
                                kv_stride) +
                          kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kt[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] =
                fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        scores[j] = expf(scores[j] - m);
        local_sum += scores[j];
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = red[0];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < len; ++j) {
            const float* vt = KvRow(v_pool, table, block_tokens, t0 + j,
                                    kv_stride) +
                              kv_head * head_dim;
            acc += scores[j] * vt[d];
        }
        part[d] = acc;
    }
    if (threadIdx.x == 0) {
        part[head_dim] = m;
        part[head_dim + 1] = denom;
    }
}

__global__ void CombineAttentionKernel(const float* __restrict__ partials,
                                       float* __restrict__ out,
                                       int head_dim, int splits) {
    const int head = blockIdx.x;
    const float* base =
        partials + size_t(head) * kMaxSplits * (head_dim + 2);
    __shared__ float m_glob, denom_glob;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int s = 0; s < splits; ++s) {
            m = fmaxf(m, base[size_t(s) * (head_dim + 2) + head_dim]);
        }
        float denom = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + size_t(s) * (head_dim + 2);
            denom += p[head_dim + 1] * expf(p[head_dim] - m);
        }
        m_glob = m;
        denom_glob = denom;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + size_t(s) * (head_dim + 2);
            acc += p[d] * expf(p[head_dim] - m_glob);
        }
        out[head * head_dim + d] = acc / denom_glob;
    }
}

// ---- batched decode kernels ----

constexpr int kMaxBatch = 16;

__global__ void RopeVarPosKernel(float* __restrict__ buf, int row_stride,
                                 int head_dim,
                                 const uint32_t* __restrict__ positions,
                                 float theta) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int half = head_dim / 2;
    float* h = buf + size_t(row) * row_stride + head * head_dim;
    const uint32_t pos = positions[row];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(theta, -2.0f * float(i) / float(head_dim));
        float angle = float(pos) * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = h[i];
        float b = h[i + half];
        h[i] = a * c - b * s;
        h[i + half] = b * c + a * s;
    }
}

// Row i of `src` -> dst_rows[i] (host-resolved paged destinations).
__global__ void ScatterRowsKernel(const float* __restrict__ src,
                                  float* const* __restrict__ dst_rows,
                                  int kv_dim) {
    const int row = blockIdx.x;
    float* dst = dst_rows[row];
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        dst[i] = src[size_t(row) * kv_dim + i];
    }
}

// Graph-safe KV scatter: destination is derived on device from the
// sequence's block table and current position, so the same captured
// graph replays correctly as positions advance.
__global__ void ScatterKvPagedKernel(const float* __restrict__ src,
                                     float* __restrict__ pool,
                                     const int* const* __restrict__ tables,
                                     const uint32_t* __restrict__ positions,
                                     int block_tokens, int kv_dim) {
    const int row = blockIdx.x;
    const int* table = tables[row];
    const int t = int(positions[row]);
    float* dst = pool + (size_t(table[t / block_tokens]) * block_tokens +
                         (t % block_tokens)) *
                            kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        dst[i] = src[size_t(row) * kv_dim + i];
    }
}

__global__ void BatchSplitAttentionKernel(
    const float* __restrict__ q_rows, const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int* const* __restrict__ tables, int block_tokens,
    const uint32_t* __restrict__ positions, float* __restrict__ partials,
    int head_dim, int kv_stride, int group, float scale, int splits) {
    __shared__ float scores[kMaxSegment];
    __shared__ float red[kAttnThreads];
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int seq = blockIdx.z;
    const int kv_head = head / group;
    const int context = int(positions[seq]) + 1;
    const int q_dim = gridDim.x * head_dim;
    const float* qh = q_rows + size_t(seq) * q_dim + head * head_dim;
    const int* table = tables[seq];
    float* part = partials +
                  ((size_t(seq) * gridDim.x + head) * kMaxSplits + split) *
                      (head_dim + 2);

    const int seg = (context + splits - 1) / splits;
    const int t0 = split * seg;
    const int t1 = min(context, t0 + seg);
    const int len = t1 - t0;
    if (len <= 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            part[d] = 0.0f;
        }
        if (threadIdx.x == 0) {
            part[head_dim] = -1e30f;
            part[head_dim + 1] = 0.0f;
        }
        return;
    }

    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float* kt = KvRow(k_pool, table, block_tokens, t0 + j,
                                kv_stride) +
                          kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kt[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] =
                fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        scores[j] = expf(scores[j] - m);
        local_sum += scores[j];
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = red[0];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < len; ++j) {
            const float* vt = KvRow(v_pool, table, block_tokens, t0 + j,
                                    kv_stride) +
                              kv_head * head_dim;
            acc += scores[j] * vt[d];
        }
        part[d] = acc;
    }
    if (threadIdx.x == 0) {
        part[head_dim] = m;
        part[head_dim + 1] = denom;
    }
}

__global__ void BatchCombineAttentionKernel(
    const float* __restrict__ partials, float* __restrict__ out,
    int head_dim, int splits) {
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int q_dim = gridDim.x * head_dim;
    const float* base = partials +
                        (size_t(seq) * gridDim.x + head) * kMaxSplits *
                            (head_dim + 2);
    __shared__ float m_glob, denom_glob;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int s = 0; s < splits; ++s) {
            m = fmaxf(m, base[size_t(s) * (head_dim + 2) + head_dim]);
        }
        float denom = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + size_t(s) * (head_dim + 2);
            denom += p[head_dim + 1] * expf(p[head_dim] - m);
        }
        m_glob = m;
        denom_glob = denom;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + size_t(s) * (head_dim + 2);
            acc += p[d] * expf(p[head_dim] - m_glob);
        }
        out[size_t(seq) * q_dim + head * head_dim + d] = acc / denom_glob;
    }
}

// Batched GEMV with compile-time batch size (register accumulators; a
// dynamically-indexed accumulator array would spill to local memory).
constexpr int kBgemvChunk = 1024;

template <typename W, int B>
__global__ void MatVecBatchKernel(W weight, const float* __restrict__ x,
                                  const float* __restrict__ bias,
                                  int in_dim, float* __restrict__ y,
                                  int ldy) {
    __shared__ float wsh[kBgemvChunk];
    __shared__ float red[128];
    const size_t row = blockIdx.x;

    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    for (int k0 = 0; k0 < in_dim; k0 += kBgemvChunk) {
        const int len = min(kBgemvChunk, in_dim - k0);
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            wsh[i] = weight.Load(row, k0 + i);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            const float wv = wsh[i];
#pragma unroll
            for (int b = 0; b < B; ++b) {
                acc[b] += wv * x[size_t(b) * in_dim + k0 + i];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int b = 0; b < B; ++b) {
        red[threadIdx.x] = acc[b];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            y[size_t(b) * ldy + row] =
                red[0] + (bias != nullptr ? bias[row] : 0.0f);
        }
        __syncthreads();
    }
}


// ---- fused decode kernels (Phase 4 kernel fusion, batch <= 8) ----
//
// The decode dependency chain is execution-latency bound, so the lever is
// fewer kernels: RMSNorm folds into projections as a precomputed
// per-sequence scale, Q/K/V share one launch, gate/up/SwiGLU collapse to
// one kernel, and residual adds ride the projection epilogue. All sums
// keep a fixed order (deterministic); the norm fold regroups
// multiplications only.

// Per-row inverse RMS: scales[row] = rsqrt(mean(x^2) + eps).
__global__ void NormScaleKernel(const float* __restrict__ x, float eps,
                                int dim, float* __restrict__ scales) {
    __shared__ float partial[256];
    const float* xr = x + size_t(blockIdx.x) * dim;
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local += xr[i] * xr[i];
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scales[blockIdx.x] = rsqrtf(partial[0] / float(dim) + eps);
    }
}

// One launch for Q, K and V: grid rows cover q_dim + 2*kv_dim, each block
// picks its matrix segment. Norm is folded: acc = sum w * (x * norm_w),
// scaled by the per-sequence inverse RMS at the epilogue.
template <typename W, int B>
__global__ void FusedQkvKernel(
    W qw, W kw, W vw, const float* __restrict__ q_bias,
    const float* __restrict__ k_bias, const float* __restrict__ v_bias,
    const float* __restrict__ x, const float* __restrict__ norm_w,
    const float* __restrict__ scales, int in_dim, int q_dim, int kv_dim,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out) {
    __shared__ float red[128];
    const int row = blockIdx.x;
    W w = qw;
    const float* bias = q_bias;
    float* dst = q_out;
    int ld = q_dim;
    size_t local = row;
    if (row >= q_dim + kv_dim) {
        w = vw;
        bias = v_bias;
        dst = v_out;
        ld = kv_dim;
        local = row - q_dim - kv_dim;
    } else if (row >= q_dim) {
        w = kw;
        bias = k_bias;
        dst = k_out;
        ld = kv_dim;
        local = row - q_dim;
    }

    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    const int chunks = in_dim >> 3;
    for (int ci = threadIdx.x; ci < chunks; ci += blockDim.x) {
        const int i0 = ci << 3;
        float wv[8];
        w.Load8(local, i0, wv);
        const float4* nwp =
            reinterpret_cast<const float4*>(norm_w + i0);
        const float4 n0 = nwp[0];
        const float4 n1 = nwp[1];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* xp = reinterpret_cast<const float4*>(
                x + size_t(b) * in_dim + i0);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            acc[b] += wv[0] * (x0.x * n0.x) + wv[1] * (x0.y * n0.y) +
                      wv[2] * (x0.z * n0.z) + wv[3] * (x0.w * n0.w) +
                      wv[4] * (x1.x * n1.x) + wv[5] * (x1.y * n1.y) +
                      wv[6] * (x1.z * n1.z) + wv[7] * (x1.w * n1.w);
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        red[threadIdx.x] = acc[b];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            dst[size_t(b) * ld + local] =
                red[0] * scales[b] + bias[local];
        }
        __syncthreads();
    }
}

// gate/up/SwiGLU in one kernel: each block computes the gate and up dot
// products for its row and writes silu(gate) * up directly.
template <typename W, int B>
__global__ void FusedGateUpSwigluKernel(
    W gate_w, W up_w, const float* __restrict__ x,
    const float* __restrict__ norm_w, const float* __restrict__ scales,
    int in_dim, float* __restrict__ out, int inter) {
    __shared__ float red[128];
    const size_t row = blockIdx.x;
    float accg[B];
    float accu[B];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        accg[b] = 0.0f;
        accu[b] = 0.0f;
    }
    const int chunks = in_dim >> 3;
    for (int ci = threadIdx.x; ci < chunks; ci += blockDim.x) {
        const int i0 = ci << 3;
        float gw[8];
        float uw[8];
        gate_w.Load8(row, i0, gw);
        up_w.Load8(row, i0, uw);
        const float4* nwp = reinterpret_cast<const float4*>(norm_w + i0);
        const float4 n0 = nwp[0];
        const float4 n1 = nwp[1];
        const float nw[8] = {n0.x, n0.y, n0.z, n0.w,
                             n1.x, n1.y, n1.z, n1.w};
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* xp = reinterpret_cast<const float4*>(
                x + size_t(b) * in_dim + i0);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            const float xn[8] = {x0.x * nw[0], x0.y * nw[1],
                                 x0.z * nw[2], x0.w * nw[3],
                                 x1.x * nw[4], x1.y * nw[5],
                                 x1.z * nw[6], x1.w * nw[7]};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                accg[b] += gw[j] * xn[j];
                accu[b] += uw[j] * xn[j];
            }
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        red[threadIdx.x] = accg[b];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float g = red[0] * scales[b];
        __syncthreads();
        red[threadIdx.x] = accu[b];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float u = red[0] * scales[b];
            const float silu = g / (1.0f + expf(-g));
            out[size_t(b) * inter + row] = silu * u;
        }
        __syncthreads();
    }
}

// Batched GEMV with a fused epilogue: optional residual accumulation and
// optional folded norm (head projection).
template <typename W, int B, bool Accumulate, bool FoldNorm>
__global__ void MatVecEpilogueKernel(
    W w, const float* __restrict__ x, const float* __restrict__ norm_w,
    const float* __restrict__ scales, int in_dim, float* __restrict__ y,
    int ldy) {
    __shared__ float red[128];
    const size_t row = blockIdx.x;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;
    const int chunks = in_dim >> 3;
    for (int ci = threadIdx.x; ci < chunks; ci += blockDim.x) {
        const int i0 = ci << 3;
        float wv[8];
        w.Load8(row, i0, wv);
        float nw[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        if (FoldNorm) {
            const float4* nwp =
                reinterpret_cast<const float4*>(norm_w + i0);
            const float4 n0 = nwp[0];
            const float4 n1 = nwp[1];
            nw[0] = n0.x; nw[1] = n0.y; nw[2] = n0.z; nw[3] = n0.w;
            nw[4] = n1.x; nw[5] = n1.y; nw[6] = n1.z; nw[7] = n1.w;
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4* xp = reinterpret_cast<const float4*>(
                x + size_t(b) * in_dim + i0);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            acc[b] += wv[0] * (x0.x * nw[0]) + wv[1] * (x0.y * nw[1]) +
                      wv[2] * (x0.z * nw[2]) + wv[3] * (x0.w * nw[3]) +
                      wv[4] * (x1.x * nw[4]) + wv[5] * (x1.y * nw[5]) +
                      wv[6] * (x1.z * nw[6]) + wv[7] * (x1.w * nw[7]);
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        red[threadIdx.x] = acc[b];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float r = red[0];
            if (FoldNorm) r *= scales[b];
            if (Accumulate) {
                y[size_t(b) * ldy + row] += r;
            } else {
                y[size_t(b) * ldy + row] = r;
            }
        }
        __syncthreads();
    }
}

// RoPE for Q and K in one launch: grid.x covers query heads then KV heads.
__global__ void RopeQkKernel(float* __restrict__ q, float* __restrict__ k,
                             int num_heads, int num_kv_heads, int head_dim,
                             const uint32_t* __restrict__ positions,
                             float theta) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int half = head_dim / 2;
    float* h;
    if (head < num_heads) {
        h = q + size_t(row) * num_heads * head_dim + head * head_dim;
    } else {
        h = k + size_t(row) * num_kv_heads * head_dim +
            (head - num_heads) * head_dim;
    }
    const uint32_t pos = positions[row];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(theta, -2.0f * float(i) / float(head_dim));
        float angle = float(pos) * freq;
        float c = cosf(angle);
        float sn = sinf(angle);
        float a = h[i];
        float bb = h[i + half];
        h[i] = a * c - bb * sn;
        h[i + half] = bb * c + a * sn;
    }
}

// K and V scatter in one launch.
__global__ void ScatterKv2Kernel(const float* __restrict__ ksrc,
                                 const float* __restrict__ vsrc,
                                 float* __restrict__ kpool,
                                 float* __restrict__ vpool,
                                 const int* const* __restrict__ tables,
                                 const uint32_t* __restrict__ positions,
                                 int block_tokens, int kv_dim) {
    const int row = blockIdx.x;
    const int* table = tables[row];
    const int t = int(positions[row]);
    const size_t off = (size_t(table[t / block_tokens]) * block_tokens +
                        (t % block_tokens)) *
                       kv_dim;
    for (int i = threadIdx.x; i < 2 * kv_dim; i += blockDim.x) {
        if (i < kv_dim) {
            kpool[off + i] = ksrc[size_t(row) * kv_dim + i];
        } else {
            vpool[off + i - kv_dim] = vsrc[size_t(row) * kv_dim + i - kv_dim];
        }
    }
}

// ---- batched prefill kernels ----


constexpr int kChunk = 128;
constexpr int kGemmTile = 16;

template <typename W>
__global__ void GemmXWtKernel(const float* __restrict__ x, W weight,
                              const float* __restrict__ bias, int n,
                              int in_dim, int out_dim,
                              float* __restrict__ c, int ldc) {
    __shared__ float xs[kGemmTile][kGemmTile];
    __shared__ float ws[kGemmTile][kGemmTile];
    const int token = blockIdx.y * kGemmTile + threadIdx.y;
    const int orow = blockIdx.x * kGemmTile + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < in_dim; k0 += kGemmTile) {
        const int kx = k0 + threadIdx.x;
        xs[threadIdx.y][threadIdx.x] =
            (token < n && kx < in_dim) ? x[size_t(token) * in_dim + kx]
                                       : 0.0f;
        const int wrow = blockIdx.x * kGemmTile + threadIdx.y;
        ws[threadIdx.y][threadIdx.x] =
            (wrow < out_dim && kx < in_dim)
                ? weight.Load(size_t(wrow), kx)
                : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < kGemmTile; ++kk) {
            acc += xs[threadIdx.y][kk] * ws[threadIdx.x][kk];
        }
        __syncthreads();
    }
    if (token < n && orow < out_dim) {
        c[size_t(token) * ldc + orow] =
            acc + (bias != nullptr ? bias[orow] : 0.0f);
    }
}

template <typename WT>
__global__ void GatherEmbedRowsKernel(const WT* __restrict__ embed,
                                      const uint32_t* __restrict__ tokens,
                                      int hidden, float* __restrict__ out) {
    const int row = blockIdx.x;
    const WT* src = embed + size_t(tokens[row]) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        out[size_t(row) * hidden + i] = LoadWeight(src, i);
    }
}

__global__ void RmsNormRowsKernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  float eps, int dim,
                                  float* __restrict__ out) {
    __shared__ float partial[256];
    const float* xr = x + size_t(blockIdx.x) * dim;
    float* outr = out + size_t(blockIdx.x) * dim;
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local += xr[i] * xr[i];
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(dim) + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = xr[i] * scale * weight[i];
    }
}

__global__ void RopeRowsKernel(float* __restrict__ buf, int row_stride,
                               int head_dim, uint32_t pos0, float theta) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int half = head_dim / 2;
    float* h = buf + size_t(row) * row_stride + head * head_dim;
    const uint32_t pos = pos0 + row;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(theta, -2.0f * float(i) / float(head_dim));
        float angle = float(pos) * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = h[i];
        float b = h[i + half];
        h[i] = a * c - b * s;
        h[i + half] = b * c + a * s;
    }
}

// Places chunk row i at paged position pos0 + i.
__global__ void ScatterChunkKvKernel(const float* __restrict__ src,
                                     float* __restrict__ pool,
                                     const int* __restrict__ table,
                                     int block_tokens, uint32_t pos0,
                                     int kv_dim) {
    const int row = blockIdx.x;
    const int t = int(pos0) + row;
    float* dst = pool + (size_t(table[t / block_tokens]) * block_tokens +
                         (t % block_tokens)) *
                            kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        dst[i] = src[size_t(row) * kv_dim + i];
    }
}

__global__ void ChunkAttentionKernel(
    const float* __restrict__ q_buf, const float* __restrict__ k_pool,
    const float* __restrict__ v_pool, const int* __restrict__ table,
    int block_tokens, float* __restrict__ out, uint32_t pos0, int head_dim,
    int kv_stride, int group, float scale) {
    const int head = blockIdx.x;
    const int i = blockIdx.y;
    const int kv_head = head / group;
    const int context = int(pos0) + i + 1;
    const int q_dim = gridDim.x * head_dim;
    const int d = threadIdx.x;

    __shared__ float qs[128], acc[128], red[128];
    __shared__ float m_s, denom_s, p_s, corr_s;

    qs[d] = q_buf[size_t(i) * q_dim + head * head_dim + d] * scale;
    acc[d] = 0.0f;
    if (d == 0) {
        m_s = -1e30f;
        denom_s = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < context; ++t) {
        const float* kt =
            KvRow(k_pool, table, block_tokens, t, kv_stride) +
            kv_head * head_dim;
        red[d] = qs[d] * kt[d];
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float s = red[0];
            const float m_new = fmaxf(m_s, s);
            corr_s = expf(m_s - m_new);
            p_s = expf(s - m_new);
            m_s = m_new;
            denom_s = denom_s * corr_s + p_s;
        }
        __syncthreads();
        const float* vt =
            KvRow(v_pool, table, block_tokens, t, kv_stride) +
            kv_head * head_dim;
        acc[d] = acc[d] * corr_s + p_s * vt[d];
        __syncthreads();
    }
    out[size_t(i) * q_dim + head * head_dim + d] = acc[d] / denom_s;
}

__global__ void SwigluKernel(float* gate, const float* up, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));
        gate[i] = silu * up[i];
    }
}

__global__ void AddKernel(float* y, const float* x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) y[i] += x[i];
}

// -------------------------------------------------------------- buffers

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() {
        if (ptr != nullptr) cudaFree(ptr);
    }
    Status AllocBytes(size_t bytes) {
        LYKURO_CUDA_CHECK(cudaMalloc(&ptr, bytes),
                          "device allocation failed");
        return Status::Ok();
    }
    float* f32() const { return static_cast<float*>(ptr); }
    int* i32() const { return static_cast<int*>(ptr); }
};

struct WeightBuffer {
    DeviceBuffer buf;
    DeviceBuffer scale_buf;  // quantized modes only
    bool bf16 = false;
    WeightQuant quant = WeightQuant::kNone;
    int in_dim = 0;
    int groups_per_row = 0;

    Status Upload(const SafetensorsFile& file, const std::string& name,
                  WeightQuant want = WeightQuant::kNone) {
        const TensorInfo* info = file.FindTensor(name);
        const uint8_t* data = file.TensorData(name);
        if (info == nullptr || data == nullptr) {
            return Status(ErrorCode::kArtifactVerificationFailed,
                          "expected weight tensor missing", kComponent);
        }
        // Matrix geometry (vectors are never quantized here).
        if (info->shape.size() == 2) {
            in_dim = int(info->shape[1]);
        }
        if (want != WeightQuant::kNone) {
            if (info->shape.size() != 2 || (in_dim % 2) != 0) {
                return Status(ErrorCode::kUnsupportedModel,
                              "weight shape unsupported for quantization",
                              kComponent);
            }
            std::vector<float> host(info->element_count);
            switch (info->dtype) {
                case Dtype::kF32:
                    std::memcpy(host.data(), data, info->data_size);
                    break;
                case Dtype::kBf16:
                    Bf16ToFloatArray(
                        reinterpret_cast<const uint16_t*>(data),
                        host.data(), info->element_count);
                    break;
                case Dtype::kF16:
                    Fp16ToFloatArray(
                        reinterpret_cast<const uint16_t*>(data),
                        host.data(), info->element_count);
                    break;
            }
            return Quantize(host, size_t(info->shape[0]), want);
        }
        if (info->dtype == Dtype::kBf16) {
            bf16 = true;
            Status s = buf.AllocBytes(info->data_size);
            if (!s.ok()) return s;
            LYKURO_CUDA_CHECK(cudaMemcpy(buf.ptr, data, info->data_size,
                                         cudaMemcpyHostToDevice),
                              "weight upload failed");
            return Status::Ok();
        }
        std::vector<float> host(info->element_count);
        if (info->dtype == Dtype::kF32) {
            std::memcpy(host.data(), data, info->data_size);
        } else {
            Fp16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             host.data(), info->element_count);
        }
        bf16 = false;
        Status s = buf.AllocBytes(host.size() * sizeof(float));
        if (!s.ok()) return s;
        LYKURO_CUDA_CHECK(cudaMemcpy(buf.ptr, host.data(),
                                     host.size() * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          "weight upload failed");
        return Status::Ok();
    }

private:
    Status Quantize(const std::vector<float>& host, size_t out_dim,
                    WeightQuant want) {
        quant = want;
        if (want == WeightQuant::kInt8) {
            // Per-output-row absmax scale.
            std::vector<int8_t> q(host.size());
            std::vector<float> scales(out_dim);
            for (size_t r = 0; r < out_dim; ++r) {
                const float* row = host.data() + r * in_dim;
                float absmax = 0.0f;
                for (int i = 0; i < in_dim; ++i) {
                    absmax = std::max(absmax, std::abs(row[i]));
                }
                const float scale = absmax > 0 ? absmax / 127.0f : 1.0f;
                scales[r] = scale;
                for (int i = 0; i < in_dim; ++i) {
                    q[r * in_dim + i] = int8_t(std::max(
                        -127.0f,
                        std::min(127.0f, std::round(row[i] / scale))));
                }
            }
            Status s = buf.AllocBytes(q.size());
            if (!s.ok()) return s;
            LYKURO_CUDA_CHECK(cudaMemcpy(buf.ptr, q.data(), q.size(),
                                         cudaMemcpyHostToDevice),
                              "weight upload failed");
            s = scale_buf.AllocBytes(scales.size() * sizeof(float));
            if (!s.ok()) return s;
            LYKURO_CUDA_CHECK(
                cudaMemcpy(scale_buf.ptr, scales.data(),
                           scales.size() * sizeof(float),
                           cudaMemcpyHostToDevice),
                "weight upload failed");
            return Status::Ok();
        }
        // INT4: per-row, per-group absmax scale; values in [-8, 7],
        // stored as nibble + 8.
        groups_per_row = (in_dim + kQuantGroup - 1) / kQuantGroup;
        std::vector<uint8_t> packed(host.size() / 2);
        std::vector<float> scales(out_dim * size_t(groups_per_row));
        for (size_t r = 0; r < out_dim; ++r) {
            const float* row = host.data() + r * in_dim;
            for (int g = 0; g < groups_per_row; ++g) {
                const int g0 = g * kQuantGroup;
                const int g1 = std::min(in_dim, g0 + kQuantGroup);
                float absmax = 0.0f;
                for (int i = g0; i < g1; ++i) {
                    absmax = std::max(absmax, std::abs(row[i]));
                }
                scales[r * groups_per_row + g] =
                    absmax > 0 ? absmax / 7.0f : 1.0f;
            }
            for (int i = 0; i < in_dim; i += 2) {
                auto nib = [&](int j) {
                    const float scale =
                        scales[r * groups_per_row + j / kQuantGroup];
                    const int v = int(std::max(
                        -8.0f,
                        std::min(7.0f, std::round(row[j] / scale))));
                    return uint8_t(v + 8);
                };
                packed[(r * in_dim + i) / 2] =
                    uint8_t(nib(i) | (nib(i + 1) << 4));
            }
        }
        Status s = buf.AllocBytes(packed.size());
        if (!s.ok()) return s;
        LYKURO_CUDA_CHECK(cudaMemcpy(buf.ptr, packed.data(), packed.size(),
                                     cudaMemcpyHostToDevice),
                          "weight upload failed");
        s = scale_buf.AllocBytes(scales.size() * sizeof(float));
        if (!s.ok()) return s;
        LYKURO_CUDA_CHECK(cudaMemcpy(scale_buf.ptr, scales.data(),
                                     scales.size() * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          "weight upload failed");
        return Status::Ok();
    }
};

Status UploadF32(const SafetensorsFile& file, const std::string& name,
                 DeviceBuffer& dst) {
    const TensorInfo* info = file.FindTensor(name);
    const uint8_t* data = file.TensorData(name);
    if (info == nullptr || data == nullptr) {
        return Status(ErrorCode::kArtifactVerificationFailed,
                      "expected weight tensor missing", kComponent);
    }
    std::vector<float> host(info->element_count);
    switch (info->dtype) {
        case Dtype::kF32:
            std::memcpy(host.data(), data, info->data_size);
            break;
        case Dtype::kBf16:
            Bf16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             host.data(), info->element_count);
            break;
        case Dtype::kF16:
            Fp16ToFloatArray(reinterpret_cast<const uint16_t*>(data),
                             host.data(), info->element_count);
            break;
    }
    Status s = dst.AllocBytes(host.size() * sizeof(float));
    if (!s.ok()) return s;
    LYKURO_CUDA_CHECK(cudaMemcpy(dst.ptr, host.data(),
                                 host.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "weight upload failed");
    return Status::Ok();
}

// Invokes fn with the correctly-typed weight accessor.
template <typename Fn>
void WithWeightView(const WeightBuffer& w, int in_dim, Fn&& fn) {
    switch (w.quant) {
        case WeightQuant::kInt8:
            fn(Int8Weight{static_cast<const int8_t*>(w.buf.ptr),
                          w.scale_buf.f32(), in_dim});
            return;
        case WeightQuant::kInt4:
            fn(Int4Weight{static_cast<const uint8_t*>(w.buf.ptr),
                          w.scale_buf.f32(), in_dim, w.groups_per_row});
            return;
        case WeightQuant::kNone:
            break;
    }
    if (w.bf16) {
        fn(Bf16Weight{static_cast<const __nv_bfloat16*>(w.buf.ptr),
                      in_dim});
    } else {
        fn(F32Weight{static_cast<const float*>(w.buf.ptr), in_dim});
    }
}

void LaunchMatVec(const WeightBuffer& w, const float* x, const float* bias,
                  int in_dim, int out_dim, float* y) {
    WithWeightView(w, in_dim, [&](auto view) {
        MatVecKernel<<<out_dim, 128>>>(view, x, bias, in_dim, y);
    });
}

void LaunchMatVecBatch(const float* x, const WeightBuffer& w,
                       const float* bias, int batch, int in_dim,
                       int out_dim, float* y, int ldy,
                       cudaStream_t stream = nullptr) {
    WithWeightView(w, in_dim, [&](auto view) {
        if (batch <= 1) {
            MatVecKernel<<<out_dim, 128, 0, stream>>>(view, x, bias,
                                                      in_dim, y);
        } else if (batch <= 2) {
            MatVecBatchKernel<decltype(view), 2>
                <<<out_dim, 128, 0, stream>>>(view, x, bias, in_dim, y,
                                              ldy);
        } else if (batch <= 4) {
            MatVecBatchKernel<decltype(view), 4>
                <<<out_dim, 128, 0, stream>>>(view, x, bias, in_dim, y,
                                              ldy);
        } else if (batch <= 8) {
            MatVecBatchKernel<decltype(view), 8>
                <<<out_dim, 128, 0, stream>>>(view, x, bias, in_dim, y,
                                              ldy);
        } else {
            MatVecBatchKernel<decltype(view), 16>
                <<<out_dim, 128, 0, stream>>>(view, x, bias, in_dim, y,
                                              ldy);
        }
    });
}

template <int B>
void LaunchFusedQkvB(const WeightBuffer& qw, const WeightBuffer& kw,
                     const WeightBuffer& vw, const float* q_bias,
                     const float* k_bias, const float* v_bias,
                     const float* x, const float* norm_w,
                     const float* scales, int in_dim, int q_dim,
                     int kv_dim, float* q_out, float* k_out, float* v_out,
                     cudaStream_t stream) {
    const int rows = q_dim + 2 * kv_dim;
    WithWeightView(qw, in_dim, [&](auto qv) {
        WithWeightView(kw, in_dim, [&](auto kv) {
            WithWeightView(vw, in_dim, [&](auto vv) {
                // decltype through by-reference lambda captures yields
                // reference types; decay before comparing/instantiating.
                using QT = std::decay_t<decltype(qv)>;
                using KT = std::decay_t<decltype(kv)>;
                using VT = std::decay_t<decltype(vv)>;
                if constexpr (std::is_same_v<QT, KT> &&
                              std::is_same_v<KT, VT>) {
                    FusedQkvKernel<QT, B><<<rows, 128, 0, stream>>>(
                        qv, kv, vv, q_bias, k_bias, v_bias, x, norm_w,
                        scales, in_dim, q_dim, kv_dim, q_out, k_out,
                        v_out);
                }
            });
        });
    });
}

template <int B>
void LaunchFusedGateUpB(const WeightBuffer& gate_w,
                        const WeightBuffer& up_w, const float* x,
                        const float* norm_w, const float* scales,
                        int in_dim, float* out, int inter,
                        cudaStream_t stream) {
    WithWeightView(gate_w, in_dim, [&](auto gv) {
        WithWeightView(up_w, in_dim, [&](auto uv) {
            using GT = std::decay_t<decltype(gv)>;
            using UT = std::decay_t<decltype(uv)>;
            if constexpr (std::is_same_v<GT, UT>) {
                FusedGateUpSwigluKernel<GT, B>
                    <<<inter, 128, 0, stream>>>(gv, uv, x, norm_w, scales,
                                                in_dim, out, inter);
            }
        });
    });
}

template <int B, bool Accumulate, bool FoldNorm>
void LaunchMatVecEpiB(const WeightBuffer& w, const float* x,
                      const float* norm_w, const float* scales, int in_dim,
                      int out_dim, float* y, int ldy,
                      cudaStream_t stream) {
    WithWeightView(w, in_dim, [&](auto view) {
        MatVecEpilogueKernel<decltype(view), B, Accumulate, FoldNorm>
            <<<out_dim, 128, 0, stream>>>(view, x, norm_w, scales, in_dim,
                                          y, ldy);
    });
}

void LaunchGemm(const float* x, const WeightBuffer& w, const float* bias,
                int n, int in_dim, int out_dim, float* c, int ldc,
                cudaStream_t stream = nullptr) {
    dim3 grid((out_dim + kGemmTile - 1) / kGemmTile,
              (n + kGemmTile - 1) / kGemmTile);
    dim3 block(kGemmTile, kGemmTile);
    WithWeightView(w, in_dim, [&](auto view) {
        GemmXWtKernel<<<grid, block, 0, stream>>>(x, view, bias, n, in_dim,
                                                  out_dim, c, ldc);
    });
}

// ------------------------------------------------------- paged KV pool

uint64_t Fnv64(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Block pool shared by every sequence of one model instance (spec §16.3).
// Block ids are shared across layers: block b maps to the same row range
// in each layer's K and V pool tensors. Full prompt blocks may be
// registered in the prefix cache (chain-hash keyed, scope-seeded);
// blocks with refcount 0 stay cached until LRU eviction reclaims them.
class KvPool {
public:
    Status Init(uint32_t num_layers, uint32_t kv_stride,
                uint32_t block_tokens, uint32_t total_tokens) {
        block_tokens_ = block_tokens;
        num_blocks_ = (total_tokens + block_tokens - 1) / block_tokens;
        k_layers_.resize(num_layers);
        v_layers_.resize(num_layers);
        const size_t bytes =
            size_t(num_blocks_) * block_tokens * kv_stride * sizeof(float);
        for (uint32_t l = 0; l < num_layers; ++l) {
            Status s = k_layers_[l].AllocBytes(bytes);
            if (!s.ok()) return s;
            s = v_layers_[l].AllocBytes(bytes);
            if (!s.ok()) return s;
        }
        meta_.assign(num_blocks_, BlockMeta{});
        free_.resize(num_blocks_);
        for (uint32_t b = 0; b < num_blocks_; ++b) {
            free_[num_blocks_ - 1 - b] = int(b);
        }
        return Status::Ok();
    }

    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t num_blocks() const { return num_blocks_; }
    uint32_t free_blocks() const {
        uint32_t evictable = 0;
        for (const BlockMeta& m : meta_) {
            if (m.cached && m.refcount == 0) ++evictable;
        }
        return uint32_t(free_.size()) + evictable;
    }
    float* k_layer(uint32_t l) { return k_layers_[l].f32(); }
    float* v_layer(uint32_t l) { return v_layers_[l].f32(); }

    // Allocates a private block: free list first, then LRU eviction of an
    // unreferenced cached block.
    bool Allocate(int& block_out) {
        if (!free_.empty()) {
            block_out = free_.back();
            free_.pop_back();
            meta_[block_out] = BlockMeta{};
            meta_[block_out].refcount = 1;
            return true;
        }
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t b = 0; b < num_blocks_; ++b) {
            if (meta_[b].cached && meta_[b].refcount == 0 &&
                meta_[b].lru_tick < oldest) {
                oldest = meta_[b].lru_tick;
                victim = int(b);
            }
        }
        if (victim < 0) return false;
        by_hash_.erase(meta_[victim].hash);
        meta_[victim] = BlockMeta{};
        meta_[victim].refcount = 1;
        block_out = victim;
        return true;
    }

    void Release(int block) {
        BlockMeta& m = meta_[block];
        if (m.refcount > 0) --m.refcount;
        if (m.refcount == 0) {
            if (m.cached) {
                m.lru_tick = ++tick_;  // stays cached, evictable
            } else {
                free_.push_back(block);
            }
        }
    }

    // Prefix-cache lookup; on hit takes a reference.
    bool LookupCached(uint64_t hash, int& block_out) {
        auto it = by_hash_.find(hash);
        if (it == by_hash_.end()) return false;
        block_out = it->second;
        ++meta_[block_out].refcount;
        meta_[block_out].lru_tick = ++tick_;
        return true;
    }

    // Registers an owned full block under its chain hash (first writer
    // wins on the astronomically-unlikely 64-bit hash collision;
    // production adds full-token comparison).
    void RegisterCached(uint64_t hash, int block) {
        if (by_hash_.count(hash)) return;
        meta_[block].cached = true;
        meta_[block].hash = hash;
        meta_[block].lru_tick = ++tick_;
        by_hash_.emplace(hash, block);
    }

private:
    struct BlockMeta {
        int refcount = 0;
        bool cached = false;
        uint64_t hash = 0;
        uint64_t lru_tick = 0;
    };

    uint32_t block_tokens_ = 64;
    uint32_t num_blocks_ = 0;
    std::vector<DeviceBuffer> k_layers_;
    std::vector<DeviceBuffer> v_layers_;
    std::vector<BlockMeta> meta_;
    std::vector<int> free_;
    std::unordered_map<uint64_t, int> by_hash_;
    uint64_t tick_ = 0;
};

}  // namespace

struct QwenCudaModel::Impl {
    struct Layer {
        DeviceBuffer input_norm, q_b, k_b, v_b, post_norm;
        WeightBuffer q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
    };

    WeightBuffer embed;
    // Quantized copy of the lm head (decode logits projection). The
    // embedding table itself stays checkpoint-dtype for lookups; the
    // Metal backend shipped a quantized head with parity-tested quality,
    // and BF16 head reads were the single largest decode cost.
    WeightBuffer head_q;
    std::vector<Layer> layers;
    DeviceBuffer final_norm;
    WeightBuffer lm_head;
    bool tied = true;
    const WeightBuffer& head() const {
        if (head_q.buf.ptr != nullptr) return head_q;
        return tied ? embed : lm_head;
    }

    KvPool pool;
    uint64_t prefix_hit_tokens = 0;

    DeviceBuffer hidden, normed, q, attn_out, proj, gate, up, logits;
    DeviceBuffer attn_partials;

    DeviceBuffer x_rows, xn_rows, q_rows, attn_rows, proj_rows, gate_rows,
        up_rows;
    DeviceBuffer d_tokens;
    bool chunked_prefill = false;

    DeviceBuffer d_positions;
    DeviceBuffer d_norm_scales;  // [kMaxBatch] fused-path inverse RMS
    DeviceBuffer d_seq_tables;  // [kMaxBatch] const int*
    DeviceBuffer kv_k_rows, kv_v_rows;
    DeviceBuffer batch_logits;
    DeviceBuffer batch_partials;

    // CUDA Graphs (decode fast path): one captured graph per batch
    // bucket. Every per-token input lives in device memory (tokens,
    // positions, per-sequence table pointers), so a graph replays with
    // fixed kernel parameters and grid shapes.
    cudaStream_t stream = nullptr;
    // Keyed by (batch bucket {1,2,4,8,16}, splits bucket {1,8,32}).
    cudaGraphExec_t decode_graphs[5][3] = {};
    DeviceBuffer pad_table;  // int[1]: reserved scratch block id
    int scratch_block = -1;  // sink for padding-row KV writes

    ~Impl() {
        for (auto& row : decode_graphs) {
            for (cudaGraphExec_t g : row) {
                if (g != nullptr) cudaGraphExecDestroy(g);
            }
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

namespace {

// Fused decode graph body for batch <= 8 (see the fused-kernel section):
// ~10 kernels per layer instead of ~18 on the dependency chain.
template <int B>
void RunFusedDecode(QwenCudaModel::Impl& impl, const QwenConfig& c,
                    const uint32_t* positions_dev,
                    const int* const* tables_dev, int splits,
                    cudaStream_t stream) {
    const int h = int(c.hidden_size);
    const int q_dim = int(c.num_heads * c.head_dim);
    const int kv_dim = int(c.num_kv_heads * c.head_dim);
    const int group = int(c.num_heads / c.num_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
    const int threads = 256;

    if (impl.embed.bf16) {
        GatherEmbedRowsKernel<__nv_bfloat16><<<B, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(impl.embed.buf.ptr),
            static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
            impl.x_rows.f32());
    } else {
        GatherEmbedRowsKernel<float><<<B, threads, 0, stream>>>(
            static_cast<const float*>(impl.embed.buf.ptr),
            static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
            impl.x_rows.f32());
    }

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        QwenCudaModel::Impl::Layer& layer = impl.layers[l];
        NormScaleKernel<<<B, 256, 0, stream>>>(
            impl.x_rows.f32(), c.rms_norm_eps, h,
            impl.d_norm_scales.f32());
        LaunchFusedQkvB<B>(layer.q_w, layer.k_w, layer.v_w,
                           layer.q_b.f32(), layer.k_b.f32(),
                           layer.v_b.f32(), impl.x_rows.f32(),
                           layer.input_norm.f32(),
                           impl.d_norm_scales.f32(), h, q_dim, kv_dim,
                           impl.q_rows.f32(), impl.kv_k_rows.f32(),
                           impl.kv_v_rows.f32(), stream);
        {
            dim3 grid_rope(c.num_heads + c.num_kv_heads, B);
            RopeQkKernel<<<grid_rope, 32, 0, stream>>>(
                impl.q_rows.f32(), impl.kv_k_rows.f32(),
                int(c.num_heads), int(c.num_kv_heads), int(c.head_dim),
                positions_dev, c.rope_theta);
            ScatterKv2Kernel<<<B, threads, 0, stream>>>(
                impl.kv_k_rows.f32(), impl.kv_v_rows.f32(),
                impl.pool.k_layer(l), impl.pool.v_layer(l), tables_dev,
                positions_dev, int(impl.pool.block_tokens()), kv_dim);
            dim3 grid_attn(c.num_heads, splits, B);
            BatchSplitAttentionKernel
                <<<grid_attn, kAttnThreads, 0, stream>>>(
                    impl.q_rows.f32(), impl.pool.k_layer(l),
                    impl.pool.v_layer(l), tables_dev,
                    int(impl.pool.block_tokens()), positions_dev,
                    impl.batch_partials.f32(), int(c.head_dim), kv_dim,
                    group, attn_scale, splits);
            dim3 grid_combine(c.num_heads, B);
            BatchCombineAttentionKernel
                <<<grid_combine, c.head_dim, 0, stream>>>(
                    impl.batch_partials.f32(), impl.attn_rows.f32(),
                    int(c.head_dim), splits);
        }
        LaunchMatVecEpiB<B, true, false>(
            layer.o_w, impl.attn_rows.f32(), nullptr, nullptr, q_dim, h,
            impl.x_rows.f32(), h, stream);

        NormScaleKernel<<<B, 256, 0, stream>>>(
            impl.x_rows.f32(), c.rms_norm_eps, h,
            impl.d_norm_scales.f32());
        LaunchFusedGateUpB<B>(layer.gate_w, layer.up_w, impl.x_rows.f32(),
                              layer.post_norm.f32(),
                              impl.d_norm_scales.f32(), h,
                              impl.gate_rows.f32(),
                              int(c.intermediate_size), stream);
        LaunchMatVecEpiB<B, true, false>(
            layer.down_w, impl.gate_rows.f32(), nullptr, nullptr,
            int(c.intermediate_size), h, impl.x_rows.f32(), h, stream);
    }

    NormScaleKernel<<<B, 256, 0, stream>>>(impl.x_rows.f32(),
                                           c.rms_norm_eps, h,
                                           impl.d_norm_scales.f32());
    const WeightBuffer& head = impl.head();
    LaunchMatVecEpiB<B, false, true>(head, impl.x_rows.f32(),
                                     impl.final_norm.f32(),
                                     impl.d_norm_scales.f32(), h,
                                     int(c.vocab_size),
                                     impl.batch_logits.f32(),
                                     int(c.vocab_size), stream);
}

// Paged per-sequence state: logical capacity is max_tokens; physical
// blocks are allocated on demand and released on destruction (spec §16.1
// ownership; release is idempotent via pool refcounts).
class CudaSequenceState final : public SequenceState {
public:
    static Status Create(KvPool* pool, uint32_t max_tokens,
                         const SequenceOptions& options,
                         std::unique_ptr<CudaSequenceState>& out) {
        auto state = std::unique_ptr<CudaSequenceState>(
            new CudaSequenceState());
        state->pool_ = pool;
        state->max_tokens_ = max_tokens;
        state->options_ = options;
        const uint32_t max_blocks =
            (max_tokens + pool->block_tokens() - 1) / pool->block_tokens();
        Status s = state->d_table_.AllocBytes(
            std::max<uint32_t>(1, max_blocks) * sizeof(int));
        if (!s.ok()) return s;
        out = std::move(state);
        return Status::Ok();
    }

    ~CudaSequenceState() override {
        for (int b : blocks_) pool_->Release(b);
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }
    const SequenceOptions& options() const { return options_; }

    // Ensures blocks exist for positions [0, pos]; uploads the device
    // table when it grew. Fails with capacity_exhausted when the pool is
    // out of blocks.
    Status EnsureCapacity(uint32_t pos) {
        const uint32_t bt = pool_->block_tokens();
        const uint32_t needed = pos / bt + 1;
        bool grew = false;
        while (blocks_.size() < needed) {
            int block = -1;
            if (!pool_->Allocate(block)) {
                return Status(ErrorCode::kCapacityExhausted,
                              "kv block pool exhausted", kComponent);
            }
            blocks_.push_back(block);
            grew = true;
        }
        if (grew) {
            LYKURO_CUDA_CHECK(
                cudaMemcpy(d_table_.ptr, blocks_.data(),
                           blocks_.size() * sizeof(int),
                           cudaMemcpyHostToDevice),
                "block table upload failed");
        }
        return Status::Ok();
    }

    // Adopts shared prefix blocks (already referenced via the pool).
    Status AttachPrefix(const std::vector<int>& shared_blocks,
                        uint32_t tokens) {
        blocks_ = shared_blocks;
        length_ = tokens;
        LYKURO_CUDA_CHECK(
            cudaMemcpy(d_table_.ptr, blocks_.data(),
                       blocks_.size() * sizeof(int),
                       cudaMemcpyHostToDevice),
            "block table upload failed");
        return Status::Ok();
    }

    float* KeyRowHost(KvPool& pool, uint32_t layer, uint32_t t,
                      uint32_t kv_stride) {
        const uint32_t bt = pool.block_tokens();
        return pool.k_layer(layer) +
               (size_t(blocks_[t / bt]) * bt + t % bt) * kv_stride;
    }
    float* ValueRowHost(KvPool& pool, uint32_t layer, uint32_t t,
                        uint32_t kv_stride) {
        const uint32_t bt = pool.block_tokens();
        return pool.v_layer(layer) +
               (size_t(blocks_[t / bt]) * bt + t % bt) * kv_stride;
    }

    const int* device_table() const { return d_table_.i32(); }
    const std::vector<int>& blocks() const { return blocks_; }
    void Advance() { ++length_; }

private:
    CudaSequenceState() = default;
    KvPool* pool_ = nullptr;
    uint32_t max_tokens_ = 0;
    uint32_t length_ = 0;
    SequenceOptions options_;
    std::vector<int> blocks_;
    DeviceBuffer d_table_;
};

}  // namespace

QwenCudaModel::~QwenCudaModel() = default;

QwenCudaModel::LoadResult QwenCudaModel::Load(const ModelManifest& manifest,
                                              const SafetensorsFile& weights,
                                              int device_id) {
    CudaModelOptions options;
    options.device_id = device_id;
    return Load(manifest, weights, options);
}

QwenCudaModel::LoadResult QwenCudaModel::Load(
    const ModelManifest& manifest, const SafetensorsFile& weights,
    const CudaModelOptions& options) {
    LoadResult result;
    auto model = std::unique_ptr<QwenCudaModel>(new QwenCudaModel());

    result.status = QwenConfig::FromManifest(manifest, model->config_);
    if (!result.status.ok()) return result;
    const QwenConfig& c = model->config_;
    model->limits_.vocab_size = c.vocab_size;
    model->limits_.max_context_tokens = c.max_context_tokens;
    model->limits_.eos_token_ids = c.eos_token_ids;
    model->device_id_ = options.device_id;

    if (cudaSetDevice(options.device_id) != cudaSuccess) {
        result.status = Status(ErrorCode::kGpuUnhealthy,
                               "cuda device not usable", kComponent);
        return result;
    }
    model->impl_ = std::make_unique<Impl>();
    Impl& impl = *model->impl_;
    impl.tied = c.tie_word_embeddings;

    auto up_w = [&](const std::string& name, WeightBuffer& dst) {
        if (result.status.ok()) {
            result.status =
                dst.Upload(weights, name, options.quantization);
        }
    };
    auto up_native = [&](const std::string& name, WeightBuffer& dst) {
        if (result.status.ok()) result.status = dst.Upload(weights, name);
    };
    auto up_f = [&](const std::string& name, DeviceBuffer& dst) {
        if (result.status.ok()) {
            result.status = UploadF32(weights, name, dst);
        }
    };

    // The embedding table keeps the checkpoint dtype (lookup quality);
    // in quantized modes the lm head additionally gets a quantized copy
    // for the decode logits projection (same policy the Metal backend
    // ships as metal-q8/q4).
    up_native("model.embed_tokens.weight", impl.embed);
    if (options.quantization != WeightQuant::kNone &&
        result.status.ok()) {
        result.status = impl.head_q.Upload(
            weights,
            c.tie_word_embeddings ? "model.embed_tokens.weight"
                                  : "lm_head.weight",
            options.quantization);
    }
    impl.layers.resize(c.num_layers);
    for (uint32_t l = 0; l < c.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        Impl::Layer& layer = impl.layers[l];
        up_f(p + "input_layernorm.weight", layer.input_norm);
        up_w(p + "self_attn.q_proj.weight", layer.q_w);
        up_f(p + "self_attn.q_proj.bias", layer.q_b);
        up_w(p + "self_attn.k_proj.weight", layer.k_w);
        up_f(p + "self_attn.k_proj.bias", layer.k_b);
        up_w(p + "self_attn.v_proj.weight", layer.v_w);
        up_f(p + "self_attn.v_proj.bias", layer.v_b);
        up_w(p + "self_attn.o_proj.weight", layer.o_w);
        up_f(p + "post_attention_layernorm.weight", layer.post_norm);
        up_w(p + "mlp.gate_proj.weight", layer.gate_w);
        up_w(p + "mlp.up_proj.weight", layer.up_w);
        up_w(p + "mlp.down_proj.weight", layer.down_w);
    }
    up_f("model.norm.weight", impl.final_norm);
    if (!c.tie_word_embeddings) {
        up_native("lm_head.weight", impl.lm_head);
    }
    if (!result.status.ok()) return result;

    // Paged KV pool (spec §16.3). Default capacity: 4x max context.
    const uint32_t kv_stride = c.num_kv_heads * c.head_dim;
    const uint32_t pool_tokens = options.kv_pool_tokens != 0
                                     ? options.kv_pool_tokens
                                     : c.max_context_tokens * 4;
    result.status = impl.pool.Init(c.num_layers, kv_stride,
                                   options.kv_block_tokens, pool_tokens);
    if (!result.status.ok()) return result;

    const size_t q_dim = size_t(c.num_heads) * c.head_dim;
    Status s = impl.hidden.AllocBytes(c.hidden_size * sizeof(float));
    if (s.ok()) s = impl.normed.AllocBytes(c.hidden_size * sizeof(float));
    if (s.ok()) s = impl.q.AllocBytes(q_dim * sizeof(float));
    if (s.ok()) s = impl.attn_out.AllocBytes(q_dim * sizeof(float));
    if (s.ok()) s = impl.proj.AllocBytes(c.hidden_size * sizeof(float));
    if (s.ok()) {
        s = impl.gate.AllocBytes(c.intermediate_size * sizeof(float));
    }
    if (s.ok()) s = impl.up.AllocBytes(c.intermediate_size * sizeof(float));
    if (s.ok()) s = impl.logits.AllocBytes(c.vocab_size * sizeof(float));
    if (s.ok()) {
        s = impl.attn_partials.AllocBytes(size_t(c.num_heads) * kMaxSplits *
                                          (c.head_dim + 2) * sizeof(float));
    }

    impl.chunked_prefill =
        c.head_dim <= 128 && (c.head_dim & (c.head_dim - 1)) == 0;
    if (s.ok() && impl.chunked_prefill) {
        const size_t rows = kChunk;
        if (s.ok()) s = impl.x_rows.AllocBytes(rows * c.hidden_size * 4);
        if (s.ok()) s = impl.xn_rows.AllocBytes(rows * c.hidden_size * 4);
        if (s.ok()) s = impl.q_rows.AllocBytes(rows * q_dim * 4);
        if (s.ok()) s = impl.attn_rows.AllocBytes(rows * q_dim * 4);
        if (s.ok()) s = impl.proj_rows.AllocBytes(rows * c.hidden_size * 4);
        if (s.ok()) {
            s = impl.gate_rows.AllocBytes(rows * c.intermediate_size * 4);
        }
        if (s.ok()) {
            s = impl.up_rows.AllocBytes(rows * c.intermediate_size * 4);
        }
        if (s.ok()) s = impl.d_tokens.AllocBytes(rows * sizeof(uint32_t));

        const size_t kv_dim_sz = size_t(c.num_kv_heads) * c.head_dim;
        if (s.ok()) {
            s = impl.d_positions.AllocBytes(kMaxBatch * sizeof(uint32_t));
        }
        if (s.ok()) {
            s = impl.d_norm_scales.AllocBytes(kMaxBatch * sizeof(float));
        }
        if (s.ok()) {
            s = impl.d_seq_tables.AllocBytes(kMaxBatch * sizeof(int*));
        }
        if (s.ok() &&
            cudaStreamCreate(&impl.stream) != cudaSuccess) {
            s = Status(ErrorCode::kGpuUnhealthy, "stream creation failed",
                       kComponent);
        }
        // Reserve one pool block as the write sink for padding rows in
        // graph-replayed decode batches.
        if (s.ok()) {
            if (!impl.pool.Allocate(impl.scratch_block)) {
                s = Status(ErrorCode::kGpuOom, "kv pool too small",
                           kComponent);
            }
        }
        if (s.ok()) s = impl.pad_table.AllocBytes(sizeof(int));
        if (s.ok() &&
            cudaMemcpy(impl.pad_table.ptr, &impl.scratch_block,
                       sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            s = Status(ErrorCode::kGpuUnhealthy, "pad table upload failed",
                       kComponent);
        }
        if (s.ok()) {
            s = impl.kv_k_rows.AllocBytes(rows * kv_dim_sz * 4);
        }
        if (s.ok()) {
            s = impl.kv_v_rows.AllocBytes(rows * kv_dim_sz * 4);
        }
        if (s.ok()) {
            s = impl.batch_logits.AllocBytes(size_t(kMaxBatch) *
                                             c.vocab_size * 4);
        }
        if (s.ok()) {
            s = impl.batch_partials.AllocBytes(
                size_t(kMaxBatch) * c.num_heads * kMaxSplits *
                (c.head_dim + 2) * 4);
        }
    }
    if (!s.ok()) {
        result.status = s;
        return result;
    }

    result.model = std::move(model);
    return result;
}

uint32_t QwenCudaModel::kv_blocks_total() const {
    // Usable blocks: the padding scratch block is permanently reserved.
    return impl_->pool.num_blocks() -
           (impl_->scratch_block >= 0 ? 1 : 0);
}
uint32_t QwenCudaModel::kv_blocks_free() const {
    return impl_->pool.free_blocks();
}
uint64_t QwenCudaModel::prefix_cache_hit_tokens() const {
    return impl_->prefix_hit_tokens;
}

Status QwenCudaModel::CreateSequence(uint32_t max_tokens,
                                     std::unique_ptr<SequenceState>& out) {
    return CreateSequenceEx(max_tokens, SequenceOptions{}, out);
}

Status QwenCudaModel::CreateSequenceEx(uint32_t max_tokens,
                                       const SequenceOptions& options,
                                       std::unique_ptr<SequenceState>& out) {
    if (cudaSetDevice(device_id_) != cudaSuccess) {
        return Status(ErrorCode::kGpuUnhealthy, "cuda device not usable",
                      kComponent);
    }
    std::unique_ptr<CudaSequenceState> state;
    Status s = CudaSequenceState::Create(&impl_->pool, max_tokens, options,
                                         state);
    if (!s.ok()) return s;
    out = std::move(state);
    return Status::Ok();
}

Status QwenCudaModel::ForwardToken(uint32_t token, uint32_t pos,
                                   void* sequence_state,
                                   std::vector<float>& logits_out,
                                   bool want_logits) {
    auto& state = *static_cast<CudaSequenceState*>(sequence_state);
    const QwenConfig& c = config_;
    Impl& impl = *impl_;
    const int h = int(c.hidden_size);
    const int q_dim = int(c.num_heads * c.head_dim);
    const int kv_dim = int(c.num_kv_heads * c.head_dim);
    const int group = int(c.num_heads / c.num_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
    const int threads = 256;

    LYKURO_CUDA_CHECK(cudaSetDevice(device_id_), "cuda device not usable");
    Status cap = state.EnsureCapacity(pos);
    if (!cap.ok()) return cap;

    if (impl.embed.bf16) {
        GatherEmbedKernel<__nv_bfloat16><<<4, threads>>>(
            static_cast<const __nv_bfloat16*>(impl.embed.buf.ptr), token, h,
            impl.hidden.f32());
    } else {
        GatherEmbedKernel<float><<<4, threads>>>(
            static_cast<const float*>(impl.embed.buf.ptr), token, h,
            impl.hidden.f32());
    }

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        Impl::Layer& layer = impl.layers[l];
        float* k_dst = state.KeyRowHost(impl.pool, l, pos, kv_dim);
        float* v_dst = state.ValueRowHost(impl.pool, l, pos, kv_dim);

        RmsNormKernel<<<1, 256>>>(impl.hidden.f32(), layer.input_norm.f32(),
                                  c.rms_norm_eps, h, impl.normed.f32());
        LaunchMatVec(layer.q_w, impl.normed.f32(), layer.q_b.f32(), h,
                     q_dim, impl.q.f32());
        LaunchMatVec(layer.k_w, impl.normed.f32(), layer.k_b.f32(), h,
                     kv_dim, k_dst);
        LaunchMatVec(layer.v_w, impl.normed.f32(), layer.v_b.f32(), h,
                     kv_dim, v_dst);

        RopeKernel<<<c.num_heads, 32>>>(impl.q.f32(), int(c.head_dim), pos,
                                        c.rope_theta);
        RopeKernel<<<c.num_kv_heads, 32>>>(k_dst, int(c.head_dim), pos,
                                           c.rope_theta);
        {
            const int context = int(pos + 1);
            int splits = (context + 127) / 128;
            if (splits > kMaxSplits) splits = kMaxSplits;
            dim3 grid(c.num_heads, splits);
            SplitAttentionKernel<<<grid, kAttnThreads>>>(
                impl.q.f32(), impl.pool.k_layer(l), impl.pool.v_layer(l),
                state.device_table(), int(impl.pool.block_tokens()),
                impl.attn_partials.f32(), context, int(c.head_dim), kv_dim,
                group, attn_scale, splits);
            CombineAttentionKernel<<<c.num_heads, c.head_dim>>>(
                impl.attn_partials.f32(), impl.attn_out.f32(),
                int(c.head_dim), splits);
        }

        LaunchMatVec(layer.o_w, impl.attn_out.f32(), nullptr, q_dim, h,
                     impl.proj.f32());
        AddKernel<<<(h + threads - 1) / threads, threads>>>(
            impl.hidden.f32(), impl.proj.f32(), h);

        RmsNormKernel<<<1, 256>>>(impl.hidden.f32(), layer.post_norm.f32(),
                                  c.rms_norm_eps, h, impl.normed.f32());
        LaunchMatVec(layer.gate_w, impl.normed.f32(), nullptr, h,
                     int(c.intermediate_size), impl.gate.f32());
        LaunchMatVec(layer.up_w, impl.normed.f32(), nullptr, h,
                     int(c.intermediate_size), impl.up.f32());
        SwigluKernel<<<(int(c.intermediate_size) + threads - 1) / threads,
                       threads>>>(impl.gate.f32(), impl.up.f32(),
                                  int(c.intermediate_size));
        LaunchMatVec(layer.down_w, impl.gate.f32(), nullptr,
                     int(c.intermediate_size), h, impl.proj.f32());
        AddKernel<<<(h + threads - 1) / threads, threads>>>(
            impl.hidden.f32(), impl.proj.f32(), h);
    }

    if (want_logits) {
        RmsNormKernel<<<1, 256>>>(impl.hidden.f32(), impl.final_norm.f32(),
                                  c.rms_norm_eps, h, impl.normed.f32());
        const WeightBuffer& head = impl.head();
        LaunchMatVec(head, impl.normed.f32(), nullptr, h,
                     int(c.vocab_size), impl.logits.f32());
        logits_out.resize(c.vocab_size);
        LYKURO_CUDA_CHECK(
            cudaMemcpy(logits_out.data(), impl.logits.ptr,
                       c.vocab_size * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "logits download failed");
        for (float v : logits_out) {
            if (!std::isfinite(v)) {
                return Status(ErrorCode::kInferenceFailed,
                              "logits contain non-finite values",
                              kComponent);
            }
        }
    }
    LYKURO_CUDA_CHECK(cudaGetLastError(), "kernel launch failed");
    return Status::Ok();
}

Status QwenCudaModel::ForwardChunk(const uint32_t* tokens, uint32_t n,
                                   uint32_t pos0, void* sequence_state,
                                   std::vector<float>& logits_out,
                                   bool want_logits_of_last) {
    auto& state = *static_cast<CudaSequenceState*>(sequence_state);
    const QwenConfig& c = config_;
    Impl& impl = *impl_;
    const int h = int(c.hidden_size);
    const int q_dim = int(c.num_heads * c.head_dim);
    const int kv_dim = int(c.num_kv_heads * c.head_dim);
    const int group = int(c.num_heads / c.num_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
    const int threads = 256;
    const int rows_elems_h = int(n) * h;
    const int rows_elems_i = int(n) * int(c.intermediate_size);
    const int bt = int(impl.pool.block_tokens());

    LYKURO_CUDA_CHECK(cudaSetDevice(device_id_), "cuda device not usable");
    Status cap = state.EnsureCapacity(pos0 + n - 1);
    if (!cap.ok()) return cap;
    LYKURO_CUDA_CHECK(
        cudaMemcpy(impl.d_tokens.ptr, tokens, n * sizeof(uint32_t),
                   cudaMemcpyHostToDevice),
        "token upload failed");

    if (impl.embed.bf16) {
        GatherEmbedRowsKernel<__nv_bfloat16><<<n, threads>>>(
            static_cast<const __nv_bfloat16*>(impl.embed.buf.ptr),
            static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
            impl.x_rows.f32());
    } else {
        GatherEmbedRowsKernel<float><<<n, threads>>>(
            static_cast<const float*>(impl.embed.buf.ptr),
            static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
            impl.x_rows.f32());
    }

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        Impl::Layer& layer = impl.layers[l];

        RmsNormRowsKernel<<<n, 256>>>(impl.x_rows.f32(),
                                      layer.input_norm.f32(),
                                      c.rms_norm_eps, h, impl.xn_rows.f32());
        LaunchGemm(impl.xn_rows.f32(), layer.q_w, layer.q_b.f32(), int(n),
                   h, q_dim, impl.q_rows.f32(), q_dim);
        LaunchGemm(impl.xn_rows.f32(), layer.k_w, layer.k_b.f32(), int(n),
                   h, kv_dim, impl.kv_k_rows.f32(), kv_dim);
        LaunchGemm(impl.xn_rows.f32(), layer.v_w, layer.v_b.f32(), int(n),
                   h, kv_dim, impl.kv_v_rows.f32(), kv_dim);

        {
            dim3 grid_q(c.num_heads, n);
            dim3 grid_kv(c.num_kv_heads, n);
            RopeRowsKernel<<<grid_q, 32>>>(impl.q_rows.f32(), q_dim,
                                           int(c.head_dim), pos0,
                                           c.rope_theta);
            RopeRowsKernel<<<grid_kv, 32>>>(impl.kv_k_rows.f32(), kv_dim,
                                            int(c.head_dim), pos0,
                                            c.rope_theta);
            ScatterChunkKvKernel<<<n, threads>>>(
                impl.kv_k_rows.f32(), impl.pool.k_layer(l),
                state.device_table(), bt, pos0, kv_dim);
            ScatterChunkKvKernel<<<n, threads>>>(
                impl.kv_v_rows.f32(), impl.pool.v_layer(l),
                state.device_table(), bt, pos0, kv_dim);
            dim3 grid_attn(c.num_heads, n);
            ChunkAttentionKernel<<<grid_attn, c.head_dim>>>(
                impl.q_rows.f32(), impl.pool.k_layer(l),
                impl.pool.v_layer(l), state.device_table(), bt,
                impl.attn_rows.f32(), pos0, int(c.head_dim), kv_dim, group,
                attn_scale);
        }

        LaunchGemm(impl.attn_rows.f32(), layer.o_w, nullptr, int(n), q_dim,
                   h, impl.proj_rows.f32(), h);
        AddKernel<<<(rows_elems_h + threads - 1) / threads, threads>>>(
            impl.x_rows.f32(), impl.proj_rows.f32(), rows_elems_h);

        RmsNormRowsKernel<<<n, 256>>>(impl.x_rows.f32(),
                                      layer.post_norm.f32(), c.rms_norm_eps,
                                      h, impl.xn_rows.f32());
        LaunchGemm(impl.xn_rows.f32(), layer.gate_w, nullptr, int(n), h,
                   int(c.intermediate_size), impl.gate_rows.f32(),
                   int(c.intermediate_size));
        LaunchGemm(impl.xn_rows.f32(), layer.up_w, nullptr, int(n), h,
                   int(c.intermediate_size), impl.up_rows.f32(),
                   int(c.intermediate_size));
        SwigluKernel<<<(rows_elems_i + threads - 1) / threads, threads>>>(
            impl.gate_rows.f32(), impl.up_rows.f32(), rows_elems_i);
        LaunchGemm(impl.gate_rows.f32(), layer.down_w, nullptr, int(n),
                   int(c.intermediate_size), h, impl.proj_rows.f32(), h);
        AddKernel<<<(rows_elems_h + threads - 1) / threads, threads>>>(
            impl.x_rows.f32(), impl.proj_rows.f32(), rows_elems_h);
    }

    if (want_logits_of_last) {
        const float* last_hidden = impl.x_rows.f32() + size_t(n - 1) * h;
        RmsNormKernel<<<1, 256>>>(last_hidden, impl.final_norm.f32(),
                                  c.rms_norm_eps, h, impl.normed.f32());
        const WeightBuffer& head = impl.head();
        LaunchMatVec(head, impl.normed.f32(), nullptr, h,
                     int(c.vocab_size), impl.logits.f32());
        logits_out.resize(c.vocab_size);
        LYKURO_CUDA_CHECK(
            cudaMemcpy(logits_out.data(), impl.logits.ptr,
                       c.vocab_size * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "logits download failed");
        for (float v : logits_out) {
            if (!std::isfinite(v)) {
                return Status(ErrorCode::kInferenceFailed,
                              "logits contain non-finite values",
                              kComponent);
            }
        }
    }
    LYKURO_CUDA_CHECK(cudaGetLastError(), "kernel launch failed");
    return Status::Ok();
}

Status QwenCudaModel::Prefill(SequenceState& state,
                              const std::vector<uint32_t>& tokens,
                              std::vector<float>& logits) {
    auto& seq = static_cast<CudaSequenceState&>(state);
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

    Impl& impl = *impl_;
    KvPool& pool = impl.pool;
    const uint32_t bt = pool.block_tokens();
    uint32_t start = 0;

    // Prefix cache (spec §16.3; default OFF per request, spec §16.2).
    // Chain hashes are seeded with the scope key, so cross-scope reuse is
    // structurally impossible (spec §16.1).
    std::vector<uint64_t> block_hashes;
    if (seq.options().allow_prefix_cache && impl.chunked_prefill) {
        uint64_t h = 1469598103934665603ULL;
        h = Fnv64(h, seq.options().scope_key.data(),
                  seq.options().scope_key.size());
        // The final token's block is never attached: the last position is
        // always recomputed so prefill can return logits.
        const uint32_t max_attach_blocks =
            (uint32_t(tokens.size()) - 1) / bt;
        const uint32_t full_blocks = uint32_t(tokens.size()) / bt;
        std::vector<int> attached;
        for (uint32_t b = 0; b < full_blocks; ++b) {
            h = Fnv64(h, tokens.data() + size_t(b) * bt,
                      bt * sizeof(uint32_t));
            block_hashes.push_back(h);
            if (b < max_attach_blocks && attached.size() == b) {
                int block = -1;
                if (pool.LookupCached(block_hashes[b], block)) {
                    attached.push_back(block);
                }
            }
        }
        if (!attached.empty()) {
            start = uint32_t(attached.size()) * bt;
            Status s = seq.AttachPrefix(attached, start);
            if (!s.ok()) return s;
            impl.prefix_hit_tokens += start;
        }
    }

    if (impl.chunked_prefill) {
        size_t done = start;
        while (done < tokens.size()) {
            const uint32_t n =
                uint32_t(std::min<size_t>(kChunk, tokens.size() - done));
            const bool last_chunk = done + n == tokens.size();
            Status s = ForwardChunk(tokens.data() + done, n,
                                    uint32_t(done), &seq, logits,
                                    last_chunk);
            if (!s.ok()) return s;
            for (uint32_t i = 0; i < n; ++i) seq.Advance();
            done += n;
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool last = i + 1 == tokens.size();
            Status s =
                ForwardToken(tokens[i], uint32_t(i), &seq, logits, last);
            if (!s.ok()) return s;
            seq.Advance();
        }
    }

    // Register this prompt's full blocks for future prefix reuse.
    if (seq.options().allow_prefix_cache && !block_hashes.empty()) {
        const uint32_t full_blocks = uint32_t(tokens.size()) / bt;
        for (uint32_t b = 0; b < full_blocks && b < seq.blocks().size();
             ++b) {
            pool.RegisterCached(block_hashes[b], seq.blocks()[b]);
        }
    }
    return Status::Ok();
}

Status QwenCudaModel::Decode(SequenceState& state, uint32_t token,
                             std::vector<float>& logits) {
    if (impl_->chunked_prefill) {
        // Single-sequence decode replays the bucket-1 CUDA graph.
        std::vector<DecodeBatchItem> items = {{&state, token, &logits}};
        std::vector<Status> per_item;
        Status s = DecodeBatch(items, per_item);
        if (!s.ok()) return s;
        return per_item[0];
    }
    auto& seq = static_cast<CudaSequenceState&>(state);
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

Status QwenCudaModel::DecodeBatch(std::vector<DecodeBatchItem>& items,
                                  std::vector<Status>& per_item) {
    per_item.assign(items.size(), Status::Ok());
    if (!impl_->chunked_prefill) {
        return GenerativeModel::DecodeBatch(items, per_item);
    }
    const QwenConfig& c = config_;
    Impl& impl = *impl_;
    const int h = int(c.hidden_size);
    const int q_dim = int(c.num_heads * c.head_dim);
    const int kv_dim = int(c.num_kv_heads * c.head_dim);
    const int group = int(c.num_heads / c.num_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
    const int threads = 256;
    cudaStream_t stream = impl.stream;

    LYKURO_CUDA_CHECK(cudaSetDevice(device_id_), "cuda device not usable");

    std::vector<size_t> valid;
    for (size_t i = 0; i < items.size(); ++i) {
        auto* state = static_cast<CudaSequenceState*>(items[i].state);
        if (items[i].token >= c.vocab_size) {
            per_item[i] = Status(ErrorCode::kInvalidRequest,
                                 "token id out of vocab range", kComponent);
        } else if (state->length() >= state->capacity()) {
            per_item[i] = Status(ErrorCode::kContextLengthExceeded,
                                 "kv cache capacity exhausted", kComponent);
        } else {
            Status cap = state->EnsureCapacity(state->length());
            if (!cap.ok()) {
                per_item[i] = cap;
            } else {
                valid.push_back(i);
            }
        }
    }

    for (size_t group_start = 0; group_start < valid.size();
         group_start += kMaxBatch) {
        const uint32_t b = uint32_t(std::min<size_t>(
            kMaxBatch, valid.size() - group_start));
        // Graph bucket: smallest of {1,2,4,8,16} that fits `b`. Grid
        // shapes are baked per bucket; padding rows write into the
        // reserved scratch block and their outputs are ignored.
        int bucket_idx = 0;
        uint32_t bucket = 1;
        while (bucket < b) {
            bucket <<= 1;
            ++bucket_idx;
        }

        uint32_t max_context = 0;
        for (uint32_t i = 0; i < b; ++i) {
            const size_t item_idx = valid[group_start + i];
            auto* state = static_cast<CudaSequenceState*>(
                items[item_idx].state);
            max_context = std::max(max_context, state->length() + 1);
        }
        // Split-count buckets keep short-context replays cheap while the
        // graph's grid shape stays fixed per key.
        int splits_idx;
        int splits;
        if (max_context <= 128) {
            splits_idx = 0;
            splits = 1;
        } else if (max_context <= 1024) {
            splits_idx = 1;
            splits = 8;
        } else {
            splits_idx = 2;
            splits = kMaxSplits;
        }

        uint32_t tokens[kMaxBatch];
        uint32_t positions[kMaxBatch];
        const int* seq_tables[kMaxBatch];
        for (uint32_t i = 0; i < bucket; ++i) {
            if (i < b) {
                const size_t item_idx = valid[group_start + i];
                auto* state = static_cast<CudaSequenceState*>(
                    items[item_idx].state);
                tokens[i] = items[item_idx].token;
                positions[i] = state->length();
                seq_tables[i] = state->device_table();
            } else {
                tokens[i] = 0;
                positions[i] = 0;
                seq_tables[i] = impl.pad_table.i32();
            }
        }
        LYKURO_CUDA_CHECK(
            cudaMemcpyAsync(impl.d_tokens.ptr, tokens,
                            bucket * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "token upload failed");
        LYKURO_CUDA_CHECK(
            cudaMemcpyAsync(impl.d_positions.ptr, positions,
                            bucket * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "position upload failed");
        LYKURO_CUDA_CHECK(
            cudaMemcpyAsync(impl.d_seq_tables.ptr, seq_tables,
                            bucket * sizeof(const int*),
                            cudaMemcpyHostToDevice, stream),
            "table pointer upload failed");

        auto* positions_dev =
            static_cast<const uint32_t*>(impl.d_positions.ptr);
        auto* tables_dev =
            static_cast<const int* const*>(impl.d_seq_tables.ptr);

        // The full decode pipeline for a fixed bucket size; every
        // per-token value is read from device memory, so this sequence
        // of launches is capturable and replayable.
        auto pipeline = [&](uint32_t nb) {
            if (nb <= 8) {
                switch (nb) {
                    case 1:
                        RunFusedDecode<1>(impl, c, positions_dev,
                                          tables_dev, splits, stream);
                        return;
                    case 2:
                        RunFusedDecode<2>(impl, c, positions_dev,
                                          tables_dev, splits, stream);
                        return;
                    case 4:
                        RunFusedDecode<4>(impl, c, positions_dev,
                                          tables_dev, splits, stream);
                        return;
                    default:
                        RunFusedDecode<8>(impl, c, positions_dev,
                                          tables_dev, splits, stream);
                        return;
                }
            }
            auto project = [&](const float* x, const WeightBuffer& w,
                               const float* bias, int in_dim, int out_dim,
                               float* y, int ldy) {
                if (nb > 8) {
                    LaunchGemm(x, w, bias, int(nb), in_dim, out_dim, y,
                               ldy, stream);
                } else {
                    LaunchMatVecBatch(x, w, bias, int(nb), in_dim,
                                      out_dim, y, ldy, stream);
                }
            };
            if (impl.embed.bf16) {
                GatherEmbedRowsKernel<__nv_bfloat16>
                    <<<nb, threads, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(
                            impl.embed.buf.ptr),
                        static_cast<const uint32_t*>(impl.d_tokens.ptr),
                        h, impl.x_rows.f32());
            } else {
                GatherEmbedRowsKernel<float><<<nb, threads, 0, stream>>>(
                    static_cast<const float*>(impl.embed.buf.ptr),
                    static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
                    impl.x_rows.f32());
            }
            const int rows_h = int(nb) * h;
            const int rows_i = int(nb) * int(c.intermediate_size);
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                Impl::Layer& layer = impl.layers[l];
                RmsNormRowsKernel<<<nb, 256, 0, stream>>>(
                    impl.x_rows.f32(), layer.input_norm.f32(),
                    c.rms_norm_eps, h, impl.xn_rows.f32());
                project(impl.xn_rows.f32(), layer.q_w, layer.q_b.f32(), h,
                        q_dim, impl.q_rows.f32(), q_dim);
                project(impl.xn_rows.f32(), layer.k_w, layer.k_b.f32(), h,
                        kv_dim, impl.kv_k_rows.f32(), kv_dim);
                project(impl.xn_rows.f32(), layer.v_w, layer.v_b.f32(), h,
                        kv_dim, impl.kv_v_rows.f32(), kv_dim);
                {
                    dim3 grid_q(c.num_heads, nb);
                    dim3 grid_kv(c.num_kv_heads, nb);
                    RopeVarPosKernel<<<grid_q, 32, 0, stream>>>(
                        impl.q_rows.f32(), q_dim, int(c.head_dim),
                        positions_dev, c.rope_theta);
                    RopeVarPosKernel<<<grid_kv, 32, 0, stream>>>(
                        impl.kv_k_rows.f32(), kv_dim, int(c.head_dim),
                        positions_dev, c.rope_theta);
                    ScatterKvPagedKernel<<<nb, threads, 0, stream>>>(
                        impl.kv_k_rows.f32(), impl.pool.k_layer(l),
                        tables_dev, positions_dev,
                        int(impl.pool.block_tokens()), kv_dim);
                    ScatterKvPagedKernel<<<nb, threads, 0, stream>>>(
                        impl.kv_v_rows.f32(), impl.pool.v_layer(l),
                        tables_dev, positions_dev,
                        int(impl.pool.block_tokens()), kv_dim);
                    // Fixed split count keeps the grid shape constant
                    // across replays; empty splits contribute nothing.
                    dim3 grid_attn(c.num_heads, splits, nb);
                    BatchSplitAttentionKernel
                        <<<grid_attn, kAttnThreads, 0, stream>>>(
                            impl.q_rows.f32(), impl.pool.k_layer(l),
                            impl.pool.v_layer(l), tables_dev,
                            int(impl.pool.block_tokens()), positions_dev,
                            impl.batch_partials.f32(), int(c.head_dim),
                            kv_dim, group, attn_scale, splits);
                    dim3 grid_combine(c.num_heads, nb);
                    BatchCombineAttentionKernel
                        <<<grid_combine, c.head_dim, 0, stream>>>(
                            impl.batch_partials.f32(),
                            impl.attn_rows.f32(), int(c.head_dim),
                            splits);
                }
                project(impl.attn_rows.f32(), layer.o_w, nullptr, q_dim,
                        h, impl.proj_rows.f32(), h);
                AddKernel<<<(rows_h + threads - 1) / threads, threads, 0,
                            stream>>>(impl.x_rows.f32(),
                                      impl.proj_rows.f32(), rows_h);
                RmsNormRowsKernel<<<nb, 256, 0, stream>>>(
                    impl.x_rows.f32(), layer.post_norm.f32(),
                    c.rms_norm_eps, h, impl.xn_rows.f32());
                project(impl.xn_rows.f32(), layer.gate_w, nullptr, h,
                        int(c.intermediate_size), impl.gate_rows.f32(),
                        int(c.intermediate_size));
                project(impl.xn_rows.f32(), layer.up_w, nullptr, h,
                        int(c.intermediate_size), impl.up_rows.f32(),
                        int(c.intermediate_size));
                SwigluKernel<<<(rows_i + threads - 1) / threads, threads,
                               0, stream>>>(impl.gate_rows.f32(),
                                            impl.up_rows.f32(), rows_i);
                project(impl.gate_rows.f32(), layer.down_w, nullptr,
                        int(c.intermediate_size), h, impl.proj_rows.f32(),
                        h);
                AddKernel<<<(rows_h + threads - 1) / threads, threads, 0,
                            stream>>>(impl.x_rows.f32(),
                                      impl.proj_rows.f32(), rows_h);
            }
            RmsNormRowsKernel<<<nb, 256, 0, stream>>>(
                impl.x_rows.f32(), impl.final_norm.f32(), c.rms_norm_eps,
                h, impl.xn_rows.f32());
            const WeightBuffer& head = impl.head();
            project(impl.xn_rows.f32(), head, nullptr, h,
                    int(c.vocab_size), impl.batch_logits.f32(),
                    int(c.vocab_size));
        };

        // Capture once per bucket, then replay (or launch directly for
        // kernel-level profiling).
        const bool no_graph = std::getenv("LYKURO_CUDA_NO_GRAPH") != nullptr;
        if (!no_graph &&
            impl.decode_graphs[bucket_idx][splits_idx] == nullptr) {
            if (cudaStreamBeginCapture(stream,
                                       cudaStreamCaptureModeGlobal) !=
                cudaSuccess) {
                return Status(ErrorCode::kGpuUnhealthy,
                              "graph capture failed", kComponent);
            }
            pipeline(bucket);
            cudaGraph_t graph = nullptr;
            if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
                cudaGraphInstantiate(
                    &impl.decode_graphs[bucket_idx][splits_idx], graph,
                    nullptr, nullptr, 0) != cudaSuccess) {
                if (graph != nullptr) cudaGraphDestroy(graph);
                return Status(ErrorCode::kGpuUnhealthy,
                              "graph instantiation failed", kComponent);
            }
            cudaGraphDestroy(graph);
        }
        const bool prof = std::getenv("LYKURO_CUDA_PROF") != nullptr;
        static cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        if (prof && ev0 == nullptr) {
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
        }
        if (prof) cudaEventRecord(ev0, stream);
        if (no_graph) {
            pipeline(bucket);
        } else {
            LYKURO_CUDA_CHECK(
                cudaGraphLaunch(impl.decode_graphs[bucket_idx][splits_idx],
                                stream),
                "graph launch failed");
        }
        if (prof) cudaEventRecord(ev1, stream);

        std::vector<float> host_logits(size_t(b) * c.vocab_size);
        LYKURO_CUDA_CHECK(
            cudaMemcpyAsync(host_logits.data(), impl.batch_logits.ptr,
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "logits download failed");
        LYKURO_CUDA_CHECK(cudaStreamSynchronize(stream),
                          "device execution failed");
        LYKURO_CUDA_CHECK(cudaGetLastError(), "kernel launch failed");
        if (prof) {
            static double gpu_ms = 0;
            static int n = 0;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            gpu_ms += ms;
            if (++n % 128 == 0) {
                std::fprintf(stderr, "[prof] cuda graph %.2fms/tok\n",
                             gpu_ms / n);
            }
        }

        for (uint32_t i = 0; i < b; ++i) {
            const size_t item_idx = valid[group_start + i];
            auto* state =
                static_cast<CudaSequenceState*>(items[item_idx].state);
            std::vector<float>& out = *items[item_idx].logits;
            out.assign(host_logits.begin() + size_t(i) * c.vocab_size,
                       host_logits.begin() + size_t(i + 1) * c.vocab_size);
            bool finite = true;
            for (float v : out) {
                if (!std::isfinite(v)) {
                    finite = false;
                    break;
                }
            }
            if (!finite) {
                per_item[item_idx] =
                    Status(ErrorCode::kInferenceFailed,
                           "logits contain non-finite values", kComponent);
            } else {
                state->Advance();
            }
        }
    }
    return Status::Ok();
}

}  // namespace lykuro::nie
