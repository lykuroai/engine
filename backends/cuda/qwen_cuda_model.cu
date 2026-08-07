#include "backends/cuda/qwen_cuda_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

__device__ inline const float* KvRow(const float* pool,
                                     const int* __restrict__ table,
                                     int block_tokens, int t,
                                     int kv_stride) {
    return pool + (size_t(table[t / block_tokens]) * block_tokens +
                   (t % block_tokens)) *
                      kv_stride;
}

template <typename WT>
__global__ void MatVecKernel(const WT* __restrict__ w,
                             const float* __restrict__ x,
                             const float* __restrict__ bias, int in_dim,
                             float* __restrict__ y) {
    __shared__ float partial[128];
    const size_t row = blockIdx.x;
    const WT* wr = w + row * size_t(in_dim);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += LoadWeight(wr, i) * x[i];
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

template <typename WT, int B>
__global__ void MatVecBatchKernel(const WT* __restrict__ w,
                                  const float* __restrict__ x,
                                  const float* __restrict__ bias,
                                  int in_dim, float* __restrict__ y,
                                  int ldy) {
    __shared__ float wsh[kBgemvChunk];
    __shared__ float red[128];
    const size_t row = blockIdx.x;
    const WT* wr = w + row * size_t(in_dim);

    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    for (int k0 = 0; k0 < in_dim; k0 += kBgemvChunk) {
        const int len = min(kBgemvChunk, in_dim - k0);
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            wsh[i] = LoadWeight(wr, k0 + i);
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

// ---- batched prefill kernels ----

constexpr int kChunk = 128;
constexpr int kGemmTile = 16;

template <typename WT>
__global__ void GemmXWtKernel(const float* __restrict__ x,
                              const WT* __restrict__ w,
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
                ? LoadWeight(w, size_t(wrow) * in_dim + kx)
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
    bool bf16 = false;

    Status Upload(const SafetensorsFile& file, const std::string& name) {
        const TensorInfo* info = file.FindTensor(name);
        const uint8_t* data = file.TensorData(name);
        if (info == nullptr || data == nullptr) {
            return Status(ErrorCode::kArtifactVerificationFailed,
                          "expected weight tensor missing", kComponent);
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

void LaunchMatVec(const WeightBuffer& w, const float* x, const float* bias,
                  int in_dim, int out_dim, float* y) {
    if (w.bf16) {
        MatVecKernel<__nv_bfloat16><<<out_dim, 128>>>(
            static_cast<const __nv_bfloat16*>(w.buf.ptr), x, bias, in_dim,
            y);
    } else {
        MatVecKernel<float><<<out_dim, 128>>>(
            static_cast<const float*>(w.buf.ptr), x, bias, in_dim, y);
    }
}

template <typename WT>
void LaunchMatVecBatchTyped(const WT* w, const float* x, const float* bias,
                            int batch, int in_dim, int out_dim, float* y,
                            int ldy) {
    if (batch <= 2) {
        MatVecBatchKernel<WT, 2><<<out_dim, 128>>>(w, x, bias, in_dim, y,
                                                   ldy);
    } else if (batch <= 4) {
        MatVecBatchKernel<WT, 4><<<out_dim, 128>>>(w, x, bias, in_dim, y,
                                                   ldy);
    } else if (batch <= 8) {
        MatVecBatchKernel<WT, 8><<<out_dim, 128>>>(w, x, bias, in_dim, y,
                                                   ldy);
    } else {
        MatVecBatchKernel<WT, 16><<<out_dim, 128>>>(w, x, bias, in_dim, y,
                                                    ldy);
    }
}

void LaunchMatVecBatch(const float* x, const WeightBuffer& w,
                       const float* bias, int batch, int in_dim,
                       int out_dim, float* y, int ldy) {
    if (w.bf16) {
        LaunchMatVecBatchTyped(
            static_cast<const __nv_bfloat16*>(w.buf.ptr), x, bias, batch,
            in_dim, out_dim, y, ldy);
    } else {
        LaunchMatVecBatchTyped(static_cast<const float*>(w.buf.ptr), x,
                               bias, batch, in_dim, out_dim, y, ldy);
    }
}

void LaunchGemm(const float* x, const WeightBuffer& w, const float* bias,
                int n, int in_dim, int out_dim, float* c, int ldc) {
    dim3 grid((out_dim + kGemmTile - 1) / kGemmTile,
              (n + kGemmTile - 1) / kGemmTile);
    dim3 block(kGemmTile, kGemmTile);
    if (w.bf16) {
        GemmXWtKernel<__nv_bfloat16><<<grid, block>>>(
            x, static_cast<const __nv_bfloat16*>(w.buf.ptr), bias, n,
            in_dim, out_dim, c, ldc);
    } else {
        GemmXWtKernel<float><<<grid, block>>>(
            x, static_cast<const float*>(w.buf.ptr), bias, n, in_dim,
            out_dim, c, ldc);
    }
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
    std::vector<Layer> layers;
    DeviceBuffer final_norm;
    WeightBuffer lm_head;
    bool tied = true;

    KvPool pool;
    uint64_t prefix_hit_tokens = 0;

    DeviceBuffer hidden, normed, q, attn_out, proj, gate, up, logits;
    DeviceBuffer attn_partials;

    DeviceBuffer x_rows, xn_rows, q_rows, attn_rows, proj_rows, gate_rows,
        up_rows;
    DeviceBuffer d_tokens;
    bool chunked_prefill = false;

    DeviceBuffer d_positions;
    DeviceBuffer d_kv_ptrs;     // [layers, 2, kMaxBatch] float* rows
    DeviceBuffer d_seq_tables;  // [kMaxBatch] const int*
    DeviceBuffer kv_k_rows, kv_v_rows;
    DeviceBuffer batch_logits;
    DeviceBuffer batch_partials;
};

namespace {

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
        if (result.status.ok()) result.status = dst.Upload(weights, name);
    };
    auto up_f = [&](const std::string& name, DeviceBuffer& dst) {
        if (result.status.ok()) {
            result.status = UploadF32(weights, name, dst);
        }
    };

    up_w("model.embed_tokens.weight", impl.embed);
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
        up_w("lm_head.weight", impl.lm_head);
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
            s = impl.d_kv_ptrs.AllocBytes(size_t(c.num_layers) * 2 *
                                          kMaxBatch * sizeof(float*));
        }
        if (s.ok()) {
            s = impl.d_seq_tables.AllocBytes(kMaxBatch * sizeof(int*));
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
    return impl_->pool.num_blocks();
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
        const WeightBuffer& head = impl.tied ? impl.embed : impl.lm_head;
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
        const WeightBuffer& head = impl.tied ? impl.embed : impl.lm_head;
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
    if (items.size() <= 1 || !impl_->chunked_prefill) {
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

        uint32_t tokens[kMaxBatch];
        uint32_t positions[kMaxBatch];
        const int* seq_tables[kMaxBatch];
        uint32_t max_context = 0;
        std::vector<float*> kv_ptrs(size_t(c.num_layers) * 2 * b);
        for (uint32_t i = 0; i < b; ++i) {
            const size_t item_idx = valid[group_start + i];
            auto* state =
                static_cast<CudaSequenceState*>(items[item_idx].state);
            tokens[i] = items[item_idx].token;
            positions[i] = state->length();
            seq_tables[i] = state->device_table();
            max_context = std::max(max_context, positions[i] + 1);
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                kv_ptrs[(size_t(l) * 2 + 0) * b + i] =
                    state->KeyRowHost(impl.pool, l, positions[i], kv_dim);
                kv_ptrs[(size_t(l) * 2 + 1) * b + i] =
                    state->ValueRowHost(impl.pool, l, positions[i], kv_dim);
            }
        }
        LYKURO_CUDA_CHECK(
            cudaMemcpy(impl.d_tokens.ptr, tokens, b * sizeof(uint32_t),
                       cudaMemcpyHostToDevice),
            "token upload failed");
        LYKURO_CUDA_CHECK(
            cudaMemcpy(impl.d_positions.ptr, positions,
                       b * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "position upload failed");
        LYKURO_CUDA_CHECK(
            cudaMemcpy(impl.d_kv_ptrs.ptr, kv_ptrs.data(),
                       kv_ptrs.size() * sizeof(float*),
                       cudaMemcpyHostToDevice),
            "kv pointer upload failed");
        LYKURO_CUDA_CHECK(
            cudaMemcpy(impl.d_seq_tables.ptr, seq_tables,
                       b * sizeof(const int*), cudaMemcpyHostToDevice),
            "table pointer upload failed");

        // Projection dispatch: shared-staged batched GEMV amortizes
        // weight reads for small batches; the tiled GEMM wins once the
        // batch approaches its 16-wide tile.
        auto project = [&](const float* x, const WeightBuffer& w,
                           const float* bias, int in_dim, int out_dim,
                           float* y, int ldy) {
            if (b > 8) {
                LaunchGemm(x, w, bias, int(b), in_dim, out_dim, y, ldy);
            } else {
                LaunchMatVecBatch(x, w, bias, int(b), in_dim, out_dim, y,
                                  ldy);
            }
        };

        int splits = (int(max_context) + 127) / 128;
        if (splits > kMaxSplits) splits = kMaxSplits;
        auto* positions_dev =
            static_cast<const uint32_t*>(impl.d_positions.ptr);
        auto* ptr_table = static_cast<float* const*>(impl.d_kv_ptrs.ptr);
        auto* tables_dev =
            static_cast<const int* const*>(impl.d_seq_tables.ptr);

        if (impl.embed.bf16) {
            GatherEmbedRowsKernel<__nv_bfloat16><<<b, threads>>>(
                static_cast<const __nv_bfloat16*>(impl.embed.buf.ptr),
                static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
                impl.x_rows.f32());
        } else {
            GatherEmbedRowsKernel<float><<<b, threads>>>(
                static_cast<const float*>(impl.embed.buf.ptr),
                static_cast<const uint32_t*>(impl.d_tokens.ptr), h,
                impl.x_rows.f32());
        }

        const int rows_h = int(b) * h;
        const int rows_i = int(b) * int(c.intermediate_size);
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            Impl::Layer& layer = impl.layers[l];
            float* const* k_rows = ptr_table + (size_t(l) * 2 + 0) * b;
            float* const* v_rows = ptr_table + (size_t(l) * 2 + 1) * b;

            RmsNormRowsKernel<<<b, 256>>>(
                impl.x_rows.f32(), layer.input_norm.f32(), c.rms_norm_eps,
                h, impl.xn_rows.f32());
            project(impl.xn_rows.f32(), layer.q_w, layer.q_b.f32(), h,
                    q_dim, impl.q_rows.f32(), q_dim);
            project(impl.xn_rows.f32(), layer.k_w, layer.k_b.f32(), h,
                    kv_dim, impl.kv_k_rows.f32(), kv_dim);
            project(impl.xn_rows.f32(), layer.v_w, layer.v_b.f32(), h,
                    kv_dim, impl.kv_v_rows.f32(), kv_dim);

            {
                dim3 grid_q(c.num_heads, b);
                dim3 grid_kv(c.num_kv_heads, b);
                RopeVarPosKernel<<<grid_q, 32>>>(impl.q_rows.f32(), q_dim,
                                                 int(c.head_dim),
                                                 positions_dev,
                                                 c.rope_theta);
                RopeVarPosKernel<<<grid_kv, 32>>>(impl.kv_k_rows.f32(),
                                                  kv_dim, int(c.head_dim),
                                                  positions_dev,
                                                  c.rope_theta);
                ScatterRowsKernel<<<b, threads>>>(impl.kv_k_rows.f32(),
                                                  k_rows, kv_dim);
                ScatterRowsKernel<<<b, threads>>>(impl.kv_v_rows.f32(),
                                                  v_rows, kv_dim);
                dim3 grid_attn(c.num_heads, splits, b);
                BatchSplitAttentionKernel<<<grid_attn, kAttnThreads>>>(
                    impl.q_rows.f32(), impl.pool.k_layer(l),
                    impl.pool.v_layer(l), tables_dev,
                    int(impl.pool.block_tokens()), positions_dev,
                    impl.batch_partials.f32(), int(c.head_dim), kv_dim,
                    group, attn_scale, splits);
                dim3 grid_combine(c.num_heads, b);
                BatchCombineAttentionKernel<<<grid_combine, c.head_dim>>>(
                    impl.batch_partials.f32(), impl.attn_rows.f32(),
                    int(c.head_dim), splits);
            }

            project(impl.attn_rows.f32(), layer.o_w, nullptr, q_dim, h,
                    impl.proj_rows.f32(), h);
            AddKernel<<<(rows_h + threads - 1) / threads, threads>>>(
                impl.x_rows.f32(), impl.proj_rows.f32(), rows_h);

            RmsNormRowsKernel<<<b, 256>>>(
                impl.x_rows.f32(), layer.post_norm.f32(), c.rms_norm_eps,
                h, impl.xn_rows.f32());
            project(impl.xn_rows.f32(), layer.gate_w, nullptr, h,
                    int(c.intermediate_size), impl.gate_rows.f32(),
                    int(c.intermediate_size));
            project(impl.xn_rows.f32(), layer.up_w, nullptr, h,
                    int(c.intermediate_size), impl.up_rows.f32(),
                    int(c.intermediate_size));
            SwigluKernel<<<(rows_i + threads - 1) / threads, threads>>>(
                impl.gate_rows.f32(), impl.up_rows.f32(), rows_i);
            project(impl.gate_rows.f32(), layer.down_w, nullptr,
                    int(c.intermediate_size), h, impl.proj_rows.f32(), h);
            AddKernel<<<(rows_h + threads - 1) / threads, threads>>>(
                impl.x_rows.f32(), impl.proj_rows.f32(), rows_h);
        }

        RmsNormRowsKernel<<<b, 256>>>(impl.x_rows.f32(),
                                      impl.final_norm.f32(),
                                      c.rms_norm_eps, h,
                                      impl.xn_rows.f32());
        const WeightBuffer& head = impl.tied ? impl.embed : impl.lm_head;
        project(impl.xn_rows.f32(), head, nullptr, h, int(c.vocab_size),
                impl.batch_logits.f32(), int(c.vocab_size));

        std::vector<float> host_logits(size_t(b) * c.vocab_size);
        LYKURO_CUDA_CHECK(
            cudaMemcpy(host_logits.data(), impl.batch_logits.ptr,
                       host_logits.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "logits download failed");
        LYKURO_CUDA_CHECK(cudaGetLastError(), "kernel launch failed");

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
