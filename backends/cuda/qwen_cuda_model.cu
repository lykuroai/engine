#include "backends/cuda/qwen_cuda_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>

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
// Phase 4 design: weights stay in their checkpoint dtype on device
// (BF16 checkpoints are never widened, halving decode bandwidth) while
// every accumulation runs in FP32 with a fixed reduction order, so
// results are deterministic and semantically identical to the CPU
// reference, which also computes fp32 over the same bf16 weight values.

__device__ inline float LoadWeight(const float* w, size_t i) { return w[i]; }
__device__ inline float LoadWeight(const __nv_bfloat16* w, size_t i) {
    return __bfloat162float(w[i]);
}

// y = W x + bias. W row-major [out_dim, in_dim]; one block per output row,
// strided fp32 accumulation with a tree reduction (deterministic).
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

// hidden = embed[token], converted to fp32.
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

// ---- split-K decode attention (long-context path) ----
//
// The context is divided into up to kMaxSplits contiguous segments; each
// (head, split) block computes a partial softmax over its segment
// (segment max, partial denominator, partial weighted value sum), and a
// combine kernel merges partials in fixed split order. All reductions use
// fixed-order loops, so results are deterministic run-to-run.

constexpr int kMaxSplits = 32;
constexpr int kMaxSegment = 1024;  // ceil(32768 / 32); shared score cap
constexpr int kAttnThreads = 128;

// Partials layout per (head, split): [head_dim floats acc][m][denom].
__global__ void SplitAttentionKernel(const float* __restrict__ q,
                                     const float* __restrict__ k_cache,
                                     const float* __restrict__ v_cache,
                                     float* __restrict__ partials,
                                     int context, int head_dim,
                                     int kv_stride, int group, float scale,
                                     int splits) {
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
            part[head_dim] = -1e30f;  // m
            part[head_dim + 1] = 0.0f;  // denom
        }
        return;
    }

    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float* kt =
            k_cache + size_t(t0 + j) * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kt[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Segment max (per-thread subset max, then fixed-order tree reduce).
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x],
                                     red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    // exp + partial denominator.
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

    // Partial weighted value sum, parallel over head_dim lanes.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < len; ++j) {
            const float* vt =
                v_cache + size_t(t0 + j) * kv_stride + kv_head * head_dim;
            acc += scores[j] * vt[d];
        }
        part[d] = acc;
    }
    if (threadIdx.x == 0) {
        part[head_dim] = m;
        part[head_dim + 1] = denom;
    }
}

// Merges split partials in fixed order. grid.x = head.
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

// ---- batched prefill kernels (Phase 4 GEMM path) ----

constexpr int kChunk = 128;      // prompt tokens per chunk
constexpr int kGemmTile = 16;    // tiled GEMM block edge

// C[token, orow] = sum_k X[token, k] * W[orow, k] (+ bias[orow]).
// X row-major [n, in_dim]; W row-major [out_dim, in_dim]; C row `token`
// starts at C + token*ldc. Shared-memory tiling reuses each W tile across
// the token dimension; fixed k-order keeps results deterministic.
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

// Gathers embeddings for a chunk of tokens into fp32 rows.
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

// Row-wise RMSNorm over a chunk. grid.x = row.
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

// RoPE over a chunk: grid (head, row). `row_stride` is the float stride
// between consecutive rows (q buffer: heads*head_dim; K cache: kv_stride).
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

// Causal attention for a chunk of query positions using an online
// (streaming) softmax: no score scratch, deterministic sequential-t
// recurrence, parallel over (head, query) blocks and head_dim lanes.
// blockDim.x == head_dim (power of two, <=128, validated at load).
__global__ void ChunkAttentionKernel(const float* __restrict__ q_buf,
                                     const float* __restrict__ k_cache,
                                     const float* __restrict__ v_cache,
                                     float* __restrict__ out, uint32_t pos0,
                                     int head_dim, int kv_stride, int group,
                                     float scale) {
    const int head = blockIdx.x;
    const int i = blockIdx.y;             // query index within the chunk
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
            k_cache + size_t(t) * kv_stride + kv_head * head_dim;
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
            v_cache + size_t(t) * kv_stride + kv_head * head_dim;
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
};

// Weight tensor kept in checkpoint dtype (bf16 or fp32).
struct WeightBuffer {
    DeviceBuffer buf;
    bool bf16 = false;

    // Uploads the tensor without widening BF16. F16 is widened to fp32
    // (fp16 checkpoints are rare in the certified set; correctness over
    // bandwidth there).
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

// Small vectors (norm weights, biases) are always fp32 on device.
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

}  // namespace

struct QwenCudaModel::Impl {
    struct Layer {
        DeviceBuffer input_norm, q_b, k_b, v_b, post_norm;
        WeightBuffer q_w, k_w, v_w, o_w, gate_w, up_w, down_w;
    };

    WeightBuffer embed;
    std::vector<Layer> layers;
    DeviceBuffer final_norm;
    WeightBuffer lm_head;  // unset when tied (embed is used)
    bool tied = true;

    DeviceBuffer hidden, normed, q, attn_out, proj, gate, up, logits;
    DeviceBuffer attn_partials;  // [heads, kMaxSplits, head_dim + 2]

    // Chunked-prefill work buffers ([kChunk, dim] rows) and token staging.
    DeviceBuffer x_rows, xn_rows, q_rows, attn_rows, proj_rows, gate_rows,
        up_rows;
    DeviceBuffer d_tokens;  // kChunk uint32
    bool chunked_prefill = false;  // head_dim power-of-two <= 128
};

namespace {

class CudaSequenceState final : public SequenceState {
public:
    static Status Create(const QwenConfig& config, uint32_t max_tokens,
                         std::unique_ptr<CudaSequenceState>& out) {
        auto state = std::unique_ptr<CudaSequenceState>(
            new CudaSequenceState());
        state->kv_stride_ = config.num_kv_heads * config.head_dim;
        state->max_tokens_ = max_tokens;
        state->keys_.resize(config.num_layers);
        state->values_.resize(config.num_layers);
        const size_t bytes =
            size_t(max_tokens) * state->kv_stride_ * sizeof(float);
        for (uint32_t l = 0; l < config.num_layers; ++l) {
            Status s = state->keys_[l].AllocBytes(bytes);
            if (!s.ok()) return s;
            s = state->values_[l].AllocBytes(bytes);
            if (!s.ok()) return s;
        }
        out = std::move(state);
        return Status::Ok();
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }

    float* KeyAt(uint32_t layer, uint32_t token_index) {
        return keys_[layer].f32() + size_t(token_index) * kv_stride_;
    }
    float* ValueAt(uint32_t layer, uint32_t token_index) {
        return values_[layer].f32() + size_t(token_index) * kv_stride_;
    }
    float* KeyBase(uint32_t layer) { return keys_[layer].f32(); }
    float* ValueBase(uint32_t layer) { return values_[layer].f32(); }
    void Advance() { ++length_; }

private:
    CudaSequenceState() = default;
    uint32_t kv_stride_ = 0;
    uint32_t max_tokens_ = 0;
    uint32_t length_ = 0;
    std::vector<DeviceBuffer> keys_;
    std::vector<DeviceBuffer> values_;
};

}  // namespace

QwenCudaModel::~QwenCudaModel() = default;

QwenCudaModel::LoadResult QwenCudaModel::Load(const ModelManifest& manifest,
                                              const SafetensorsFile& weights,
                                              int device_id) {
    LoadResult result;
    auto model = std::unique_ptr<QwenCudaModel>(new QwenCudaModel());

    result.status = QwenConfig::FromManifest(manifest, model->config_);
    if (!result.status.ok()) return result;
    const QwenConfig& c = model->config_;
    model->limits_.vocab_size = c.vocab_size;
    model->limits_.max_context_tokens = c.max_context_tokens;
    model->limits_.eos_token_ids = c.eos_token_ids;
    model->device_id_ = device_id;

    if (cudaSetDevice(device_id) != cudaSuccess) {
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

    // Chunked prefill needs blockDim == head_dim in the attention kernel.
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
    }
    if (!s.ok()) {
        result.status = s;
        return result;
    }

    result.model = std::move(model);
    return result;
}

Status QwenCudaModel::CreateSequence(uint32_t max_tokens,
                                     std::unique_ptr<SequenceState>& out) {
    if (cudaSetDevice(device_id_) != cudaSuccess) {
        return Status(ErrorCode::kGpuUnhealthy, "cuda device not usable",
                      kComponent);
    }
    std::unique_ptr<CudaSequenceState> state;
    Status s = CudaSequenceState::Create(config_, max_tokens, state);
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
        float* k_dst = state.KeyAt(l, pos);
        float* v_dst = state.ValueAt(l, pos);

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
                impl.q.f32(), state.KeyBase(l), state.ValueBase(l),
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
        // cudaMemcpy above already synchronized the stream.
        LYKURO_CUDA_CHECK(cudaGetLastError(), "kernel launch failed");
        return Status::Ok();
    }
    // Prefill fast path: no host sync per token; launches queue on the
    // stream and errors surface at the next logits download.
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
    const int kv_stride = kv_dim;
    const int group = int(c.num_heads / c.num_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
    const int threads = 256;
    const int rows_elems_h = int(n) * h;
    const int rows_elems_i = int(n) * int(c.intermediate_size);

    LYKURO_CUDA_CHECK(cudaSetDevice(device_id_), "cuda device not usable");
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
        float* k0 = state.KeyAt(l, pos0);
        float* v0 = state.ValueAt(l, pos0);

        RmsNormRowsKernel<<<n, 256>>>(impl.x_rows.f32(),
                                      layer.input_norm.f32(),
                                      c.rms_norm_eps, h, impl.xn_rows.f32());
        LaunchGemm(impl.xn_rows.f32(), layer.q_w, layer.q_b.f32(), int(n),
                   h, q_dim, impl.q_rows.f32(), q_dim);
        LaunchGemm(impl.xn_rows.f32(), layer.k_w, layer.k_b.f32(), int(n),
                   h, kv_dim, k0, kv_stride);
        LaunchGemm(impl.xn_rows.f32(), layer.v_w, layer.v_b.f32(), int(n),
                   h, kv_dim, v0, kv_stride);

        {
            dim3 grid_q(c.num_heads, n);
            dim3 grid_kv(c.num_kv_heads, n);
            RopeRowsKernel<<<grid_q, 32>>>(impl.q_rows.f32(), q_dim,
                                           int(c.head_dim), pos0,
                                           c.rope_theta);
            RopeRowsKernel<<<grid_kv, 32>>>(k0, kv_stride, int(c.head_dim),
                                            pos0, c.rope_theta);
            dim3 grid_attn(c.num_heads, n);
            ChunkAttentionKernel<<<grid_attn, c.head_dim>>>(
                impl.q_rows.f32(), state.KeyBase(l), state.ValueBase(l),
                impl.attn_rows.f32(), pos0, int(c.head_dim), kv_stride,
                group, attn_scale);
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
        const float* last_hidden =
            impl.x_rows.f32() + size_t(n - 1) * h;
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
    if (impl_->chunked_prefill) {
        size_t done = 0;
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
        return Status::Ok();
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool last = i + 1 == tokens.size();
        Status s = ForwardToken(tokens[i], uint32_t(i), &seq, logits, last);
        if (!s.ok()) return s;
        seq.Advance();
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

}  // namespace lykuro::nie
