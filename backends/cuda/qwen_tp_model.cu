#include "backends/cuda/qwen_tp_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_tp";
constexpr int kNumShards = 2;

Status CudaError(cudaError_t err, const char* what) {
    if (err == cudaErrorMemoryAllocation) {
        return Status(ErrorCode::kGpuOom, what, kComponent);
    }
    return Status(ErrorCode::kGpuUnhealthy, what, kComponent);
}

#define TP_CUDA_CHECK(expr, what)                       \
    do {                                                \
        cudaError_t tp_err_ = (expr);                   \
        if (tp_err_ != cudaSuccess) {                   \
            return CudaError(tp_err_, what);            \
        }                                               \
    } while (0)

// ---- minimal per-token kernels (correctness PoC; FP32) ----

__global__ void TpMatVecKernel(const float* __restrict__ w,
                               const float* __restrict__ x,
                               const float* __restrict__ bias, int in_dim,
                               float* __restrict__ y) {
    __shared__ float partial[128];
    const size_t row = blockIdx.x;
    const float* wr = w + row * size_t(in_dim);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += wr[i] * x[i];
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

__global__ void TpRmsNormKernel(const float* x, const float* weight,
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

__global__ void TpRopeKernel(float* vec, int head_dim, uint32_t pos,
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

// Simple causal attention over the local heads (serial softmax by
// thread 0 — correctness PoC).
__global__ void TpAttentionKernel(const float* __restrict__ q,
                                  const float* __restrict__ k_cache,
                                  const float* __restrict__ v_cache,
                                  float* __restrict__ scores,
                                  float* __restrict__ out, int context,
                                  int head_dim, int kv_stride, int group,
                                  int max_context, float scale) {
    const int head = blockIdx.x;
    const int kv_head = head / group;
    const float* qh = q + head * head_dim;
    float* row = scores + size_t(head) * max_context;

    for (int t = threadIdx.x; t < context; t += blockDim.x) {
        const float* kt =
            k_cache + size_t(t) * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kt[d];
        }
        row[t] = dot * scale;
    }
    __syncthreads();

    __shared__ float denom_s;
    if (threadIdx.x == 0) {
        float max_score = row[0];
        for (int t = 1; t < context; ++t) {
            max_score = fmaxf(max_score, row[t]);
        }
        float denom = 0.0f;
        for (int t = 0; t < context; ++t) {
            row[t] = expf(row[t] - max_score);
            denom += row[t];
        }
        denom_s = denom;
    }
    __syncthreads();

    float* oh = out + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < context; ++t) {
            const float* vt =
                v_cache + size_t(t) * kv_stride + kv_head * head_dim;
            acc += (row[t] / denom_s) * vt[d];
        }
        oh[d] = acc;
    }
}

__global__ void TpSwigluKernel(float* gate, const float* up, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));
        gate[i] = silu * up[i];
    }
}

__global__ void TpAddKernel(float* y, const float* x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) y[i] += x[i];
}

// ---- device buffers ----

struct TpBuffer {
    void* ptr = nullptr;
    int device = -1;
    ~TpBuffer() {
        if (ptr != nullptr) {
            cudaSetDevice(device);
            cudaFree(ptr);
        }
    }
    Status Alloc(int dev, size_t bytes) {
        device = dev;
        TP_CUDA_CHECK(cudaSetDevice(dev), "cuda device not usable");
        TP_CUDA_CHECK(cudaMalloc(&ptr, bytes), "device allocation failed");
        return Status::Ok();
    }
    Status Upload(int dev, const std::vector<float>& host) {
        Status s = Alloc(dev, host.size() * sizeof(float));
        if (!s.ok()) return s;
        TP_CUDA_CHECK(cudaMemcpy(ptr, host.data(),
                                 host.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "weight upload failed");
        return Status::Ok();
    }
    float* f32() const { return static_cast<float*>(ptr); }
};

// Reads a tensor as host FP32.
Status TensorToHost(const SafetensorsFile& file, const std::string& name,
                    std::vector<float>& out, std::vector<uint64_t>* shape) {
    const TensorInfo* info = file.FindTensor(name);
    const uint8_t* data = file.TensorData(name);
    if (info == nullptr || data == nullptr) {
        return Status(ErrorCode::kArtifactVerificationFailed,
                      "expected weight tensor missing", kComponent);
    }
    if (shape != nullptr) *shape = info->shape;
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

std::vector<float> SliceRows(const std::vector<float>& w, size_t in_dim,
                             size_t r0, size_t r1) {
    return std::vector<float>(w.begin() + r0 * in_dim,
                              w.begin() + r1 * in_dim);
}

std::vector<float> SliceCols(const std::vector<float>& w, size_t rows,
                             size_t in_dim, size_t c0, size_t c1) {
    std::vector<float> out((c1 - c0) * rows);
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(out.data() + r * (c1 - c0),
                    w.data() + r * in_dim + c0, (c1 - c0) * sizeof(float));
    }
    return out;
}

}  // namespace

struct QwenTpModel::Impl {
    struct Shard {
        int device = 0;
        cudaStream_t stream = nullptr;

        struct Layer {
            TpBuffer input_norm, post_norm;      // replicated
            TpBuffer q_w, q_b, k_w, k_b, v_w, v_b;  // local heads
            TpBuffer o_w;                        // [h, local_q_dim]
            TpBuffer gate_w, up_w;               // [local_inter, h]
            TpBuffer down_w;                     // [h, local_inter]
        };
        std::vector<Layer> layers;

        // Work buffers.
        TpBuffer hidden, normed, q, attn_out, partial, gate, up, scores;

        ~Shard() {
            if (stream != nullptr) {
                cudaSetDevice(device);
                cudaStreamDestroy(stream);
            }
        }
    };

    Shard shards[kNumShards];
    TpBuffer embed;       // device 0, [vocab, h]
    TpBuffer final_norm;  // device 0
    TpBuffer lm_head;     // device 0 (empty when tied)
    TpBuffer logits;      // device 0, [vocab]
    bool tied = true;

    // Local head geometry.
    uint32_t local_heads = 0;
    uint32_t local_kv_heads = 0;
    uint32_t local_inter = 0;

    // Host staging for the fixed-order all-reduce.
    std::vector<float> host_a, host_b, host_hidden;
};

namespace {

// Per-sequence KV cache: each shard owns its local heads only.
class TpSequenceState final : public SequenceState {
public:
    static Status Create(const QwenConfig& config, uint32_t local_kv_heads,
                         const int (&devices)[kNumShards],
                         uint32_t max_tokens,
                         std::unique_ptr<TpSequenceState>& out) {
        auto state = std::unique_ptr<TpSequenceState>(new TpSequenceState());
        state->kv_stride_ = local_kv_heads * config.head_dim;
        state->max_tokens_ = max_tokens;
        const size_t bytes =
            size_t(max_tokens) * state->kv_stride_ * sizeof(float);
        for (int s = 0; s < kNumShards; ++s) {
            state->keys_[s].resize(config.num_layers);
            state->values_[s].resize(config.num_layers);
            for (uint32_t l = 0; l < config.num_layers; ++l) {
                Status st = state->keys_[s][l].Alloc(devices[s], bytes);
                if (!st.ok()) return st;
                st = state->values_[s][l].Alloc(devices[s], bytes);
                if (!st.ok()) return st;
            }
        }
        out = std::move(state);
        return Status::Ok();
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }

    float* KeyAt(int shard, uint32_t layer, uint32_t t) {
        return keys_[shard][layer].f32() + size_t(t) * kv_stride_;
    }
    float* ValueAt(int shard, uint32_t layer, uint32_t t) {
        return values_[shard][layer].f32() + size_t(t) * kv_stride_;
    }
    float* KeyBase(int shard, uint32_t layer) {
        return keys_[shard][layer].f32();
    }
    float* ValueBase(int shard, uint32_t layer) {
        return values_[shard][layer].f32();
    }
    void Advance() { ++length_; }

private:
    TpSequenceState() = default;
    uint32_t kv_stride_ = 0;
    uint32_t max_tokens_ = 0;
    uint32_t length_ = 0;
    std::vector<TpBuffer> keys_[kNumShards];
    std::vector<TpBuffer> values_[kNumShards];
};

}  // namespace

QwenTpModel::~QwenTpModel() = default;

QwenTpModel::LoadResult QwenTpModel::Load(const ModelManifest& manifest,
                                          const SafetensorsFile& weights,
                                          const Options& options) {
    LoadResult result;
    auto model = std::unique_ptr<QwenTpModel>(new QwenTpModel());

    result.status = QwenConfig::FromManifest(manifest, model->config_);
    if (!result.status.ok()) return result;
    const QwenConfig& c = model->config_;
    model->limits_.vocab_size = c.vocab_size;
    model->limits_.max_context_tokens = c.max_context_tokens;
    model->limits_.eos_token_ids = c.eos_token_ids;

    if (options.device_ids.size() != kNumShards) {
        result.status = Status(ErrorCode::kInvalidRequest,
                               "tensor parallel PoC requires exactly two "
                               "devices",
                               kComponent);
        return result;
    }
    // Even sharding or refusal (spec §0.2: no silent guessing).
    if (c.num_heads % kNumShards != 0 ||
        c.num_kv_heads % kNumShards != 0 ||
        c.intermediate_size % kNumShards != 0) {
        result.status = Status(ErrorCode::kUnsupportedModel,
                               "model dimensions do not shard evenly",
                               kComponent);
        return result;
    }

    model->impl_ = std::make_unique<Impl>();
    Impl& impl = *model->impl_;
    impl.tied = c.tie_word_embeddings;
    impl.local_heads = c.num_heads / kNumShards;
    impl.local_kv_heads = c.num_kv_heads / kNumShards;
    impl.local_inter = c.intermediate_size / kNumShards;

    const size_t h = c.hidden_size;
    const size_t hd = c.head_dim;
    const size_t local_q = impl.local_heads * hd;
    const size_t local_kv = impl.local_kv_heads * hd;

    std::vector<float> host;
    std::vector<uint64_t> shape;

    for (int s = 0; s < kNumShards; ++s) {
        Impl::Shard& shard = impl.shards[s];
        shard.device = options.device_ids[s];
        if (cudaSetDevice(shard.device) != cudaSuccess ||
            cudaStreamCreate(&shard.stream) != cudaSuccess) {
            result.status = Status(ErrorCode::kGpuUnhealthy,
                                   "cuda device not usable", kComponent);
            return result;
        }
        shard.layers.resize(c.num_layers);
    }

    auto load = [&](const std::string& name) -> Status {
        return TensorToHost(weights, name, host, &shape);
    };

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard::Layer& layer = impl.shards[s].layers[l];
            const int dev = impl.shards[s].device;
            Status st = load(p + "input_layernorm.weight");
            if (st.ok()) st = layer.input_norm.Upload(dev, host);
            if (st.ok()) st = load(p + "post_attention_layernorm.weight");
            if (st.ok()) st = layer.post_norm.Upload(dev, host);

            if (st.ok()) st = load(p + "self_attn.q_proj.weight");
            if (st.ok()) {
                st = layer.q_w.Upload(
                    dev, SliceRows(host, h, s * local_q, (s + 1) * local_q));
            }
            if (st.ok()) st = load(p + "self_attn.q_proj.bias");
            if (st.ok()) {
                st = layer.q_b.Upload(
                    dev, std::vector<float>(
                             host.begin() + s * local_q,
                             host.begin() + (s + 1) * local_q));
            }
            if (st.ok()) st = load(p + "self_attn.k_proj.weight");
            if (st.ok()) {
                st = layer.k_w.Upload(
                    dev,
                    SliceRows(host, h, s * local_kv, (s + 1) * local_kv));
            }
            if (st.ok()) st = load(p + "self_attn.k_proj.bias");
            if (st.ok()) {
                st = layer.k_b.Upload(
                    dev, std::vector<float>(
                             host.begin() + s * local_kv,
                             host.begin() + (s + 1) * local_kv));
            }
            if (st.ok()) st = load(p + "self_attn.v_proj.weight");
            if (st.ok()) {
                st = layer.v_w.Upload(
                    dev,
                    SliceRows(host, h, s * local_kv, (s + 1) * local_kv));
            }
            if (st.ok()) st = load(p + "self_attn.v_proj.bias");
            if (st.ok()) {
                st = layer.v_b.Upload(
                    dev, std::vector<float>(
                             host.begin() + s * local_kv,
                             host.begin() + (s + 1) * local_kv));
            }
            if (st.ok()) st = load(p + "self_attn.o_proj.weight");
            if (st.ok()) {
                st = layer.o_w.Upload(
                    dev, SliceCols(host, h, size_t(c.num_heads) * hd,
                                   s * local_q, (s + 1) * local_q));
            }
            if (st.ok()) st = load(p + "mlp.gate_proj.weight");
            if (st.ok()) {
                st = layer.gate_w.Upload(
                    dev, SliceRows(host, h, s * impl.local_inter,
                                   (s + 1) * impl.local_inter));
            }
            if (st.ok()) st = load(p + "mlp.up_proj.weight");
            if (st.ok()) {
                st = layer.up_w.Upload(
                    dev, SliceRows(host, h, s * impl.local_inter,
                                   (s + 1) * impl.local_inter));
            }
            if (st.ok()) st = load(p + "mlp.down_proj.weight");
            if (st.ok()) {
                st = layer.down_w.Upload(
                    dev, SliceCols(host, h, c.intermediate_size,
                                   s * impl.local_inter,
                                   (s + 1) * impl.local_inter));
            }
            if (!st.ok()) {
                result.status = st;
                return result;
            }
        }
    }

    // Device-0 resident: embed, final norm, (lm_head).
    {
        const int dev0 = impl.shards[0].device;
        Status st = load("model.embed_tokens.weight");
        if (st.ok()) st = impl.embed.Upload(dev0, host);
        if (st.ok()) st = load("model.norm.weight");
        if (st.ok()) st = impl.final_norm.Upload(dev0, host);
        if (st.ok() && !c.tie_word_embeddings) {
            st = load("lm_head.weight");
            if (st.ok()) st = impl.lm_head.Upload(dev0, host);
        }
        if (st.ok()) st = impl.logits.Alloc(dev0, c.vocab_size * 4);
        if (!st.ok()) {
            result.status = st;
            return result;
        }
    }

    // Work buffers per shard.
    for (int s = 0; s < kNumShards; ++s) {
        Impl::Shard& shard = impl.shards[s];
        const int dev = shard.device;
        Status st = shard.hidden.Alloc(dev, h * 4);
        if (st.ok()) st = shard.normed.Alloc(dev, h * 4);
        if (st.ok()) st = shard.q.Alloc(dev, local_q * 4);
        if (st.ok()) st = shard.attn_out.Alloc(dev, local_q * 4);
        if (st.ok()) st = shard.partial.Alloc(dev, h * 4);
        if (st.ok()) st = shard.gate.Alloc(dev, impl.local_inter * 4);
        if (st.ok()) st = shard.up.Alloc(dev, impl.local_inter * 4);
        if (st.ok()) {
            st = shard.scores.Alloc(
                dev, size_t(impl.local_heads) * c.max_context_tokens * 4);
        }
        if (!st.ok()) {
            result.status = st;
            return result;
        }
    }
    impl.host_a.resize(h);
    impl.host_b.resize(h);
    impl.host_hidden.resize(h);

    result.model = std::move(model);
    return result;
}

Status QwenTpModel::CreateSequence(uint32_t max_tokens,
                                   std::unique_ptr<SequenceState>& out) {
    Impl& impl = *impl_;
    int devices[kNumShards] = {impl.shards[0].device,
                               impl.shards[1].device};
    std::unique_ptr<TpSequenceState> state;
    Status s = TpSequenceState::Create(config_, impl.local_kv_heads,
                                       devices, max_tokens, state);
    if (!s.ok()) return s;
    out = std::move(state);
    return Status::Ok();
}

Status QwenTpModel::ForwardToken(uint32_t token, uint32_t pos,
                                 void* sequence_state,
                                 std::vector<float>& logits_out,
                                 bool want_logits) {
    auto& state = *static_cast<TpSequenceState*>(sequence_state);
    const QwenConfig& c = config_;
    Impl& impl = *impl_;
    const int h = int(c.hidden_size);
    const int hd = int(c.head_dim);
    const int local_q = int(impl.local_heads) * hd;
    const int local_kv_dim = int(impl.local_kv_heads) * hd;
    const int group = int(impl.local_heads / impl.local_kv_heads);
    const float attn_scale = 1.0f / std::sqrt(float(hd));
    const int threads = 256;

    // Embed on device 0, replicate hidden to both shards through the
    // host (identical bytes -> hidden stays bit-identical across shards).
    {
        Impl::Shard& s0 = impl.shards[0];
        TP_CUDA_CHECK(cudaSetDevice(s0.device), "cuda device not usable");
        TP_CUDA_CHECK(
            cudaMemcpy(s0.hidden.ptr,
                       impl.embed.f32() + size_t(token) * h, h * 4,
                       cudaMemcpyDeviceToDevice),
            "embedding lookup failed");
        TP_CUDA_CHECK(cudaMemcpy(impl.host_hidden.data(), s0.hidden.ptr,
                                 h * 4, cudaMemcpyDeviceToHost),
                      "hidden download failed");
        Impl::Shard& s1 = impl.shards[1];
        TP_CUDA_CHECK(cudaSetDevice(s1.device), "cuda device not usable");
        TP_CUDA_CHECK(cudaMemcpy(s1.hidden.ptr, impl.host_hidden.data(),
                                 h * 4, cudaMemcpyHostToDevice),
                      "hidden upload failed");
    }

    for (uint32_t l = 0; l < c.num_layers; ++l) {
        // Attention block on both shards.
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            Impl::Shard::Layer& layer = shard.layers[l];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            float* k_dst = state.KeyAt(s, l, pos);
            float* v_dst = state.ValueAt(s, l, pos);
            TpRmsNormKernel<<<1, 256, 0, shard.stream>>>(
                shard.hidden.f32(), layer.input_norm.f32(),
                c.rms_norm_eps, h, shard.normed.f32());
            TpMatVecKernel<<<local_q, 128, 0, shard.stream>>>(
                layer.q_w.f32(), shard.normed.f32(), layer.q_b.f32(), h,
                shard.q.f32());
            TpMatVecKernel<<<local_kv_dim, 128, 0, shard.stream>>>(
                layer.k_w.f32(), shard.normed.f32(), layer.k_b.f32(), h,
                k_dst);
            TpMatVecKernel<<<local_kv_dim, 128, 0, shard.stream>>>(
                layer.v_w.f32(), shard.normed.f32(), layer.v_b.f32(), h,
                v_dst);
            TpRopeKernel<<<impl.local_heads, 32, 0, shard.stream>>>(
                shard.q.f32(), hd, pos, c.rope_theta);
            TpRopeKernel<<<impl.local_kv_heads, 32, 0, shard.stream>>>(
                k_dst, hd, pos, c.rope_theta);
            TpAttentionKernel<<<impl.local_heads, 128, 0, shard.stream>>>(
                shard.q.f32(), state.KeyBase(s, l), state.ValueBase(s, l),
                shard.scores.f32(), shard.attn_out.f32(), int(pos + 1),
                hd, local_kv_dim, group, int(c.max_context_tokens),
                attn_scale);
            TpMatVecKernel<<<h, 128, 0, shard.stream>>>(
                layer.o_w.f32(), shard.attn_out.f32(), nullptr, local_q,
                shard.partial.f32());
        }
        // Fixed-order host all-reduce of the o_proj partials.
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            float* dst = s == 0 ? impl.host_a.data() : impl.host_b.data();
            TP_CUDA_CHECK(
                cudaMemcpyAsync(dst, shard.partial.ptr, h * 4,
                                cudaMemcpyDeviceToHost, shard.stream),
                "partial download failed");
        }
        for (int s = 0; s < kNumShards; ++s) {
            TP_CUDA_CHECK(cudaSetDevice(impl.shards[s].device),
                          "cuda device not usable");
            TP_CUDA_CHECK(cudaStreamSynchronize(impl.shards[s].stream),
                          "device execution failed");
        }
        for (int i = 0; i < h; ++i) {
            impl.host_a[i] += impl.host_b[i];  // fixed order: shard0+shard1
        }
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            TP_CUDA_CHECK(
                cudaMemcpyAsync(shard.partial.ptr, impl.host_a.data(),
                                h * 4, cudaMemcpyHostToDevice,
                                shard.stream),
                "reduced upload failed");
            TpAddKernel<<<(h + threads - 1) / threads, threads, 0,
                          shard.stream>>>(shard.hidden.f32(),
                                          shard.partial.f32(), h);
        }

        // MLP block on both shards.
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            Impl::Shard::Layer& layer = shard.layers[l];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            TpRmsNormKernel<<<1, 256, 0, shard.stream>>>(
                shard.hidden.f32(), layer.post_norm.f32(), c.rms_norm_eps,
                h, shard.normed.f32());
            TpMatVecKernel<<<int(impl.local_inter), 128, 0,
                             shard.stream>>>(layer.gate_w.f32(),
                                             shard.normed.f32(), nullptr,
                                             h, shard.gate.f32());
            TpMatVecKernel<<<int(impl.local_inter), 128, 0,
                             shard.stream>>>(layer.up_w.f32(),
                                             shard.normed.f32(), nullptr,
                                             h, shard.up.f32());
            TpSwigluKernel<<<(int(impl.local_inter) + threads - 1) /
                                 threads,
                             threads, 0, shard.stream>>>(
                shard.gate.f32(), shard.up.f32(), int(impl.local_inter));
            TpMatVecKernel<<<h, 128, 0, shard.stream>>>(
                layer.down_w.f32(), shard.gate.f32(), nullptr,
                int(impl.local_inter), shard.partial.f32());
        }
        // Fixed-order host all-reduce of the down_proj partials.
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            float* dst = s == 0 ? impl.host_a.data() : impl.host_b.data();
            TP_CUDA_CHECK(
                cudaMemcpyAsync(dst, shard.partial.ptr, h * 4,
                                cudaMemcpyDeviceToHost, shard.stream),
                "partial download failed");
        }
        for (int s = 0; s < kNumShards; ++s) {
            TP_CUDA_CHECK(cudaSetDevice(impl.shards[s].device),
                          "cuda device not usable");
            TP_CUDA_CHECK(cudaStreamSynchronize(impl.shards[s].stream),
                          "device execution failed");
        }
        for (int i = 0; i < h; ++i) {
            impl.host_a[i] += impl.host_b[i];
        }
        for (int s = 0; s < kNumShards; ++s) {
            Impl::Shard& shard = impl.shards[s];
            TP_CUDA_CHECK(cudaSetDevice(shard.device),
                          "cuda device not usable");
            TP_CUDA_CHECK(
                cudaMemcpyAsync(shard.partial.ptr, impl.host_a.data(),
                                h * 4, cudaMemcpyHostToDevice,
                                shard.stream),
                "reduced upload failed");
            TpAddKernel<<<(h + threads - 1) / threads, threads, 0,
                          shard.stream>>>(shard.hidden.f32(),
                                          shard.partial.f32(), h);
        }
    }

    if (want_logits) {
        // Head on device 0 only.
        Impl::Shard& s0 = impl.shards[0];
        TP_CUDA_CHECK(cudaSetDevice(s0.device), "cuda device not usable");
        TpRmsNormKernel<<<1, 256, 0, s0.stream>>>(
            s0.hidden.f32(), impl.final_norm.f32(), c.rms_norm_eps, h,
            s0.normed.f32());
        const float* head =
            impl.tied ? impl.embed.f32() : impl.lm_head.f32();
        TpMatVecKernel<<<int(c.vocab_size), 128, 0, s0.stream>>>(
            head, s0.normed.f32(), nullptr, h, impl.logits.f32());
        logits_out.resize(c.vocab_size);
        TP_CUDA_CHECK(
            cudaMemcpyAsync(logits_out.data(), impl.logits.ptr,
                            c.vocab_size * 4, cudaMemcpyDeviceToHost,
                            s0.stream),
            "logits download failed");
        TP_CUDA_CHECK(cudaStreamSynchronize(s0.stream),
                      "device execution failed");
        for (float v : logits_out) {
            if (!std::isfinite(v)) {
                return Status(ErrorCode::kInferenceFailed,
                              "logits contain non-finite values",
                              kComponent);
            }
        }
    } else {
        for (int s = 0; s < kNumShards; ++s) {
            TP_CUDA_CHECK(cudaSetDevice(impl.shards[s].device),
                          "cuda device not usable");
            TP_CUDA_CHECK(cudaStreamSynchronize(impl.shards[s].stream),
                          "device execution failed");
        }
    }
    return Status::Ok();
}

Status QwenTpModel::Prefill(SequenceState& state,
                            const std::vector<uint32_t>& tokens,
                            std::vector<float>& logits) {
    auto& seq = static_cast<TpSequenceState&>(state);
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

Status QwenTpModel::Decode(SequenceState& state, uint32_t token,
                           std::vector<float>& logits) {
    auto& seq = static_cast<TpSequenceState&>(state);
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
