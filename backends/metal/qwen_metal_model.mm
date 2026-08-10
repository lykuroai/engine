#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "backends/metal/qwen_metal_model.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_metal";
constexpr uint32_t kCtxBucket = 128;  // context bucket granularity
constexpr uint32_t kPrefillChunk = 32;  // prompt tokens per graph run

Status MetalFailed(const char* what) {
    return Status(ErrorCode::kGpuUnhealthy, what, kComponent);
}

// Reads a verified tensor as host FP32.
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

// Transposes a row-major [rows, cols] matrix to [cols, rows].
std::vector<float> Transpose(const std::vector<float>& w, size_t rows,
                             size_t cols) {
    std::vector<float> out(w.size());
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = w[r * cols + c];
        }
    }
    return out;
}

}  // namespace


// RMSNorm with FP32-internal reduction (FP16 sum of squares overflows).
// Returns a tensor in `compute_dt`.
static MPSGraphTensor* RmsNormGraph(MPSGraph* g, MPSGraphTensor* x,
                                    MPSGraphTensor* w, float eps,
                                    bool fp16) {
    MPSGraphTensor* xf =
        fp16 ? [g castTensor:x toType:MPSDataTypeFloat32 name:nil] : x;
    MPSGraphTensor* sq = [g squareWithTensor:xf name:nil];
    MPSGraphTensor* mean = [g meanOfTensor:sq axes:@[ @1 ] name:nil];
    MPSGraphTensor* epsc = [g constantWithScalar:double(eps)
                                           shape:@[ @1, @1 ]
                                        dataType:MPSDataTypeFloat32];
    MPSGraphTensor* rs = [g
        reverseSquareRootWithTensor:[g additionWithPrimaryTensor:mean
                                                 secondaryTensor:epsc
                                                            name:nil]
                               name:nil];
    MPSGraphTensor* n = [g multiplicationWithPrimaryTensor:xf
                                          secondaryTensor:rs
                                                     name:nil];
    MPSGraphTensor* wf =
        fp16 ? [g castTensor:w toType:MPSDataTypeFloat32 name:nil] : w;
    MPSGraphTensor* out = [g multiplicationWithPrimaryTensor:n
                                             secondaryTensor:wf
                                                        name:nil];
    return fp16 ? [g castTensor:out toType:MPSDataTypeFloat16 name:nil]
                : out;
}


// Matmul with FP32 accumulation. FP16 operands are widened for the
// reduction (MPSGraph fp16 matmul accumulates in fp16 and overflows on
// the large hidden/intermediate/vocab reductions), then the result is
// returned in the compute dtype. Storage/bandwidth stay FP16; only the
// in-flight accumulator is FP32.
static MPSGraphTensor* MatMulAcc(MPSGraph* g, MPSGraphTensor* a,
                                 MPSGraphTensor* b, bool fp16) {
    if (!fp16) {
        return [g matrixMultiplicationWithPrimaryTensor:a
                                        secondaryTensor:b
                                                   name:nil];
    }
    MPSGraphTensor* af = [g castTensor:a toType:MPSDataTypeFloat32
                                 name:nil];
    MPSGraphTensor* bf = [g castTensor:b toType:MPSDataTypeFloat32
                                 name:nil];
    MPSGraphTensor* c = [g matrixMultiplicationWithPrimaryTensor:af
                                                 secondaryTensor:bf
                                                            name:nil];
    return [g castTensor:c toType:MPSDataTypeFloat16 name:nil];
}

struct QwenMetalModel::Impl {
    // Compute/storage dtype: FP16 (addendum initial standard) or FP32
    // (parity anchor). Logits are always emitted as FP32.
    bool fp16 = false;
    MPSDataType dt = MPSDataTypeFloat32;
    size_t esz = sizeof(float);
    std::vector<uint16_t> embed_fp16;  // fp16 row source when fp16 mode

    // Unified Memory admission (addendum §10): a runtime budget plus staged
    // watermarks over the recommended working set. weight_bytes is the
    // resident weight footprint; kv_committed tracks physical KV backing
    // summed across live sequences. A NEW sequence is refused once
    // projected usage crosses the soft watermark (load shedding); an
    // EXISTING sequence's KV cannot grow past the hard watermark. Fractions
    // and (for tests) the absolute budget are overridable by env var.
    uint64_t um_budget = 0;   // bytes; 0 disables runtime enforcement
    uint64_t weight_bytes = 0;
    double um_soft = 0.80;
    double um_hard = 0.92;
    std::atomic<uint64_t> kv_committed{0};

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandQueue> pf_queue = nil;  // prefill-only queue
    MPSGraphDevice* graph_device = nil;

    struct Layer {
        // Transposed weights ([in, out]) in unified-memory buffers with
        // their per-call MPSGraphTensorData views (created once).
        id<MTLBuffer> input_norm, post_norm;
        id<MTLBuffer> qw_t, kw_t, vw_t, ow_t, gate_t, up_t, down_t;
        id<MTLBuffer> qb, kb, vb;
        MPSGraphTensorData *input_norm_td, *post_norm_td;
        MPSGraphTensorData *qw_td, *kw_td, *vw_td, *ow_td, *gate_td,
            *up_td, *down_td;
        MPSGraphTensorData *qb_td, *kb_td, *vb_td;
    };
    std::vector<Layer> layers;
    id<MTLBuffer> final_norm = nil;
    id<MTLBuffer> head_t = nil;  // [h, vocab]
    MPSGraphTensorData* final_norm_td = nil;
    MPSGraphTensorData* head_td = nil;
    std::vector<float> embed_host;  // [vocab, h] for host-side lookup

    // Resident per-call input buffers (unified memory; host writes into
    // contents() directly — no NSData allocation per token).
    id<MTLBuffer> in_x = nil;      // [1, h]
    id<MTLBuffer> in_cos = nil;    // [1, half]
    id<MTLBuffer> in_sin = nil;    // [1, half]
    id<MTLBuffer> in_mask = nil;   // [1, max_bucket + 1]
    // Staging outputs the GPU writes into; host memcpys from contents().
    std::vector<id<MTLBuffer>> stage_k, stage_v;  // per layer [1, kv_dim]
    id<MTLBuffer> stage_logits = nil;             // [1, vocab]
    std::vector<MPSGraphTensorData*> stage_k_td, stage_v_td;
    MPSGraphTensorData* stage_logits_td = nil;

    // Per-context-bucket compiled graph and its placeholder handles.
    struct BucketGraph {
        MPSGraph* graph = nil;
        MPSGraphTensor* x = nil;
        MPSGraphTensor* cos_in = nil;
        MPSGraphTensor* sin_in = nil;
        MPSGraphTensor* mask = nil;
        struct LayerIo {
            MPSGraphTensor *input_norm, *post_norm;
            MPSGraphTensor *qw, *kw, *vw, *ow, *gate, *up, *down;
            MPSGraphTensor *qb, *kb, *vb;
            MPSGraphTensor *k_cache, *v_cache;
            MPSGraphTensor *new_k, *new_v;  // outputs [1, kv_dim]
        };
        std::vector<LayerIo> layer_io;
        MPSGraphTensor* final_norm = nil;
        MPSGraphTensor* head = nil;
        MPSGraphTensor* logits = nil;  // output [1, vocab]

        // Cached call state: feeds dictionary (weights + resident input
        // views baked once; only the per-sequence KV entries mutate per
        // call) and the preallocated results mapping.
        NSMutableDictionary* feeds = nil;
        NSDictionary* results = nil;
        NSArray* targets = nil;
    };
    std::map<uint32_t, BucketGraph> bucket_graphs;

    // Multi-token prefill graphs, keyed by context bucket (chunk size is
    // fixed at kPrefillChunk; the tail chunk is padded and masked).
    struct PrefillGraph {
        MPSGraph* graph = nil;
        MPSGraphTensor* x = nil;         // [P, h]
        MPSGraphTensor* cos_in = nil;    // [P, half]
        MPSGraphTensor* sin_in = nil;    // [P, half]
        MPSGraphTensor* mask = nil;      // [P, bucket + P]
        struct LayerIo {
            MPSGraphTensor *input_norm, *post_norm;
            MPSGraphTensor *qw, *kw, *vw, *ow, *gate, *up, *down;
            MPSGraphTensor *qb, *kb, *vb;
            MPSGraphTensor *k_cache, *v_cache;
            MPSGraphTensor *new_k, *new_v;  // [P, kv_dim]
        };
        std::vector<LayerIo> layer_io;
        MPSGraphTensor* final_norm = nil;
        MPSGraphTensor* head = nil;
        MPSGraphTensor* logits = nil;  // last real row [1, vocab] via slice
        MPSGraphTensor* last_index = nil;  // [1] int32: last real row
        NSMutableDictionary* feeds = nil;
        NSDictionary* results = nil;
    };
    std::map<uint32_t, PrefillGraph> prefill_graphs;

    // Prefill resident inputs/staging (P rows).
    id<MTLBuffer> pf_x = nil;     // [P, h]
    id<MTLBuffer> pf_cos = nil;   // [P, half]
    id<MTLBuffer> pf_sin = nil;   // [P, half]
    id<MTLBuffer> pf_mask = nil;  // [P, max_bucket + P]
    id<MTLBuffer> pf_last = nil;  // [1] int32
    id<MTLBuffer> pf_logits = nil;
    MPSGraphTensorData* pf_logits_td = nil;
    std::vector<id<MTLBuffer>> pf_stage_k, pf_stage_v;  // [P, kv_dim]
    std::vector<MPSGraphTensorData*> pf_stage_k_td, pf_stage_v_td;

    ~Impl() {
        // ARC releases the ObjC objects.
    }
};

namespace {

// Bounded contiguous KV cache in unified memory (spec MVP §16).
class MetalSequenceState final : public SequenceState {
public:
    struct UmBudget {
        std::atomic<uint64_t>* committed = nullptr;  // shared, per-model
        uint64_t budget = 0;        // 0 => runtime enforcement disabled
        uint64_t weight_bytes = 0;
        double soft = 0.80;
        double hard = 0.92;
    };

    static Status Create(id<MTLDevice> device, const QwenConfig& config,
                         uint32_t max_tokens, size_t esz,
                         MPSDataType dt, const UmBudget& um,
                         std::unique_ptr<MetalSequenceState>& out) {
        auto state =
            std::unique_ptr<MetalSequenceState>(new MetalSequenceState());
        state->device_ = device;
        state->esz_ = esz;
        state->dt_ = dt;
        state->kv_stride_ = config.num_kv_heads * config.head_dim;
        state->max_tokens_ = max_tokens;
        state->um_ = um;
        state->keys_.resize(config.num_layers);
        state->values_.resize(config.num_layers);
        // Lazy allocation: start small and grow on demand. The logical
        // capacity is max_tokens, but a short request only ever backs a
        // few blocks of physical KV (avoids ~800MB/sequence at 32k
        // context — the MVP contiguous equivalent of paged KV).
        const uint32_t initial =
            std::min<uint32_t>(max_tokens, 2 * kCtxBucket);
        Status s = state->Reserve(initial);
        if (!s.ok()) return s;
        out = std::move(state);
        return Status::Ok();
    }

    // Ensures physical capacity for at least `tokens` positions, rounded
    // up to a context bucket. Growth copies existing KV forward and
    // invalidates the TensorData view cache (buffer pointers change).
    Status Reserve(uint32_t tokens) {
        const uint32_t want =
            ((tokens + kCtxBucket - 1) / kCtxBucket) * kCtxBucket;
        if (want <= alloc_tokens_) return Status::Ok();
        uint32_t grown = alloc_tokens_ == 0 ? want : alloc_tokens_;
        while (grown < want) grown *= 2;
        grown = std::min(grown,
                         ((max_tokens_ + kCtxBucket - 1) / kCtxBucket) *
                             kCtxBucket);
        const size_t new_bytes = size_t(grown) * kv_stride_ * esz_;
        const size_t old_bytes = size_t(alloc_tokens_) * kv_stride_ * esz_;

        // Unified Memory admission (addendum §10). delta is the added
        // physical KV backing across all layers (k+v). The initial reserve
        // of a fresh sequence (alloc_tokens_==0) is gated by the SOFT
        // watermark — load shedding, so an over-budget arrival is refused
        // before it displaces running work. Growth of a live sequence is
        // gated by the HARD watermark, above which the engine must not push
        // the device toward an OS-level eviction.
        const uint64_t delta =
            uint64_t(new_bytes - old_bytes) * 2 * keys_.size();
        if (um_.budget > 0 && delta > 0) {
            const uint64_t projected =
                um_.weight_bytes + um_.committed->load() + delta;
            const bool initial = (alloc_tokens_ == 0);
            const double frac = initial ? um_.soft : um_.hard;
            const uint64_t limit = uint64_t(double(um_.budget) * frac);
            if (projected > limit) {
                return Status(
                    initial ? ErrorCode::kCapacityExhausted
                            : ErrorCode::kGpuOom,
                    initial ? "unified memory soft watermark: new sequence "
                              "refused (load shedding)"
                            : "unified memory hard watermark: KV growth "
                              "refused",
                    kComponent)
                    .WithDetail("projected_bytes", int64_t(projected))
                    .WithDetail("watermark_bytes", int64_t(limit))
                    .WithDetail("budget_bytes", int64_t(um_.budget));
            }
        }

        __block bool oom = false;
        // Drain autoreleased Metal temporaries here: this runs on the
        // engine worker thread, which has no ambient pool, so buffers
        // created below would otherwise accumulate (a ~6MB/request leak).
        @autoreleasepool {
            for (uint32_t l = 0; l < keys_.size(); ++l) {
                id<MTLBuffer> nk = [device_
                    newBufferWithLength:new_bytes
                                options:MTLResourceStorageModeShared];
                id<MTLBuffer> nv = [device_
                    newBufferWithLength:new_bytes
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
        kv_td_cache_.clear();  // views point at the old buffers
        if (um_.committed != nullptr) {
            um_.committed->fetch_add(delta, std::memory_order_relaxed);
            committed_bytes_ += delta;
        }
        return Status::Ok();
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }
    uint32_t kv_stride() const { return kv_stride_; }

    id<MTLBuffer> key_buffer(uint32_t layer) { return keys_[layer]; }
    id<MTLBuffer> value_buffer(uint32_t layer) { return values_[layer]; }

    // Per-bucket cached TensorData views of the KV buffers (creation is
    // not free; the engine calls per token).
    const std::vector<MPSGraphTensorData*>& KvTds(uint32_t key,
                                                  uint32_t layers,
                                                  uint32_t kv_dim) {
        auto it = kv_td_cache_.find(key);
        if (it != kv_td_cache_.end()) return it->second;
        const uint32_t bucket = key & 0x7FFFFFFFu;
        std::vector<MPSGraphTensorData*> tds;
        tds.reserve(size_t(layers) * 2);
        for (uint32_t l = 0; l < layers; ++l) {
            tds.push_back([[MPSGraphTensorData alloc]
                initWithMTLBuffer:keys_[l]
                            shape:@[ @(bucket), @(kv_dim) ]
                         dataType:dt_]);
            tds.push_back([[MPSGraphTensorData alloc]
                initWithMTLBuffer:values_[l]
                            shape:@[ @(bucket), @(kv_dim) ]
                         dataType:dt_]);
        }
        return kv_td_cache_.emplace(key, std::move(tds)).first->second;
    }

    void* KeyRow(uint32_t layer, uint32_t t) {
        return static_cast<uint8_t*>(keys_[layer].contents) +
               size_t(t) * kv_stride_ * esz_;
    }
    void* ValueRow(uint32_t layer, uint32_t t) {
        return static_cast<uint8_t*>(values_[layer].contents) +
               size_t(t) * kv_stride_ * esz_;
    }
    void Advance() { ++length_; }

    // Release the KV buffers and cached TensorData views inside an
    // autoreleasepool. The engine worker thread has no ambient pool, so
    // Metal's deferred autoreleases from -dealloc would otherwise
    // accumulate (~6MB per destroyed sequence).
    ~MetalSequenceState() {
        if (um_.committed != nullptr && committed_bytes_ > 0) {
            um_.committed->fetch_sub(committed_bytes_,
                                     std::memory_order_relaxed);
        }
        @autoreleasepool {
            kv_td_cache_.clear();
            for (auto& b : keys_) b = nil;
            for (auto& b : values_) b = nil;
        }
    }

private:
    MetalSequenceState() = default;
    id<MTLDevice> device_ = nil;
    uint32_t kv_stride_ = 0;
    uint32_t max_tokens_ = 0;    // logical capacity
    uint32_t alloc_tokens_ = 0;  // physical capacity (grows on demand)
    uint32_t length_ = 0;
    std::vector<id<MTLBuffer>> keys_;
    std::vector<id<MTLBuffer>> values_;
    std::map<uint32_t, std::vector<MPSGraphTensorData*>> kv_td_cache_;
    size_t esz_ = sizeof(float);
    MPSDataType dt_ = MPSDataTypeFloat32;
    UmBudget um_{};              // shared budget context (set in Create)
    uint64_t committed_bytes_ = 0;  // this sequence's contribution
};

NSArray<NSNumber*>* Shape2(uint64_t a, uint64_t b) {
    return @[ @(a), @(b) ];
}

}  // namespace

QwenMetalModel::~QwenMetalModel() = default;

QwenMetalModel::LoadResult QwenMetalModel::Load(
    const ModelManifest& manifest, const SafetensorsFile& weights) {
    return Load(manifest, weights, MetalModelOptions{});
}

QwenMetalModel::LoadResult QwenMetalModel::Load(
    const ModelManifest& manifest, const SafetensorsFile& weights,
    const MetalModelOptions& options) {
    LoadResult result;
    @autoreleasepool {
        auto model = std::unique_ptr<QwenMetalModel>(new QwenMetalModel());
        result.status = QwenConfig::FromManifest(manifest, model->config_);
        if (!result.status.ok()) return result;
        const QwenConfig& c = model->config_;
        model->limits_.vocab_size = c.vocab_size;
        model->limits_.max_context_tokens = c.max_context_tokens;
        model->limits_.eos_token_ids = c.eos_token_ids;

        model->impl_ = std::make_unique<Impl>();
        Impl& impl = *model->impl_;
        impl.fp16 = options.fp16;
        impl.dt = options.fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        impl.esz = options.fp16 ? 2 : sizeof(float);
        impl.device = MTLCreateSystemDefaultDevice();
        if (impl.device == nil || !impl.device.hasUnifiedMemory) {
            result.status = Status(ErrorCode::kGpuUnhealthy,
                                   "metal device unavailable", kComponent);
            return result;
        }
        impl.queue = [impl.device newCommandQueue];
        impl.pf_queue = [impl.device newCommandQueue];
        impl.graph_device =
            [MPSGraphDevice deviceWithMTLDevice:impl.device];

        // Unified Memory admission (addendum §10): the FP32 weight
        // estimate must fit inside the engine budget derived from the
        // recommended working set, never the physical total. Fail closed
        // before any allocation.
        {
            const double safety_margin = 0.15;
            const uint64_t budget = uint64_t(
                double(impl.device.recommendedMaxWorkingSetSize) *
                (1.0 - safety_margin));
            // FP32 resident weights: embed + per-layer projections
            // (+ transposed head copy when tied).
            const uint64_t h64 = c.hidden_size;
            const uint64_t per_layer =
                (2 * h64 +                                   // norms
                 h64 * size_t(c.num_heads) * c.head_dim +    // q
                 2 * h64 * size_t(c.num_kv_heads) * c.head_dim +  // k,v
                 size_t(c.num_heads) * c.head_dim * h64 +    // o
                 3 * h64 * uint64_t(c.intermediate_size)) *  // mlp
                sizeof(float);
            const uint64_t bytes_per = impl.fp16 ? 2 : sizeof(float);
            const uint64_t weight_bytes =
                (uint64_t(c.vocab_size) * h64 * 2 +
                 uint64_t(c.num_layers) *
                     (per_layer / sizeof(float))) *
                bytes_per;
            if (weight_bytes > budget) {
                result.status =
                    Status(ErrorCode::kCapacityExhausted,
                           "model exceeds unified memory budget",
                           kComponent)
                        .WithDetail("budget_bytes", int64_t(budget))
                        .WithDetail("weight_bytes",
                                    int64_t(weight_bytes));
                return result;
            }

            // Seed the runtime admission budget from the same working-set
            // figure. Env overrides allow tuning and let tests exercise the
            // watermarks without a multi-GB model.
            impl.weight_bytes = weight_bytes;
            impl.um_budget = impl.device.recommendedMaxWorkingSetSize;
            if (const char* b =
                    std::getenv("LYKURO_METAL_UM_BUDGET_BYTES")) {
                impl.um_budget = std::strtoull(b, nullptr, 10);
            }
            if (const char* s = std::getenv("LYKURO_METAL_UM_SOFT")) {
                impl.um_soft = std::atof(s);
            }
            if (const char* h = std::getenv("LYKURO_METAL_UM_HARD")) {
                impl.um_hard = std::atof(h);
            }
        }

        const size_t h = c.hidden_size;
        const size_t q_dim = size_t(c.num_heads) * c.head_dim;
        const size_t kv_dim = size_t(c.num_kv_heads) * c.head_dim;

        auto upload = [&](const std::vector<float>& host,
                          id<MTLBuffer> __strong& buf) -> Status {
            if (impl.fp16) {
                std::vector<uint16_t> half_host(host.size());
                for (size_t i = 0; i < host.size(); ++i) {
                    half_host[i] = FloatToFp16(host[i]);
                }
                buf = [impl.device
                    newBufferWithBytes:half_host.data()
                                length:half_host.size() * 2
                               options:MTLResourceStorageModeShared];
            } else {
                buf = [impl.device
                    newBufferWithBytes:host.data()
                                length:host.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];
            }
            if (buf == nil) {
                return Status(ErrorCode::kGpuOom, "weight upload failed",
                              kComponent);
            }
            return Status::Ok();
        };
        auto make_td = [&](id<MTLBuffer> buf, uint64_t rows,
                           uint64_t cols) -> MPSGraphTensorData* {
            return [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buf
                            shape:Shape2(rows, cols)
                         dataType:impl.dt];
        };

        std::vector<float> host;
        std::vector<uint64_t> shape;
        impl.layers.resize(c.num_layers);
        Status s = Status::Ok();
        for (uint32_t l = 0; l < c.num_layers && s.ok(); ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            Impl::Layer& layer = impl.layers[l];

            s = TensorToHost(weights, p + "input_layernorm.weight", host,
                             nullptr);
            if (s.ok()) s = upload(host, layer.input_norm);
            if (s.ok()) {
                layer.input_norm_td = make_td(layer.input_norm, 1, h);
                s = TensorToHost(weights,
                                 p + "post_attention_layernorm.weight",
                                 host, nullptr);
            }
            if (s.ok()) s = upload(host, layer.post_norm);
            if (s.ok()) {
                layer.post_norm_td = make_td(layer.post_norm, 1, h);
            }

            auto load_t = [&](const std::string& name, size_t rows,
                              size_t cols, id<MTLBuffer> __strong& buf,
                              MPSGraphTensorData* __strong& td) {
                if (!s.ok()) return;
                s = TensorToHost(weights, name, host, &shape);
                if (!s.ok()) return;
                s = upload(Transpose(host, rows, cols), buf);
                if (s.ok()) td = make_td(buf, cols, rows);
            };
            load_t(p + "self_attn.q_proj.weight", q_dim, h, layer.qw_t,
                   layer.qw_td);
            load_t(p + "self_attn.k_proj.weight", kv_dim, h, layer.kw_t,
                   layer.kw_td);
            load_t(p + "self_attn.v_proj.weight", kv_dim, h, layer.vw_t,
                   layer.vw_td);
            load_t(p + "self_attn.o_proj.weight", h, q_dim, layer.ow_t,
                   layer.ow_td);
            load_t(p + "mlp.gate_proj.weight", c.intermediate_size, h,
                   layer.gate_t, layer.gate_td);
            load_t(p + "mlp.up_proj.weight", c.intermediate_size, h,
                   layer.up_t, layer.up_td);
            load_t(p + "mlp.down_proj.weight", h, c.intermediate_size,
                   layer.down_t, layer.down_td);

            auto load_bias = [&](const std::string& name, size_t n,
                                 id<MTLBuffer> __strong& buf,
                                 MPSGraphTensorData* __strong& td) {
                if (!s.ok()) return;
                s = TensorToHost(weights, name, host, nullptr);
                if (s.ok()) s = upload(host, buf);
                if (s.ok()) td = make_td(buf, 1, n);
            };
            load_bias(p + "self_attn.q_proj.bias", q_dim, layer.qb,
                      layer.qb_td);
            load_bias(p + "self_attn.k_proj.bias", kv_dim, layer.kb,
                      layer.kb_td);
            load_bias(p + "self_attn.v_proj.bias", kv_dim, layer.vb,
                      layer.vb_td);
        }
        if (s.ok()) {
            s = TensorToHost(weights, "model.norm.weight", host, nullptr);
        }
        if (s.ok()) s = upload(host, impl.final_norm);
        if (s.ok()) impl.final_norm_td = make_td(impl.final_norm, 1, h);

        if (s.ok()) {
            s = TensorToHost(weights, "model.embed_tokens.weight",
                             impl.embed_host, nullptr);
            if (s.ok() && impl.fp16) {
                impl.embed_fp16.resize(impl.embed_host.size());
                for (size_t i = 0; i < impl.embed_host.size(); ++i) {
                    impl.embed_fp16[i] = FloatToFp16(impl.embed_host[i]);
                }
            }
        }
        if (s.ok()) {
            if (c.tie_word_embeddings) {
                s = upload(Transpose(impl.embed_host, c.vocab_size, h),
                           impl.head_t);
            } else {
                std::vector<float> head;
                s = TensorToHost(weights, "lm_head.weight", head, nullptr);
                if (s.ok()) {
                    s = upload(Transpose(head, c.vocab_size, h),
                               impl.head_t);
                }
            }
        }
        if (s.ok()) impl.head_td = make_td(impl.head_t, h, c.vocab_size);

        // Resident per-call input and staging buffers.
        if (s.ok()) {
            const uint32_t half = c.head_dim / 2;
            const uint32_t max_bucket =
                ((c.max_context_tokens + kCtxBucket - 1) / kCtxBucket) *
                kCtxBucket;
            const size_t kv_dim = size_t(c.num_kv_heads) * c.head_dim;
            auto alloc = [&](size_t bytes) {
                return [impl.device
                    newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];
            };
            impl.in_x = alloc(h * impl.esz);
            impl.in_cos = alloc(half * impl.esz);
            impl.in_sin = alloc(half * impl.esz);
            impl.in_mask = alloc((size_t(max_bucket) + 1) * impl.esz);
            impl.stage_logits = alloc(c.vocab_size * sizeof(float));
            if (impl.in_x == nil || impl.in_cos == nil ||
                impl.in_sin == nil || impl.in_mask == nil ||
                impl.stage_logits == nil) {
                s = Status(ErrorCode::kGpuOom, "buffer allocation failed",
                           kComponent);
            }
            if (s.ok()) {
                impl.stage_logits_td = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:impl.stage_logits
                                shape:Shape2(1, c.vocab_size)
                             dataType:MPSDataTypeFloat32];
                impl.stage_k.resize(c.num_layers);
                impl.stage_v.resize(c.num_layers);
                impl.stage_k_td.resize(c.num_layers);
                impl.stage_v_td.resize(c.num_layers);
                for (uint32_t l = 0; l < c.num_layers && s.ok(); ++l) {
                    impl.stage_k[l] = alloc(kv_dim * impl.esz);
                    impl.stage_v[l] = alloc(kv_dim * impl.esz);
                    if (impl.stage_k[l] == nil || impl.stage_v[l] == nil) {
                        s = Status(ErrorCode::kGpuOom,
                                   "buffer allocation failed", kComponent);
                        break;
                    }
                    impl.stage_k_td[l] = [[MPSGraphTensorData alloc]
                        initWithMTLBuffer:impl.stage_k[l]
                                    shape:Shape2(1, kv_dim)
                                 dataType:impl.dt];
                    impl.stage_v_td[l] = [[MPSGraphTensorData alloc]
                        initWithMTLBuffer:impl.stage_v[l]
                                    shape:Shape2(1, kv_dim)
                                 dataType:impl.dt];
                }
                const uint32_t P = kPrefillChunk;
                impl.pf_x = alloc(size_t(P) * h * impl.esz);
                impl.pf_cos = alloc(size_t(P) * half * impl.esz);
                impl.pf_sin = alloc(size_t(P) * half * impl.esz);
                impl.pf_mask = alloc(size_t(P) * (max_bucket + P) *
                                     impl.esz);
                impl.pf_last = alloc(sizeof(int32_t));
                impl.pf_logits = alloc(c.vocab_size * sizeof(float));
                impl.pf_logits_td = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:impl.pf_logits
                                shape:Shape2(1, c.vocab_size)
                             dataType:MPSDataTypeFloat32];
                impl.pf_stage_k.resize(c.num_layers);
                impl.pf_stage_v.resize(c.num_layers);
                impl.pf_stage_k_td.resize(c.num_layers);
                impl.pf_stage_v_td.resize(c.num_layers);
                for (uint32_t l = 0; l < c.num_layers && s.ok(); ++l) {
                    impl.pf_stage_k[l] =
                        alloc(size_t(P) * kv_dim * impl.esz);
                    impl.pf_stage_v[l] =
                        alloc(size_t(P) * kv_dim * impl.esz);
                    if (impl.pf_stage_k[l] == nil ||
                        impl.pf_stage_v[l] == nil) {
                        s = Status(ErrorCode::kGpuOom,
                                   "buffer allocation failed", kComponent);
                        break;
                    }
                    impl.pf_stage_k_td[l] = [[MPSGraphTensorData alloc]
                        initWithMTLBuffer:impl.pf_stage_k[l]
                                    shape:Shape2(P, kv_dim)
                                 dataType:impl.dt];
                    impl.pf_stage_v_td[l] = [[MPSGraphTensorData alloc]
                        initWithMTLBuffer:impl.pf_stage_v[l]
                                    shape:Shape2(P, kv_dim)
                                 dataType:impl.dt];
                }
            }
        }
        if (!s.ok()) {
            result.status = s;
            return result;
        }

        // Pre-warm: compile the first-bucket decode and prefill graphs
        // against a throwaway sequence so the first request does not pay
        // graph specialization (part of load, mirrors the smoke test).
        {
            std::unique_ptr<SequenceState> warm;
            s = model->CreateSequence(kPrefillChunk + 2, warm);
            if (s.ok()) {
                std::vector<float> warm_logits;
                std::vector<uint32_t> warm_tokens(kPrefillChunk, 0);
                s = model->Prefill(*warm, warm_tokens, warm_logits);
                if (s.ok()) {
                    s = model->Decode(*warm, 0, warm_logits);
                }
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

Status QwenMetalModel::CreateSequence(uint32_t max_tokens,
                                      std::unique_ptr<SequenceState>& out) {
    std::unique_ptr<MetalSequenceState> state;
    MetalSequenceState::UmBudget um;
    um.committed = &impl_->kv_committed;
    um.budget = impl_->um_budget;
    um.weight_bytes = impl_->weight_bytes;
    um.soft = impl_->um_soft;
    um.hard = impl_->um_hard;
    Status s = MetalSequenceState::Create(impl_->device, config_,
                                          max_tokens, impl_->esz,
                                          impl_->dt, um, state);
    if (!s.ok()) return s;
    out = std::move(state);
    return Status::Ok();
}

uint64_t QwenMetalModel::UmBudgetBytes() const { return impl_->um_budget; }
uint64_t QwenMetalModel::UmWeightBytes() const { return impl_->weight_bytes; }
uint64_t QwenMetalModel::UmCommittedKvBytes() const {
    return impl_->kv_committed.load(std::memory_order_relaxed);
}

Status QwenMetalModel::ForwardToken(uint32_t token, uint32_t pos,
                                    void* sequence_state,
                                    std::vector<float>& logits_out,
                                    bool want_logits) {
    @autoreleasepool {
        auto& state = *static_cast<MetalSequenceState*>(sequence_state);
        const QwenConfig& c = config_;
        Impl& impl = *impl_;
        const uint32_t h = c.hidden_size;
        const uint32_t half = c.head_dim / 2;
        const uint32_t kv_dim = c.num_kv_heads * c.head_dim;
        const uint32_t bucket =
            ((pos / kCtxBucket) + 1) * kCtxBucket;  // cached rows fed

        // Grow physical KV to cover this bucket before it is viewed.
        Status reserve = state.Reserve(bucket);
        if (!reserve.ok()) return reserve;

        // Build (or reuse) the bucket-specialized graph.
        Impl::BucketGraph& bg = impl.bucket_graphs[bucket];
        if (bg.graph == nil) {
            MPSGraph* g = [[MPSGraph alloc] init];
            bg.graph = g;
            const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
            bg.x = [g placeholderWithShape:Shape2(1, h)
                                  dataType:impl.dt
                                      name:nil];
            bg.cos_in = [g placeholderWithShape:Shape2(1, half)
                                       dataType:impl.dt
                                           name:nil];
            bg.sin_in = [g placeholderWithShape:Shape2(1, half)
                                       dataType:impl.dt
                                           name:nil];
            bg.mask = [g placeholderWithShape:Shape2(1, bucket + 1)
                                     dataType:impl.dt
                                         name:nil];
            bg.layer_io.resize(c.num_layers);

            auto rmsnorm = [&](MPSGraphTensor* x, MPSGraphTensor* w) {
                return RmsNormGraph(g, x, w, c.rms_norm_eps, impl.fp16);
            };
            auto rope = [&](MPSGraphTensor* v, uint32_t heads) {
                // [1, heads*hd] -> [heads, hd] -> rotate-half -> back.
                MPSGraphTensor* r = [g
                    reshapeTensor:v
                        withShape:Shape2(heads, c.head_dim)
                             name:nil];
                MPSGraphTensor* a =
                    [g sliceTensor:r dimension:1 start:0 length:half
                              name:nil];
                MPSGraphTensor* b =
                    [g sliceTensor:r dimension:1 start:half length:half
                              name:nil];
                MPSGraphTensor* ap = [g
                    subtractionWithPrimaryTensor:
                        [g multiplicationWithPrimaryTensor:a
                                           secondaryTensor:bg.cos_in
                                                      name:nil]
                                secondaryTensor:
                                    [g multiplicationWithPrimaryTensor:b
                                                       secondaryTensor:
                                                           bg.sin_in
                                                                  name:nil]
                                           name:nil];
                MPSGraphTensor* bp = [g
                    additionWithPrimaryTensor:
                        [g multiplicationWithPrimaryTensor:b
                                           secondaryTensor:bg.cos_in
                                                      name:nil]
                              secondaryTensor:
                                  [g multiplicationWithPrimaryTensor:a
                                                     secondaryTensor:
                                                         bg.sin_in
                                                                name:nil]
                                         name:nil];
                MPSGraphTensor* cat =
                    [g concatTensors:@[ ap, bp ] dimension:1 name:nil];
                return [g reshapeTensor:cat
                              withShape:Shape2(1, heads * c.head_dim)
                                   name:nil];
            };

            MPSGraphTensor* x = bg.x;
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                Impl::BucketGraph::LayerIo& io = bg.layer_io[l];
                io.input_norm = [g placeholderWithShape:Shape2(1, h)
                                               dataType:impl.dt
                                                   name:nil];
                io.post_norm = [g placeholderWithShape:Shape2(1, h)
                                              dataType:impl.dt
                                                  name:nil];
                io.qw = [g
                    placeholderWithShape:Shape2(h, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:impl.dt
                                    name:nil];
                io.kw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.vw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.ow = [g
                    placeholderWithShape:Shape2(size_t(c.num_heads) *
                                                    c.head_dim,
                                                h)
                                dataType:impl.dt
                                    name:nil];
                io.gate = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:impl.dt
                                    name:nil];
                io.up = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:impl.dt
                                    name:nil];
                io.down = [g
                    placeholderWithShape:Shape2(c.intermediate_size, h)
                                dataType:impl.dt
                                    name:nil];
                io.qb = [g
                    placeholderWithShape:Shape2(1, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:impl.dt
                                    name:nil];
                io.kb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.vb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.k_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:impl.dt
                                                name:nil];
                io.v_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:impl.dt
                                                name:nil];

                MPSGraphTensor* normed = rmsnorm(x, io.input_norm);
                auto proj = [&](MPSGraphTensor* w, MPSGraphTensor* bias) {
                    MPSGraphTensor* y = MatMulAcc(g, normed, w, impl.fp16);
                    if (bias != nil) {
                        y = [g additionWithPrimaryTensor:y
                                         secondaryTensor:bias
                                                    name:nil];
                    }
                    return y;
                };
                MPSGraphTensor* q = rope(proj(io.qw, io.qb), c.num_heads);
                MPSGraphTensor* k =
                    rope(proj(io.kw, io.kb), c.num_kv_heads);
                MPSGraphTensor* v = proj(io.vw, io.vb);
                io.new_k = k;
                io.new_v = v;

                // K/V over cached rows + current token.
                MPSGraphTensor* k_all =
                    [g concatTensors:@[ io.k_cache, k ] dimension:0
                                name:nil];
                MPSGraphTensor* v_all =
                    [g concatTensors:@[ io.v_cache, v ] dimension:0
                                name:nil];
                const uint32_t rows = bucket + 1;
                // [rows, kv_heads, hd] -> broadcast over the query group
                // -> [heads, rows, hd].
                auto expand = [&](MPSGraphTensor* t) {
                    MPSGraphTensor* r4 = [g
                        reshapeTensor:t
                            withShape:@[
                                @(rows), @(c.num_kv_heads), @1,
                                @(c.head_dim)
                            ]
                                 name:nil];
                    MPSGraphTensor* tiled = [g
                        broadcastTensor:r4
                                toShape:@[
                                    @(rows), @(c.num_kv_heads),
                                    @(c.num_heads / c.num_kv_heads),
                                    @(c.head_dim)
                                ]
                                   name:nil];
                    MPSGraphTensor* merged = [g
                        reshapeTensor:tiled
                            withShape:@[
                                @(rows), @(c.num_heads), @(c.head_dim)
                            ]
                                 name:nil];
                    return [g transposeTensor:merged
                                    dimension:0
                                withDimension:1
                                         name:nil];  // [heads, rows, hd]
                };
                MPSGraphTensor* k_h = expand(k_all);
                MPSGraphTensor* v_h = expand(v_all);
                MPSGraphTensor* q_h = [g
                    reshapeTensor:q
                        withShape:@[ @(c.num_heads), @1, @(c.head_dim) ]
                             name:nil];
                MPSGraphTensor* k_t =
                    [g transposeTensor:k_h
                             dimension:1
                         withDimension:2
                                  name:nil];  // [heads, hd, rows]
                MPSGraphTensor* scores = MatMulAcc(g, q_h, k_t, impl.fp16);
                MPSGraphTensor* scale =
                    [g constantWithScalar:double(attn_scale)
                                    shape:@[ @1, @1, @1 ]
                                 dataType:impl.dt];
                scores = [g multiplicationWithPrimaryTensor:scores
                                            secondaryTensor:scale
                                                       name:nil];
                MPSGraphTensor* mask3 = [g
                    reshapeTensor:bg.mask
                        withShape:@[ @1, @1, @(rows) ]
                             name:nil];
                scores = [g additionWithPrimaryTensor:scores
                                      secondaryTensor:mask3
                                                 name:nil];
                MPSGraphTensor* probs =
                    [g softMaxWithTensor:scores axis:2 name:nil];
                MPSGraphTensor* ctx = MatMulAcc(g, probs, v_h, impl.fp16);
                MPSGraphTensor* ctx_flat = [g
                    reshapeTensor:ctx
                        withShape:Shape2(1, size_t(c.num_heads) *
                                                c.head_dim)
                             name:nil];
                MPSGraphTensor* o =
                    MatMulAcc(g, ctx_flat, io.ow, impl.fp16);
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:o
                                            name:nil];

                MPSGraphTensor* normed2 = rmsnorm(x, io.post_norm);
                MPSGraphTensor* gate =
                    MatMulAcc(g, normed2, io.gate, impl.fp16);
                MPSGraphTensor* up =
                    MatMulAcc(g, normed2, io.up, impl.fp16);
                MPSGraphTensor* silu = [g
                    multiplicationWithPrimaryTensor:gate
                                    secondaryTensor:[g sigmoidWithTensor:gate
                                                                    name:nil]
                                               name:nil];
                MPSGraphTensor* act =
                    [g multiplicationWithPrimaryTensor:silu
                                       secondaryTensor:up
                                                  name:nil];
                MPSGraphTensor* down =
                    MatMulAcc(g, act, io.down, impl.fp16);
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:down
                                            name:nil];
            }
            bg.final_norm = [g placeholderWithShape:Shape2(1, h)
                                           dataType:impl.dt
                                               name:nil];
            bg.head = [g placeholderWithShape:Shape2(h, c.vocab_size)
                                     dataType:impl.dt
                                         name:nil];
            MPSGraphTensor* fn = rmsnorm(x, bg.final_norm);
            MPSGraphTensor* lg = MatMulAcc(g, fn, bg.head, impl.fp16);
            bg.logits = impl.fp16 ? [g castTensor:lg
                                           toType:MPSDataTypeFloat32
                                             name:nil]
                                  : lg;
        }

        // One-time per-bucket call state: feeds dictionary with the
        // weight views and resident input views baked in; preallocated
        // results mapping so outputs land in staging buffers directly.
        const uint32_t kv_dim_u = kv_dim;
        if (bg.feeds == nil) {
            NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
            feeds[bg.x] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.in_x
                            shape:Shape2(1, h)
                         dataType:impl.dt];
            feeds[bg.cos_in] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.in_cos
                            shape:Shape2(1, half)
                         dataType:impl.dt];
            feeds[bg.sin_in] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.in_sin
                            shape:Shape2(1, half)
                         dataType:impl.dt];
            feeds[bg.mask] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.in_mask
                            shape:Shape2(1, bucket + 1)
                         dataType:impl.dt];
            NSMutableDictionary* results = [NSMutableDictionary dictionary];
            NSMutableArray* targets = [NSMutableArray array];
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                Impl::Layer& lw = impl.layers[l];
                Impl::BucketGraph::LayerIo& io = bg.layer_io[l];
                feeds[io.input_norm] = lw.input_norm_td;
                feeds[io.post_norm] = lw.post_norm_td;
                feeds[io.qw] = lw.qw_td;
                feeds[io.kw] = lw.kw_td;
                feeds[io.vw] = lw.vw_td;
                feeds[io.ow] = lw.ow_td;
                feeds[io.gate] = lw.gate_td;
                feeds[io.up] = lw.up_td;
                feeds[io.down] = lw.down_td;
                feeds[io.qb] = lw.qb_td;
                feeds[io.kb] = lw.kb_td;
                feeds[io.vb] = lw.vb_td;
                results[io.new_k] = impl.stage_k_td[l];
                results[io.new_v] = impl.stage_v_td[l];
                [targets addObject:io.new_k];
                [targets addObject:io.new_v];
            }
            feeds[bg.final_norm] = impl.final_norm_td;
            feeds[bg.head] = impl.head_td;
            results[bg.logits] = impl.stage_logits_td;
            [targets addObject:bg.logits];
            bg.feeds = feeds;
            bg.results = results;
            bg.targets = targets;
        }

        // Per-token inputs written directly into resident buffers
        // (converted to the compute dtype). FP16 mask uses -6e4 (the
        // representable large negative; -1e30 saturates to -inf).
        const float neg_inf = impl.fp16 ? -60000.0f : -1e30f;
        if (impl.fp16) {
            uint16_t* xp = static_cast<uint16_t*>(impl.in_x.contents);
            const uint16_t* src = impl.embed_fp16.data() + size_t(token) * h;
            std::memcpy(xp, src, h * sizeof(uint16_t));
            uint16_t* cos_p = static_cast<uint16_t*>(impl.in_cos.contents);
            uint16_t* sin_p = static_cast<uint16_t*>(impl.in_sin.contents);
            uint16_t* mask_p = static_cast<uint16_t*>(impl.in_mask.contents);
            for (uint32_t i = 0; i < half; ++i) {
                const float freq = std::pow(
                    c.rope_theta, -2.0f * float(i) / float(c.head_dim));
                cos_p[i] = FloatToFp16(std::cos(float(pos) * freq));
                sin_p[i] = FloatToFp16(std::sin(float(pos) * freq));
            }
            const uint16_t zero = FloatToFp16(0.0f);
            const uint16_t ninf = FloatToFp16(neg_inf);
            for (uint32_t t = 0; t < bucket; ++t) {
                mask_p[t] = t < pos ? zero : ninf;
            }
            mask_p[bucket] = zero;
        } else {
            std::memcpy(impl.in_x.contents,
                        impl.embed_host.data() + size_t(token) * h,
                        h * sizeof(float));
            float* cos_p = static_cast<float*>(impl.in_cos.contents);
            float* sin_p = static_cast<float*>(impl.in_sin.contents);
            for (uint32_t i = 0; i < half; ++i) {
                const float freq = std::pow(
                    c.rope_theta, -2.0f * float(i) / float(c.head_dim));
                cos_p[i] = std::cos(float(pos) * freq);
                sin_p[i] = std::sin(float(pos) * freq);
            }
            float* mask_p = static_cast<float*>(impl.in_mask.contents);
            for (uint32_t t = 0; t < pos; ++t) mask_p[t] = 0.0f;
            for (uint32_t t = pos; t < bucket; ++t) mask_p[t] = -1e30f;
            mask_p[bucket] = 0.0f;
        }

        // Per-sequence KV views go into a per-call copy of the cached
        // (weight + resident-input) feeds. Mutating the model-lifetime
        // bg.feeds with per-sequence TensorData would retain each
        // completed sequence's KV buffers (a ~6MB/request leak); the copy
        // drains with this autoreleasepool instead.
        NSMutableDictionary* call_feeds = [bg.feeds mutableCopy];
        const auto& kv_tds =
            state.KvTds(bucket, c.num_layers, kv_dim_u);
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            Impl::BucketGraph::LayerIo& io = bg.layer_io[l];
            call_feeds[io.k_cache] = kv_tds[size_t(l) * 2];
            call_feeds[io.v_cache] = kv_tds[size_t(l) * 2 + 1];
        }

        [bg.graph runWithMTLCommandQueue:impl.queue
                                   feeds:call_feeds
                        targetOperations:nil
                       resultsDictionary:bg.results];

        // Staging -> cache (unified memory memcpy, no readbacks).
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            std::memcpy(state.KeyRow(l, pos), impl.stage_k[l].contents,
                        kv_dim_u * impl.esz);
            std::memcpy(state.ValueRow(l, pos), impl.stage_v[l].contents,
                        kv_dim_u * impl.esz);
        }
        if (want_logits) {
            const float* lp =
                static_cast<const float*>(impl.stage_logits.contents);
            logits_out.assign(lp, lp + c.vocab_size);
            for (float v : logits_out) {
                if (!std::isfinite(v)) {
                    return Status(ErrorCode::kInferenceFailed,
                                  "logits contain non-finite values",
                                  kComponent);
                }
            }
        }
        return Status::Ok();
    }
}


Status QwenMetalModel::ForwardChunk(const uint32_t* tokens, uint32_t pos0,
                                    void* sequence_state,
                                    std::vector<float>& logits_out,
                                    bool want_logits) {
    @autoreleasepool {
        auto& state = *static_cast<MetalSequenceState*>(sequence_state);
        const QwenConfig& c = config_;
        Impl& impl = *impl_;
        const uint32_t P = kPrefillChunk;
        const uint32_t h = c.hidden_size;
        const uint32_t half = c.head_dim / 2;
        const uint32_t kv_dim = c.num_kv_heads * c.head_dim;
        const uint32_t bucket = std::max(
            kCtxBucket,
            ((pos0 + kCtxBucket - 1) / kCtxBucket) * kCtxBucket);
        const uint32_t rows = bucket + P;

        // Grow physical KV to cover the chunk's writes (pos0 + P).
        Status reserve = state.Reserve(pos0 + P);
        if (!reserve.ok()) return reserve;

        Impl::PrefillGraph& pg = impl.prefill_graphs[bucket];
        if (pg.graph == nil) {
            MPSGraph* g = [[MPSGraph alloc] init];
            pg.graph = g;
            const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
            pg.x = [g placeholderWithShape:Shape2(P, h)
                                  dataType:impl.dt
                                      name:nil];
            pg.cos_in = [g placeholderWithShape:Shape2(P, half)
                                       dataType:impl.dt
                                           name:nil];
            pg.sin_in = [g placeholderWithShape:Shape2(P, half)
                                       dataType:impl.dt
                                           name:nil];
            pg.mask = [g placeholderWithShape:Shape2(P, rows)
                                     dataType:impl.dt
                                         name:nil];
            pg.layer_io.resize(c.num_layers);

            auto rmsnorm = [&](MPSGraphTensor* x, MPSGraphTensor* w) {
                return RmsNormGraph(g, x, w, c.rms_norm_eps, impl.fp16);
            };
            // Rotate-half RoPE over [P, heads*hd] with per-row angles.
            auto rope = [&](MPSGraphTensor* v, uint32_t heads) {
                MPSGraphTensor* r = [g
                    reshapeTensor:v
                        withShape:@[ @(P), @(heads), @(c.head_dim) ]
                             name:nil];
                MPSGraphTensor* a =
                    [g sliceTensor:r dimension:2 start:0 length:half
                              name:nil];
                MPSGraphTensor* b =
                    [g sliceTensor:r dimension:2 start:half length:half
                              name:nil];
                MPSGraphTensor* cos3 = [g
                    reshapeTensor:pg.cos_in
                        withShape:@[ @(P), @1, @(half) ]
                             name:nil];
                MPSGraphTensor* sin3 = [g
                    reshapeTensor:pg.sin_in
                        withShape:@[ @(P), @1, @(half) ]
                             name:nil];
                MPSGraphTensor* ap = [g
                    subtractionWithPrimaryTensor:
                        [g multiplicationWithPrimaryTensor:a
                                           secondaryTensor:cos3
                                                      name:nil]
                                secondaryTensor:
                                    [g multiplicationWithPrimaryTensor:b
                                                       secondaryTensor:sin3
                                                                  name:nil]
                                           name:nil];
                MPSGraphTensor* bp = [g
                    additionWithPrimaryTensor:
                        [g multiplicationWithPrimaryTensor:b
                                           secondaryTensor:cos3
                                                      name:nil]
                              secondaryTensor:
                                  [g multiplicationWithPrimaryTensor:a
                                                     secondaryTensor:sin3
                                                                name:nil]
                                         name:nil];
                MPSGraphTensor* cat =
                    [g concatTensors:@[ ap, bp ] dimension:2 name:nil];
                return [g reshapeTensor:cat
                              withShape:Shape2(P, heads * c.head_dim)
                                   name:nil];
            };

            MPSGraphTensor* x = pg.x;
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                Impl::PrefillGraph::LayerIo& io = pg.layer_io[l];
                io.input_norm = [g placeholderWithShape:Shape2(1, h)
                                               dataType:impl.dt
                                                   name:nil];
                io.post_norm = [g placeholderWithShape:Shape2(1, h)
                                              dataType:impl.dt
                                                  name:nil];
                io.qw = [g
                    placeholderWithShape:Shape2(h, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:impl.dt
                                    name:nil];
                io.kw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.vw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.ow = [g
                    placeholderWithShape:Shape2(size_t(c.num_heads) *
                                                    c.head_dim,
                                                h)
                                dataType:impl.dt
                                    name:nil];
                io.gate = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:impl.dt
                                    name:nil];
                io.up = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:impl.dt
                                    name:nil];
                io.down = [g
                    placeholderWithShape:Shape2(c.intermediate_size, h)
                                dataType:impl.dt
                                    name:nil];
                io.qb = [g
                    placeholderWithShape:Shape2(1, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:impl.dt
                                    name:nil];
                io.kb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.vb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:impl.dt
                                           name:nil];
                io.k_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:impl.dt
                                                name:nil];
                io.v_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:impl.dt
                                                name:nil];

                MPSGraphTensor* normed = rmsnorm(x, io.input_norm);
                auto proj = [&](MPSGraphTensor* w, MPSGraphTensor* bias) {
                    MPSGraphTensor* y = MatMulAcc(g, normed, w, impl.fp16);
                    if (bias != nil) {
                        y = [g additionWithPrimaryTensor:y
                                         secondaryTensor:bias
                                                    name:nil];
                    }
                    return y;
                };
                MPSGraphTensor* q = rope(proj(io.qw, io.qb), c.num_heads);
                MPSGraphTensor* k =
                    rope(proj(io.kw, io.kb), c.num_kv_heads);
                MPSGraphTensor* v = proj(io.vw, io.vb);
                io.new_k = k;
                io.new_v = v;

                MPSGraphTensor* k_all =
                    [g concatTensors:@[ io.k_cache, k ] dimension:0
                                name:nil];
                MPSGraphTensor* v_all =
                    [g concatTensors:@[ io.v_cache, v ] dimension:0
                                name:nil];
                auto expand = [&](MPSGraphTensor* t) {
                    MPSGraphTensor* r4 = [g
                        reshapeTensor:t
                            withShape:@[
                                @(rows), @(c.num_kv_heads), @1,
                                @(c.head_dim)
                            ]
                                 name:nil];
                    MPSGraphTensor* tiled = [g
                        broadcastTensor:r4
                                toShape:@[
                                    @(rows), @(c.num_kv_heads),
                                    @(c.num_heads / c.num_kv_heads),
                                    @(c.head_dim)
                                ]
                                   name:nil];
                    MPSGraphTensor* merged = [g
                        reshapeTensor:tiled
                            withShape:@[
                                @(rows), @(c.num_heads), @(c.head_dim)
                            ]
                                 name:nil];
                    return [g transposeTensor:merged
                                    dimension:0
                                withDimension:1
                                         name:nil];
                };
                MPSGraphTensor* k_h = expand(k_all);
                MPSGraphTensor* v_h = expand(v_all);
                MPSGraphTensor* q_h = [g
                    transposeTensor:[g reshapeTensor:q
                                           withShape:@[
                                               @(P), @(c.num_heads),
                                               @(c.head_dim)
                                           ]
                                                name:nil]
                          dimension:0
                      withDimension:1
                               name:nil];  // [heads, P, hd]
                MPSGraphTensor* k_t =
                    [g transposeTensor:k_h
                             dimension:1
                         withDimension:2
                                  name:nil];  // [heads, hd, rows]
                MPSGraphTensor* scores = MatMulAcc(g, q_h, k_t, impl.fp16);
                MPSGraphTensor* scale =
                    [g constantWithScalar:double(attn_scale)
                                    shape:@[ @1, @1, @1 ]
                                 dataType:impl.dt];
                scores = [g multiplicationWithPrimaryTensor:scores
                                            secondaryTensor:scale
                                                       name:nil];
                MPSGraphTensor* mask3 = [g
                    reshapeTensor:pg.mask
                        withShape:@[ @1, @(P), @(rows) ]
                             name:nil];
                scores = [g additionWithPrimaryTensor:scores
                                      secondaryTensor:mask3
                                                 name:nil];
                MPSGraphTensor* probs =
                    [g softMaxWithTensor:scores axis:2 name:nil];
                MPSGraphTensor* ctx = MatMulAcc(g, probs, v_h, impl.fp16);
                MPSGraphTensor* ctx_flat = [g
                    reshapeTensor:[g transposeTensor:ctx
                                           dimension:0
                                       withDimension:1
                                                name:nil]
                        withShape:Shape2(P, size_t(c.num_heads) *
                                                c.head_dim)
                             name:nil];
                MPSGraphTensor* o =
                    MatMulAcc(g, ctx_flat, io.ow, impl.fp16);
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:o
                                            name:nil];

                MPSGraphTensor* normed2 = rmsnorm(x, io.post_norm);
                MPSGraphTensor* gate =
                    MatMulAcc(g, normed2, io.gate, impl.fp16);
                MPSGraphTensor* up =
                    MatMulAcc(g, normed2, io.up, impl.fp16);
                MPSGraphTensor* silu = [g
                    multiplicationWithPrimaryTensor:gate
                                    secondaryTensor:
                                        [g sigmoidWithTensor:gate
                                                        name:nil]
                                               name:nil];
                MPSGraphTensor* act =
                    [g multiplicationWithPrimaryTensor:silu
                                       secondaryTensor:up
                                                  name:nil];
                MPSGraphTensor* down =
                    MatMulAcc(g, act, io.down, impl.fp16);
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:down
                                            name:nil];
            }
            pg.final_norm = [g placeholderWithShape:Shape2(1, h)
                                           dataType:impl.dt
                                               name:nil];
            pg.head = [g placeholderWithShape:Shape2(h, c.vocab_size)
                                     dataType:impl.dt
                                         name:nil];
            // The overlap-chunk contract puts the last real token in the
            // final row, so a static slice suffices.
            MPSGraphTensor* last =
                [g sliceTensor:x dimension:0 start:P - 1 length:1
                          name:nil];
            MPSGraphTensor* fn = rmsnorm(last, pg.final_norm);
            MPSGraphTensor* lg = MatMulAcc(g, fn, pg.head, impl.fp16);
            pg.logits = impl.fp16 ? [g castTensor:lg
                                           toType:MPSDataTypeFloat32
                                             name:nil]
                                  : lg;
        }

        if (pg.feeds == nil) {
            NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
            feeds[pg.x] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.pf_x
                            shape:Shape2(P, h)
                         dataType:impl.dt];
            feeds[pg.cos_in] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.pf_cos
                            shape:Shape2(P, half)
                         dataType:impl.dt];
            feeds[pg.sin_in] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.pf_sin
                            shape:Shape2(P, half)
                         dataType:impl.dt];
            feeds[pg.mask] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:impl.pf_mask
                            shape:Shape2(P, rows)
                         dataType:impl.dt];
            NSMutableDictionary* results = [NSMutableDictionary dictionary];
            for (uint32_t l = 0; l < c.num_layers; ++l) {
                Impl::Layer& lw = impl.layers[l];
                Impl::PrefillGraph::LayerIo& io = pg.layer_io[l];
                feeds[io.input_norm] = lw.input_norm_td;
                feeds[io.post_norm] = lw.post_norm_td;
                feeds[io.qw] = lw.qw_td;
                feeds[io.kw] = lw.kw_td;
                feeds[io.vw] = lw.vw_td;
                feeds[io.ow] = lw.ow_td;
                feeds[io.gate] = lw.gate_td;
                feeds[io.up] = lw.up_td;
                feeds[io.down] = lw.down_td;
                feeds[io.qb] = lw.qb_td;
                feeds[io.kb] = lw.kb_td;
                feeds[io.vb] = lw.vb_td;
                results[io.new_k] = impl.pf_stage_k_td[l];
                results[io.new_v] = impl.pf_stage_v_td[l];
            }
            feeds[pg.final_norm] = impl.final_norm_td;
            feeds[pg.head] = impl.head_td;
            results[pg.logits] = impl.pf_logits_td;
            pg.feeds = feeds;
            pg.results = results;
        }

        // Inputs: embeddings, angles, causal mask (row stride = rows).
        const float neg_inf = impl.fp16 ? -60000.0f : -1e30f;
        if (impl.fp16) {
            uint16_t* xp = static_cast<uint16_t*>(impl.pf_x.contents);
            for (uint32_t i = 0; i < P; ++i) {
                std::memcpy(xp + size_t(i) * h,
                            impl.embed_fp16.data() + size_t(tokens[i]) * h,
                            h * sizeof(uint16_t));
            }
            uint16_t* cos_p = static_cast<uint16_t*>(impl.pf_cos.contents);
            uint16_t* sin_p = static_cast<uint16_t*>(impl.pf_sin.contents);
            for (uint32_t i = 0; i < P; ++i) {
                for (uint32_t j = 0; j < half; ++j) {
                    const float freq = std::pow(
                        c.rope_theta, -2.0f * float(j) / float(c.head_dim));
                    cos_p[i * half + j] =
                        FloatToFp16(std::cos(float(pos0 + i) * freq));
                    sin_p[i * half + j] =
                        FloatToFp16(std::sin(float(pos0 + i) * freq));
                }
            }
            uint16_t* mask_p = static_cast<uint16_t*>(impl.pf_mask.contents);
            const uint16_t zero = FloatToFp16(0.0f);
            const uint16_t ninf = FloatToFp16(neg_inf);
            for (uint32_t i = 0; i < P; ++i) {
                uint16_t* row = mask_p + size_t(i) * rows;
                for (uint32_t t = 0; t < bucket; ++t) {
                    row[t] = t < pos0 ? zero : ninf;
                }
                for (uint32_t j = 0; j < P; ++j) {
                    row[bucket + j] = j <= i ? zero : ninf;
                }
            }
        } else {
            float* xp = static_cast<float*>(impl.pf_x.contents);
            for (uint32_t i = 0; i < P; ++i) {
                std::memcpy(xp + size_t(i) * h,
                            impl.embed_host.data() + size_t(tokens[i]) * h,
                            h * sizeof(float));
            }
            float* cos_p = static_cast<float*>(impl.pf_cos.contents);
            float* sin_p = static_cast<float*>(impl.pf_sin.contents);
            for (uint32_t i = 0; i < P; ++i) {
                for (uint32_t j = 0; j < half; ++j) {
                    const float freq = std::pow(
                        c.rope_theta, -2.0f * float(j) / float(c.head_dim));
                    cos_p[i * half + j] = std::cos(float(pos0 + i) * freq);
                    sin_p[i * half + j] = std::sin(float(pos0 + i) * freq);
                }
            }
            float* mask_p = static_cast<float*>(impl.pf_mask.contents);
            for (uint32_t i = 0; i < P; ++i) {
                float* row = mask_p + size_t(i) * rows;
                for (uint32_t t = 0; t < bucket; ++t) {
                    row[t] = t < pos0 ? 0.0f : -1e30f;
                }
                for (uint32_t j = 0; j < P; ++j) {
                    row[bucket + j] = j <= i ? 0.0f : -1e30f;
                }
            }
        }

        // Distinct TD namespace: sharing TensorData objects between the
        // prefill and decode graphs degrades later decode runs. Per-call
        // feeds copy so completed sequences' KV buffers are not retained.
        NSMutableDictionary* call_feeds = [pg.feeds mutableCopy];
        const auto& kv_tds =
            state.KvTds(bucket | 0x80000000u, c.num_layers, kv_dim);
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            Impl::PrefillGraph::LayerIo& io = pg.layer_io[l];
            call_feeds[io.k_cache] = kv_tds[size_t(l) * 2];
            call_feeds[io.v_cache] = kv_tds[size_t(l) * 2 + 1];
        }

        [pg.graph runWithMTLCommandQueue:impl.pf_queue
                                   feeds:call_feeds
                        targetOperations:nil
                       resultsDictionary:pg.results];

        for (uint32_t l = 0; l < c.num_layers; ++l) {
            std::memcpy(state.KeyRow(l, pos0),
                        impl.pf_stage_k[l].contents,
                        size_t(P) * kv_dim * impl.esz);
            std::memcpy(state.ValueRow(l, pos0),
                        impl.pf_stage_v[l].contents,
                        size_t(P) * kv_dim * impl.esz);
        }
        if (want_logits) {
            const float* lp =
                static_cast<const float*>(impl.pf_logits.contents);
            logits_out.assign(lp, lp + c.vocab_size);
            for (float v : logits_out) {
                if (!std::isfinite(v)) {
                    return Status(ErrorCode::kInferenceFailed,
                                  "logits contain non-finite values",
                                  kComponent);
                }
            }
        }
        return Status::Ok();
    }
}

Status QwenMetalModel::Prefill(SequenceState& state,
                               const std::vector<uint32_t>& tokens,
                               std::vector<float>& logits) {
    auto& seq = static_cast<MetalSequenceState&>(state);
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
    const uint32_t n = uint32_t(tokens.size());
    if (n >= kPrefillChunk) {
        // Full chunks, then an overlapping tail chunk that re-derives the
        // last (n mod P) tokens' K/V (bit-identical rewrite) so the last
        // real token is always the final row.
        const uint32_t full = (n / kPrefillChunk) * kPrefillChunk;
        for (uint32_t pos0 = 0; pos0 < full; pos0 += kPrefillChunk) {
            const bool last_chunk = pos0 + kPrefillChunk == n;
            Status s = ForwardChunk(tokens.data() + pos0, pos0, &seq,
                                    logits, last_chunk);
            if (!s.ok()) return s;
            for (uint32_t i = 0; i < kPrefillChunk; ++i) seq.Advance();
        }
        if (full < n) {
            const uint32_t pos0 = n - kPrefillChunk;
            Status s = ForwardChunk(tokens.data() + pos0, pos0, &seq,
                                    logits, true);
            if (!s.ok()) return s;
            for (uint32_t i = full; i < n; ++i) seq.Advance();
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

Status QwenMetalModel::Decode(SequenceState& state, uint32_t token,
                              std::vector<float>& logits) {
    auto& seq = static_cast<MetalSequenceState&>(state);
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
