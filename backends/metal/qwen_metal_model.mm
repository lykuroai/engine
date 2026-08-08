#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "backends/metal/qwen_metal_model.h"

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "qwen_metal";
constexpr uint32_t kCtxBucket = 128;  // context bucket granularity

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

struct QwenMetalModel::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
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
    };
    std::map<uint32_t, BucketGraph> bucket_graphs;

    ~Impl() {
        // ARC releases the ObjC objects.
    }
};

namespace {

// Bounded contiguous KV cache in unified memory (spec MVP §16).
class MetalSequenceState final : public SequenceState {
public:
    static Status Create(id<MTLDevice> device, const QwenConfig& config,
                         uint32_t max_tokens,
                         std::unique_ptr<MetalSequenceState>& out) {
        auto state =
            std::unique_ptr<MetalSequenceState>(new MetalSequenceState());
        state->kv_stride_ = config.num_kv_heads * config.head_dim;
        state->max_tokens_ = max_tokens;
        // Rounded up so a [bucket, kv] view never exceeds the buffer.
        const uint32_t alloc_tokens =
            ((max_tokens + kCtxBucket - 1) / kCtxBucket) * kCtxBucket;
        const size_t bytes =
            size_t(alloc_tokens) * state->kv_stride_ * sizeof(float);
        state->keys_.resize(config.num_layers);
        state->values_.resize(config.num_layers);
        for (uint32_t l = 0; l < config.num_layers; ++l) {
            id<MTLBuffer> k =
                [device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> v =
                [device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
            if (k == nil || v == nil) {
                return Status(ErrorCode::kGpuOom,
                              "kv cache allocation failed", kComponent);
            }
            // Zeroed so masked padding rows can never inject NaN/Inf.
            std::memset(k.contents, 0, bytes);
            std::memset(v.contents, 0, bytes);
            state->keys_[l] = k;
            state->values_[l] = v;
        }
        out = std::move(state);
        return Status::Ok();
    }

    uint32_t length() const override { return length_; }
    uint32_t capacity() const override { return max_tokens_; }
    uint32_t kv_stride() const { return kv_stride_; }

    id<MTLBuffer> key_buffer(uint32_t layer) { return keys_[layer]; }
    id<MTLBuffer> value_buffer(uint32_t layer) { return values_[layer]; }

    float* KeyRow(uint32_t layer, uint32_t t) {
        return static_cast<float*>(keys_[layer].contents) +
               size_t(t) * kv_stride_;
    }
    float* ValueRow(uint32_t layer, uint32_t t) {
        return static_cast<float*>(values_[layer].contents) +
               size_t(t) * kv_stride_;
    }
    void Advance() { ++length_; }

private:
    MetalSequenceState() = default;
    uint32_t kv_stride_ = 0;
    uint32_t max_tokens_ = 0;
    uint32_t length_ = 0;
    std::vector<id<MTLBuffer>> keys_;
    std::vector<id<MTLBuffer>> values_;
};

NSArray<NSNumber*>* Shape2(uint64_t a, uint64_t b) {
    return @[ @(a), @(b) ];
}

}  // namespace

QwenMetalModel::~QwenMetalModel() = default;

QwenMetalModel::LoadResult QwenMetalModel::Load(
    const ModelManifest& manifest, const SafetensorsFile& weights) {
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
        impl.device = MTLCreateSystemDefaultDevice();
        if (impl.device == nil || !impl.device.hasUnifiedMemory) {
            result.status = Status(ErrorCode::kGpuUnhealthy,
                                   "metal device unavailable", kComponent);
            return result;
        }
        impl.queue = [impl.device newCommandQueue];
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
            const uint64_t weight_bytes =
                uint64_t(c.vocab_size) * h64 * sizeof(float) * 2 +
                uint64_t(c.num_layers) * per_layer;
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
        }

        const size_t h = c.hidden_size;
        const size_t q_dim = size_t(c.num_heads) * c.head_dim;
        const size_t kv_dim = size_t(c.num_kv_heads) * c.head_dim;

        auto upload = [&](const std::vector<float>& host,
                          id<MTLBuffer> __strong& buf) -> Status {
            buf = [impl.device
                newBufferWithBytes:host.data()
                            length:host.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
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
                         dataType:MPSDataTypeFloat32];
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
        if (!s.ok()) {
            result.status = s;
            return result;
        }
        result.model = std::move(model);
        return result;
    }
}

Status QwenMetalModel::CreateSequence(uint32_t max_tokens,
                                      std::unique_ptr<SequenceState>& out) {
    std::unique_ptr<MetalSequenceState> state;
    Status s = MetalSequenceState::Create(impl_->device, config_,
                                          max_tokens, state);
    if (!s.ok()) return s;
    out = std::move(state);
    return Status::Ok();
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

        // Build (or reuse) the bucket-specialized graph.
        Impl::BucketGraph& bg = impl.bucket_graphs[bucket];
        if (bg.graph == nil) {
            MPSGraph* g = [[MPSGraph alloc] init];
            bg.graph = g;
            const float attn_scale = 1.0f / std::sqrt(float(c.head_dim));
            bg.x = [g placeholderWithShape:Shape2(1, h)
                                  dataType:MPSDataTypeFloat32
                                      name:nil];
            bg.cos_in = [g placeholderWithShape:Shape2(1, half)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
            bg.sin_in = [g placeholderWithShape:Shape2(1, half)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
            bg.mask = [g placeholderWithShape:Shape2(1, bucket + 1)
                                     dataType:MPSDataTypeFloat32
                                         name:nil];
            bg.layer_io.resize(c.num_layers);

            auto rmsnorm = [&](MPSGraphTensor* x, MPSGraphTensor* w) {
                MPSGraphTensor* sq = [g squareWithTensor:x name:nil];
                MPSGraphTensor* mean =
                    [g meanOfTensor:sq axes:@[ @1 ] name:nil];
                MPSGraphTensor* eps =
                    [g constantWithScalar:double(c.rms_norm_eps)
                                    shape:Shape2(1, 1)
                                 dataType:MPSDataTypeFloat32];
                MPSGraphTensor* rs = [g
                    reverseSquareRootWithTensor:[g additionWithPrimaryTensor:mean
                                                          secondaryTensor:eps
                                                                     name:nil]
                                           name:nil];
                MPSGraphTensor* n =
                    [g multiplicationWithPrimaryTensor:x
                                       secondaryTensor:rs
                                                  name:nil];
                return [g multiplicationWithPrimaryTensor:n
                                          secondaryTensor:w
                                                     name:nil];
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
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
                io.post_norm = [g placeholderWithShape:Shape2(1, h)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
                io.qw = [g
                    placeholderWithShape:Shape2(h, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.kw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
                io.vw = [g placeholderWithShape:Shape2(h, kv_dim)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
                io.ow = [g
                    placeholderWithShape:Shape2(size_t(c.num_heads) *
                                                    c.head_dim,
                                                h)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.gate = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.up = [g
                    placeholderWithShape:Shape2(h, c.intermediate_size)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.down = [g
                    placeholderWithShape:Shape2(c.intermediate_size, h)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.qb = [g
                    placeholderWithShape:Shape2(1, size_t(c.num_heads) *
                                                       c.head_dim)
                                dataType:MPSDataTypeFloat32
                                    name:nil];
                io.kb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
                io.vb = [g placeholderWithShape:Shape2(1, kv_dim)
                                       dataType:MPSDataTypeFloat32
                                           name:nil];
                io.k_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
                io.v_cache = [g placeholderWithShape:Shape2(bucket, kv_dim)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];

                MPSGraphTensor* normed = rmsnorm(x, io.input_norm);
                auto proj = [&](MPSGraphTensor* w, MPSGraphTensor* bias) {
                    MPSGraphTensor* y =
                        [g matrixMultiplicationWithPrimaryTensor:normed
                                                 secondaryTensor:w
                                                            name:nil];
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
                MPSGraphTensor* scores =
                    [g matrixMultiplicationWithPrimaryTensor:q_h
                                             secondaryTensor:k_t
                                                        name:nil];
                MPSGraphTensor* scale =
                    [g constantWithScalar:double(attn_scale)
                                    shape:@[ @1, @1, @1 ]
                                 dataType:MPSDataTypeFloat32];
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
                MPSGraphTensor* ctx =
                    [g matrixMultiplicationWithPrimaryTensor:probs
                                             secondaryTensor:v_h
                                                        name:nil];
                MPSGraphTensor* ctx_flat = [g
                    reshapeTensor:ctx
                        withShape:Shape2(1, size_t(c.num_heads) *
                                                c.head_dim)
                             name:nil];
                MPSGraphTensor* o =
                    [g matrixMultiplicationWithPrimaryTensor:ctx_flat
                                             secondaryTensor:io.ow
                                                        name:nil];
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:o
                                            name:nil];

                MPSGraphTensor* normed2 = rmsnorm(x, io.post_norm);
                MPSGraphTensor* gate =
                    [g matrixMultiplicationWithPrimaryTensor:normed2
                                             secondaryTensor:io.gate
                                                        name:nil];
                MPSGraphTensor* up =
                    [g matrixMultiplicationWithPrimaryTensor:normed2
                                             secondaryTensor:io.up
                                                        name:nil];
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
                    [g matrixMultiplicationWithPrimaryTensor:act
                                             secondaryTensor:io.down
                                                        name:nil];
                x = [g additionWithPrimaryTensor:x
                                 secondaryTensor:down
                                            name:nil];
            }
            bg.final_norm = [g placeholderWithShape:Shape2(1, h)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
            bg.head = [g placeholderWithShape:Shape2(h, c.vocab_size)
                                     dataType:MPSDataTypeFloat32
                                         name:nil];
            MPSGraphTensor* fn = rmsnorm(x, bg.final_norm);
            bg.logits =
                [g matrixMultiplicationWithPrimaryTensor:fn
                                         secondaryTensor:bg.head
                                                    name:nil];
        }

        // --- feeds ---
        NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
        // Hidden = embedding row (host lookup; unified memory).
        std::vector<float> x_host(impl.embed_host.begin() +
                                      size_t(token) * h,
                                  impl.embed_host.begin() +
                                      size_t(token) * h + h);
        feeds[bg.x] = [[MPSGraphTensorData alloc]
            initWithDevice:impl.graph_device
                      data:[NSData dataWithBytes:x_host.data()
                                          length:h * sizeof(float)]
                     shape:Shape2(1, h)
                  dataType:MPSDataTypeFloat32];
        // RoPE angles for this position.
        std::vector<float> cos_host(half), sin_host(half);
        for (uint32_t i = 0; i < half; ++i) {
            const float freq =
                std::pow(c.rope_theta, -2.0f * float(i) / float(c.head_dim));
            cos_host[i] = std::cos(float(pos) * freq);
            sin_host[i] = std::sin(float(pos) * freq);
        }
        feeds[bg.cos_in] = [[MPSGraphTensorData alloc]
            initWithDevice:impl.graph_device
                      data:[NSData dataWithBytes:cos_host.data()
                                          length:half * sizeof(float)]
                     shape:Shape2(1, half)
                  dataType:MPSDataTypeFloat32];
        feeds[bg.sin_in] = [[MPSGraphTensorData alloc]
            initWithDevice:impl.graph_device
                      data:[NSData dataWithBytes:sin_host.data()
                                          length:half * sizeof(float)]
                     shape:Shape2(1, half)
                  dataType:MPSDataTypeFloat32];
        // Additive mask: cached rows [0, pos) and the current token are
        // visible; padding rows are -1e30.
        std::vector<float> mask_host(bucket + 1, -1e30f);
        for (uint32_t t = 0; t < pos; ++t) mask_host[t] = 0.0f;
        mask_host[bucket] = 0.0f;
        feeds[bg.mask] = [[MPSGraphTensorData alloc]
            initWithDevice:impl.graph_device
                      data:[NSData dataWithBytes:mask_host.data()
                                          length:(bucket + 1) *
                                                 sizeof(float)]
                     shape:Shape2(1, bucket + 1)
                  dataType:MPSDataTypeFloat32];

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
            feeds[io.k_cache] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:state.key_buffer(l)
                            shape:Shape2(bucket, kv_dim)
                         dataType:MPSDataTypeFloat32];
            feeds[io.v_cache] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:state.value_buffer(l)
                            shape:Shape2(bucket, kv_dim)
                         dataType:MPSDataTypeFloat32];
            [targets addObject:io.new_k];
            [targets addObject:io.new_v];
        }
        feeds[bg.final_norm] = impl.final_norm_td;
        feeds[bg.head] = impl.head_td;
        if (want_logits) [targets addObject:bg.logits];

        NSDictionary* results =
            [bg.graph runWithMTLCommandQueue:impl.queue
                                       feeds:feeds
                               targetTensors:targets
                            targetOperations:nil];
        if (results == nil) {
            return MetalFailed("graph execution failed");
        }

        // Write the new K/V rows into the cache (unified memory).
        for (uint32_t l = 0; l < c.num_layers; ++l) {
            Impl::BucketGraph::LayerIo& io = bg.layer_io[l];
            MPSGraphTensorData* kd = results[io.new_k];
            MPSGraphTensorData* vd = results[io.new_v];
            if (kd == nil || vd == nil) {
                return MetalFailed("kv outputs missing");
            }
            [[kd mpsndarray] readBytes:state.KeyRow(l, pos)
                           strideBytes:nil];
            [[vd mpsndarray] readBytes:state.ValueRow(l, pos)
                           strideBytes:nil];
        }
        if (want_logits) {
            MPSGraphTensorData* ld = results[bg.logits];
            if (ld == nil) return MetalFailed("logits output missing");
            logits_out.resize(c.vocab_size);
            [[ld mpsndarray] readBytes:logits_out.data() strideBytes:nil];
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
