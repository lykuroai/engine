// Decode benchmark (spec §25.3 subset): TTFT (prefill + first sample)
// and steady-state decode tokens/second for one sequence.
//
//   bench_decode <artifact_dir> cpu|cuda[:N] [decode_steps]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/generation/sampler.h"
#include "model/loader/artifact_loader.h"
#include "model/tokenizer/prompt_template.h"
#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
#endif
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/qwen_metal_model.h"
#endif

using namespace lykuro::nie;

#ifdef LYKURO_HAVE_CUDA
// Parses "cuda[-int8|-int4][:device]".
static bool ParseCudaBackend(const std::string& backend,
                             lykuro::nie::CudaModelOptions& options) {
    if (backend.rfind("cuda", 0) != 0) return false;
    std::string rest = backend.substr(4);
    if (rest.rfind("-int8", 0) == 0) {
        options.quantization = lykuro::nie::WeightQuant::kInt8;
        rest = rest.substr(5);
    } else if (rest.rfind("-int4", 0) == 0) {
        options.quantization = lykuro::nie::WeightQuant::kInt4;
        rest = rest.substr(5);
    }
    if (!rest.empty() && rest[0] == ':') {
        options.device_id = std::atoi(rest.c_str() + 1);
    }
    return true;
}
#endif


int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: bench_decode <artifact_dir> cpu|cuda[:N] "
                     "[decode_steps]\n");
        return 2;
    }
    const std::string backend = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 128;
    const int prompt_repeat = argc > 4 ? std::atoi(argv[4]) : 1;
    const int batch = argc > 5 ? std::atoi(argv[5]) : 1;

    ArtifactLoadOptions options;
    options.allow_unsigned_dev = true;
    auto t_load0 = std::chrono::steady_clock::now();
    ArtifactLoadResult loaded = LoadArtifact(argv[1], options);
    if (!loaded.status.ok()) {
        std::fprintf(stderr, "load failed: %s\n",
                     loaded.status.message().c_str());
        return 1;
    }
    std::unique_ptr<GenerativeModel> model = std::move(loaded.artifact.model);
    if (backend == "metal") {
#ifdef LYKURO_HAVE_METAL
        auto metal = QwenMetalModel::Load(loaded.artifact.manifest,
                                          *loaded.artifact.weights);
        if (!metal.status.ok()) {
            std::fprintf(stderr, "metal load failed: %s\n",
                         metal.status.message().c_str());
            return 1;
        }
        model = std::move(metal.model);
#else
        std::fprintf(stderr, "no Metal in this build\n");
        return 1;
#endif
    } else if (backend.rfind("cuda", 0) == 0) {
#ifdef LYKURO_HAVE_CUDA
        CudaModelOptions cuda_options;
        if (!ParseCudaBackend(backend, cuda_options)) {
            std::fprintf(stderr, "bad backend spec\n");
            return 2;
        }
        auto cuda = QwenCudaModel::Load(loaded.artifact.manifest,
                                        *loaded.artifact.weights,
                                        cuda_options);
        if (!cuda.status.ok()) {
            std::fprintf(stderr, "cuda load failed: %s\n",
                         cuda.status.message().c_str());
            return 1;
        }
        model = std::move(cuda.model);
#else
        std::fprintf(stderr, "no CUDA in this build\n");
        return 1;
#endif
    }
    auto t_load1 = std::chrono::steady_clock::now();

    std::string user_text;
    for (int i = 0; i < prompt_repeat; ++i) {
        user_text += "Write a short story about a lighthouse keeper. ";
    }
    std::vector<uint32_t> prompt;
    Status s = QwenChatTemplate::BuildPrompt(
        *loaded.artifact.tokenizer, {{Role::kUser, user_text}}, prompt);
    if (!s.ok()) {
        std::fprintf(stderr, "prompt build failed\n");
        return 1;
    }

    std::vector<std::unique_ptr<SequenceState>> states(batch);
    std::vector<std::vector<float>> logits(batch);
    std::vector<uint32_t> tokens(batch);
    for (int i = 0; i < batch; ++i) {
        s = model->CreateSequence(uint32_t(prompt.size()) + steps + 8,
                                  states[i]);
        if (!s.ok()) {
            std::fprintf(stderr, "sequence alloc failed: %s\n",
                         s.message().c_str());
            return 1;
        }
    }

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    std::vector<Sampler> samplers(batch, Sampler(greedy));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < batch; ++i) {
        s = model->Prefill(*states[i], prompt, logits[i]);
        if (!s.ok()) {
            std::fprintf(stderr, "prefill failed: %s\n",
                         s.message().c_str());
            return 1;
        }
        if (!samplers[i].Sample(logits[i], tokens[i]).ok()) return 1;
    }
    auto t1 = std::chrono::steady_clock::now();

    int produced = 0;
    for (int step = 0; step < steps; ++step) {
        std::vector<GenerativeModel::DecodeBatchItem> items;
        for (int i = 0; i < batch; ++i) {
            items.push_back({states[i].get(), tokens[i], &logits[i]});
        }
        std::vector<Status> per_item;
        if (!model->DecodeBatch(items, per_item).ok()) break;
        bool all_ok = true;
        for (int i = 0; i < batch; ++i) {
            if (!per_item[i].ok() ||
                !samplers[i].Sample(logits[i], tokens[i]).ok()) {
                all_ok = false;
            }
        }
        if (!all_ok) break;
        produced += batch;
    }
    auto t2 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
                   .count() /
               1000.0;
    };
    const double decode_ms = ms(t1, t2);
    std::printf(
        "backend=%s batch=%d load_ms=%.0f prompt_tokens=%zu ttft_ms=%.1f "
        "decode_tokens=%d decode_ms=%.1f tokens_per_s=%.1f\n",
        backend.c_str(), batch, ms(t_load0, t_load1), prompt.size(),
        ms(t0, t1) / batch, produced, decode_ms,
        produced * 1000.0 / decode_ms);
    return 0;
}
