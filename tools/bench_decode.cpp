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

using namespace lykuro::nie;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: bench_decode <artifact_dir> cpu|cuda[:N] "
                     "[decode_steps]\n");
        return 2;
    }
    const std::string backend = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 128;

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
    if (backend.rfind("cuda", 0) == 0) {
#ifdef LYKURO_HAVE_CUDA
        int device = backend.size() > 5 ? std::atoi(backend.c_str() + 5) : 0;
        auto cuda = QwenCudaModel::Load(loaded.artifact.manifest,
                                        *loaded.artifact.weights, device);
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

    std::vector<uint32_t> prompt;
    Status s = QwenChatTemplate::BuildPrompt(
        *loaded.artifact.tokenizer,
        {{Role::kUser, "Write a short story about a lighthouse keeper."}},
        prompt);
    if (!s.ok()) {
        std::fprintf(stderr, "prompt build failed\n");
        return 1;
    }

    std::unique_ptr<SequenceState> state;
    s = model->CreateSequence(uint32_t(prompt.size()) + steps + 8, state);
    if (!s.ok()) {
        std::fprintf(stderr, "sequence alloc failed: %s\n",
                     s.message().c_str());
        return 1;
    }

    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    s = model->Prefill(*state, prompt, logits);
    if (!s.ok()) {
        std::fprintf(stderr, "prefill failed: %s\n", s.message().c_str());
        return 1;
    }
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler sampler(greedy);
    uint32_t token = 0;
    if (!sampler.Sample(logits, token).ok()) return 1;
    auto t1 = std::chrono::steady_clock::now();

    int produced = 0;
    for (int i = 0; i < steps; ++i) {
        if (!model->Decode(*state, token, logits).ok()) break;
        if (!sampler.Sample(logits, token).ok()) break;
        ++produced;
    }
    auto t2 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
                   .count() /
               1000.0;
    };
    const double decode_ms = ms(t1, t2);
    std::printf(
        "backend=%s load_ms=%.0f prompt_tokens=%zu ttft_ms=%.1f "
        "decode_tokens=%d decode_ms=%.1f tokens_per_s=%.1f\n",
        backend.c_str(), ms(t_load0, t_load1), prompt.size(), ms(t0, t1),
        produced, decode_ms, produced * 1000.0 / decode_ms);
    return 0;
}
