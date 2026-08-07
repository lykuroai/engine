// Golden verification against an external reference oracle (spec §30.2).
//
//   verify_reference <artifact_dir> <reference.json> cpu|cuda[:N]
//
// Loads the artifact (dev mode), replays each reference case's prompt
// token ids, and reports:
//   - max |logit diff| on the oracle's logits snapshot (first 64 + argmax)
//   - greedy-token agreement over the 32-token continuation
// Exit code 0 when every case matches within tolerance.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/engine/json.h"
#include "core/generation/sampler.h"
#include "model/loader/artifact_loader.h"
#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
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


namespace {

constexpr float kLogitTolerance = 2e-2f;  // FP32, differing op order

std::string ReadAll(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: verify_reference <artifact_dir> "
                     "<reference.json> cpu|cuda[:N]\n");
        return 2;
    }
    const std::string artifact_dir = argv[1];
    const std::string backend = argv[3];

    ArtifactLoadOptions options;
    options.allow_unsigned_dev = true;
    ArtifactLoadResult loaded = LoadArtifact(artifact_dir, options);
    if (!loaded.status.ok()) {
        std::fprintf(stderr, "artifact load failed: %s\n",
                     loaded.status.message().c_str());
        return 1;
    }

    std::unique_ptr<GenerativeModel> model = std::move(loaded.artifact.model);
    if (backend.rfind("cuda", 0) == 0) {
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

    json::ParseResult ref = json::Parse(ReadAll(argv[2]));
    if (!ref.ok() || ref.value->Find("cases") == nullptr) {
        std::fprintf(stderr, "reference json invalid\n");
        return 1;
    }

    int failures = 0;
    for (const auto& case_value : ref.value->Find("cases")->as_array()) {
        const json::Value& c = *case_value;
        std::vector<uint32_t> prompt;
        for (const auto& id : c.Find("prompt_ids")->as_array()) {
            prompt.push_back(uint32_t(id->as_int()));
        }
        const auto& expected_head = c.Find("logits_head64")->as_array();
        const uint32_t expected_argmax =
            uint32_t(c.Find("logits_argmax")->as_int());
        const float expected_max = float(c.Find("logits_max")->as_double());

        std::unique_ptr<SequenceState> state;
        Status s = model->CreateSequence(uint32_t(prompt.size()) + 40, state);
        std::vector<float> logits;
        if (s.ok()) s = model->Prefill(*state, prompt, logits);
        if (!s.ok()) {
            std::fprintf(stderr, "prefill failed: %s\n",
                         s.message().c_str());
            return 1;
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < expected_head.size(); ++i) {
            max_diff = std::max(
                max_diff, std::abs(logits[i] -
                                   float(expected_head[i]->as_double())));
        }
        uint32_t argmax = uint32_t(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        max_diff = std::max(max_diff,
                            std::abs(logits[argmax] - expected_max));

        // Greedy continuation.
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        Sampler sampler(greedy);
        const auto& expected_greedy = c.Find("greedy_ids")->as_array();
        size_t agree = 0;
        bool diverged = false;
        for (size_t step = 0; step < expected_greedy.size(); ++step) {
            uint32_t token = 0;
            if (!sampler.Sample(logits, token).ok()) break;
            const uint32_t expected_token =
                uint32_t(expected_greedy[step]->as_int());
            if (!diverged && token == expected_token) {
                ++agree;
            } else {
                diverged = true;
                break;
            }
            if (!model->Decode(*state, token, logits).ok()) break;
        }

        const bool logits_ok = max_diff <= kLogitTolerance;
        const bool argmax_ok = argmax == expected_argmax;
        const bool greedy_ok = agree == expected_greedy.size();
        if (!logits_ok || !argmax_ok || !greedy_ok) ++failures;
        std::printf(
            "case \"%s\": logits_max_diff=%.5f (%s) argmax=%u/%u (%s) "
            "greedy=%zu/%zu (%s)\n",
            c.Find("prompt")->as_string().c_str(), double(max_diff),
            logits_ok ? "OK" : "FAIL", argmax, expected_argmax,
            argmax_ok ? "OK" : "FAIL", agree, expected_greedy.size(),
            greedy_ok ? "OK" : "FAIL");
    }

    std::printf("%s: %d failing case(s)\n", backend.c_str(), failures);
    return failures == 0 ? 0 : 1;
}
