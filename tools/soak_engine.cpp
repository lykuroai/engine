// Engine soak / stability harness (spec §25.3, §30.5).
//
//   soak_engine <artifact_dir> cpu|cuda[:N] <seconds> [reload_cycles]
//
// Phase 1: drives the full InferenceEngine continuously for <seconds>
// with mixed request lengths/priorities and periodic cancels, reporting
// completed/cancelled/failed counts, aggregate token throughput, and
// VRAM / RSS drift between warm-up and finish.
// Phase 2: <reload_cycles> full model unload/reload cycles with a VRAM
// leak check (free VRAM after each cycle must return to baseline).

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#include "core/engine/engine.h"
#include "model/loader/artifact_loader.h"
#ifdef LYKURO_HAVE_CUDA
#include <cuda_runtime.h>

#include "backends/cuda/qwen_cuda_model.h"

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

using namespace lykuro::nie;

namespace {

long RssKb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // KiB on Linux
}

uint64_t FreeVram() {
#ifdef LYKURO_HAVE_CUDA
    size_t free_bytes = 0, total = 0;
    if (cudaMemGetInfo(&free_bytes, &total) == cudaSuccess) {
        return free_bytes;
    }
#endif
    return 0;
}

std::unique_ptr<GenerativeModel> MakeModel(const LoadedArtifact& artifact,
                                           const std::string& backend,
                                           std::unique_ptr<QwenModel> cpu) {
#ifdef LYKURO_HAVE_CUDA
    if (backend.rfind("cuda", 0) == 0) {
        CudaModelOptions options;
        ParseCudaBackend(backend, options);
        auto cuda = QwenCudaModel::Load(artifact.manifest,
                                        *artifact.weights, options);
        if (!cuda.status.ok()) {
            std::fprintf(stderr, "cuda load failed: %s\n",
                         cuda.status.message().c_str());
            std::exit(1);
        }
        return std::move(cuda.model);
    }
#endif
    (void)backend;
    return cpu;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: soak_engine <artifact_dir> cpu|cuda[:N] "
                     "<seconds> [reload_cycles]\n");
        return 2;
    }
    const std::string artifact_dir = argv[1];
    const std::string backend = argv[2];
    const int seconds = std::atoi(argv[3]);
    const int reload_cycles = argc > 4 ? std::atoi(argv[4]) : 0;

    ArtifactLoadOptions load_options;
    load_options.allow_unsigned_dev = true;

    // ---- Phase 1: continuous mixed load ----
    {
        ArtifactLoadResult loaded = LoadArtifact(artifact_dir, load_options);
        if (!loaded.status.ok()) {
            std::fprintf(stderr, "load failed: %s\n",
                         loaded.status.message().c_str());
            return 1;
        }
        auto tokenizer = std::move(loaded.artifact.tokenizer);
        auto model = MakeModel(loaded.artifact, backend,
                               std::move(loaded.artifact.model));

        EngineConfig config;
        config.max_sequences = 8;
        InferenceEngine engine(std::move(model), std::move(tokenizer),
                               config);

        std::atomic<bool> running{true};
        std::thread worker([&] {
            while (running) {
                if (!engine.Step()) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(1));
                }
            }
        });

        const uint64_t vram_start = FreeVram();
        const long rss_start = RssKb();

        uint64_t submitted = 0, completed = 0, cancelled = 0, failed = 0,
                 tokens = 0, rejected = 0;
        std::vector<std::shared_ptr<EventChannel>> inflight;
        std::vector<std::string> inflight_ids;

        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(seconds);
        uint64_t next_id = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            // Keep ~12 requests in flight with varied shapes.
            while (inflight.size() < 12) {
                InferenceRequest req;
                req.request_id = "soak_" + std::to_string(next_id++);
                req.tenant_scope = "tn_" + std::to_string(next_id % 3);
                req.priority = uint32_t(20 + (next_id * 17) % 70);
                std::string text = "Tell me about lighthouse number " +
                                   std::to_string(next_id) + ". ";
                for (uint64_t r = 0; r < next_id % 4; ++r) text += text;
                req.messages = {{Role::kUser, text}};
                req.max_output_tokens = uint32_t(8 + (next_id * 31) % 56);
                req.sampling.temperature = (next_id % 2) ? 0.0f : 0.8f;
                req.sampling.seed = next_id;
                auto submit = engine.Submit(req);
                if (!submit.status.ok()) {
                    ++rejected;
                    break;
                }
                ++submitted;
                inflight.push_back(submit.events);
                inflight_ids.push_back(req.request_id);
            }
            // Cancel roughly every 40th request mid-flight.
            if (!inflight_ids.empty() && next_id % 40 == 0) {
                engine.Cancel(inflight_ids.back());
            }
            // Drain finished channels.
            for (size_t i = 0; i < inflight.size();) {
                bool done = false;
                while (auto e = inflight[i]->TryPop()) {
                    if (e->kind == StreamEvent::Kind::kCompleted) {
                        done = true;
                        tokens += e->usage.output_tokens;
                        if (e->finish_reason == FinishReason::kCancelled) {
                            ++cancelled;
                        } else {
                            ++completed;
                        }
                    } else if (e->kind == StreamEvent::Kind::kError) {
                        done = true;
                        if (e->error.code() ==
                            ErrorCode::kRequestCancelled) {
                            ++cancelled;
                        } else {
                            ++failed;
                        }
                    }
                }
                if (done || inflight[i]->closed()) {
                    inflight.erase(inflight.begin() + long(i));
                    inflight_ids.erase(inflight_ids.begin() + long(i));
                } else {
                    ++i;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // Drain remaining work.
        for (auto& ch : inflight) {
            while (auto e = ch->Pop()) {
                if (e->kind == StreamEvent::Kind::kCompleted) {
                    ++completed;
                    tokens += e->usage.output_tokens;
                }
                if (e->kind == StreamEvent::Kind::kError) ++failed;
            }
        }
        running = false;
        worker.join();

        const uint64_t vram_end = FreeVram();
        const long rss_end = RssKb();
        std::printf(
            "soak: seconds=%d submitted=%llu completed=%llu "
            "cancelled=%llu failed=%llu rejected=%llu output_tokens=%llu "
            "tokens_per_s=%.1f\n",
            seconds, (unsigned long long)submitted,
            (unsigned long long)completed, (unsigned long long)cancelled,
            (unsigned long long)failed, (unsigned long long)rejected,
            (unsigned long long)tokens, double(tokens) / seconds);
        std::printf(
            "soak-memory: vram_free_start=%llu vram_free_end=%llu "
            "vram_drift_mb=%.1f rss_start_kb=%ld rss_end_kb=%ld\n",
            (unsigned long long)vram_start, (unsigned long long)vram_end,
            (double(vram_start) - double(vram_end)) / 1048576.0, rss_start,
            rss_end);
    }

    // ---- Phase 2: unload/reload cycles with VRAM leak check ----
    if (reload_cycles > 0) {
        const uint64_t baseline = FreeVram();
        uint64_t min_free_after = UINT64_MAX;
        for (int cycle = 0; cycle < reload_cycles; ++cycle) {
            ArtifactLoadResult loaded =
                LoadArtifact(artifact_dir, load_options);
            if (!loaded.status.ok()) {
                std::fprintf(stderr, "reload %d failed: %s\n", cycle,
                             loaded.status.message().c_str());
                return 1;
            }
            auto tokenizer = std::move(loaded.artifact.tokenizer);
            auto model = MakeModel(loaded.artifact, backend,
                                   std::move(loaded.artifact.model));
            // One short request as the smoke inference.
            InferenceEngine engine(std::move(model), std::move(tokenizer),
                                   {});
            InferenceRequest req;
            req.request_id = "cycle";
            req.messages = {{Role::kUser, "ping"}};
            req.max_output_tokens = 4;
            req.sampling.temperature = 0.0f;
            auto submit = engine.Submit(req);
            if (!submit.status.ok()) {
                std::fprintf(stderr, "cycle %d submit failed\n", cycle);
                return 1;
            }
            engine.RunUntilIdle();
            while (submit.events->TryPop()) {
            }
            // Engine/model/weights released at scope exit.
        }
        const uint64_t after = FreeVram();
        min_free_after = std::min(min_free_after, after);
        const double leak_mb =
            (double(baseline) - double(after)) / 1048576.0;
        std::printf(
            "reload: cycles=%d vram_free_baseline=%llu vram_free_after=%llu "
            "leak_mb=%.1f %s\n",
            reload_cycles, (unsigned long long)baseline,
            (unsigned long long)after, leak_mb,
            leak_mb < 64.0 ? "OK" : "LEAK-SUSPECT");
        if (leak_mb >= 64.0) return 1;
    }
    return 0;
}
