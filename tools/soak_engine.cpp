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

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "core/engine/engine.h"
#include "model/loader/artifact_loader.h"
#ifdef LYKURO_HAVE_CUDA
#include <cuda_runtime.h>

#include "backends/cuda/qwen_cuda_model.h"

static bool ParseCudaBackend(const std::string& backend,  // NOLINT
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
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/metal_backend.h"
#include "backends/metal/qwen_metal_model.h"
#endif

using namespace lykuro::nie;

namespace {

// Current process resident/footprint in bytes. On macOS this is
// phys_footprint (the true unified-memory leak signal, and unlike
// ru_maxrss it can decrease); on Linux ru_maxrss (KiB -> bytes).
uint64_t ProcessMemoryBytes() {
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return info.phys_footprint;
    }
    return 0;
#else
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return uint64_t(usage.ru_maxrss) * 1024;  // KiB -> bytes
#endif
}

uint64_t FreeVram() {
#ifdef LYKURO_HAVE_CUDA
    size_t free_bytes = 0, total = 0;
    if (cudaMemGetInfo(&free_bytes, &total) == cudaSuccess) {
        return free_bytes;
    }
#endif
#ifdef LYKURO_HAVE_METAL
    // Unified memory: report device-resident allocation (drift, not free).
    MetalDeviceInfo info;
    if (InspectMetalDevice(info).ok()) {
        // Negate so "drift = start - end" reads as growth like the CUDA
        // free-memory convention.
        return UINT64_MAX - info.current_allocated_bytes;
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
#ifdef LYKURO_HAVE_METAL
    if (backend == "metal" || backend == "metal-fp16") {
        MetalModelOptions options;
        options.fp16 = (backend == "metal-fp16");
        auto metal = QwenMetalModel::Load(artifact.manifest,
                                          *artifact.weights, options);
        if (!metal.status.ok()) {
            std::fprintf(stderr, "metal load failed: %s\n",
                         metal.status.message().c_str());
            std::exit(1);
        }
        return std::move(metal.model);
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

        uint64_t submitted = 0, completed = 0, cancelled = 0, failed = 0,
                 tokens = 0, rejected = 0;
        std::vector<std::shared_ptr<EventChannel>> inflight;
        std::vector<std::string> inflight_ids;

        // Baseline is captured after a warm-up window so the reported
        // drift is steady-state growth, not the one-time allocator/graph
        // working-set ramp (MPSGraph caches, model residency).
        const int warmup_s = std::min(seconds / 10, 15);
        uint64_t vram_start = 0, mem_start = 0;
        long rss_start = 0;
        bool baselined = false;

        const auto start_time = std::chrono::steady_clock::now();
        const auto deadline = start_time + std::chrono::seconds(seconds);
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
            if (!baselined &&
                std::chrono::steady_clock::now() - start_time >=
                    std::chrono::seconds(warmup_s)) {
                vram_start = FreeVram();
                mem_start = ProcessMemoryBytes();
                rss_start = long(mem_start / 1024);
                baselined = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!baselined) {  // very short runs
            vram_start = FreeVram();
            mem_start = ProcessMemoryBytes();
            rss_start = long(mem_start / 1024);
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
        const long rss_end = long(ProcessMemoryBytes() / 1024);
        const uint64_t mem_end = ProcessMemoryBytes();
        std::printf(
            "soak: seconds=%d submitted=%llu completed=%llu "
            "cancelled=%llu failed=%llu rejected=%llu output_tokens=%llu "
            "tokens_per_s=%.1f\n",
            seconds, (unsigned long long)submitted,
            (unsigned long long)completed, (unsigned long long)cancelled,
            (unsigned long long)failed, (unsigned long long)rejected,
            (unsigned long long)tokens, double(tokens) / seconds);
        // Process footprint drift is the meaningful signal on unified
        // memory; CUDA free-VRAM drift is reported alongside on Linux.
        std::printf(
            "soak-memory: footprint_start_mb=%.1f footprint_end_mb=%.1f "
            "footprint_drift_mb=%.1f vram_free_start=%llu vram_free_end=%llu "
            "rss_start_kb=%ld rss_end_kb=%ld\n",
            double(mem_start) / 1048576.0, double(mem_end) / 1048576.0,
            (double(mem_end) - double(mem_start)) / 1048576.0,
            (unsigned long long)vram_start, (unsigned long long)vram_end,
            rss_start, rss_end);
    }

    // ---- Phase 2: unload/reload cycles with a memory leak check ----
    // On unified-memory hosts (macOS) the gate is process footprint;
    // elsewhere it is CUDA free-VRAM return-to-baseline.
    if (reload_cycles > 0) {
        uint64_t baseline = 0, foot_baseline = 0;
        for (int cycle = 0; cycle < reload_cycles; ++cycle) {
            // Baseline after the first cycle: the initial cycle warms the
            // process-wide MPSGraph pipeline cache, which must not be
            // charged as a per-cycle leak.
            if (cycle == 1) {
                baseline = FreeVram();
                foot_baseline = ProcessMemoryBytes();
            }
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
        const uint64_t foot_after = ProcessMemoryBytes();
#ifdef __APPLE__
        // Unified memory: footprint must return near baseline. A 100-cycle
        // reload allocator warm-up settles within a small band.
        const double leak_mb =
            (double(foot_after) - double(foot_baseline)) / 1048576.0;
        const char* metric = "footprint";
#else
        const double leak_mb =
            (double(baseline) - double(after)) / 1048576.0;
        const char* metric = "vram_free";
        (void)foot_baseline;
        (void)foot_after;
#endif
        std::printf(
            "reload: cycles=%d metric=%s baseline_mb=%.1f after_mb=%.1f "
            "leak_mb=%.1f %s\n",
            reload_cycles, metric,
#ifdef __APPLE__
            double(foot_baseline) / 1048576.0,
            double(foot_after) / 1048576.0,
#else
            double(baseline) / 1048576.0, double(after) / 1048576.0,
#endif
            leak_mb, leak_mb < 128.0 ? "OK" : "LEAK-SUSPECT");
        (void)baseline;
        (void)after;
        if (leak_mb >= 128.0) return 1;
    }
    return 0;
}
