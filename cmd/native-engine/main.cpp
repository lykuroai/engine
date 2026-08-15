#include <sys/stat.h>

#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cmd/native-engine/generator.h"
#include "core/engine/config.h"
#include "core/generation/sampler.h"
#include "model/architectures/generative_model.h"
#include "model/convert/hf_convert.h"
#include "model/loader/artifact_loader.h"
#include "model/tokenizer/bpe_tokenizer.h"
#include "model/tokenizer/prompt_template.h"
#include "observability/log.h"
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/qwen_metal_fast.h"
#include "backends/metal/qwen_metal_model.h"
#endif
#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
#endif

#ifdef LYKURO_HAVE_GRPC
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "api/server/grpc_server.h"
#endif

namespace lykuro::nie {
int RunHttpServe(int argc, char** argv);  // http_api.cpp
}

namespace {

constexpr const char kVersion[] = "1.0.4";

void PrintUsage() {
    std::printf(
        "lykuro-native-engine %s\n"
        "\n"
        "Usage:\n"
        "  native-engine pull <hf_repo> [out_dir]   download + convert a model\n"
        "  native-engine list                       list local models\n"
        "  native-engine run <model_dir> [\"prompt\"] [options]   "
        "standalone inference (no config)\n"
        "  native-engine serve [--host 0.0.0.0] [--port 11434]  HTTP API "
        "(Ollama + OpenAI)\n"
        "  native-engine serve --config <path>      gRPC engine server\n"
        "  native-engine convert <hf_dir> <out_dir> HF checkpoint -> artifact\n"
        "  native-engine --version | --help\n"
        "\n"
        "run options:\n"
        "  --backend <cpu|metal|metal-fp16|metal-fast|metal-q8|metal-q4|"
        "cuda[:N]|cuda-q8|cuda-q4>  (default: best built)\n"
        "  --max-tokens <N>       (default 512)\n"
        "  --temperature <T>      (default 0 = greedy)\n"
        "  --system \"<text>\"      optional system prompt\n"
        "  --seed <N>             PRNG seed for temperature>0\n"
        "\n"
        "With no \"prompt\", `run` starts an interactive chat (Ctrl-D to "
        "exit).\n",
        kVersion);
}

// Model resolution / pull helpers live in generator.cpp (cli::) and are
// shared with the HTTP API.

// Config-less, single-command standalone inference. Loads a model directory
// and generates text directly — no JSON config, no mTLS, no server.
int RunGenerate(int argc, char** argv) {
    using namespace lykuro::nie;

    std::string dir, prompt, system_prompt;
#if defined(LYKURO_HAVE_METAL)
    std::string backend = "metal-q4";  // fastest local default
#elif defined(LYKURO_HAVE_CUDA)
    std::string backend = "cuda";
#else
    std::string backend = "cpu";
#endif
    int max_tokens = 512;
    float temperature = 0.0f;
    uint64_t seed = 0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* n) -> const char* {
            if (i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "%s needs a value\n", n);
            std::exit(2);
        };
        if (a == "--backend") backend = val("--backend");
        else if (a == "--max-tokens") max_tokens = std::atoi(val("--max-tokens"));
        else if (a == "--temperature") temperature = std::atof(val("--temperature"));
        else if (a == "--system") system_prompt = val("--system");
        else if (a == "--seed") seed = std::strtoull(val("--seed"), nullptr, 10);
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        } else if (dir.empty()) dir = a;
        else if (prompt.empty()) prompt = a;
        else { prompt += " "; prompt += a; }
    }
    if (dir.empty()) {
        PrintUsage();
        return 2;
    }
    // single-command: a HF repo id auto-pulls if not already local.
    if (int rc = cli::ResolveModelArg(dir)) return rc;

    ArtifactLoadOptions opt;
    opt.allow_unsigned_dev = true;  // local eval; no signing config here
    ArtifactLoadResult loaded = LoadArtifact(dir, opt);
    if (!loaded.status.ok()) {
        std::fprintf(stderr, "load failed: %s\n",
                     loaded.status.message().c_str());
        return 1;
    }
    std::unique_ptr<GenerativeModel> model = std::move(loaded.artifact.model);

    if (backend == "metal-fast" || backend == "metal-q8" ||
        backend == "metal-q4") {
#ifdef LYKURO_HAVE_METAL
        MetalFastOptions fo;
        if (backend == "metal-q8") fo.quant = MetalFastOptions::Quant::kInt8;
        if (backend == "metal-q4") fo.quant = MetalFastOptions::Quant::kInt4;
        auto m = QwenMetalFastModel::Load(loaded.artifact.manifest,
                                          *loaded.artifact.weights, fo);
        if (!m.status.ok()) {
            std::fprintf(stderr, "metal load failed: %s\n",
                         m.status.message().c_str());
            return 1;
        }
        model = std::move(m.model);
#else
        std::fprintf(stderr, "no Metal in this build; try --backend cpu\n");
        return 1;
#endif
    } else if (backend == "metal" || backend == "metal-fp16") {
#ifdef LYKURO_HAVE_METAL
        MetalModelOptions mo;
        mo.fp16 = (backend == "metal-fp16");
        auto m = QwenMetalModel::Load(loaded.artifact.manifest,
                                      *loaded.artifact.weights, mo);
        if (!m.status.ok()) {
            std::fprintf(stderr, "metal load failed: %s\n",
                         m.status.message().c_str());
            return 1;
        }
        model = std::move(m.model);
#else
        std::fprintf(stderr, "no Metal in this build; try --backend cpu\n");
        return 1;
#endif
    } else if (backend.rfind("cuda", 0) == 0) {
#ifdef LYKURO_HAVE_CUDA
        CudaModelOptions co;
        std::string cb = backend;
        auto p = cb.find(':');
        if (p != std::string::npos) {
            co.device_id = std::atoi(cb.c_str() + p + 1);
            cb = cb.substr(0, p);
        }
        if (cb == "cuda-q8") co.quantization = WeightQuant::kInt8;
        else if (cb == "cuda-q4") co.quantization = WeightQuant::kInt4;
        else if (cb != "cuda") {
            std::fprintf(stderr, "unknown backend: %s\n", backend.c_str());
            return 1;
        }
        auto c = QwenCudaModel::Load(loaded.artifact.manifest,
                                     *loaded.artifact.weights, co);
        if (!c.status.ok()) {
            std::fprintf(stderr, "cuda load failed: %s\n",
                         c.status.message().c_str());
            return 1;
        }
        model = std::move(c.model);
#else
        std::fprintf(stderr, "no CUDA in this build; try --backend cpu\n");
        return 1;
#endif
    } else if (backend != "cpu") {
        std::fprintf(stderr, "unknown backend: %s\n", backend.c_str());
        return 1;
    }

    const BpeTokenizer& tok = *loaded.artifact.tokenizer;
    const std::vector<uint32_t> eos = loaded.artifact.manifest.eos_token_ids;
    auto is_eos = [&](uint32_t t) {
        for (uint32_t e : eos)
            if (e == t) return true;
        return false;
    };

    std::vector<ChatMessage> history;
    if (!system_prompt.empty())
        history.push_back({Role::kSystem, system_prompt});

    auto generate = [&](const std::string& user) -> int {
        history.push_back({Role::kUser, user});
        std::vector<uint32_t> prompt_ids;
        Status s = QwenChatTemplate::BuildPrompt(tok, history, prompt_ids);
        if (!s.ok()) {
            std::fprintf(stderr, "prompt build failed: %s\n",
                         s.message().c_str());
            return 1;
        }
        std::unique_ptr<SequenceState> st;
        s = model->CreateSequence(
            uint32_t(prompt_ids.size()) + uint32_t(max_tokens) + 8, st);
        if (!s.ok()) {
            std::fprintf(stderr, "sequence alloc failed: %s\n",
                         s.message().c_str());
            return 1;
        }
        std::vector<float> logits;
        s = model->Prefill(*st, prompt_ids, logits);
        if (!s.ok()) {
            std::fprintf(stderr, "prefill failed: %s\n", s.message().c_str());
            return 1;
        }
        SamplingParams sp;
        sp.temperature = temperature;
        sp.seed = seed;
        if (temperature > 0.0f) sp.top_p = 0.95f;
        Sampler sampler(sp);

        uint32_t token = 0;
        if (!sampler.Sample(logits, token).ok()) return 1;
        std::vector<uint32_t> gen;
        std::string printed;
        std::string bytes;              // incremental detokenization
        std::vector<uint32_t> one(1);
        std::vector<uint32_t> pending;  // greedy-run token queue
        size_t pending_i = 0;
        const bool greedy_fast =
            !(temperature > 0.0f) && model->SupportsGreedyRun();
        for (int n = 0; n < max_tokens; ++n) {
            if (is_eos(token)) break;
            gen.push_back(token);
            one[0] = token;
            tok.DecodeBytes(one, bytes);
            if (bytes.size() > printed.size()) {
                std::fwrite(bytes.data() + printed.size(), 1,
                            bytes.size() - printed.size(), stdout);
                std::fflush(stdout);
                printed = bytes;
            }
            if (greedy_fast) {
                if (pending_i >= pending.size()) {
                    pending.clear();
                    pending_i = 0;
                    s = model->GreedyRun(*st, token,
                                         uint32_t(max_tokens - n), pending);
                    if (!s.ok() || pending.empty()) {
                        std::fprintf(stderr, "\ndecode failed: %s\n",
                                     s.message().c_str());
                        return 1;
                    }
                }
                token = pending[pending_i++];
            } else {
                s = model->Decode(*st, token, logits);
                if (!s.ok()) {
                    std::fprintf(stderr, "\ndecode failed: %s\n",
                                 s.message().c_str());
                    return 1;
                }
                if (!sampler.Sample(logits, token).ok()) return 1;
            }
        }
        std::printf("\n");
        std::string reply;
        tok.DecodeText(gen, reply);
        history.push_back({Role::kAssistant, reply});
        return 0;
    };

    if (!prompt.empty()) return generate(prompt);

    std::fprintf(stderr,
                 "lykuro %s — %s backend, model %s. Enter a prompt "
                 "(Ctrl-D to exit).\n",
                 kVersion, backend.c_str(), dir.c_str());
    std::string line;
    while (true) {
        std::fprintf(stderr, ">>> ");
        std::fflush(stderr);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        int rc = generate(line);
        if (rc != 0) return rc;
    }
    std::fprintf(stderr, "\n");
    return 0;
}

// `list` — show locally available models (local model list).
int RunList(int /*argc*/, char** /*argv*/) {
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    const std::string base =
        (home ? std::string(home) : ".") + "/.lykuro/models";
    auto human = [](uintmax_t b) -> std::string {
        char buf[32];
        if (b >= (1ull << 30))
            std::snprintf(buf, sizeof(buf), "%.1f GB", double(b) / (1 << 30));
        else
            std::snprintf(buf, sizeof(buf), "%.0f MB", double(b) / (1 << 20));
        return buf;
    };
    std::printf("%-44s %10s  %s\n", "NAME", "SIZE", "MODIFIED");
    std::error_code ec;
    int n = 0;
    if (fs::exists(base, ec)) {
        std::vector<fs::directory_entry> dirs;
        for (const auto& e : fs::directory_iterator(base, ec))
            if (e.is_directory() && fs::exists(e.path() / "manifest.json"))
                dirs.push_back(e);
        for (const auto& e : dirs) {
            uintmax_t sz = fs::file_size(
                e.path() / "weights" / "model.safetensors", ec);
            if (ec) sz = 0;
            struct stat st{};
            char when[20] = "-";
            if (::stat((e.path() / "manifest.json").c_str(), &st) == 0) {
                std::tm tmv{};
                localtime_r(&st.st_mtime, &tmv);
                std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M", &tmv);
            }
            std::printf("%-44s %10s  %s\n",
                        e.path().filename().string().c_str(),
                        human(sz).c_str(), when);
            ++n;
        }
    }
    if (n == 0)
        std::fprintf(stderr,
                     "no models yet. pull one: native-engine pull "
                     "Qwen/Qwen2.5-0.5B-Instruct\n");
    return 0;
}

// `convert <hf_dir> <out_dir>` — native HF->artifact conversion, no Python.
int RunConvert(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: native-engine convert <hf_dir> <out_dir>\n");
        return 2;
    }
    lykuro::nie::Status s = lykuro::nie::ConvertHfQwen(argv[2], argv[3]);
    if (!s.ok()) {
        std::fprintf(stderr, "convert failed: %s\n", s.message().c_str());
        return 1;
    }
    std::fprintf(stderr, "artifact written to %s\n", argv[3]);
    return 0;
}

// `pull <hf_repo> [out_dir]` — single-command: curl the HF files (curl sets no
// Gatekeeper quarantine) then convert natively.
int RunPull(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: native-engine pull <hf_repo> [out_dir]\n"
                     "  e.g. native-engine pull Qwen/Qwen2.5-0.5B-Instruct\n");
        return 2;
    }
    using namespace lykuro::nie;
    const std::string repo = argv[2];
    const std::string out = argc > 3 ? argv[3] : cli::DefaultModelDir(repo);
    int rc = cli::PullModel(repo, out);
    if (rc != 0) return rc;
    std::fprintf(stderr,
                 "pulled -> %s\n  run it: native-engine run %s \"...\"\n",
                 out.c_str(), out.c_str());
    return 0;
}

#ifdef LYKURO_HAVE_GRPC

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop = true; }

int RunServer(const lykuro::nie::FileConfig& config) {
    using namespace lykuro::nie;

    Logger::Get().SetLevel(config.log_level == "debug"  ? LogLevel::kDebug
                           : config.log_level == "warn" ? LogLevel::kWarn
                           : config.log_level == "error"
                               ? LogLevel::kError
                               : LogLevel::kInfo);

    ServerConfig server_config;
    server_config.listen_address =
        config.listen_address + ":" + std::to_string(config.grpc_port);
    server_config.control_identities = config.control_identities;
    server_config.data_identities = config.data_identities;
    server_config.engine.max_sequences = config.max_sequences;
    server_config.engine.scheduler.max_queue = config.max_queue;
    server_config.engine.max_output_tokens_limit = config.max_output_tokens;
    server_config.engine.max_input_bytes = config.max_input_bytes;
    server_config.hardware_backend = config.hardware_backend;
    server_config.device_id = config.device_id;
    server_config.metrics_enabled = config.metrics_enabled;
    server_config.metrics_port = config.metrics_port;
    server_config.load_options.allow_unsigned_dev = config.allow_unsigned_dev;
    for (const std::string& hex : config.trusted_signing_keys_hex) {
        std::array<uint8_t, 32> key;
        Status s = ParsePublicKeyHex(hex, key);
        if (!s.ok()) {
            std::fprintf(stderr, "invalid trusted signing key in config\n");
            return 1;
        }
        server_config.load_options.trusted_keys.keys.push_back(key);
    }

    if (config.mtls_required) {
        Status s = MakeMtlsServerCredentials(
            config.server_cert_path, config.server_key_path,
            config.client_ca_path, server_config.credentials);
        if (!s.ok()) {
            std::fprintf(stderr, "mtls setup failed: %s\n",
                         s.message().c_str());
            return 1;
        }
    } else {
        // Development only; production configs must keep mtls_required.
        server_config.credentials = grpc::InsecureServerCredentials();
        Logger::Get().Log(LogLevel::kWarn, "main", "mtls_disabled");
    }

    EngineServer server(std::move(server_config));
    int port = 0;
    uint16_t metrics_port = 0;
    Status s = server.Start(&port, &metrics_port);
    if (!s.ok()) {
        std::fprintf(stderr, "server start failed: %s\n",
                     s.message().c_str());
        return 1;
    }
    Logger::Get().Log(LogLevel::kInfo, "main", "server_started",
                      {{"engine_id", config.engine_id}},
                      {{"grpc_port", port}, {"metrics_port", metrics_port}});

    if (!config.artifact_path.empty()) {
        s = server.LoadModel(config.artifact_path);
        if (!s.ok()) {
            // Refuse to run without the configured model (fail fast for
            // the Model Manager to observe).
            std::fprintf(stderr, "model load failed: %s\n",
                         s.message().c_str());
            server.Shutdown();
            return 1;
        }
        Logger::Get().Log(LogLevel::kInfo, "main", "model_loaded");
    }

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Logger::Get().Log(LogLevel::kInfo, "main", "shutdown_requested");
    server.Shutdown();
    return 0;
}

// `serve --config <path>` (or `serve <path>`) — the gRPC engine server.
int RunServe(int argc, char** argv) {
    using namespace lykuro::nie;
    const char* cfg = nullptr;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfg = argv[++i];
        else if (!a.empty() && a[0] != '-' && cfg == nullptr) cfg = argv[i];
        else {
            std::fprintf(stderr, "unknown serve argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (cfg == nullptr) {
        std::fprintf(stderr, "usage: native-engine serve --config <path>\n");
        return 2;
    }
    FileConfigResult config = LoadFileConfig(cfg);
    if (!config.status.ok()) {
        std::fprintf(stderr, "config error: %s\n",
                     config.status.message().c_str());
        return 1;
    }
    return RunServer(config.config);
}

#endif  // LYKURO_HAVE_GRPC

}  // namespace

int main(int argc, char** argv) {
    // Unified subcommands (single-command). These run before legacy flag
    // parsing; `pull`/`run`/`convert` need no JSON config at all.
    if (argc >= 2 && std::strcmp(argv[1], "run") == 0)
        return RunGenerate(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "pull") == 0)
        return RunPull(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "convert") == 0)
        return RunConvert(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "list") == 0)
        return RunList(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) {
        // `serve --config <path>` runs the gRPC server; bare `serve` (or
        // `serve --http`) starts the Ollama/OpenAI HTTP API by default.
        bool has_config = false;
        for (int i = 2; i < argc; ++i)
            if (std::strcmp(argv[i], "--config") == 0) has_config = true;
        if (!has_config) return lykuro::nie::RunHttpServe(argc, argv);
#ifdef LYKURO_HAVE_GRPC
        return RunServe(argc, argv);
#else
        std::fprintf(stderr,
                     "this build has no gRPC transport; rebuild with "
                     "-DLYKURO_ENABLE_GRPC=ON\n");
        return 1;
#endif
    }

    const char* config_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("%s\n", kVersion);
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0) {
            PrintUsage();
            return 0;
        }
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
            continue;
        }
        std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
        return 2;
    }
    if (config_path == nullptr) {
        PrintUsage();
        return 2;
    }

    lykuro::nie::FileConfigResult config =
        lykuro::nie::LoadFileConfig(config_path);
    if (!config.status.ok()) {
        std::fprintf(stderr, "config error: %s\n",
                     config.status.message().c_str());
        return 1;
    }

#ifdef LYKURO_HAVE_GRPC
    return RunServer(config.config);
#else
    std::fprintf(stderr,
                 "this build has no gRPC transport; rebuild with "
                 "-DLYKURO_ENABLE_GRPC=ON\n");
    return 1;
#endif
}
