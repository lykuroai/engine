#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/engine/config.h"
#include "core/generation/sampler.h"
#include "model/architectures/generative_model.h"
#include "model/loader/artifact_loader.h"
#include "model/tokenizer/bpe_tokenizer.h"
#include "model/tokenizer/prompt_template.h"
#include "observability/log.h"
#ifdef LYKURO_HAVE_METAL
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

namespace {

constexpr const char kVersion[] = "1.0.0";

void PrintUsage() {
    std::printf(
        "lykuro-native-engine %s\n"
        "\n"
        "Usage:\n"
        "  native-engine run <model_dir> [\"prompt\"] [options]   "
        "standalone inference (no config)\n"
        "  native-engine --config <path>                          "
        "gRPC engine server\n"
        "  native-engine --version | --help\n"
        "\n"
        "run options:\n"
        "  --backend <cpu|metal|metal-fp16|cuda[:N]>  (default: best built)\n"
        "  --max-tokens <N>       (default 512)\n"
        "  --temperature <T>      (default 0 = greedy)\n"
        "  --system \"<text>\"      optional system prompt\n"
        "  --seed <N>             PRNG seed for temperature>0\n"
        "\n"
        "With no \"prompt\", `run` starts an interactive chat (Ctrl-D to "
        "exit).\n",
        kVersion);
}

// Config-less, Ollama-style standalone inference. Loads a model directory
// and generates text directly — no JSON config, no mTLS, no server.
int RunGenerate(int argc, char** argv) {
    using namespace lykuro::nie;

    std::string dir, prompt, system_prompt;
#if defined(LYKURO_HAVE_METAL)
    std::string backend = "metal";
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

    ArtifactLoadOptions opt;
    opt.allow_unsigned_dev = true;  // local eval; no signing config here
    ArtifactLoadResult loaded = LoadArtifact(dir, opt);
    if (!loaded.status.ok()) {
        std::fprintf(stderr, "load failed: %s\n",
                     loaded.status.message().c_str());
        return 1;
    }
    std::unique_ptr<GenerativeModel> model = std::move(loaded.artifact.model);

    if (backend == "metal" || backend == "metal-fp16") {
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
        auto p = backend.find(':');
        if (p != std::string::npos) co.device_id = std::atoi(backend.c_str() + p + 1);
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
        for (int n = 0; n < max_tokens; ++n) {
            if (is_eos(token)) break;
            gen.push_back(token);
            std::string bytes;
            tok.DecodeBytes(gen, bytes);
            if (bytes.size() > printed.size()) {
                std::fwrite(bytes.data() + printed.size(), 1,
                            bytes.size() - printed.size(), stdout);
                std::fflush(stdout);
                printed = bytes;
            }
            s = model->Decode(*st, token, logits);
            if (!s.ok()) {
                std::fprintf(stderr, "\ndecode failed: %s\n",
                             s.message().c_str());
                return 1;
            }
            if (!sampler.Sample(logits, token).ok()) return 1;
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

#endif  // LYKURO_HAVE_GRPC

}  // namespace

int main(int argc, char** argv) {
    // Ollama-style standalone inference: `native-engine run <model> ...`.
    // Handled before config parsing — it needs no JSON config at all.
    if (argc >= 2 && std::strcmp(argv[1], "run") == 0) {
        return RunGenerate(argc, argv);
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
