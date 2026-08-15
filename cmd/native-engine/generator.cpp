#include "cmd/native-engine/generator.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "core/generation/sampler.h"
#include "model/convert/hf_convert.h"
#include "model/tokenizer/bpe_tokenizer.h"
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/qwen_metal_fast.h"
#include "backends/metal/qwen_metal_model.h"
#endif
#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
#endif

namespace lykuro::nie::cli {
namespace fs = std::filesystem;

std::string DefaultBackend() {
#if defined(LYKURO_HAVE_METAL)
    // Fastest local default (kernel path, INT4 weight-only — same
    // default precision class as Ollama's q4 models). `metal` remains
    // the FP32 MPSGraph parity anchor, selectable with --backend.
    return "metal-q4";
#elif defined(LYKURO_HAVE_CUDA)
    return "cuda";
#else
    return "cpu";
#endif
}

bool LooksLikeHfRepo(const std::string& s) {
    if (s.find('/') == std::string::npos) return false;
    for (char ch : s)
        if (!std::isalnum((unsigned char)ch) && ch != '/' && ch != '-' &&
            ch != '_' && ch != '.')
            return false;
    return true;
}

std::string DefaultModelDir(const std::string& repo) {
    const char* home = std::getenv("HOME");
    std::string name = repo;
    for (char& ch : name)
        if (ch == '/') ch = '_';
    return (home ? std::string(home) : ".") + "/.lykuro/models/" + name;
}

std::string DisplayModelName(const std::string& dir) {
    // The canonical user-facing name: the HF repo id the artifact came
    // from (recorded by pull in a `source_repo` sidecar), falling back
    // to deriving it from the directory name — HF owner names cannot
    // contain '_', so the first '_' is the '/' the pull replaced. Every
    // name this returns round-trips through DefaultModelDir back to the
    // same directory, so list/tags output is always valid pull/run
    // input.
    const fs::path p(dir);
    std::ifstream f(p / "source_repo");
    std::string repo;
    if (f && std::getline(f, repo) && !repo.empty() &&
        DefaultModelDir(repo) ==
            DefaultModelDir(p.filename().string())) {
        return repo;
    }
    std::string name = p.filename().string();
    const size_t us = name.find('_');
    if (us != std::string::npos && us > 0 && us + 1 < name.size()) {
        name[us] = '/';
    }
    return name;
}

int PullModel(const std::string& repo, const std::string& out) {
    // Already-local names succeed as a no-op (Ollama-style semantics). This
    // also makes the names `list` / /api/tags report — local directory
    // names like "Qwen_Qwen2.5-0.5B-Instruct", which are not valid HF
    // repo ids — round-trip through pull.
    if (fs::exists(fs::path(out) / "manifest.json")) {
        std::fprintf(stderr, "already local: %s\n", out.c_str());
        return 0;
    }
    if (repo.find('/') == std::string::npos) {
        const std::string local = DefaultModelDir(repo);
        if (fs::exists(fs::path(local) / "manifest.json")) {
            std::fprintf(stderr, "already local: %s\n", local.c_str());
            return 0;
        }
    }
    if (!LooksLikeHfRepo(repo)) {
        std::fprintf(stderr, "invalid repo id: %s\n", repo.c_str());
        return 2;
    }
    std::error_code ec;
    const std::string tmp = out + "/.hf-download";
    fs::create_directories(tmp, ec);
    auto dl = [&](const char* file, bool required) -> bool {
        const std::string url =
            "https://huggingface.co/" + repo + "/resolve/main/" + file;
        const std::string cmd = "curl -fSL --retry 3 -o '" + tmp + "/" + file +
                                "' '" + url + "'";
        std::fprintf(stderr, "pull: %s\n", file);
        int rc = std::system(cmd.c_str());
        if (rc != 0 && required)
            std::fprintf(stderr, "download failed: %s\n", file);
        return rc == 0;
    };
    if (!dl("config.json", true)) return 1;
    if (!dl("tokenizer.json", true)) return 1;
    if (!dl("model.safetensors", true)) {
        std::fprintf(stderr,
                     "note: only single-file model.safetensors is supported "
                     "(sharded checkpoints are not yet merged)\n");
        return 1;
    }
    dl("generation_config.json", false);
    Status s = ConvertHfQwen(tmp, out);
    if (!s.ok()) {
        std::fprintf(stderr, "convert failed: %s\n", s.message().c_str());
        return 1;
    }
    fs::remove_all(tmp, ec);
    // Record the source repo id so `list` / the HTTP tag endpoints can
    // report exactly the name pull was given.
    std::ofstream(fs::path(out) / "source_repo") << repo << "\n";
    return 0;
}

int ResolveModelArg(std::string& dir) {
    if (fs::exists(fs::path(dir) / "manifest.json")) return 0;
    // A bare local name as listed by `/v1/models` and `/api/tags`
    // (e.g. "Qwen_Qwen2.5-1.5B-Instruct") resolves under ~/.lykuro/models.
    if (dir.find('/') == std::string::npos) {
        std::string local = DefaultModelDir(dir);
        if (fs::exists(fs::path(local) / "manifest.json")) {
            dir = local;
            return 0;
        }
    }
    if (LooksLikeHfRepo(dir)) {
        std::string resolved = DefaultModelDir(dir);
        if (!fs::exists(fs::path(resolved) / "manifest.json")) {
            std::fprintf(stderr, "model not found locally; pulling %s ...\n",
                         dir.c_str());
            int rc = PullModel(dir, resolved);
            if (rc != 0) return rc;
        }
        dir = resolved;
    }
    return 0;
}

Status LoadSession(const std::string& dir_in, const std::string& backend_in,
                   Session& out) {
    std::string dir = dir_in;
    if (int rc = ResolveModelArg(dir)) {
        return Status(ErrorCode::kInvalidRequest, "model resolve/pull failed",
                      "generator");
    }
    out.dir = dir;
    out.backend = backend_in.empty() ? DefaultBackend() : backend_in;

    ArtifactLoadOptions opt;
    opt.allow_unsigned_dev = true;
    // A GPU backend replaces the model anyway; building the CPU
    // reference (a full BF16->FP32 conversion plus a CPU smoke forward)
    // doubled the load time of every first request.
    opt.skip_reference_model = (out.backend != "cpu");
    const bool prof = std::getenv("LYKURO_LOAD_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    out.loaded = LoadArtifact(dir, opt);
    if (prof) {
        std::fprintf(stderr, "[load] artifact verify+open: %.2fs\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
        t0 = std::chrono::steady_clock::now();
    }
    if (!out.loaded.status.ok()) return out.loaded.status;
    out.model = std::move(out.loaded.artifact.model);

    const std::string& b = out.backend;
    if (b == "metal-fast" || b == "metal-q8" || b == "metal-q4") {
#ifdef LYKURO_HAVE_METAL
        MetalFastOptions fo;
        if (b == "metal-q8") fo.quant = MetalFastOptions::Quant::kInt8;
        if (b == "metal-q4") fo.quant = MetalFastOptions::Quant::kInt4;
        auto m = QwenMetalFastModel::Load(out.loaded.artifact.manifest,
                                          *out.loaded.artifact.weights, fo);
        if (!m.status.ok()) return m.status;
        out.model = std::move(m.model);
#else
        return Status(ErrorCode::kInvalidRequest, "no Metal in this build",
                      "generator");
#endif
    } else if (b == "metal" || b == "metal-fp16") {
#ifdef LYKURO_HAVE_METAL
        MetalModelOptions mo;
        mo.fp16 = (b == "metal-fp16");
        auto m = QwenMetalModel::Load(out.loaded.artifact.manifest,
                                      *out.loaded.artifact.weights, mo);
        if (!m.status.ok()) return m.status;
        out.model = std::move(m.model);
#else
        return Status(ErrorCode::kInvalidRequest, "no Metal in this build",
                      "generator");
#endif
    } else if (b.rfind("cuda", 0) == 0) {
#ifdef LYKURO_HAVE_CUDA
        CudaModelOptions co;
        // cuda[:N] | cuda-q8[:N] | cuda-q4[:N] (weight-only quantization,
        // same scheme the Metal backend exposes as metal-q8/q4).
        std::string cb = b;
        auto p = cb.find(':');
        if (p != std::string::npos) {
            co.device_id = std::atoi(cb.c_str() + p + 1);
            cb = cb.substr(0, p);
        }
        if (cb == "cuda-q8") co.quantization = WeightQuant::kInt8;
        else if (cb == "cuda-q4") co.quantization = WeightQuant::kInt4;
        else if (cb != "cuda") {
            return Status(ErrorCode::kInvalidRequest, "unknown backend: " + b,
                          "generator");
        }
        auto c = QwenCudaModel::Load(out.loaded.artifact.manifest,
                                     *out.loaded.artifact.weights, co);
        if (!c.status.ok()) return c.status;
        out.model = std::move(c.model);
#else
        return Status(ErrorCode::kInvalidRequest, "no CUDA in this build",
                      "generator");
#endif
    } else if (b != "cpu") {
        return Status(ErrorCode::kInvalidRequest, "unknown backend: " + b,
                      "generator");
    }
    if (prof) {
        std::fprintf(stderr, "[load] backend model build: %.2fs\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
    }
    return Status::Ok();
}

Status Generate(Session& s, const std::vector<ChatMessage>& messages,
                const GenParams& params,
                const std::function<bool(const std::string&)>& on_delta,
                std::string& full_out, uint32_t& prompt_tokens,
                uint32_t& completion_tokens) {
    const BpeTokenizer& tok = *s.loaded.artifact.tokenizer;
    const std::vector<uint32_t> eos = s.loaded.artifact.manifest.eos_token_ids;
    auto is_eos = [&](uint32_t t) {
        for (uint32_t e : eos)
            if (e == t) return true;
        return false;
    };

    std::vector<uint32_t> prompt_ids;
    Status st = QwenChatTemplate::BuildPrompt(tok, messages, prompt_ids);
    if (!st.ok()) return st;
    prompt_tokens = uint32_t(prompt_ids.size());

    std::unique_ptr<SequenceState> seq;
    st = s.model->CreateSequence(
        uint32_t(prompt_ids.size()) + uint32_t(params.max_tokens) + 8, seq);
    if (!st.ok()) return st;
    std::vector<float> logits;
    st = s.model->Prefill(*seq, prompt_ids, logits);
    if (!st.ok()) return st;

    SamplingParams sp;
    sp.temperature = params.temperature;
    sp.seed = params.seed;
    if (params.temperature > 0.0f) {
        sp.top_p = params.top_p > 0 ? params.top_p : 0.95f;
        sp.top_k = params.top_k;
    }
    Sampler sampler(sp);

    uint32_t token = 0;
    if (!sampler.Sample(logits, token).ok())
        return Status(ErrorCode::kInferenceFailed, "sample failed", "generator");
    std::vector<uint32_t> gen;
    std::string printed;
    std::string bytes;              // accumulated raw output bytes
    std::vector<uint32_t> one_token(1);
    std::vector<uint32_t> pending;  // greedy-run token queue
    size_t pending_i = 0;
    completion_tokens = 0;
    for (int n = 0; n < params.max_tokens; ++n) {
        if (is_eos(token)) break;
        gen.push_back(token);
        ++completion_tokens;
        // Incremental detokenization: DecodeBytes is per-token
        // concatenative, so append just the new token's bytes instead of
        // re-decoding the whole sequence every step (O(n^2) at 400 tok/s
        // is real money).
        one_token[0] = token;
        tok.DecodeBytes(one_token, bytes);
        if (bytes.size() > printed.size()) {
            std::string delta = bytes.substr(printed.size());
            printed = bytes;
            if (on_delta && !on_delta(delta)) break;  // consumer gone
        }
        // Greedy fast path: the backend runs several decode steps per
        // call with on-GPU argmax (same rule as Sampler's greedy), so
        // tokens arrive in small batches instead of one CPU/GPU round
        // trip each. Falls back to Decode+Sample for temperature > 0.
        if (!(params.temperature > 0.0f) && s.model->SupportsGreedyRun()) {
            if (pending_i >= pending.size()) {
                pending.clear();
                pending_i = 0;
                st = s.model->GreedyRun(
                    *seq, token, uint32_t(params.max_tokens - n), pending);
                if (!st.ok()) return st;
                if (pending.empty())
                    return Status(ErrorCode::kInferenceFailed,
                                  "empty greedy run", "generator");
            }
            token = pending[pending_i++];
        } else {
            st = s.model->Decode(*seq, token, logits);
            if (!st.ok()) return st;
            if (!sampler.Sample(logits, token).ok())
                return Status(ErrorCode::kInferenceFailed, "sample failed",
                              "generator");
        }
    }
    full_out = printed;
    return Status::Ok();
}

}  // namespace lykuro::nie::cli
