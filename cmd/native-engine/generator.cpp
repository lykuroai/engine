#include "cmd/native-engine/generator.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "core/generation/sampler.h"
#include "model/convert/hf_convert.h"
#include "model/tokenizer/bpe_tokenizer.h"
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/qwen_metal_model.h"
#endif
#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
#endif

namespace lykuro::nie::cli {
namespace fs = std::filesystem;

std::string DefaultBackend() {
#if defined(LYKURO_HAVE_METAL)
    return "metal";
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

int PullModel(const std::string& repo, const std::string& out) {
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
    out.loaded = LoadArtifact(dir, opt);
    if (!out.loaded.status.ok()) return out.loaded.status;
    out.model = std::move(out.loaded.artifact.model);

    const std::string& b = out.backend;
    if (b == "metal" || b == "metal-fp16") {
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
        auto p = b.find(':');
        if (p != std::string::npos) co.device_id = std::atoi(b.c_str() + p + 1);
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
    completion_tokens = 0;
    for (int n = 0; n < params.max_tokens; ++n) {
        if (is_eos(token)) break;
        gen.push_back(token);
        ++completion_tokens;
        std::string bytes;
        tok.DecodeBytes(gen, bytes);
        if (bytes.size() > printed.size()) {
            std::string delta = bytes.substr(printed.size());
            printed = bytes;
            if (on_delta && !on_delta(delta)) break;  // consumer gone
        }
        st = s.model->Decode(*seq, token, logits);
        if (!st.ok()) return st;
        if (!sampler.Sample(logits, token).ok())
            return Status(ErrorCode::kInferenceFailed, "sample failed", "generator");
    }
    full_out = printed;
    return Status::Ok();
}

}  // namespace lykuro::nie::cli
