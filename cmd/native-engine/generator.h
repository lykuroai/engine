#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/engine/error.h"
#include "model/architectures/generative_model.h"
#include "model/loader/artifact_loader.h"
#include "model/tokenizer/prompt_template.h"

// Shared model-resolution + text-generation core, used by both the `run`
// subcommand and the Ollama/OpenAI-compatible HTTP API.
namespace lykuro::nie::cli {

// --- model resolution (single-command / auto-pull) ---
bool LooksLikeHfRepo(const std::string& s);
std::string DefaultModelDir(const std::string& repo);

// Canonical user-facing name for a local artifact directory (the HF
// repo id it came from). Guaranteed to resolve back to `dir` via
// DefaultModelDir, so it is always valid run/pull input.
std::string DisplayModelName(const std::string& dir);
// Download an HF checkpoint via curl + native convert into `out`. 0 on ok.
int PullModel(const std::string& repo, const std::string& out);
// Resolve a model arg to a local artifact dir; a HF repo id auto-pulls if
// not present. Mutates `dir` to the resolved path. 0 on ok.
int ResolveModelArg(std::string& dir);

struct GenParams {
    int max_tokens = 512;
    float temperature = 0.0f;
    float top_p = 0.95f;
    uint32_t top_k = 0;
    uint64_t seed = 0;
    std::vector<std::string> stop;  // reserved (not yet enforced)
};

// A loaded model + everything needed to generate from it. Owns the
// artifact (weights/tokenizer/manifest stay alive for the backend model).
struct Session {
    ArtifactLoadResult loaded;  // owns manifest/weights/tokenizer
    std::unique_ptr<GenerativeModel> model;  // active backend model
    std::string backend;
    std::string dir;  // resolved artifact dir
};

// Load `dir` (auto-pull if it is a repo id) and build the backend model.
// backend: "cpu"|"metal"|"metal-fp16"|"cuda[:N]" ("" = best built).
Status LoadSession(const std::string& dir, const std::string& backend,
                   Session& out);

// Generate from chat messages. `on_delta` is called with each newly
// decoded text fragment (for streaming) and returns whether to continue;
// returning false (e.g. the client disconnected) stops decoding early and
// Generate returns Ok with the partial text. `full_out` receives the
// decoded reply. `prompt_tokens`/`completion_tokens` receive token counts.
Status Generate(Session& s, const std::vector<ChatMessage>& messages,
                const GenParams& params,
                const std::function<bool(const std::string&)>& on_delta,
                std::string& full_out, uint32_t& prompt_tokens,
                uint32_t& completion_tokens);

// Best built-in backend name for this binary (metal > cuda > cpu).
std::string DefaultBackend();

}  // namespace lykuro::nie::cli
