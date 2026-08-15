// Ollama- and OpenAI-compatible HTTP API for the Lykuro Native Inference
// Engine. A minimal from-scratch HTTP/1.1 server (POSIX sockets, no
// third-party HTTP runtime) exposing:
//   Ollama:  GET /api/version, GET /api/tags, POST /api/generate,
//            POST /api/chat, POST /api/pull
//   OpenAI:  GET /v1/models, POST /v1/completions, POST /v1/chat/completions
//
// Models are referenced by a HuggingFace repo id (or a local artifact dir)
// and auto-pulled on first use. Generation is serialized by a global mutex
// (one device, correctness first).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <csignal>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cmd/native-engine/generator.h"
#include "core/engine/json.h"

namespace lykuro::nie {
namespace {
namespace fs = std::filesystem;

// ---- small helpers -------------------------------------------------------

std::string JsonEsc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    char u[8];
                    std::snprintf(u, sizeof(u), "\\u%04x", c);
                    o += u;
                } else {
                    o.push_back(char(c));
                }
        }
    }
    return o;
}

std::string NowIso() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    gmtime_r(&t, &tmv);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

// ---- model cache (serialize generation on one device) --------------------

std::mutex g_mu;
std::map<std::string, std::shared_ptr<cli::Session>> g_cache;
std::string g_backend;  // requested backend for all models

// Returns a loaded session for `model` (repo id or dir), auto-pulling and
// caching. Caller must hold g_mu. Returns nullptr + sets err on failure.
std::shared_ptr<cli::Session> GetSession(const std::string& model,
                                         std::string& err) {
    auto it = g_cache.find(model);
    if (it != g_cache.end()) return it->second;
    auto sess = std::make_shared<cli::Session>();
    Status s = cli::LoadSession(model, g_backend, *sess);
    if (!s.ok()) {
        err = s.message();
        return nullptr;
    }
    g_cache[model] = sess;
    return sess;
}

// ---- HTTP connection -----------------------------------------------------

struct Request {
    std::string method, path, body;
};

bool ReadRequest(int fd, Request& req) {
    std::string buf;
    char tmp[4096];
    size_t header_end = std::string::npos;
    // read until headers complete
    while (header_end == std::string::npos) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf.append(tmp, size_t(n));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20)) return false;  // header too large
    }
    // request line
    size_t sp1 = buf.find(' ');
    size_t sp2 = buf.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = buf.substr(0, sp1);
    req.path = buf.substr(sp1 + 1, sp2 - sp1 - 1);
    // content-length
    size_t clen = 0;
    {
        std::string lower = buf.substr(0, header_end);
        for (char& c : lower) c = char(std::tolower((unsigned char)c));
        size_t p = lower.find("content-length:");
        if (p != std::string::npos)
            clen = size_t(std::strtoul(buf.c_str() + p + 15, nullptr, 10));
    }
    std::string body = buf.substr(header_end + 4);
    while (body.size() < clen) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n <= 0) break;
        body.append(tmp, size_t(n));
    }
    req.body = body.substr(0, clen ? clen : body.size());
    return true;
}

bool WriteAll(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = write(fd, s.data() + off, s.size() - off);
        if (n <= 0) return false;  // EPIPE etc.: client is gone
        off += size_t(n);
    }
    return true;
}

void SendJson(int fd, int code, const std::string& json) {
    const char* reason = code == 200 ? "OK" : code == 404 ? "Not Found"
                                          : code == 400   ? "Bad Request"
                                                          : "Error";
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + reason +
                    "\r\nContent-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(json.size()) +
                    "\r\nConnection: close\r\n\r\n";
    WriteAll(fd, h + json);
}

// Begin a chunked (streaming) response; write chunks with SendChunk; end
// with EndChunked.
void BeginChunked(int fd, const char* content_type) {
    std::string h = std::string("HTTP/1.1 200 OK\r\nContent-Type: ") +
                    content_type +
                    "\r\nTransfer-Encoding: chunked\r\n"
                    "Connection: close\r\n\r\n";
    WriteAll(fd, h);
}
bool SendChunk(int fd, const std::string& data) {
    if (data.empty()) return true;
    char len[32];
    std::snprintf(len, sizeof(len), "%zx\r\n", data.size());
    return WriteAll(fd, len) && WriteAll(fd, data) && WriteAll(fd, "\r\n");
}
void EndChunked(int fd) { WriteAll(fd, "0\r\n\r\n"); }

// ---- request-body parsing (in-tree JSON) ---------------------------------

std::string JStr(const json::Value* o, const char* k, const std::string& d) {
    if (!o) return d;
    const json::Value* v = o->Find(k);
    return (v && v->is_string()) ? v->as_string() : d;
}
bool JBool(const json::Value* o, const char* k, bool d) {
    if (!o) return d;
    const json::Value* v = o->Find(k);
    return (v && v->is_bool()) ? v->as_bool() : d;
}
double JNum(const json::Value* o, const char* k, double d) {
    if (!o) return d;
    const json::Value* v = o->Find(k);
    return (v && v->is_number()) ? v->as_double() : d;
}

cli::GenParams ParamsFrom(const json::Value* root, const json::Value* options) {
    cli::GenParams p;
    // OpenAI puts params at top level; Ollama under "options".
    const json::Value* o = options ? options : root;
    p.max_tokens = int(JNum(root, "max_tokens",
                            JNum(o, "num_predict", 512)));
    if (p.max_tokens <= 0) p.max_tokens = 512;
    p.temperature = float(JNum(root, "temperature", JNum(o, "temperature", 0)));
    p.top_p = float(JNum(root, "top_p", JNum(o, "top_p", 0.95)));
    p.top_k = uint32_t(JNum(o, "top_k", 0));
    p.seed = uint64_t(JNum(o, "seed", 0));
    return p;
}

std::vector<ChatMessage> MessagesFrom(const json::Value* arr) {
    std::vector<ChatMessage> msgs;
    if (!arr || !arr->is_array()) return msgs;
    for (const auto& m : arr->as_array()) {
        if (!m->is_object()) continue;
        std::string role = JStr(m.get(), "role", "user");
        std::string content = JStr(m.get(), "content", "");
        Role r = Role::kUser;
        if (role == "system") r = Role::kSystem;
        else if (role == "assistant") r = Role::kAssistant;
        else if (role == "developer") r = Role::kDeveloper;
        else if (role == "tool") r = Role::kTool;
        msgs.push_back({r, content});
    }
    return msgs;
}

// ---- endpoint handlers ---------------------------------------------------

void HandleTags(int fd) {
    // GET /api/tags -> list ~/.lykuro/models
    const char* home = std::getenv("HOME");
    std::string base = (home ? std::string(home) : ".") + "/.lykuro/models";
    std::string out = "{\"models\":[";
    bool first = true;
    std::error_code ec;
    if (fs::exists(base, ec)) {
        for (const auto& e : fs::directory_iterator(base, ec)) {
            if (!e.is_directory()) continue;
            if (!fs::exists(e.path() / "manifest.json")) continue;
            if (!first) out += ",";
            first = false;
            std::string nm = e.path().filename().string();
            out += "{\"name\":\"" + JsonEsc(nm) + "\",\"model\":\"" +
                   JsonEsc(nm) + "\",\"modified_at\":\"" + NowIso() +
                   "\",\"size\":0}";
        }
    }
    out += "]}";
    SendJson(fd, 200, out);
}

void HandleVersion(int fd) {
    SendJson(fd, 200,
             "{\"version\":\"1.0.1\",\"engine\":\"lykuro-native-engine\"}");
}

void HandleModels(int fd) {  // OpenAI GET /v1/models
    const char* home = std::getenv("HOME");
    std::string base = (home ? std::string(home) : ".") + "/.lykuro/models";
    std::string out = "{\"object\":\"list\",\"data\":[";
    bool first = true;
    std::error_code ec;
    if (fs::exists(base, ec)) {
        for (const auto& e : fs::directory_iterator(base, ec)) {
            if (!e.is_directory() ||
                !fs::exists(e.path() / "manifest.json"))
                continue;
            if (!first) out += ",";
            first = false;
            std::string nm = e.path().filename().string();
            out += "{\"id\":\"" + JsonEsc(nm) +
                   "\",\"object\":\"model\",\"owned_by\":\"lykuro\"}";
        }
    }
    out += "]}";
    SendJson(fd, 200, out);
}

// Shared generation for a resolved (model, messages, params). Streams via
// on_delta when streaming; returns full text + counts.
Status RunGen(const std::string& model,
              const std::vector<ChatMessage>& msgs, const cli::GenParams& p,
              const std::function<bool(const std::string&)>& on_delta,
              std::string& full, uint32_t& pt, uint32_t& ct,
              std::string& err) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto sess = GetSession(model, err);
    if (!sess) return Status(ErrorCode::kInvalidRequest, err, "http");
    return cli::Generate(*sess, msgs, p, on_delta, full, pt, ct);
}

void HandleOllamaGenerate(int fd, const json::Value* root, bool chat) {
    std::string model = JStr(root, "model", "");
    bool stream = JBool(root, "stream", true);
    if (model.empty()) {
        SendJson(fd, 400, "{\"error\":\"missing model\"}");
        return;
    }
    std::vector<ChatMessage> msgs;
    if (chat) {
        msgs = MessagesFrom(root->Find("messages"));
    } else {
        std::string sys = JStr(root, "system", "");
        if (!sys.empty()) msgs.push_back({Role::kSystem, sys});
        msgs.push_back({Role::kUser, JStr(root, "prompt", "")});
    }
    cli::GenParams p = ParamsFrom(root, root->Find("options"));

    auto line = [&](const std::string& delta, bool done) -> std::string {
        std::string j = "{\"model\":\"" + JsonEsc(model) +
                        "\",\"created_at\":\"" + NowIso() + "\",";
        if (chat)
            j += "\"message\":{\"role\":\"assistant\",\"content\":\"" +
                 JsonEsc(delta) + "\"},";
        else
            j += "\"response\":\"" + JsonEsc(delta) + "\",";
        j += done ? "\"done\":true,\"done_reason\":\"stop\"}"
                  : "\"done\":false}";
        return j + "\n";
    };

    std::string full, err;
    uint32_t pt = 0, ct = 0;
    if (stream) {
        BeginChunked(fd, "application/x-ndjson");
        Status s = RunGen(
            model, msgs, p,
            [&](const std::string& d) { return SendChunk(fd, line(d, false)); },
            full, pt, ct, err);
        if (!s.ok()) {
            SendChunk(fd, "{\"error\":\"" + JsonEsc(err.empty()
                                                        ? s.message()
                                                        : err) + "\"}\n");
        } else {
            SendChunk(fd, line("", true));
        }
        EndChunked(fd);
    } else {
        Status s = RunGen(model, msgs, p, nullptr, full, pt, ct, err);
        if (!s.ok()) {
            SendJson(fd, 400,
                     "{\"error\":\"" + JsonEsc(err.empty() ? s.message()
                                                           : err) + "\"}");
            return;
        }
        std::string j = "{\"model\":\"" + JsonEsc(model) +
                        "\",\"created_at\":\"" + NowIso() + "\",";
        if (chat)
            j += "\"message\":{\"role\":\"assistant\",\"content\":\"" +
                 JsonEsc(full) + "\"},";
        else
            j += "\"response\":\"" + JsonEsc(full) + "\",";
        j += "\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":" +
             std::to_string(pt) + ",\"eval_count\":" + std::to_string(ct) + "}";
        SendJson(fd, 200, j);
    }
}

void HandleOllamaPull(int fd, const json::Value* root) {
    std::string name = JStr(root, "name", JStr(root, "model", ""));
    if (name.empty()) {
        SendJson(fd, 400, "{\"error\":\"missing name\"}");
        return;
    }
    BeginChunked(fd, "application/x-ndjson");
    SendChunk(fd, "{\"status\":\"pulling " + JsonEsc(name) + "\"}\n");
    int rc = cli::PullModel(name, cli::DefaultModelDir(name));
    if (rc != 0)
        SendChunk(fd, "{\"error\":\"pull failed\"}\n");
    else
        SendChunk(fd, "{\"status\":\"success\"}\n");
    EndChunked(fd);
}

void HandleOpenAiChat(int fd, const json::Value* root, bool chat) {
    std::string model = JStr(root, "model", "");
    bool stream = JBool(root, "stream", false);
    if (model.empty()) {
        SendJson(fd, 400,
                 "{\"error\":{\"message\":\"missing model\"}}");
        return;
    }
    std::vector<ChatMessage> msgs;
    if (chat)
        msgs = MessagesFrom(root->Find("messages"));
    else
        msgs.push_back({Role::kUser, JStr(root, "prompt", "")});
    cli::GenParams p = ParamsFrom(root, nullptr);

    const std::string id = "chatcmpl-lykuro";
    const std::string obj = chat ? "chat.completion" : "text_completion";
    const std::string created = std::to_string(std::time(nullptr));

    std::string full, err;
    uint32_t pt = 0, ct = 0;
    if (stream) {
        BeginChunked(fd, "text/event-stream");
        auto chunk = [&](const std::string& delta) {
            std::string d = "{\"id\":\"" + id +
                            "\",\"object\":\"chat.completion.chunk\","
                            "\"created\":" + created + ",\"model\":\"" +
                            JsonEsc(model) + "\",\"choices\":[{\"index\":0,"
                            "\"delta\":{\"content\":\"" + JsonEsc(delta) +
                            "\"},\"finish_reason\":null}]}";
            return SendChunk(fd, "data: " + d + "\n\n");
        };
        Status s = RunGen(model, msgs, p, chunk, full, pt, ct, err);
        std::string fin = "{\"id\":\"" + id +
                          "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                          created + ",\"model\":\"" + JsonEsc(model) +
                          "\",\"choices\":[{\"index\":0,\"delta\":{},"
                          "\"finish_reason\":\"stop\"}]}";
        SendChunk(fd, "data: " + fin + "\n\n");
        SendChunk(fd, "data: [DONE]\n\n");
        EndChunked(fd);
    } else {
        Status s = RunGen(model, msgs, p, nullptr, full, pt, ct, err);
        if (!s.ok()) {
            SendJson(fd, 400, "{\"error\":{\"message\":\"" +
                                  JsonEsc(err.empty() ? s.message() : err) +
                                  "\"}}");
            return;
        }
        std::string msg =
            chat ? "\"message\":{\"role\":\"assistant\",\"content\":\"" +
                       JsonEsc(full) + "\"}"
                 : "\"text\":\"" + JsonEsc(full) + "\"";
        std::string j =
            "{\"id\":\"" + id + "\",\"object\":\"" + obj +
            "\",\"created\":" + created + ",\"model\":\"" + JsonEsc(model) +
            "\",\"choices\":[{\"index\":0," + msg +
            ",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" +
            std::to_string(pt) + ",\"completion_tokens\":" +
            std::to_string(ct) + ",\"total_tokens\":" +
            std::to_string(pt + ct) + "}}";
        SendJson(fd, 200, j);
    }
}

void HandleConnection(int fd) {
    Request req;
    if (ReadRequest(fd, req)) {
        json::ParseResult pr;
        const json::Value* root = nullptr;
        if (!req.body.empty()) {
            pr = json::Parse(req.body);
            root = pr.value.get();
        }
        if (req.method == "GET" && req.path == "/api/version")
            HandleVersion(fd);
        else if (req.method == "GET" && req.path == "/api/tags")
            HandleTags(fd);
        else if (req.method == "GET" && req.path == "/v1/models")
            HandleModels(fd);
        else if (req.method == "POST" && req.path == "/api/generate" && root)
            HandleOllamaGenerate(fd, root, /*chat=*/false);
        else if (req.method == "POST" && req.path == "/api/chat" && root)
            HandleOllamaGenerate(fd, root, /*chat=*/true);
        else if (req.method == "POST" && req.path == "/api/pull" && root)
            HandleOllamaPull(fd, root);
        else if (req.method == "POST" &&
                 req.path == "/v1/chat/completions" && root)
            HandleOpenAiChat(fd, root, /*chat=*/true);
        else if (req.method == "POST" && req.path == "/v1/completions" && root)
            HandleOpenAiChat(fd, root, /*chat=*/false);
        else
            SendJson(fd, 404, "{\"error\":\"not found\"}");
    }
    close(fd);
}

}  // namespace

// Entry point: `native-engine serve --http [--port N] [--backend b]`.
int RunHttpServe(int argc, char** argv) {
    int port = 11434;
    g_backend = "";  // best built
    std::string host = "127.0.0.1";  // loopback by default (safe)
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--http") continue;
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) g_backend = argv[++i];
        else if ((a == "--host" || a == "--addr") && i + 1 < argc)
            host = argv[++i];
    }
    if (host == "*" || host.empty()) host = "0.0.0.0";

    // A client aborting mid-stream must surface as EPIPE on write(),
    // not kill the whole server with SIGPIPE.
    std::signal(SIGPIPE, SIG_IGN);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        std::perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::fprintf(stderr, "invalid --host address: %s\n", host.c_str());
        return 2;
    }
    addr.sin_port = htons(uint16_t(port));
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::perror("bind");
        return 1;
    }
    if (listen(srv, 64) < 0) {
        std::perror("listen");
        return 1;
    }
    const bool loopback = (host == "127.0.0.1");
    std::fprintf(stderr,
                 "lykuro HTTP API on http://%s:%d  (Ollama /api/* + "
                 "OpenAI /v1/*)\n",
                 host.c_str(), port);
    if (!loopback)
        std::fprintf(stderr,
                     "WARNING: bound to %s — the HTTP API has NO "
                     "authentication. Expose only on a trusted internal "
                     "network (firewall it); for authenticated access use "
                     "the gRPC mTLS server (serve --config).\n",
                     host.c_str());
    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread(HandleConnection, fd).detach();
    }
    return 0;
}

}  // namespace lykuro::nie
