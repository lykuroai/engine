#include "observability/metrics.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <thread>

namespace lykuro::nie {

MetricsRegistry::Counter* MetricsRegistry::GetCounter(
    const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = counters_.find(name);
    if (it == counters_.end()) {
        CounterEntry entry;
        entry.help = help;
        entry.counter = std::make_unique<Counter>();
        it = counters_.emplace(name, std::move(entry)).first;
    }
    return it->second.counter.get();
}

void MetricsRegistry::RegisterGauge(const std::string& name,
                                    const std::string& help, GaugeFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    gauges_[name] = GaugeEntry{help, std::move(fn)};
}

std::string MetricsRegistry::Render() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string out;
    for (const auto& [name, entry] : counters_) {
        out += "# HELP " + name + " " + entry.help + "\n";
        out += "# TYPE " + name + " counter\n";
        out += name + " " + std::to_string(entry.counter->value()) + "\n";
    }
    for (const auto& [name, entry] : gauges_) {
        out += "# HELP " + name + " " + entry.help + "\n";
        out += "# TYPE " + name + " gauge\n";
        out += name + " " + std::to_string(entry.fn()) + "\n";
    }
    return out;
}

MetricsHttpServer::MetricsHttpServer(const MetricsRegistry& registry)
    : registry_(registry) {}

MetricsHttpServer::~MetricsHttpServer() { Stop(); }

bool MetricsHttpServer::Start(uint16_t port, uint16_t* bound_port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    int reuse = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    // Loopback only: the metrics port is never exposed beyond the host.
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
               sizeof(addr)) != 0 ||
        ::listen(listen_fd_, 8) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (bound_port != nullptr) {
        sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &len);
        *bound_port = ntohs(bound.sin_port);
    }
    running_ = true;
    thread_ = std::make_unique<std::thread>([this] { Loop(); });
    return true;
}

void MetricsHttpServer::Stop() {
    running_ = false;
    if (listen_fd_ >= 0) {
        // Closing the fd unblocks accept().
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (thread_ && thread_->joinable()) thread_->join();
    thread_.reset();
}

void MetricsHttpServer::Loop() {
    while (running_) {
        int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            if (!running_) break;
            continue;
        }
        char buf[1024];
        ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        std::string response;
        if (n > 0) {
            buf[n] = '\0';
            if (std::strncmp(buf, "GET /metrics", 12) == 0) {
                std::string body = registry_.Render();
                response =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/plain; version=0.0.4\r\n"
                    "Content-Length: " + std::to_string(body.size()) +
                    "\r\nConnection: close\r\n\r\n" + body;
            } else {
                response =
                    "HTTP/1.1 404 Not Found\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
            }
        }
        if (!response.empty()) {
            size_t off = 0;
            while (off < response.size()) {
                ssize_t sent = ::send(client, response.data() + off,
                                      response.size() - off, 0);
                if (sent <= 0) break;
                off += size_t(sent);
            }
        }
        ::close(client);
    }
}

}  // namespace lykuro::nie
