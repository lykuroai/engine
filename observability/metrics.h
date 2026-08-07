#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lykuro::nie {

// Content-free metrics registry (spec §24). Metric names and label values
// are fixed identifiers supplied by engine code; there is no API path by
// which request content can become a label (spec §24.2).
class MetricsRegistry {
public:
    class Counter {
    public:
        void Increment(uint64_t delta = 1) {
            value_.fetch_add(delta, std::memory_order_relaxed);
        }
        uint64_t value() const {
            return value_.load(std::memory_order_relaxed);
        }

    private:
        std::atomic<uint64_t> value_{0};
    };

    using GaugeFn = std::function<int64_t()>;

    // Registers (or returns the existing) counter with the given name.
    Counter* GetCounter(const std::string& name, const std::string& help);

    // Registers a pull-style gauge evaluated at render time.
    void RegisterGauge(const std::string& name, const std::string& help,
                       GaugeFn fn);

    // Prometheus text exposition format.
    std::string Render() const;

private:
    struct CounterEntry {
        std::string help;
        std::unique_ptr<Counter> counter;
    };
    struct GaugeEntry {
        std::string help;
        GaugeFn fn;
    };

    mutable std::mutex mutex_;
    std::map<std::string, CounterEntry> counters_;
    std::map<std::string, GaugeEntry> gauges_;
};

// Minimal loopback-only HTTP endpoint serving GET /metrics (spec §9.1:
// health/metrics may be management-network HTTP). Single-threaded accept
// loop; anything other than GET /metrics gets 404.
class MetricsHttpServer {
public:
    explicit MetricsHttpServer(const MetricsRegistry& registry);
    ~MetricsHttpServer();

    // Binds 127.0.0.1:<port> (0 = ephemeral); returns the bound port.
    bool Start(uint16_t port, uint16_t* bound_port = nullptr);
    void Stop();

private:
    void Loop();

    const MetricsRegistry& registry_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> thread_;
};

}  // namespace lykuro::nie
