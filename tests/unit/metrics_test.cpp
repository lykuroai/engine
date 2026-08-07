#include "observability/metrics.h"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>

namespace lykuro::nie {
namespace {

TEST(MetricsRegistryTest, CountersAccumulate) {
    MetricsRegistry registry;
    auto* c = registry.GetCounter("nie_test_total", "test counter");
    c->Increment();
    c->Increment(4);
    EXPECT_EQ(c->value(), 5u);
    // Same name returns the same counter.
    EXPECT_EQ(registry.GetCounter("nie_test_total", "test counter"), c);
}

TEST(MetricsRegistryTest, RendersPrometheusText) {
    MetricsRegistry registry;
    registry.GetCounter("nie_a_total", "counter a")->Increment(3);
    registry.RegisterGauge("nie_b", "gauge b", [] { return 42; });
    std::string text = registry.Render();
    EXPECT_NE(text.find("# TYPE nie_a_total counter"), std::string::npos);
    EXPECT_NE(text.find("nie_a_total 3"), std::string::npos);
    EXPECT_NE(text.find("# TYPE nie_b gauge"), std::string::npos);
    EXPECT_NE(text.find("nie_b 42"), std::string::npos);
}

std::string HttpGet(uint16_t port, const std::string& path) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) !=
        0) {
        ::close(fd);
        return {};
    }
    std::string request = "GET " + path + " HTTP/1.1\r\nHost: l\r\n\r\n";
    ::send(fd, request.data(), request.size(), 0);
    std::string response;
    char buf[4096];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
        response.append(buf, size_t(n));
    }
    ::close(fd);
    return response;
}

TEST(MetricsHttpTest, ServesMetricsOnLoopback) {
    MetricsRegistry registry;
    registry.GetCounter("nie_http_total", "http test")->Increment(7);
    MetricsHttpServer server(registry);
    uint16_t port = 0;
    ASSERT_TRUE(server.Start(0, &port));
    ASSERT_GT(port, 0);

    std::string response = HttpGet(port, "/metrics");
    EXPECT_NE(response.find("200 OK"), std::string::npos);
    EXPECT_NE(response.find("nie_http_total 7"), std::string::npos);

    std::string not_found = HttpGet(port, "/secrets");
    EXPECT_NE(not_found.find("404"), std::string::npos);

    server.Stop();
}

}  // namespace
}  // namespace lykuro::nie
