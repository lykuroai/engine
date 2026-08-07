#include "core/streaming/event_channel.h"

#include <gtest/gtest.h>

#include <thread>

namespace lykuro::nie {
namespace {

StreamEvent Delta(uint64_t seq, const std::string& text) {
    StreamEvent e;
    e.kind = StreamEvent::Kind::kOutputTextDelta;
    e.sequence = seq;
    e.text = text;
    return e;
}

TEST(EventChannelTest, PreservesOrder) {
    EventChannel ch(10);
    ASSERT_TRUE(ch.TryPush(Delta(0, "a")));
    ASSERT_TRUE(ch.TryPush(Delta(1, "b")));
    ASSERT_TRUE(ch.TryPush(Delta(2, "c")));
    EXPECT_EQ(ch.TryPop()->text, "a");
    EXPECT_EQ(ch.TryPop()->text, "b");
    EXPECT_EQ(ch.TryPop()->text, "c");
    EXPECT_FALSE(ch.TryPop().has_value());
}

TEST(EventChannelTest, TryPushFailsWhenFull) {
    EventChannel ch(2);
    EXPECT_TRUE(ch.TryPush(Delta(0, "a")));
    EXPECT_TRUE(ch.TryPush(Delta(1, "b")));
    EXPECT_FALSE(ch.TryPush(Delta(2, "c")));
    EXPECT_EQ(ch.size(), 2u);
}

TEST(EventChannelTest, PushTerminalEvictsOldest) {
    EventChannel ch(2);
    ASSERT_TRUE(ch.TryPush(Delta(0, "a")));
    ASSERT_TRUE(ch.TryPush(Delta(1, "b")));
    StreamEvent err;
    err.kind = StreamEvent::Kind::kError;
    err.sequence = 2;
    EXPECT_TRUE(ch.PushTerminal(std::move(err)));
    EXPECT_EQ(ch.size(), 2u);
    EXPECT_EQ(ch.TryPop()->text, "b");
    EXPECT_EQ(ch.TryPop()->kind, StreamEvent::Kind::kError);
}

TEST(EventChannelTest, ClosedChannelRejectsPushes) {
    EventChannel ch(2);
    ch.Close();
    EXPECT_FALSE(ch.TryPush(Delta(0, "a")));
    EXPECT_FALSE(ch.PushTerminal(Delta(1, "b")));
    EXPECT_FALSE(ch.Pop().has_value());
}

TEST(EventChannelTest, BlockingPopDrainsThenEndsAfterClose) {
    EventChannel ch(4);
    ASSERT_TRUE(ch.TryPush(Delta(0, "a")));
    ch.Close();
    EXPECT_EQ(ch.Pop()->text, "a");  // drains buffered events
    EXPECT_FALSE(ch.Pop().has_value());
}

TEST(EventChannelTest, CrossThreadDelivery) {
    EventChannel ch(64);
    std::thread producer([&] {
        for (int i = 0; i < 50; ++i) {
            while (!ch.TryPush(Delta(uint64_t(i), std::to_string(i)))) {
            }
        }
        ch.Close();
    });
    int expected = 0;
    while (auto e = ch.Pop()) {
        EXPECT_EQ(e->text, std::to_string(expected));
        ++expected;
    }
    EXPECT_EQ(expected, 50);
    producer.join();
}

}  // namespace
}  // namespace lykuro::nie
