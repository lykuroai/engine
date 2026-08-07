#include "core/scheduler/scheduler.h"

#include <gtest/gtest.h>

namespace lykuro::nie {
namespace {

ScheduledRequest MakeRequest(const std::string& id, uint32_t priority,
                             const std::string& tenant = "tn_a") {
    ScheduledRequest r;
    r.request_id = id;
    r.tenant_scope = tenant;
    r.priority = priority;
    r.prompt_tokens = {1};
    r.max_output_tokens = 8;
    return r;
}

TEST(SchedulerTest, HigherPriorityWins) {
    Scheduler s({});
    ASSERT_TRUE(s.Enqueue(MakeRequest("low", 10), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("high", 90), 0).ok());
    auto next = s.PopNext(0, {}, 4);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->request_id, "high");
}

TEST(SchedulerTest, FifoWithinSamePriority) {
    Scheduler s({});
    ASSERT_TRUE(s.Enqueue(MakeRequest("first", 50), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("second", 50), 0).ok());
    EXPECT_EQ(s.PopNext(0, {}, 4)->request_id, "first");
    EXPECT_EQ(s.PopNext(0, {}, 4)->request_id, "second");
}

TEST(SchedulerTest, PriorityClampedToCeiling) {
    SchedulerConfig cfg;
    cfg.priority_ceiling = 60;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("clamped", 100), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("mid", 59), 0).ok());
    // 100 clamps to 60, still beats 59.
    EXPECT_EQ(s.PopNext(0, {}, 4)->request_id, "clamped");
}

TEST(SchedulerTest, AgingLiftsLongWaiters) {
    SchedulerConfig cfg;
    cfg.aging_ms_per_point = 100;
    cfg.aging_max_bonus = 20;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("old_low", 45), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("new_high", 50), 1000).ok());
    // At t=1000, old_low has +10 bonus -> 55 beats 50.
    EXPECT_EQ(s.PopNext(1000, {}, 4)->request_id, "old_low");
}

TEST(SchedulerTest, AgingBonusIsBounded) {
    SchedulerConfig cfg;
    cfg.aging_ms_per_point = 1;
    cfg.aging_max_bonus = 5;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("ancient", 10), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("high", 40), 999999).ok());
    // ancient: 10 + 5 (capped) = 15 < 40.
    EXPECT_EQ(s.PopNext(999999, {}, 4)->request_id, "high");
}

TEST(SchedulerTest, QueueFullReturnsResourceExhausted) {
    SchedulerConfig cfg;
    cfg.max_queue = 2;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("a", 50), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("b", 50), 0).ok());
    Status st = s.Enqueue(MakeRequest("c", 50), 0);
    EXPECT_FALSE(st.ok());
    EXPECT_EQ(st.code(), ErrorCode::kResourceExhausted);
    EXPECT_TRUE(st.retryable());
}

TEST(SchedulerTest, RejectsInfeasibleDeadline) {
    Scheduler s({});
    ScheduledRequest r = MakeRequest("late", 50);
    r.deadline_unix_ms = 100;
    Status st = s.Enqueue(std::move(r), /*now=*/200);
    EXPECT_FALSE(st.ok());
    EXPECT_EQ(st.code(), ErrorCode::kDeadlineRejected);
}

TEST(SchedulerTest, RejectsDuplicateId) {
    Scheduler s({});
    ASSERT_TRUE(s.Enqueue(MakeRequest("dup", 50), 0).ok());
    EXPECT_FALSE(s.Enqueue(MakeRequest("dup", 50), 0).ok());
}

TEST(SchedulerTest, CancelQueuedIsIdempotent) {
    Scheduler s({});
    ASSERT_TRUE(s.Enqueue(MakeRequest("x", 50), 0).ok());
    EXPECT_TRUE(s.Cancel("x"));
    EXPECT_FALSE(s.Cancel("x"));
    EXPECT_FALSE(s.Cancel("unknown"));
    EXPECT_EQ(s.queue_depth(), 0u);
}

TEST(SchedulerTest, PopExpiredCollectsPassedDeadlines) {
    Scheduler s({});
    ScheduledRequest a = MakeRequest("a", 50);
    a.deadline_unix_ms = 100;
    ScheduledRequest b = MakeRequest("b", 50);  // no deadline
    ASSERT_TRUE(s.Enqueue(std::move(a), 0).ok());
    ASSERT_TRUE(s.Enqueue(std::move(b), 0).ok());
    auto expired = s.PopExpired(150);
    ASSERT_EQ(expired.size(), 1u);
    EXPECT_EQ(expired[0]->request_id, "a");
    EXPECT_EQ(s.queue_depth(), 1u);
}

TEST(SchedulerTest, TenantFairnessCapSkipsGreedyTenant) {
    SchedulerConfig cfg;
    cfg.max_tenant_active_share = 0.5;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("a2", 90, "tn_a"), 0).ok());
    ASSERT_TRUE(s.Enqueue(MakeRequest("b1", 30, "tn_b"), 0).ok());
    // tn_a already holds 1 of 2 slots (cap = 1): tn_b goes next despite
    // lower priority.
    std::map<std::string, uint32_t> active = {{"tn_a", 1}};
    EXPECT_EQ(s.PopNext(0, active, 2)->request_id, "b1");
}

TEST(SchedulerTest, FairnessCapIgnoredWhenOnlyCappedTenantWaits) {
    SchedulerConfig cfg;
    cfg.max_tenant_active_share = 0.5;
    Scheduler s(cfg);
    ASSERT_TRUE(s.Enqueue(MakeRequest("a2", 50, "tn_a"), 0).ok());
    std::map<std::string, uint32_t> active = {{"tn_a", 1}};
    // No other tenant waiting: capacity is not left idle.
    EXPECT_EQ(s.PopNext(0, active, 2)->request_id, "a2");
}

}  // namespace
}  // namespace lykuro::nie
