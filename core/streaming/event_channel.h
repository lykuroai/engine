#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

#include "core/streaming/events.h"

namespace lykuro::nie {

// Bounded per-request event channel (spec §19.1). The producer (GPU/engine
// worker) never blocks: when the consumer falls behind and the buffer is
// full, TryPush fails and the engine cancels the request with
// stream_consumer_slow instead of buffering without bound.
class EventChannel {
public:
    explicit EventChannel(size_t capacity) : capacity_(capacity) {}

    // Returns false when the channel is full or closed.
    bool TryPush(StreamEvent event) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || queue_.size() >= capacity_) return false;
            queue_.push_back(std::move(event));
        }
        cv_.notify_one();
        return true;
    }

    // Push for terminal events (error/completed): when the buffer is full,
    // the oldest event is evicted so the consumer always learns how the
    // request ended. Returns false only when already closed.
    bool PushTerminal(StreamEvent event) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            if (queue_.size() >= capacity_ && !queue_.empty()) {
                queue_.pop_front();
            }
            queue_.push_back(std::move(event));
        }
        cv_.notify_one();
        return true;
    }

    // Non-blocking pop.
    std::optional<StreamEvent> TryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        StreamEvent e = std::move(queue_.front());
        queue_.pop_front();
        return e;
    }

    // Blocking pop; returns nullopt once closed and drained.
    std::optional<StreamEvent> Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return std::nullopt;
        StreamEvent e = std::move(queue_.front());
        queue_.pop_front();
        return e;
    }

    // Producer side signals no more events will arrive.
    void Close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<StreamEvent> queue_;
    bool closed_ = false;
};

}  // namespace lykuro::nie
