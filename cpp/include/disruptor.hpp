#pragma once
#include <atomic>
#include <array>
#include <vector>
#include <functional>
#include <thread>
#include <cstddef>

// LMAX Disruptor Pattern
// Ultra-low latency inter-thread message passing
// Used in HFT: market data feed → signal engine → order router
// Key: pre-allocated ring + sequence numbers, no locks, no GC

template<typename T, size_t Capacity = 65536>
class Disruptor {
    static_assert((Capacity & (Capacity-1)) == 0, "Must be power of 2");
    static constexpr size_t MASK = Capacity - 1;

    // Cache line padding prevents false sharing between producer/consumer
    struct alignas(64) PaddedSequence {
        std::atomic<int64_t> value{-1};
        char padding[64 - sizeof(std::atomic<int64_t>)];
    };

public:
    using Handler = std::function<void(const T&, int64_t, bool)>;

    Disruptor() { cursor_.value.store(-1); gating_.value.store(-1); }

    // Producer: claim next slot — O(1)
    int64_t next() {
        return cursor_.value.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    // Producer: publish to slot
    void publish(int64_t seq, const T& data) {
        ring_[seq & MASK]    = data;
        // Memory fence — ensure data visible before sequence
        published_[seq & MASK].value.store(seq, std::memory_order_release);
    }

    // Consumer: wait for sequence — spin wait (HFT style)
    const T& consume(int64_t seq) const {
        // Busy spin — lowest latency, highest CPU
        while (published_[seq & MASK].value.load(std::memory_order_acquire) < seq)
            std::this_thread::yield();
        return ring_[seq & MASK];
    }

    // Batch consume — process all available events
    int64_t available_sequence() const {
        int64_t seq = gating_.value.load(std::memory_order_acquire);
        while (published_[(seq+1) & MASK].value.load(std::memory_order_acquire) >= seq+1)
            ++seq;
        return seq;
    }

    void commit_consumer(int64_t seq) {
        gating_.value.store(seq, std::memory_order_release);
    }

    size_t capacity() const { return Capacity; }

private:
    alignas(64) std::array<T, Capacity>               ring_;
    alignas(64) std::array<PaddedSequence, Capacity>  published_;
    PaddedSequence cursor_;
    PaddedSequence gating_;
};
