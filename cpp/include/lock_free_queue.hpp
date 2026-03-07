#pragma once
#include <atomic>
#include <array>
#include <optional>
#include <cstddef>

// Single-Producer Single-Consumer lock-free queue
// Uses cache-line padding to prevent false sharing
// Zero mutex, zero syscall — pure atomic CAS operations
template<typename T, size_t Capacity = 4096>
class LockFreeQueue {
    static_assert((Capacity & (Capacity-1)) == 0, "Capacity must be power of 2");

public:
    LockFreeQueue() : head_(0), tail_(0) {}

    // Producer thread only
    bool push(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = (tail + 1) & mask_;
        if (next == head_.load(std::memory_order_acquire))
            return false;  // queue full
        buffer_[tail] = item;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer thread only
    std::optional<T> pop() {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire))
            return std::nullopt;  // queue empty
        T item = buffer_[head];
        head_.store((head + 1) & mask_, std::memory_order_release);
        return item;
    }

    size_t size() const {
        size_t tail = tail_.load(std::memory_order_acquire);
        size_t head = head_.load(std::memory_order_acquire);
        return (tail - head) & mask_;
    }

    bool empty() const { return size() == 0; }
    bool full()  const { return size() == Capacity - 1; }

private:
    static constexpr size_t mask_ = Capacity - 1;
    // Pad to cache line (64 bytes) to prevent false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) std::array<T, Capacity> buffer_;
};

// Tick data for the queue
struct Tick {
    uint64_t timestamp_ns;
    double   price;
    double   bid;
    double   ask;
    double   volume;
    char     symbol[8];
};
