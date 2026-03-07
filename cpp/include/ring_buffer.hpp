#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>

// Circular tick store — O(1) write, O(1) random access
// Fixed memory, never allocates, perfect for tick replay
template<typename T, size_t Capacity = 1048576>  // 1M ticks default
class RingBuffer {
    static_assert((Capacity & (Capacity-1)) == 0, "Must be power of 2");

public:
    RingBuffer() : write_pos_(0), count_(0) {}

    void push(const T& item) {
        buffer_[write_pos_ & mask_] = item;
        ++write_pos_;
        if (count_ < Capacity) ++count_;
    }

    // Access by age: 0 = newest, 1 = one before, etc.
    const T& operator[](size_t age) const {
        size_t idx = (write_pos_ - 1 - age) & mask_;
        return buffer_[idx];
    }

    // Replay a range of ticks into a callback
    void replay(size_t from_age, size_t to_age,
                std::function<void(const T&)> callback) const {
        for (size_t age = to_age; age >= from_age; --age) {
            callback((*this)[age]);
            if (age == 0) break;
        }
    }

    size_t size()     const { return count_; }
    size_t capacity() const { return Capacity; }
    bool   empty()    const { return count_ == 0; }
    bool   full()     const { return count_ == Capacity; }

    // Compute rolling stats on last N ticks
    struct Stats { double mean, variance, min, max; };
    Stats rolling_stats(size_t n, std::function<double(const T&)> extractor) const {
        n = std::min(n, count_);
        if (n == 0) return {0,0,0,0};
        double sum=0, sum_sq=0, mn=1e18, mx=-1e18;
        for (size_t i=0; i<n; ++i) {
            double v = extractor((*this)[i]);
            sum += v; sum_sq += v*v;
            mn = std::min(mn, v); mx = std::max(mx, v);
        }
        double mean = sum/n;
        return {mean, sum_sq/n - mean*mean, mn, mx};
    }

private:
    static constexpr size_t mask_ = Capacity - 1;
    std::array<T, Capacity> buffer_;
    size_t write_pos_;
    size_t count_;
};
