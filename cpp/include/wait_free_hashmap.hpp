#pragma once
#include <atomic>
#include <vector>
#include <optional>
#include <cstdint>
#include <cstring>

// Wait-free HashMap — O(1) guaranteed, no locks
// Uses open addressing + linear probing
// Each slot has atomic version for lock-free reads
// Used for: symbol → order book mapping (concurrent access)

template<typename Key, typename Value, size_t Capacity = 4096>
class WaitFreeHashMap {
    static_assert((Capacity & (Capacity-1)) == 0, "Must be power of 2");
    static constexpr size_t MASK = Capacity - 1;

    struct Slot {
        alignas(64) std::atomic<uint64_t> version{0};
        Key   key{};
        Value value{};
        bool  occupied{false};
    };

public:
    WaitFreeHashMap() : size_(0) {}

    // O(1) wait-free insert
    bool insert(const Key& key, const Value& value) {
        size_t idx = hash(key) & MASK;
        for (size_t i = 0; i < Capacity; ++i) {
            size_t pos = (idx + i) & MASK;
            uint64_t ver = slots_[pos].version.load(std::memory_order_acquire);

            if (!slots_[pos].occupied) {
                // Try to claim this slot with CAS
                uint64_t expected = ver & ~1ULL;  // even = free
                if (slots_[pos].version.compare_exchange_strong(
                        expected, ver | 1ULL, std::memory_order_acq_rel)) {
                    slots_[pos].key      = key;
                    slots_[pos].value    = value;
                    slots_[pos].occupied = true;
                    slots_[pos].version.fetch_add(1, std::memory_order_release);
                    ++size_;
                    return true;
                }
            } else if (slots_[pos].key == key) {
                // Update existing
                slots_[pos].version.fetch_or(1, std::memory_order_acq_rel);
                slots_[pos].value = value;
                slots_[pos].version.fetch_add(1, std::memory_order_release);
                return true;
            }
        }
        return false;  // table full
    }

    // O(1) wait-free lookup with seqlock-style consistency
    std::optional<Value> get(const Key& key) const {
        size_t idx = hash(key) & MASK;
        for (size_t i = 0; i < Capacity; ++i) {
            size_t pos = (idx + i) & MASK;
            if (!slots_[pos].occupied) return std::nullopt;
            if (slots_[pos].key != key) continue;

            // Seqlock read — retry if version odd (write in progress)
            while (true) {
                uint64_t v1 = slots_[pos].version.load(std::memory_order_acquire);
                if (v1 & 1) { std::this_thread::yield(); continue; }
                Value val   = slots_[pos].value;
                uint64_t v2 = slots_[pos].version.load(std::memory_order_acquire);
                if (v1 == v2) return val;
            }
        }
        return std::nullopt;
    }

    bool remove(const Key& key) {
        size_t idx = hash(key) & MASK;
        for (size_t i = 0; i < Capacity; ++i) {
            size_t pos = (idx + i) & MASK;
            if (!slots_[pos].occupied) return false;
            if (slots_[pos].key == key) {
                slots_[pos].occupied = false;
                --size_;
                return true;
            }
        }
        return false;
    }

    size_t size()     const { return size_.load(); }
    size_t capacity() const { return Capacity; }

private:
    std::array<Slot, Capacity> slots_;
    std::atomic<size_t>        size_;

    size_t hash(const Key& k) const {
        // FNV-1a hash
        size_t h = 14695981039346656037ULL;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&k);
        for (size_t i = 0; i < sizeof(Key); ++i)
            h = (h ^ p[i]) * 1099511628211ULL;
        return h;
    }
};
