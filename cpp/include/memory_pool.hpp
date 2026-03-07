#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <array>

// Pre-allocated slab allocator — zero malloc in the hot path
// All memory claimed upfront at startup, O(1) alloc/free
template<typename T, size_t PoolSize = 65536>
class MemoryPool {
public:
    MemoryPool() {
        for (size_t i = 0; i < PoolSize - 1; ++i)
            reinterpret_cast<Slot*>(&storage_[i])->next = &storage_[i+1];
        reinterpret_cast<Slot*>(&storage_[PoolSize-1])->next = nullptr;
        free_list_ = &storage_[0];
        available_ = PoolSize;
    }

    T* allocate() {
        if (!free_list_) return nullptr;  // pool exhausted
        Slot* slot = free_list_;
        free_list_ = slot->next;
        --available_;
        return reinterpret_cast<T*>(slot);
    }

    void deallocate(T* ptr) {
        Slot* slot  = reinterpret_cast<Slot*>(ptr);
        slot->next  = free_list_;
        free_list_  = slot;
        ++available_;
    }

    size_t available() const { return available_; }
    size_t capacity()  const { return PoolSize; }
    size_t used()      const { return PoolSize - available_; }

private:
    union Slot {
        alignas(T) std::byte data[sizeof(T)];
        Slot* next;
    };
    std::array<Slot, PoolSize> storage_;
    Slot*  free_list_;
    size_t available_;
};

// Order struct for the pool
struct Order {
    uint64_t id;
    double   price;
    double   quantity;
    bool     is_bid;
    uint64_t timestamp_ns;
};
