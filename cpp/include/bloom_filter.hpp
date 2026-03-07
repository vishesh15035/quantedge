#pragma once
#include <array>
#include <string>
#include <cstdint>
#include <cmath>

// Bloom Filter — fixed size array, no dynamic allocation
// O(1) duplicate order detection, zero false negatives

class BloomFilter {
    static constexpr size_t BITS = 1 << 20;  // 1M bits = 128KB fixed
    static constexpr size_t K    = 7;         // 7 hash functions

public:
    BloomFilter() : bits_{}, count_(0) {}

    void insert(uint64_t id) {
        for (size_t i = 0; i < K; ++i) set_bit(hash(id, i) % BITS);
        ++count_;
    }

    bool possibly_seen(uint64_t id) const {
        for (size_t i = 0; i < K; ++i)
            if (!get_bit(hash(id, i) % BITS)) return false;
        return true;
    }

    bool is_duplicate(uint64_t id) {
        if (possibly_seen(id)) return true;
        insert(id); return false;
    }

    double estimated_fpr() const {
        double filled = (double)count_ * K / BITS;
        return std::pow(1.0 - std::exp(-filled), K);
    }

    size_t bit_count()     const { return BITS; }
    size_t hash_count()    const { return K; }
    size_t element_count() const { return count_; }

private:
    std::array<uint64_t, BITS/64> bits_;
    size_t count_;

    void set_bit(size_t pos) { bits_[pos/64] |=  (1ULL << (pos%64)); }
    bool get_bit(size_t pos) const { return (bits_[pos/64] >> (pos%64)) & 1; }

    uint64_t hash(uint64_t val, size_t seed) const {
        val ^= seed * 0x9e3779b97f4a7c15ULL;
        val ^= val >> 30; val *= 0xbf58476d1ce4e5b9ULL;
        val ^= val >> 27; val *= 0x94d049bb133111ebULL;
        return val ^ (val >> 31);
    }
};
