#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <functional>

// ── Fenwick Tree (Binary Indexed Tree) ─────────────────
// O(log n) prefix sum queries — used for rolling VWAP
// Classic DSA: each index stores partial sums
template<typename T>
class FenwickTree {
public:
    explicit FenwickTree(size_t n) : n_(n), tree_(n+1, T{}) {}

    // Update index i (1-based) by delta
    void update(size_t i, T delta) {
        for (; i <= n_; i += i & (-i))
            tree_[i] += delta;
    }

    // Prefix sum [1..i]
    T query(size_t i) const {
        T sum{};
        for (; i > 0; i -= i & (-i))
            sum += tree_[i];
        return sum;
    }

    // Range sum [l..r]
    T range_query(size_t l, size_t r) const {
        return query(r) - (l > 1 ? query(l-1) : T{});
    }

    size_t size() const { return n_; }

private:
    size_t         n_;
    std::vector<T> tree_;
};

// Rolling VWAP using two Fenwick Trees
// O(log n) per update and query
class FenwickVWAP {
public:
    explicit FenwickVWAP(size_t capacity)
        : cap_(capacity), pos_(0), count_(0),
          pv_tree_(capacity), vol_tree_(capacity) {}

    void update(double price, double volume) {
        size_t idx = (pos_ % cap_) + 1;
        // Subtract old value at this position before overwriting
        if (count_ >= cap_) {
            double old_pv  = old_pv_[idx-1];
            double old_vol = old_vol_[idx-1];
            pv_tree_.update(idx, -old_pv);
            vol_tree_.update(idx, -old_vol);
        }
        double pv = price * volume;
        pv_tree_.update(idx, pv);
        vol_tree_.update(idx, volume);
        old_pv_[idx-1]  = pv;
        old_vol_[idx-1] = volume;
        ++pos_; ++count_;
    }

    double vwap() const {
        double total_pv  = pv_tree_.query(cap_);
        double total_vol = vol_tree_.query(cap_);
        return total_vol > 0 ? total_pv / total_vol : 0.0;
    }

    size_t count() const { return std::min(count_, cap_); }

private:
    size_t cap_, pos_, count_;
    FenwickTree<double>   pv_tree_, vol_tree_;
    std::vector<double>   old_pv_  = std::vector<double>(cap_, 0.0);
    std::vector<double>   old_vol_ = std::vector<double>(cap_, 0.0);
};

// ── Segment Tree ────────────────────────────────────────
// O(log n) range min/max/sum queries on price history
// Supports point updates and arbitrary range queries
template<typename T>
class SegmentTree {
public:
    struct Node {
        T    min_val;
        T    max_val;
        T    sum;
        T    lazy;
    };

    explicit SegmentTree(size_t n)
        : n_(n), tree_(4*n, {std::numeric_limits<T>::max(),
                              std::numeric_limits<T>::lowest(), T{}, T{}}) {}

    void build(const std::vector<T>& arr, size_t node=1,
               size_t start=0, size_t end=SIZE_MAX) {
        if (end == SIZE_MAX) end = n_ - 1;
        if (start == end) {
            tree_[node] = {arr[start], arr[start], arr[start], T{}};
            return;
        }
        size_t mid = (start + end) / 2;
        build(arr, 2*node,   start, mid);
        build(arr, 2*node+1, mid+1, end);
        pull_up(node);
    }

    void update(size_t idx, T val, size_t node=1,
                size_t start=0, size_t end=SIZE_MAX) {
        if (end == SIZE_MAX) end = n_ - 1;
        if (start == end) {
            tree_[node] = {val, val, val, T{}};
            return;
        }
        size_t mid = (start + end) / 2;
        if (idx <= mid) update(idx, val, 2*node,   start, mid);
        else            update(idx, val, 2*node+1, mid+1, end);
        pull_up(node);
    }

    // Range min query [l, r]
    T range_min(size_t l, size_t r, size_t node=1,
                size_t start=0, size_t end=SIZE_MAX) const {
        if (end == SIZE_MAX) end = n_ - 1;
        if (r < start || end < l) return std::numeric_limits<T>::max();
        if (l <= start && end <= r) return tree_[node].min_val;
        size_t mid = (start + end) / 2;
        return std::min(range_min(l,r,2*node,start,mid),
                        range_min(l,r,2*node+1,mid+1,end));
    }

    // Range max query [l, r]
    T range_max(size_t l, size_t r, size_t node=1,
                size_t start=0, size_t end=SIZE_MAX) const {
        if (end == SIZE_MAX) end = n_ - 1;
        if (r < start || end < l) return std::numeric_limits<T>::lowest();
        if (l <= start && end <= r) return tree_[node].max_val;
        size_t mid = (start + end) / 2;
        return std::max(range_max(l,r,2*node,start,mid),
                        range_max(l,r,2*node+1,mid+1,end));
    }

    // Range sum query [l, r]
    T range_sum(size_t l, size_t r, size_t node=1,
                size_t start=0, size_t end=SIZE_MAX) const {
        if (end == SIZE_MAX) end = n_ - 1;
        if (r < start || end < l) return T{};
        if (l <= start && end <= r) return tree_[node].sum;
        size_t mid = (start + end) / 2;
        return range_sum(l,r,2*node,start,mid) +
               range_sum(l,r,2*node+1,mid+1,end);
    }

    size_t size() const { return n_; }

private:
    size_t            n_;
    std::vector<Node> tree_;

    void pull_up(size_t node) {
        tree_[node].min_val = std::min(tree_[2*node].min_val, tree_[2*node+1].min_val);
        tree_[node].max_val = std::max(tree_[2*node].max_val, tree_[2*node+1].max_val);
        tree_[node].sum     = tree_[2*node].sum + tree_[2*node+1].sum;
    }
};
