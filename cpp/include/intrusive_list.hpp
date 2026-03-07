#pragma once
#include <cstddef>
#include <functional>
#include <iostream>
#include <cassert>

// Intrusive Doubly-Linked List
// Zero allocation — nodes embed list pointers
// O(1) insert/remove with pointer to node
// Used for order chains: each price level has intrusive list of orders

struct IntrusiveNode {
    IntrusiveNode* prev = nullptr;
    IntrusiveNode* next = nullptr;
};

template<typename T, IntrusiveNode T::*NodePtr>
class IntrusiveList {
public:
    IntrusiveList() : size_(0) {
        head_.next = &tail_;
        tail_.prev = &head_;
        head_.prev = nullptr;
        tail_.next = nullptr;
    }

    // O(1) push front — no allocation
    void push_front(T* item) {
        IntrusiveNode* node = &(item->*NodePtr);
        node->next          = head_.next;
        node->prev          = &head_;
        head_.next->prev    = node;
        head_.next          = node;
        ++size_;
    }

    // O(1) push back
    void push_back(T* item) {
        IntrusiveNode* node = &(item->*NodePtr);
        node->prev          = tail_.prev;
        node->next          = &tail_;
        tail_.prev->next    = node;
        tail_.prev          = node;
        ++size_;
    }

    // O(1) remove any node — just need pointer
    void remove(T* item) {
        IntrusiveNode* node = &(item->*NodePtr);
        node->prev->next    = node->next;
        node->next->prev    = node->prev;
        node->prev = node->next = nullptr;
        --size_;
    }

    // O(1) pop front
    T* pop_front() {
        if (empty()) return nullptr;
        IntrusiveNode* node = head_.next;
        head_.next          = node->next;
        node->next->prev    = &head_;
        node->prev = node->next = nullptr;
        --size_;
        return container_of(node);
    }

    T* front() const {
        if (empty()) return nullptr;
        return container_of(head_.next);
    }

    T* back() const {
        if (empty()) return nullptr;
        return container_of(tail_.prev);
    }

    bool   empty() const { return size_ == 0; }
    size_t size()  const { return size_; }

    void for_each(std::function<void(T*)> fn) const {
        IntrusiveNode* cur = head_.next;
        while (cur != &tail_) {
            IntrusiveNode* nxt = cur->next;
            fn(container_of(cur));
            cur = nxt;
        }
    }

private:
    IntrusiveNode head_, tail_;
    size_t        size_;

    static T* container_of(IntrusiveNode* node) {
        // Compute offset of NodePtr member and get parent pointer
        T* dummy = nullptr;
        size_t offset = reinterpret_cast<size_t>(&(dummy->*NodePtr));
        return reinterpret_cast<T*>(reinterpret_cast<char*>(node) - offset);
    }
};

// Order with intrusive node — zero extra allocation
struct IntrusiveOrder {
    uint64_t      id;
    double        price;
    double        quantity;
    bool          is_bid;
    uint64_t      timestamp_ns;
    IntrusiveNode list_node;   // embedded — no extra alloc

    IntrusiveOrder(uint64_t id, double p, double q, bool bid, uint64_t ts)
        : id(id), price(p), quantity(q), is_bid(bid), timestamp_ns(ts) {}
};

using OrderList = IntrusiveList<IntrusiveOrder, &IntrusiveOrder::list_node>;
