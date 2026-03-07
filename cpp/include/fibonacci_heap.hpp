#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include <limits>
#include <iostream>

// Fibonacci Heap — amortized O(1) insert, O(log n) extract-min
// Used as priority queue for best bid/ask tracking
// Key advantage: O(1) decrease-key vs O(log n) in binary heap

template<typename Key, typename Value, typename Compare = std::less<Key>>
class FibonacciHeap {
public:
    struct Node {
        Key    key;
        Value  value;
        int    degree;
        bool   marked;
        Node*  parent;
        Node*  child;
        Node*  left;
        Node*  right;

        Node(Key k, Value v)
            : key(k), value(v), degree(0), marked(false),
              parent(nullptr), child(nullptr), left(this), right(this) {}
    };

    FibonacciHeap() : min_(nullptr), size_(0) {}

    ~FibonacciHeap() { clear(); }

    // O(1) amortized
    Node* insert(const Key& k, const Value& v) {
        Node* n = new Node(k, v);
        add_to_root(n);
        if (!min_ || cmp_(n->key, min_->key)) min_ = n;
        ++size_;
        return n;
    }

    // O(1)
    Node* top() const { return min_; }

    // O(log n) amortized
    Node* extract_min() {
        if (!min_) return nullptr;
        Node* z = min_;

        // Add children to root list
        if (z->child) {
            std::vector<Node*> children;
            Node* c = z->child;
            do { children.push_back(c); c = c->right; } while (c != z->child);
            for (auto* ch : children) {
                remove_from_list(ch);
                add_to_root(ch);
                ch->parent = nullptr;
            }
        }

        remove_from_list(z);
        if (z == z->right) { min_ = nullptr; }
        else { min_ = z->right; consolidate(); }
        --size_;
        return z;  // caller owns memory
    }

    // O(1) amortized — key advantage over binary heap
    void decrease_key(Node* x, const Key& new_key) {
        if (cmp_(x->key, new_key)) return;  // new key must be smaller
        x->key = new_key;
        Node* p = x->parent;
        if (p && cmp_(x->key, p->key)) {
            cut(x, p);
            cascading_cut(p);
        }
        if (cmp_(x->key, min_->key)) min_ = x;
    }

    size_t size()  const { return size_; }
    bool   empty() const { return size_ == 0; }

private:
    Node*  min_;
    size_t size_;
    Compare cmp_;

    void add_to_root(Node* n) {
        if (!min_) { n->left = n->right = n; }
        else {
            n->right       = min_;
            n->left        = min_->left;
            min_->left->right = n;
            min_->left     = n;
        }
    }

    void remove_from_list(Node* n) {
        n->left->right = n->right;
        n->right->left = n->left;
    }

    void link(Node* y, Node* x) {
        remove_from_list(y);
        y->parent = x;
        if (!x->child) { x->child = y; y->left = y->right = y; }
        else {
            y->right = x->child;
            y->left  = x->child->left;
            x->child->left->right = y;
            x->child->left        = y;
        }
        ++x->degree;
        y->marked = false;
    }

    void consolidate() {
        int max_deg = (int)(std::log2(size_) + 2);
        std::vector<Node*> A(max_deg + 1, nullptr);

        std::vector<Node*> roots;
        Node* cur = min_;
        do { roots.push_back(cur); cur = cur->right; } while (cur != min_);

        for (auto* w : roots) {
            Node* x = w;
            int   d = x->degree;
            while (d < (int)A.size() && A[d]) {
                Node* y = A[d];
                if (cmp_(y->key, x->key)) std::swap(x, y);
                link(y, x);
                A[d++] = nullptr;
            }
            if (d >= (int)A.size()) A.resize(d+1, nullptr);
            A[d] = x;
        }

        min_ = nullptr;
        for (auto* n : A) {
            if (!n) continue;
            if (!min_) { min_ = n; n->left = n->right = n; }
            else {
                add_to_root(n);
                if (cmp_(n->key, min_->key)) min_ = n;
            }
        }
    }

    void cut(Node* x, Node* y) {
        if (x->right == x) y->child = nullptr;
        else {
            remove_from_list(x);
            if (y->child == x) y->child = x->right;
        }
        --y->degree;
        add_to_root(x);
        x->parent = nullptr;
        x->marked = false;
    }

    void cascading_cut(Node* y) {
        Node* z = y->parent;
        if (z) {
            if (!y->marked) { y->marked = true; }
            else { cut(y, z); cascading_cut(z); }
        }
    }

    void clear() {
        if (!min_) return;
        std::vector<Node*> to_del;
        collect(min_, to_del);
        for (auto* n : to_del) delete n;
        min_ = nullptr; size_ = 0;
    }

    void collect(Node* n, std::vector<Node*>& out) {
        if (!n) return;
        Node* cur = n;
        do {
            out.push_back(cur);
            if (cur->child) collect(cur->child, out);
            cur = cur->right;
        } while (cur != n);
    }
};
