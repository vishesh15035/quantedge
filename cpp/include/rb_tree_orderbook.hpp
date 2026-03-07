#pragma once
#include <cstdint>
#include <functional>
#include <cassert>
#include <iostream>
#include <iomanip>

// Red-Black Tree Order Book
// O(log n) insert/delete/find vs std::map's same complexity
// but with guaranteed rebalancing and no allocator overhead
// Color: 0=Black, 1=Red

template<typename Key, typename Value, typename Compare = std::less<Key>>
class RBTree {
public:
    enum Color { BLACK=0, RED=1 };

    struct Node {
        Key    key;
        Value  value;
        Color  color;
        Node*  left;
        Node*  right;
        Node*  parent;
        Node(Key k, Value v)
            : key(k), value(v), color(RED),
              left(nullptr), right(nullptr), parent(nullptr) {}
    };

    RBTree() : root_(nullptr), size_(0), nil_(new Node(Key{}, Value{})) {
        nil_->color = BLACK;
        nil_->left = nil_->right = nil_->parent = nil_;
        root_ = nil_;
    }

    ~RBTree() { clear(root_); delete nil_; }

    void insert(const Key& k, const Value& v) {
        Node* z = new Node(k, v);
        z->left = z->right = z->parent = nil_;
        Node* y = nil_;
        Node* x = root_;
        while (x != nil_) {
            y = x;
            if (cmp_(k, x->key))      x = x->left;
            else if (cmp_(x->key, k)) x = x->right;
            else { x->value = v; delete z; return; } // update
        }
        z->parent = y;
        if (y == nil_)                root_    = z;
        else if (cmp_(k, y->key))     y->left  = z;
        else                          y->right = z;
        ++size_;
        insert_fixup(z);
    }

    bool remove(const Key& k) {
        Node* z = find_node(k);
        if (z == nil_) return false;
        delete_node(z);
        --size_;
        return true;
    }

    Value* find(const Key& k) {
        Node* x = find_node(k);
        return x != nil_ ? &x->value : nullptr;
    }

    // Min/Max — O(log n)
    Node* minimum() const { return tree_min(root_); }
    Node* maximum() const { return tree_max(root_); }

    size_t size()  const { return size_; }
    bool   empty() const { return size_ == 0; }

    // In-order traversal
    void inorder(std::function<void(const Key&, const Value&)> fn) const {
        inorder_(root_, fn);
    }

private:
    Node*   root_;
    Node*   nil_;   // sentinel
    size_t  size_;
    Compare cmp_;

    Node* tree_min(Node* x) const {
        while (x->left != nil_) x = x->left;
        return x;
    }
    Node* tree_max(Node* x) const {
        while (x->right != nil_) x = x->right;
        return x;
    }
    Node* find_node(const Key& k) const {
        Node* x = root_;
        while (x != nil_) {
            if      (cmp_(k, x->key)) x = x->left;
            else if (cmp_(x->key, k)) x = x->right;
            else                      return x;
        }
        return nil_;
    }

    void rotate_left(Node* x) {
        Node* y  = x->right;
        x->right = y->left;
        if (y->left != nil_) y->left->parent = x;
        y->parent = x->parent;
        if      (x->parent == nil_)        root_           = y;
        else if (x == x->parent->left)     x->parent->left = y;
        else                               x->parent->right= y;
        y->left   = x;
        x->parent = y;
    }

    void rotate_right(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        if (y->right != nil_) y->right->parent = x;
        y->parent = x->parent;
        if      (x->parent == nil_)        root_            = y;
        else if (x == x->parent->right)    x->parent->right = y;
        else                               x->parent->left  = y;
        y->right  = x;
        x->parent = y;
    }

    void insert_fixup(Node* z) {
        while (z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;
                if (y->color == RED) {
                    z->parent->color          = BLACK;
                    y->color                  = BLACK;
                    z->parent->parent->color  = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) { z = z->parent; rotate_left(z); }
                    z->parent->color         = BLACK;
                    z->parent->parent->color = RED;
                    rotate_right(z->parent->parent);
                }
            } else {
                Node* y = z->parent->parent->left;
                if (y->color == RED) {
                    z->parent->color         = BLACK;
                    y->color                 = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) { z = z->parent; rotate_right(z); }
                    z->parent->color         = BLACK;
                    z->parent->parent->color = RED;
                    rotate_left(z->parent->parent);
                }
            }
        }
        root_->color = BLACK;
    }

    void transplant(Node* u, Node* v) {
        if      (u->parent == nil_)      root_            = v;
        else if (u == u->parent->left)   u->parent->left  = v;
        else                             u->parent->right = v;
        v->parent = u->parent;
    }

    void delete_node(Node* z) {
        Node* y = z;
        Node* x;
        Color orig = y->color;
        if      (z->left == nil_)  { x = z->right; transplant(z, z->right); }
        else if (z->right == nil_) { x = z->left;  transplant(z, z->left);  }
        else {
            y      = tree_min(z->right);
            orig   = y->color;
            x      = y->right;
            if (y->parent == z) x->parent = y;
            else { transplant(y, y->right); y->right = z->right; y->right->parent = y; }
            transplant(z, y);
            y->left         = z->left;
            y->left->parent = y;
            y->color        = z->color;
        }
        delete z;
        if (orig == BLACK) delete_fixup(x);
    }

    void delete_fixup(Node* x) {
        while (x != root_ && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;
                if (w->color == RED) {
                    w->color          = BLACK;
                    x->parent->color  = RED;
                    rotate_left(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    w->color = RED; x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
                        w->left->color = BLACK; w->color = RED;
                        rotate_right(w); w = x->parent->right;
                    }
                    w->color          = x->parent->color;
                    x->parent->color  = BLACK;
                    w->right->color   = BLACK;
                    rotate_left(x->parent);
                    x = root_;
                }
            } else {
                Node* w = x->parent->left;
                if (w->color == RED) {
                    w->color = BLACK; x->parent->color = RED;
                    rotate_right(x->parent); w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    w->color = RED; x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
                        w->right->color = BLACK; w->color = RED;
                        rotate_left(w); w = x->parent->left;
                    }
                    w->color         = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color   = BLACK;
                    rotate_right(x->parent);
                    x = root_;
                }
            }
        }
        x->color = BLACK;
    }

    void inorder_(Node* x, std::function<void(const Key&,const Value&)>& fn) const {
        if (x == nil_) return;
        inorder_(x->left, fn);
        fn(x->key, x->value);
        inorder_(x->right, fn);
    }

    void clear(Node* x) {
        if (x == nil_) return;
        clear(x->left); clear(x->right); delete x;
    }
};

// RB-Tree based Order Book
// Bids: max at top (reverse), Asks: min at top
class RBOrderBook {
public:
    using BidTree = RBTree<double, double, std::greater<double>>;
    using AskTree = RBTree<double, double, std::less<double>>;

    void add_bid(double price, double qty) { bids_.insert(price, qty); }
    void add_ask(double price, double qty) { asks_.insert(price, qty); }
    void remove_bid(double price)          { bids_.remove(price); }
    void remove_ask(double price)          { asks_.remove(price); }

    double best_bid() const { auto* n=bids_.maximum(); return n?n->key:0.0; }
    double best_ask() const { auto* n=asks_.minimum(); return n?n->key:0.0; }
    double spread()   const { return best_ask() - best_bid(); }
    double mid()      const { return (best_bid() + best_ask()) / 2.0; }

    size_t bid_levels() const { return bids_.size(); }
    size_t ask_levels() const { return asks_.size(); }

    void print_top(int n=5) const {
        std::cout << "\n--- RB-Tree Order Book ---\n";
        int cnt=0;
        asks_.inorder([&](double p, double q){
            if(cnt++<n) std::cout<<"  ASK $"<<std::fixed<<std::setprecision(2)<<p<<" x"<<q<<"\n";
        });
        std::cout<<"  Mid: $"<<mid()<<" Spread: $"<<spread()<<"\n";
        cnt=0;
        bids_.inorder([&](double p, double q){
            if(cnt++<n) std::cout<<"  BID $"<<std::fixed<<std::setprecision(2)<<p<<" x"<<q<<"\n";
        });
    }

private:
    BidTree bids_;
    AskTree asks_;
};
