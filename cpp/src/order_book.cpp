#include "../include/order_book.hpp"
#include <chrono>

OrderBook::OrderBook(const std::string& symbol) : symbol_(symbol) {}

long long OrderBook::now_us() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

std::string OrderBook::get_symbol() const { return symbol_; }

void OrderBook::add_bid(double price, double qty) {
    if (price <= 0 || qty <= 0) return;
    bids_[price] += qty;
}

void OrderBook::add_ask(double price, double qty) {
    if (price <= 0 || qty <= 0) return;
    asks_[price] += qty;
}

void OrderBook::remove_level(double price, bool is_bid) {
    if (is_bid) bids_.erase(price);
    else        asks_.erase(price);
}

void OrderBook::update_level(double price, double qty, bool is_bid) {
    if (qty <= 0) { remove_level(price, is_bid); return; }
    if (is_bid) bids_[price] = qty;
    else        asks_[price] = qty;
}

void OrderBook::clear() { bids_.clear(); asks_.clear(); }

double OrderBook::get_spread() const {
    if (bids_.empty() || asks_.empty()) return 0.0;
    return asks_.begin()->first - bids_.begin()->first;
}

double OrderBook::get_mid_price() const {
    if (bids_.empty() || asks_.empty()) return 0.0;
    return (bids_.begin()->first + asks_.begin()->first) / 2.0;
}

double OrderBook::get_vwap(int levels) const {
    double total_pv = 0.0, total_qty = 0.0; int count = 0;
    for (auto& [p, q] : bids_) { if (count++ >= levels) break; total_pv += p*q; total_qty += q; }
    count = 0;
    for (auto& [p, q] : asks_) { if (count++ >= levels) break; total_pv += p*q; total_qty += q; }
    return total_qty > 0 ? total_pv / total_qty : 0.0;
}

double OrderBook::get_order_imbalance() const {
    double bid_qty = 0.0, ask_qty = 0.0; int count = 0;
    for (auto& [p, q] : bids_) { if (count++ >= 5) break; bid_qty += q; }
    count = 0;
    for (auto& [p, q] : asks_) { if (count++ >= 5) break; ask_qty += q; }
    double total = bid_qty + ask_qty;
    return total > 0 ? (bid_qty - ask_qty) / total : 0.0;
}

OrderBookSnapshot OrderBook::snapshot(int depth) const {
    OrderBookSnapshot snap;
    snap.timestamp_us    = now_us();
    snap.best_bid        = bids_.empty() ? 0.0 : bids_.begin()->first;
    snap.best_ask        = asks_.empty() ? 0.0 : asks_.begin()->first;
    snap.spread          = get_spread();
    snap.mid_price       = get_mid_price();
    snap.order_imbalance = get_order_imbalance();
    snap.vwap            = get_vwap(5);
    int count = 0;
    for (auto& [p, q] : bids_) { if (count++ >= depth) break; snap.bids.push_back({p, q, 1}); }
    count = 0;
    for (auto& [p, q] : asks_) { if (count++ >= depth) break; snap.asks.push_back({p, q, 1}); }
    return snap;
}
