#pragma once
#include <map>
#include <vector>
#include <string>
#include <chrono>

struct PriceLevel {
    double price;
    double quantity;
    int order_count;
};

struct OrderBookSnapshot {
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    double best_bid;
    double best_ask;
    double spread;
    double mid_price;
    double order_imbalance;
    double vwap;
    long long timestamp_us;
};

class OrderBook {
public:
    explicit OrderBook(const std::string& symbol);
    void add_bid(double price, double qty);
    void add_ask(double price, double qty);
    void remove_level(double price, bool is_bid);
    void update_level(double price, double qty, bool is_bid);
    void clear();
    OrderBookSnapshot snapshot(int depth = 10) const;
    double get_spread() const;
    double get_mid_price() const;
    double get_vwap(int levels) const;
    double get_order_imbalance() const;
    std::string get_symbol() const;
private:
    std::string symbol_;
    std::map<double, double, std::greater<double>> bids_;
    std::map<double, double> asks_;
    long long now_us() const;
};
