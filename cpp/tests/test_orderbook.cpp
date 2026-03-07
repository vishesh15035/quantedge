#include "../include/order_book.hpp"
#include "../include/indicators.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_order_book() {
    OrderBook book("TEST");
    book.add_bid(100.0, 500); book.add_bid(99.5, 300);
    book.add_ask(100.5, 400); book.add_ask(101.0, 200);
    assert(std::abs(book.get_spread() - 0.5) < 1e-9);
    assert(std::abs(book.get_mid_price() - 100.25) < 1e-9);
    assert(book.get_order_imbalance() > 0);
    std::cout << "[PASS] OrderBook\n";
}

void test_ema() {
    EMA ema(3);
    double v1=ema.update(10), v2=ema.update(20), v3=ema.update(30);
    assert(ema.ready()); assert(v3>v2 && v2>v1);
    std::cout << "[PASS] EMA\n";
}

void test_rsi() {
    RSI rsi(14);
    for (int i=0;i<20;++i) rsi.update(100.0+i);
    assert(rsi.value() > 50.0);
    std::cout << "[PASS] RSI\n";
}

void test_bollinger() {
    BollingerBands bb(5, 2.0);
    for (int i=0;i<5;++i) bb.update(100.0);
    assert(bb.ready()); assert(bb.upper()>=bb.middle()); assert(bb.lower()<=bb.middle());
    std::cout << "[PASS] Bollinger\n";
}

void test_vwap() {
    VWAP vwap;
    vwap.update(100.0, 1000); vwap.update(102.0, 2000);
    double expected = (100*1000.0 + 102*2000.0) / 3000.0;
    assert(std::abs(vwap.value() - expected) < 1e-6);
    std::cout << "[PASS] VWAP\n";
}

int main() {
    std::cout << "\n=== QuantEdge C++ Tests ===\n";
    test_order_book(); test_ema(); test_rsi(); test_bollinger(); test_vwap();
    std::cout << "=== All passed ===\n\n";
    return 0;
}
