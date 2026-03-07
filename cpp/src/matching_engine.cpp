#include "../include/order_book.hpp"
#include "../include/memory_pool.hpp"
#include "../include/lock_free_queue.hpp"
#include "../include/ring_buffer.hpp"
#include "../include/simd_math.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

using TickQueue  = LockFreeQueue<Tick, 4096>;
using TickStore  = RingBuffer<Tick, 65536>;
using OrderPool  = MemoryPool<Order, 65536>;

long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void run_latency_benchmark() {
    std::cout << "\n========================================\n";
    std::cout << "  QuantEdge v2 — Full Latency Benchmark\n";
    std::cout << "========================================\n\n";

    OrderPool  pool;
    TickQueue  queue;
    TickStore  store;
    OrderBook  book("SPY");

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0, 0.05);
    std::uniform_real_distribution<double> qty(100, 1000);

    // 8-stock EMA state for SIMD batch
    std::array<double,8> prices = {450,151,380,275,320,195,88,412};
    std::array<double,8> emas   = prices;
    double alpha = 2.0/10.0;

    std::vector<long long> pool_lat, queue_lat, ring_lat, simd_lat, bs_lat;
    const int N = 10000;

    double price = 450.0;
    for (int i = 0; i < N; ++i) {
        price += noise(rng);

        // 1. Memory pool alloc/free
        auto t0 = now_ns();
        Order* o = pool.allocate();
        o->id = i; o->price = price; o->quantity = qty(rng); o->is_bid = (i%2==0);
        o->timestamp_ns = now_ns();
        pool.deallocate(o);
        pool_lat.push_back(now_ns() - t0);

        // 2. Lock-free queue push/pop
        t0 = now_ns();
        Tick tick{(uint64_t)now_ns(), price, price-0.01, price+0.01, qty(rng), "SPY\0\0\0\0"};
        queue.push(tick);
        auto maybe = queue.pop();
        queue_lat.push_back(now_ns() - t0);

        // 3. Ring buffer write + rolling stats
        t0 = now_ns();
        store.push(tick);
        if (store.size() >= 20) {
            auto stats = store.rolling_stats(20, [](const Tick& t){ return t.price; });
            (void)stats;
        }
        ring_lat.push_back(now_ns() - t0);

        // 4. SIMD batch EMA (8 stocks at once)
        t0 = now_ns();
        for (int j=0; j<8; ++j) prices[j] += noise(rng)*0.1;
        emas = ema8(prices, emas, alpha);
        simd_lat.push_back(now_ns() - t0);

        // 5. Black-Scholes Greeks
        t0 = now_ns();
        auto res = black_scholes(price, 450.0, 30.0/365, 0.05, 0.20, true);
        (void)res;
        bs_lat.push_back(now_ns() - t0);
    }

    auto report = [&](const std::string& name, std::vector<long long>& lat) {
        std::sort(lat.begin(), lat.end());
        double mean = 0; for (auto l:lat) mean+=l; mean/=lat.size();
        std::cout << std::left << std::setw(22) << name
                  << " mean=" << std::setw(6) << (long long)mean << "ns"
                  << " p50="  << std::setw(6) << lat[lat.size()*50/100] << "ns"
                  << " p99="  << std::setw(6) << lat[lat.size()*99/100] << "ns\n";
    };

    std::cout << "--- Per-operation latency (" << N << " iterations) ---\n";
    report("MemPool alloc/free",  pool_lat);
    report("LockFreeQueue push/pop", queue_lat);
    report("RingBuffer write+stats", ring_lat);
    report("SIMD EMA x8 stocks",  simd_lat);
    report("Black-Scholes Greeks", bs_lat);

    // Options chain
    std::cout << "\n--- SPY Options Chain (S=$" << std::fixed << std::setprecision(2) << price << ", T=30d, σ=20%) ---\n";
    std::cout << std::setw(8) << "Strike" << std::setw(10) << "Call $"
              << std::setw(8) << "Delta" << std::setw(8) << "Gamma"
              << std::setw(8) << "Theta" << std::setw(8) << "Vega" << "\n";
    for (double K = price-20; K <= price+20; K += 5) {
        auto c = black_scholes(price, K, 30.0/365, 0.05, 0.20, true);
        std::cout << std::setw(8) << K
                  << std::setw(10) << c.price
                  << std::setw(8) << c.delta
                  << std::setw(8) << c.gamma
                  << std::setw(8) << c.theta
                  << std::setw(8) << c.vega << "\n";
    }

    // Pool stats
    std::cout << "\n--- Memory Pool Stats ---\n";
    std::cout << "Capacity: " << pool.capacity() << " orders\n";
    std::cout << "Used:     " << pool.used()     << " orders\n";
    std::cout << "Available:" << pool.available() << " orders\n";

    std::cout << "\n--- Ring Buffer Stats ---\n";
    std::cout << "Ticks stored: " << store.size() << "\n";
    if (store.size() >= 100) {
        auto s = store.rolling_stats(100, [](const Tick& t){ return t.price; });
        std::cout << "Last 100 ticks — mean=$" << s.mean
                  << " var=" << s.variance
                  << " min=$" << s.min << " max=$" << s.max << "\n";
    }
    std::cout << "========================================\n\n";
}

int main() { run_latency_benchmark(); return 0; }
