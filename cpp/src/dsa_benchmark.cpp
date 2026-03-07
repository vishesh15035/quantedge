#include "../include/lock_free_queue.hpp"
#include "../include/rb_tree_orderbook.hpp"
#include "../include/fibonacci_heap.hpp"
#include "../include/fenwick_segment_tree.hpp"
#include "../include/bloom_filter.hpp"
#include "../include/aho_corasick.hpp"
#include "../include/disruptor.hpp"
#include "../include/wait_free_hashmap.hpp"
#include "../include/intrusive_list.hpp"
#include "../include/memory_pool.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void bench(const std::string& name, int n, std::function<void()> fn) {
    std::vector<long long> lats; lats.reserve(n);
    for (int i=0;i<n;++i) { auto t0=now_ns(); fn(); lats.push_back(now_ns()-t0); }
    std::sort(lats.begin(), lats.end());
    double mean=0; for (auto l:lats) mean+=l; mean/=n;
    std::cout << std::left << std::setw(35) << name
              << " mean=" << std::setw(6) << (long long)mean << "ns"
              << " p50="  << std::setw(6) << lats[n*50/100] << "ns"
              << " p99="  << std::setw(6) << lats[n*99/100] << "ns\n";
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> price_dist(440.0, 460.0);
    std::uniform_real_distribution<double> qty_dist(100, 2000);
    const int N = 10000;

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║   QuantEdge v3 — Advanced DSA Benchmark              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // 1. Red-Black Tree
    std::cout << "── Red-Black Tree Order Book ──\n";
    RBOrderBook rb_book;
    bench("RB-Tree insert bid",  N, [&]{ rb_book.add_bid(price_dist(rng), qty_dist(rng)); });
    bench("RB-Tree insert ask",  N, [&]{ rb_book.add_ask(price_dist(rng), qty_dist(rng)); });
    bench("RB-Tree best_bid",    N, [&]{ volatile double x=rb_book.best_bid(); (void)x; });
    bench("RB-Tree spread",      N, [&]{ volatile double x=rb_book.spread();   (void)x; });
    std::cout << "  Levels: " << rb_book.bid_levels() << " bids / "
              << rb_book.ask_levels() << " asks\n\n";

    // 2. Fibonacci Heap
    std::cout << "── Fibonacci Heap ──\n";
    auto* fib = new FibonacciHeap<double,double,std::greater<double>>();
    bench("FibHeap insert O(1)",      N,   [&]{ fib->insert(price_dist(rng), qty_dist(rng)); });
    bench("FibHeap top O(1)",         N,   [&]{ volatile auto* x=fib->top(); (void)x; });
    bench("FibHeap extract O(log n)", 100, [&]{ if(!fib->empty()){auto* n=fib->extract_min();delete n;} });
    std::cout << "  Size: " << fib->size() << "\n\n";
    delete fib;

    // 3. Fenwick Tree VWAP
    std::cout << "── Fenwick Tree Rolling VWAP ──\n";
    FenwickVWAP fen(1000);
    bench("Fenwick update O(log n)", N, [&]{ fen.update(price_dist(rng), qty_dist(rng)); });
    bench("Fenwick VWAP query",      N, [&]{ volatile double v=fen.vwap(); (void)v; });
    std::cout << "  VWAP: $" << std::fixed << std::setprecision(4) << fen.vwap() << "\n\n";

    // 4. Segment Tree
    std::cout << "── Segment Tree Range Queries ──\n";
    std::vector<double> prices(1000);
    for (auto& p:prices) p=price_dist(rng);
    auto* seg = new SegmentTree<double>(1000);
    seg->build(prices);
    bench("SegTree update O(log n)",    N, [&]{ seg->update(rng()%1000, price_dist(rng)); });
    bench("SegTree range_min O(log n)", N, [&]{ size_t l=rng()%500; volatile double v=seg->range_min(l,l+rng()%499); (void)v; });
    bench("SegTree range_max O(log n)", N, [&]{ size_t l=rng()%500; volatile double v=seg->range_max(l,l+rng()%499); (void)v; });
    std::cout << "  min=$" << seg->range_min(0,999) << " max=$" << seg->range_max(0,999) << "\n\n";
    delete seg;

    // 5. Bloom Filter
    std::cout << "── Bloom Filter Duplicate Detection ──\n";
    auto* bloom = new BloomFilter();
    bench("Bloom insert O(1)", N, [&]{ bloom->insert((uint64_t)rng()); });
    bench("Bloom lookup O(1)", N, [&]{ volatile bool v=bloom->possibly_seen((uint64_t)rng()); (void)v; });
    std::cout << "  Bits: " << bloom->bit_count()
              << " | Hashes: " << bloom->hash_count()
              << " | FPR: " << std::scientific << bloom->estimated_fpr() << "\n\n";
    delete bloom;

    // 6. Aho-Corasick
    std::cout << "── Aho-Corasick Market Pattern Detection ──\n";
    auto ac = build_market_pattern_detector();
    std::vector<double> tick_prices;
    double p = 450.0;
    std::normal_distribution<double> noise(0, 0.5);
    for (int i=0;i<500;++i) { p+=noise(rng); tick_prices.push_back(p); }
    std::string encoded = AhoCorasick::encode_ticks(tick_prices);
    auto matches = ac.search(encoded);
    std::cout << "  Stream: " << encoded.size() << " ticks | "
              << "Patterns: " << ac.pattern_count() << " | "
              << "Matches: "  << matches.size() << "\n";
    for (size_t i=0;i<std::min(matches.size(),(size_t)3);++i)
        std::cout << "    [" << matches[i].position << "] " << matches[i].pattern << "\n";
    std::cout << "\n";

    // 7. Wait-free HashMap — small capacity to stay on heap
    std::cout << "── Wait-Free HashMap ──\n";
    auto* wfmap = new WaitFreeHashMap<uint64_t,double,4096>();
    bench("WaitFree insert O(1)", N, [&]{ wfmap->insert(rng()%4000, price_dist(rng)); });
    bench("WaitFree lookup O(1)", N, [&]{ auto v=wfmap->get(rng()%4000); (void)v; });
    std::cout << "  Entries: " << wfmap->size() << "\n\n";
    delete wfmap;

    // 8. Intrusive List
    std::cout << "── Intrusive List Zero-Allocation Orders ──\n";
    auto* pool = new MemoryPool<IntrusiveOrder,65536>();
    OrderList olist;
    std::vector<IntrusiveOrder*> orders;
    bench("Intrusive push_back O(1)", N, [&]{
        void* mem = pool->allocate();
        if (mem) {
            auto* o = new(mem) IntrusiveOrder(rng(),price_dist(rng),qty_dist(rng),true,now_ns());
            olist.push_back(o); orders.push_back(o);
        }
    });
    bench("Intrusive pop_front O(1)", std::min(N,(int)orders.size()), [&]{
        auto* o = olist.pop_front();
        if (o) pool->deallocate(o);
    });
    std::cout << "  Remaining: " << olist.size() << " | Pool used: " << pool->used() << "\n\n";
    delete pool;

    // 9. Disruptor
    std::cout << "── LMAX Disruptor Lock-free IPC ──\n";
    auto* dis = new Disruptor<Tick,65536>();
    bench("Disruptor publish O(1)", N, [&]{
        int64_t seq = dis->next();
        Tick t{(uint64_t)now_ns(), price_dist(rng), 0, 0, qty_dist(rng), {'S','P','Y',0,0,0,0,0}};
        dis->publish(seq, t);
    });
    std::cout << "\n";
    delete dis;

    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Complexity Summary                                   ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  RB-Tree insert/delete/find  O(log n)                ║\n";
    std::cout << "║  Fib-Heap insert             O(1) amortized          ║\n";
    std::cout << "║  Fib-Heap extract-min        O(log n) amortized      ║\n";
    std::cout << "║  Fib-Heap decrease-key       O(1) amortized          ║\n";
    std::cout << "║  Fenwick VWAP update/query   O(log n)                ║\n";
    std::cout << "║  Segment Tree range query    O(log n)                ║\n";
    std::cout << "║  Bloom Filter insert/lookup  O(k) ~ O(1)             ║\n";
    std::cout << "║  Aho-Corasick search         O(n + m + z)            ║\n";
    std::cout << "║  Wait-free HashMap           O(1) guaranteed         ║\n";
    std::cout << "║  Intrusive List ops          O(1) zero allocation    ║\n";
    std::cout << "║  Disruptor publish           O(1) lock-free          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    return 0;
}
