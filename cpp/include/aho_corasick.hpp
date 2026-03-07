#pragma once
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <iostream>

// Aho-Corasick — O(n + m + z) multi-pattern search
// n = text length, m = total pattern length, z = matches
// Used to detect trading patterns in tick stream:
// e.g. "BULL_RUN", "FLASH_CRASH", "PUMP_DUMP"

struct ACMatch {
    size_t      position;
    std::string pattern;
    std::string context;
};

class AhoCorasick {
public:
    struct State {
        std::unordered_map<char, int> next;
        int                           fail;
        std::vector<std::string>      output;
        State() : fail(0) {}
    };

    AhoCorasick() { states_.emplace_back(); }

    void add_pattern(const std::string& pattern) {
        int cur = 0;
        for (char c : pattern) {
            auto it = states_[cur].next.find(c);
            if (it == states_[cur].next.end()) {
                states_[cur].next[c] = (int)states_.size();
                states_.emplace_back();
            }
            cur = states_[cur].next[c];
        }
        states_[cur].output.push_back(pattern);
        patterns_.push_back(pattern);
    }

    // Build failure links — must call after all add_pattern()
    void build() {
        std::queue<int> q;
        for (auto& [ch, s] : states_[0].next) {
            states_[s].fail = 0;
            q.push(s);
        }
        while (!q.empty()) {
            int r = q.front(); q.pop();
            for (auto& [ch, s] : states_[r].next) {
                int f = states_[r].fail;
                while (f && !states_[f].next.count(ch)) f = states_[f].fail;
                auto it = states_[f].next.find(ch);
                states_[s].fail = (it != states_[f].next.end() && it->second != s)
                                   ? it->second : 0;
                // Merge output
                for (auto& p : states_[states_[s].fail].output)
                    states_[s].output.push_back(p);
                q.push(s);
            }
        }
        built_ = true;
    }

    // Search text for all patterns — O(n + m + z)
    std::vector<ACMatch> search(const std::string& text) const {
        std::vector<ACMatch> matches;
        int cur = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            while (cur && !states_[cur].next.count(c)) cur = states_[cur].fail;
            auto it = states_[cur].next.find(c);
            cur = (it != states_[cur].next.end()) ? it->second : 0;
            for (auto& pat : states_[cur].output) {
                size_t start   = i + 1 - pat.size();
                size_t ctx_s   = start > 5 ? start-5 : 0;
                size_t ctx_e   = std::min(i+5, text.size()-1);
                matches.push_back({i, pat, text.substr(ctx_s, ctx_e-ctx_s+1)});
            }
        }
        return matches;
    }

    // Encode tick stream as string for pattern detection
    // price up = 'U', down = 'D', flat = 'F', big_up = 'B', big_down = 'C'
    static std::string encode_ticks(const std::vector<double>& prices,
                                     double threshold = 0.002) {
        std::string encoded;
        for (size_t i = 1; i < prices.size(); ++i) {
            double ret = (prices[i] - prices[i-1]) / prices[i-1];
            if      (ret >  threshold*3) encoded += 'B';  // big up
            else if (ret >  threshold)   encoded += 'U';  // up
            else if (ret < -threshold*3) encoded += 'C';  // crash
            else if (ret < -threshold)   encoded += 'D';  // down
            else                         encoded += 'F';  // flat
        }
        return encoded;
    }

    size_t pattern_count() const { return patterns_.size(); }

private:
    std::vector<State>  states_;
    std::vector<std::string> patterns_;
    bool built_ = false;
};

// Pre-built pattern library for common market events
inline AhoCorasick build_market_pattern_detector() {
    AhoCorasick ac;
    // Bull patterns
    ac.add_pattern("UUUUU");    // strong uptrend
    ac.add_pattern("BUBUB");    // big up alternating
    ac.add_pattern("BUUU");     // breakout
    ac.add_pattern("FUUUU");    // slow bull
    // Bear patterns
    ac.add_pattern("DDDDD");    // strong downtrend
    ac.add_pattern("CDCDC");    // flash crash pattern
    ac.add_pattern("CDDD");     // breakdown
    ac.add_pattern("FDDDD");    // slow bear
    // Reversal patterns
    ac.add_pattern("DDDBUUU");  // V-bottom reversal
    ac.add_pattern("UUUCDDD");  // inverted V top
    ac.add_pattern("DUDU");     // consolidation
    ac.add_pattern("UDUDUD");   // range bound
    // Volatility patterns
    ac.add_pattern("BCBC");     // extreme volatility
    ac.add_pattern("CBCB");     // extreme volatility
    ac.add_pattern("BCBCBC");   // flash crash + recovery
    ac.build();
    return ac;
}
