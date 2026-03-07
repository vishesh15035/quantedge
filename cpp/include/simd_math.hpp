#pragma once
#include <cmath>
#include <array>
#include <numeric>
#include <algorithm>

// SIMD-style vectorized math for price computation
// On Apple Silicon (ARM): uses NEON intrinsics path
// On x86: uses AVX2 — compiler auto-vectorizes with -O3 -march=native
// We write the scalar version with hints for the compiler to vectorize

// Compute EMA for 8 prices simultaneously
inline std::array<double,8> ema8(
    const std::array<double,8>& prices,
    const std::array<double,8>& prev_emas,
    double alpha)
{
    std::array<double,8> result;
    // Compiler will vectorize this loop with AVX2/NEON at -O3
    for (int i = 0; i < 8; ++i)
        result[i] = alpha * prices[i] + (1.0 - alpha) * prev_emas[i];
    return result;
}

// Compute VWAP for N prices/volumes in one pass
inline double vwap_fast(const double* prices, const double* volumes, size_t n) {
    double pv = 0.0, vol = 0.0;
    // Auto-vectorized: single pass, no branch in loop
    for (size_t i = 0; i < n; ++i) { pv += prices[i]*volumes[i]; vol += volumes[i]; }
    return vol > 0 ? pv / vol : 0.0;
}

// Compute returns for N prices: ret[i] = (p[i] - p[i-1]) / p[i-1]
inline void returns_fast(const double* prices, double* rets, size_t n) {
    for (size_t i = 1; i < n; ++i)
        rets[i-1] = (prices[i] - prices[i-1]) / prices[i-1];
}

// Dot product — core of factor model: alpha = beta . factors
inline double dot_product(const double* a, const double* b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Z-score normalization for signal generation
inline void zscore_normalize(double* data, size_t n) {
    double mean = 0.0, var = 0.0;
    for (size_t i=0; i<n; ++i) mean += data[i];
    mean /= n;
    for (size_t i=0; i<n; ++i) var += (data[i]-mean)*(data[i]-mean);
    var  /= n;
    double std = std::sqrt(var) + 1e-8;
    for (size_t i=0; i<n; ++i) data[i] = (data[i]-mean)/std;
}

// Black-Scholes normal CDF — fast approximation (Abramowitz & Stegun)
inline double norm_cdf(double x) {
    double t = 1.0 / (1.0 + 0.2316419 * std::abs(x));
    double d = 0.3989422803 * std::exp(-x*x/2.0);
    double p = d * t * (0.3193815 + t*(-0.3565638 + t*(1.7814779 + t*(-1.8212560 + t*1.3302744))));
    return x >= 0 ? 1.0 - p : p;
}

// Black-Scholes option pricer — vectorizable over strikes
struct BSResult { double price, delta, gamma, theta, vega, rho; };

inline BSResult black_scholes(double S, double K, double T,
                               double r, double sigma, bool is_call) {
    if (T <= 0) return {0,0,0,0,0,0};
    double sqrtT = std::sqrt(T);
    double d1    = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT);
    double d2    = d1 - sigma * sqrtT;
    double Nd1   = norm_cdf(d1),  Nd2  = norm_cdf(d2);
    double Nnd1  = norm_cdf(-d1), Nnd2 = norm_cdf(-d2);
    double nd1   = 0.3989422803 * std::exp(-d1*d1/2.0);
    double disc  = std::exp(-r*T);

    BSResult res;
    if (is_call) {
        res.price = S*Nd1 - K*disc*Nd2;
        res.delta = Nd1;
        res.rho   = K*T*disc*Nd2 / 100.0;
    } else {
        res.price = K*disc*Nnd2 - S*Nnd1;
        res.delta = Nd1 - 1.0;
        res.rho   = -K*T*disc*Nnd2 / 100.0;
    }
    res.gamma = nd1 / (S * sigma * sqrtT);
    res.vega  = S * nd1 * sqrtT / 100.0;
    res.theta = (-S*nd1*sigma/(2*sqrtT) - r*K*disc*(is_call?Nd2:Nnd2)) / 365.0;
    return res;
}
