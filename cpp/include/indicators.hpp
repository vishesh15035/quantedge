#pragma once
#include <vector>
#include <deque>
#include <string>
#include <cmath>

class EMA {
public:
    explicit EMA(int period);
    double update(double value);
    double value() const;
    bool ready() const;
private:
    int period_; double alpha_, ema_; int count_;
};

class RSI {
public:
    explicit RSI(int period = 14);
    double update(double close);
    double value() const;
    bool ready() const;
private:
    int period_; double avg_gain_, avg_loss_, prev_close_; int count_;
};

class VWAP {
public:
    void update(double price, double volume);
    double value() const;
    void reset();
private:
    double cumulative_pv_ = 0.0, cumulative_vol_ = 0.0;
};

class BollingerBands {
public:
    explicit BollingerBands(int period = 20, double num_std = 2.0);
    void update(double price);
    double upper() const;
    double lower() const;
    double middle() const;
    double bandwidth() const;
    bool ready() const;
private:
    int period_; double num_std_; std::deque<double> window_;
};
