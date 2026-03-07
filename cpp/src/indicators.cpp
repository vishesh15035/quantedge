#include "../include/indicators.hpp"
#include <numeric>

EMA::EMA(int period) : period_(period), alpha_(2.0/(period+1)), ema_(0.0), count_(0) {}
double EMA::update(double v) { ema_ = count_==0 ? v : alpha_*v+(1.0-alpha_)*ema_; ++count_; return ema_; }
double EMA::value() const { return ema_; }
bool   EMA::ready() const { return count_ >= period_; }

RSI::RSI(int period) : period_(period), avg_gain_(0), avg_loss_(0), prev_close_(0), count_(0) {}
double RSI::update(double close) {
    if (count_==0) { prev_close_=close; ++count_; return 50.0; }
    double change=close-prev_close_, gain=change>0?change:0.0, loss=change<0?-change:0.0;
    if (count_<=period_) { avg_gain_+=gain/period_; avg_loss_+=loss/period_; }
    else { avg_gain_=(avg_gain_*(period_-1)+gain)/period_; avg_loss_=(avg_loss_*(period_-1)+loss)/period_; }
    prev_close_=close; ++count_;
    if (avg_loss_==0) return 100.0;
    return 100.0-(100.0/(1.0+avg_gain_/avg_loss_));
}
double RSI::value() const { return avg_loss_==0?100.0:100.0-(100.0/(1.0+avg_gain_/avg_loss_)); }
bool   RSI::ready() const { return count_ > period_; }

void   VWAP::update(double p, double v) { cumulative_pv_+=p*v; cumulative_vol_+=v; }
double VWAP::value() const { return cumulative_vol_>0?cumulative_pv_/cumulative_vol_:0.0; }
void   VWAP::reset() { cumulative_pv_=cumulative_vol_=0.0; }

BollingerBands::BollingerBands(int p, double s) : period_(p), num_std_(s) {}
void BollingerBands::update(double price) { window_.push_back(price); if ((int)window_.size()>period_) window_.pop_front(); }
double BollingerBands::middle() const { if(window_.empty())return 0.0; double s=0; for(auto v:window_)s+=v; return s/window_.size(); }
double BollingerBands::bandwidth() const {
    if((int)window_.size()<period_)return 0.0;
    double m=middle(), sq=0; for(auto v:window_)sq+=(v-m)*(v-m);
    return std::sqrt(sq/window_.size());
}
double BollingerBands::upper() const { return middle()+num_std_*bandwidth(); }
double BollingerBands::lower() const { return middle()-num_std_*bandwidth(); }
bool   BollingerBands::ready() const { return (int)window_.size()>=period_; }
