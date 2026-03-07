import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.data.fetcher import MarketDataFetcher
from python.signals.momentum_signal import MomentumSignal, ShortTermMomentum, VWAPMomentum
from python.signals.mean_reversion_signal import BollingerReversion, RSIReversion, ZScoreReversion
from python.backtest.engine import Backtester
import json
from pathlib import Path

print("\n" + "="*60)
print("  QuantEdge — Multi-Signal Backtest")
print("="*60)

fetcher = MarketDataFetcher(cache_dir="./data/cache")
tickers = ["SPY", "AAPL", "MSFT"]
print(f"\nFetching: {tickers}")
data = fetcher.fetch_multiple(tickers, period="5y")

signals = {
    "Momentum 12-1m":    MomentumSignal(),
    "Short Momentum 5d": ShortTermMomentum(),
    "VWAP Momentum":     VWAPMomentum(),
    "Bollinger Rev":     BollingerReversion(),
    "RSI Reversion":     RSIReversion(),
    "Z-Score Rev":       ZScoreReversion(),
}

bt      = Backtester(initial_capital=100_000, transaction_cost_bps=5, slippage_bps=2)
results = {}

print("\n" + "-"*60)
print(f"{'Signal':<22} {'Ticker':<6} {'Sharpe':>8} {'Return':>9} {'MaxDD':>8} {'Trades':>7}")
print("-"*60)

for ticker, ticker_data in data.items():
    for sig_name, signal in signals.items():
        try:
            sig_vals = signal.compute(ticker_data)
            res      = bt.run(ticker_data['Close'], sig_vals, ticker=ticker, signal_name=sig_name)
            results[f"{sig_name}_{ticker}"] = res
            print(f"{sig_name:<22} {ticker:<6} "
                  f"{res.sharpe_ratio:>8.3f} "
                  f"{res.total_return*100:>8.1f}% "
                  f"{res.max_drawdown*100:>7.1f}% "
                  f"{res.num_trades:>7}")
        except Exception as e:
            print(f"{sig_name:<22} {ticker:<6} ERROR: {e}")

best_key = max(results.items(), key=lambda x: x[1].sharpe_ratio)
print("\n" + "="*60)
print(f"  BEST: {best_key[0]}")
best_key[1].summary()

Path("./results").mkdir(exist_ok=True)
with open("./results/backtest_results.json", "w") as f:
    json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
print("\nSaved → results/backtest_results.json")
