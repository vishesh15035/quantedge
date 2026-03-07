"""
Automated Training Pipeline
Runs daily via scheduler — retrains all models, logs to MLflow
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.data.fetcher import MarketDataFetcher
from python.signals.momentum_signal import MomentumSignal, ShortTermMomentum
from python.signals.mean_reversion_signal import RSIReversion, BollingerReversion
from python.backtest.engine import Backtester
from python.universe.stocks import UNIVERSE, SECTOR_ETFS
from mlops.experiment_tracker import ExperimentTracker
from mlops.drift_detector import DriftDetector
from datetime import datetime

class TrainingPipeline:
    def __init__(self):
        self.fetcher  = MarketDataFetcher(cache_dir="./data/cache")
        self.tracker  = ExperimentTracker()
        self.detector = DriftDetector()
        self.bt       = Backtester()

    def run(self, tickers: list = None, period: str = "2y") -> dict:
        tickers = tickers or ["SPY","AAPL","MSFT","GOOGL","NVDA","JPM","JNJ"]
        print(f"\n{'='*60}")
        print(f"  QuantEdge Training Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Tickers: {tickers}")
        print(f"{'='*60}\n")

        # 1. Fetch data
        print("[1/5] Fetching market data...")
        data = self.fetcher.fetch_multiple(tickers, period=period)

        # 2. Check drift
        print("[2/5] Checking data drift...")
        close = pd.DataFrame({t: data[t]['Close'] for t in tickers if t in data})
        rets  = close.pct_change().dropna()
        if self.detector.reference_data is None:
            self.detector.set_reference(rets.iloc[:len(rets)//2])
        should_retrain = self.detector.should_retrain(rets.iloc[len(rets)//2:])
        print(f"  Should retrain: {should_retrain}")

        # 3. Run signals + backtest
        print("[3/5] Running backtests...")
        signals = {
            "momentum":       MomentumSignal(),
            "short_momentum": ShortTermMomentum(),
            "rsi":            RSIReversion(),
            "bollinger":      BollingerReversion(),
        }
        results  = []
        best_run = None
        best_sharpe = -np.inf

        for ticker in tickers:
            if ticker not in data: continue
            td = data[ticker]
            for sig_name, sig_obj in signals.items():
                try:
                    sig_vals = sig_obj.compute(td)
                    res      = self.bt.run(td['Close'], sig_vals,
                                          ticker=ticker, signal_name=sig_name)
                    run_id   = self.tracker.log_backtest(res.to_dict())
                    results.append(res.to_dict())
                    if res.sharpe_ratio > best_sharpe:
                        best_sharpe = res.sharpe_ratio
                        best_run    = (sig_name, ticker, res)
                except Exception as e:
                    print(f"  Error {sig_name}/{ticker}: {e}")

        # 4. Report best
        print(f"\n[4/5] Best model this run:")
        if best_run:
            sig_name, ticker, res = best_run
            res.summary()

        # 5. Summary
        print(f"\n[5/5] Pipeline complete")
        print(f"  Runs logged : {len(results)}")
        print(f"  Best Sharpe : {best_sharpe:.4f}")
        print(f"  MLflow UI   : mlflow ui --port 5000")

        # Show top runs from MLflow
        top = self.tracker.get_best_runs(n=5)
        if not top.empty:
            print(f"\nTop 5 runs in MLflow:")
            print(top.to_string(index=False))

        return {"runs": len(results), "best_sharpe": best_sharpe,
                "best_signal": best_run[0] if best_run else None}


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
