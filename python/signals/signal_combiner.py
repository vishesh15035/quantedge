import pandas as pd
from typing import Dict
from .base_signal import BaseSignal

class SignalCombiner:
    def __init__(self, signals: Dict[str, BaseSignal], weights: Dict[str, float] = None):
        self.signals = signals
        self.weights = weights or {k: 1/len(signals) for k in signals}

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        results = {name: sig.compute(data) for name, sig in self.signals.items()}
        df = pd.DataFrame(results)
        w  = pd.Series(self.weights)
        w  = w / w.sum()
        df['combined'] = sum(df[k] * w[k] for k in self.signals)
        return df

    def optimize_weights(self, data: pd.DataFrame, prices: pd.Series) -> Dict[str, float]:
        from python.backtest.engine import Backtester
        bt = Backtester()
        sharpes = {}
        for name, signal in self.signals.items():
            sig = signal.compute(data)
            res = bt.run(prices, sig)
            sharpes[name] = max(res.sharpe_ratio, 0)
        total = sum(sharpes.values()) + 1e-8
        self.weights = {k: v/total for k, v in sharpes.items()}
        print("[Combiner] Optimized weights:")
        for k, v in self.weights.items():
            print(f"  {k}: {v:.3f}")
        return self.weights
