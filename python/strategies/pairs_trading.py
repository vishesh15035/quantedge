import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PairResult:
    ticker1:      str
    ticker2:      str
    pvalue:       float
    hedge_ratio:  float
    half_life:    float
    is_cointegrated: bool

    def summary(self):
        print(f"\nPair: {self.ticker1} / {self.ticker2}")
        print(f"  Cointegrated : {self.is_cointegrated}")
        print(f"  P-value      : {self.pvalue:.4f}")
        print(f"  Hedge Ratio  : {self.hedge_ratio:.4f}")
        print(f"  Half Life    : {self.half_life:.1f} days")


class PairsTrader:
    def __init__(self, entry_z=2.0, exit_z=0.5, stop_z=3.5):
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.stop_z  = stop_z

    def find_cointegrated_pairs(self, prices: pd.DataFrame,
                                 pvalue_threshold=0.05) -> List[PairResult]:
        tickers = list(prices.columns)
        pairs   = []
        n       = len(tickers)
        print(f"Testing {n*(n-1)//2} pairs for cointegration...")
        for i in range(n):
            for j in range(i+1, n):
                t1, t2 = tickers[i], tickers[j]
                try:
                    s1 = np.log(prices[t1].dropna())
                    s2 = np.log(prices[t2].dropna())
                    idx = s1.index.intersection(s2.index)
                    s1, s2 = s1[idx], s2[idx]
                    if len(s1) < 100: continue
                    _, pval, _ = coint(s1, s2)
                    model      = OLS(s1, add_constant(s2)).fit()
                    hedge      = model.params.iloc[-1]
                    spread     = s1 - hedge * s2
                    half_life  = self._half_life(spread)
                    pairs.append(PairResult(
                        ticker1=t1, ticker2=t2,
                        pvalue=pval, hedge_ratio=hedge,
                        half_life=half_life,
                        is_cointegrated=pval < pvalue_threshold
                    ))
                except Exception:
                    continue
        pairs.sort(key=lambda x: x.pvalue)
        return pairs

    def _half_life(self, spread: pd.Series) -> float:
        lag    = spread.shift(1).dropna()
        delta  = spread.diff().dropna()
        idx    = lag.index.intersection(delta.index)
        model  = OLS(delta[idx], add_constant(lag[idx])).fit()
        lam    = model.params.iloc[-1]
        return -np.log(2) / lam if lam < 0 else float('inf')

    def compute_spread(self, s1: pd.Series, s2: pd.Series,
                       hedge_ratio: float) -> pd.Series:
        return np.log(s1) - hedge_ratio * np.log(s2)

    def zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        mean = spread.rolling(window).mean()
        std  = spread.rolling(window).std()
        return (spread - mean) / (std + 1e-8)

    def generate_signals(self, s1: pd.Series, s2: pd.Series,
                         hedge_ratio: float, window: int = 60) -> pd.DataFrame:
        spread = self.compute_spread(s1, s2, hedge_ratio)
        z      = self.zscore(spread, window)

        pos_s1 = pd.Series(0.0, index=z.index)
        pos_s2 = pd.Series(0.0, index=z.index)

        for i in range(1, len(z)):
            zi = z.iloc[i]
            prev_s1 = pos_s1.iloc[i-1]

            if abs(zi) > self.stop_z:          # stop loss
                pos_s1.iloc[i] = 0; pos_s2.iloc[i] = 0
            elif zi > self.entry_z and prev_s1 == 0:   # short spread
                pos_s1.iloc[i] = -1; pos_s2.iloc[i] = hedge_ratio
            elif zi < -self.entry_z and prev_s1 == 0:  # long spread
                pos_s1.iloc[i] = 1;  pos_s2.iloc[i] = -hedge_ratio
            elif abs(zi) < self.exit_z:        # exit
                pos_s1.iloc[i] = 0; pos_s2.iloc[i] = 0
            else:
                pos_s1.iloc[i] = prev_s1
                pos_s2.iloc[i] = pos_s2.iloc[i-1]

        ret1 = s1.pct_change()
        ret2 = s2.pct_change()
        pnl  = (pos_s1.shift(1)*ret1 + pos_s2.shift(1)*ret2).fillna(0)

        return pd.DataFrame({
            'spread': spread, 'zscore': z,
            'pos_s1': pos_s1, 'pos_s2': pos_s2,
            'pnl': pnl, 'cumulative_pnl': (1+pnl).cumprod()
        })

    def backtest_pair(self, s1: pd.Series, s2: pd.Series,
                      hedge_ratio: float) -> dict:
        df     = self.generate_signals(s1, s2, hedge_ratio)
        pnl    = df['pnl'].dropna()
        cum    = df['cumulative_pnl'].dropna()
        sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
        dd     = (cum - cum.cummax()) / cum.cummax()
        return {
            "total_return":  float(cum.iloc[-1] - 1),
            "sharpe_ratio":  float(sharpe),
            "max_drawdown":  float(dd.min()),
            "num_trades":    int((df['pos_s1'].diff().abs() > 0).sum()),
            "win_rate":      float((pnl > 0).mean()),
        }
