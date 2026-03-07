import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class BacktestResult:
    ticker:            str
    signal_name:       str
    total_return:      float
    annualized_return: float
    sharpe_ratio:      float
    sortino_ratio:     float
    max_drawdown:      float
    win_rate:          float
    profit_factor:     float
    num_trades:        int
    volatility:        float
    calmar_ratio:      float

    def summary(self):
        print(f"\n{'='*52}")
        print(f"  {self.signal_name} on {self.ticker}")
        print(f"{'='*52}")
        print(f"  Total Return      : {self.total_return*100:+.2f}%")
        print(f"  Annualized Return : {self.annualized_return*100:+.2f}%")
        print(f"  Sharpe Ratio      : {self.sharpe_ratio:.4f}")
        print(f"  Sortino Ratio     : {self.sortino_ratio:.4f}")
        print(f"  Max Drawdown      : {self.max_drawdown*100:.2f}%")
        print(f"  Win Rate          : {self.win_rate*100:.1f}%")
        print(f"  Profit Factor     : {self.profit_factor:.3f}")
        print(f"  Num Trades        : {self.num_trades}")
        print(f"  Volatility        : {self.volatility*100:.2f}%")
        print(f"  Calmar Ratio      : {self.calmar_ratio:.4f}")
        print(f"{'='*52}")

    def to_dict(self):
        return asdict(self)

class Backtester:
    def __init__(self, initial_capital=100_000, transaction_cost_bps=5, slippage_bps=2):
        self.capital  = initial_capital
        self.cost_bps = (transaction_cost_bps + slippage_bps) / 10_000

    def run(self, prices: pd.Series, signal: pd.Series,
            ticker="ASSET", signal_name="signal") -> BacktestResult:
        df = pd.DataFrame({'price': prices, 'signal': signal}).dropna()
        pos = pd.Series(0.0, index=df.index)
        pos[df['signal'] >  0.2] =  1.0
        pos[df['signal'] < -0.2] = -1.0
        price_rets  = df['price'].pct_change()
        strat_rets  = pos.shift(1) * price_rets
        trades      = pos.diff().abs()
        strat_rets -= trades * self.cost_bps
        strat_rets  = strat_rets.dropna()
        cumulative  = (1 + strat_rets).cumprod()
        n           = len(strat_rets)
        return BacktestResult(
            ticker            = ticker,
            signal_name       = signal_name,
            total_return      = float(cumulative.iloc[-1] - 1),
            annualized_return = float(cumulative.iloc[-1] ** (252/n) - 1),
            sharpe_ratio      = self._sharpe(strat_rets),
            sortino_ratio     = self._sortino(strat_rets),
            max_drawdown      = self._max_drawdown(cumulative),
            win_rate          = float((strat_rets > 0).mean()),
            profit_factor     = self._profit_factor(strat_rets),
            num_trades        = int(trades.sum()),
            volatility        = float(strat_rets.std() * np.sqrt(252)),
            calmar_ratio      = self._calmar(strat_rets, cumulative),
        )

    def _sharpe(self, r, rf=0.05/252):
        e = r - rf
        return float(e.mean() / (e.std() + 1e-8) * np.sqrt(252))

    def _sortino(self, r, rf=0.05/252):
        e  = r - rf
        ds = e[e < 0].std()
        return float(e.mean() / (ds + 1e-8) * np.sqrt(252))

    def _max_drawdown(self, cum):
        dd = (cum - cum.cummax()) / cum.cummax()
        return float(dd.min())

    def _profit_factor(self, r):
        g = r[r > 0].sum()
        l = abs(r[r < 0].sum())
        return float(g / (l + 1e-8))

    def _calmar(self, r, cum):
        ann = cum.iloc[-1] ** (252/len(r)) - 1
        mdd = abs(self._max_drawdown(cum))
        return float(ann / (mdd + 1e-8))
