import numpy as np
import pandas as pd
from typing import Tuple

class TradingEnv:
    """
    Custom OpenAI Gym-style trading environment
    State:  [returns(5d), rsi, bb_pos, volume_ratio, position, pnl]
    Action: 0=hold, 1=buy, 2=sell
    Reward: risk-adjusted pnl with transaction cost penalty
    """
    def __init__(self, prices: pd.Series, window: int = 20,
                 initial_capital: float = 100_000,
                 transaction_cost: float = 0.001):
        self.prices           = prices.values
        self.dates            = prices.index
        self.window           = window
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.n_actions        = 3
        self.n_features       = 10
        self.reset()

    def reset(self) -> np.ndarray:
        self.t          = self.window
        self.position   = 0       # -1, 0, 1
        self.capital    = self.initial_capital
        self.portfolio  = self.initial_capital
        self.trades     = 0
        self.returns    = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        p   = self.prices[self.t-self.window:self.t]
        ret = np.diff(p) / p[:-1]

        # Features
        f1  = ret[-1]                                    # last return
        f2  = ret[-5:].mean() if len(ret)>=5 else 0     # 5d momentum
        f3  = ret.mean()                                 # window momentum
        f4  = ret.std() + 1e-8                           # volatility

        # RSI
        gains  = np.maximum(ret, 0)
        losses = np.maximum(-ret, 0)
        rsi    = 100 - 100/(1 + gains.mean()/(losses.mean()+1e-8))
        f5     = (rsi - 50) / 50

        # Bollinger position
        sma = p.mean(); std = p.std() + 1e-8
        f6  = (p[-1] - sma) / (2*std)

        # Volume proxy (price range)
        f7  = (p.max() - p.min()) / (p.mean() + 1e-8)

        # Position and pnl
        f8  = float(self.position)
        f9  = (self.portfolio - self.initial_capital) / self.initial_capital
        f10 = float(self.trades) / 100

        return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        price_now  = self.prices[self.t]
        price_next = self.prices[self.t+1] if self.t+1 < len(self.prices) else price_now

        prev_position = self.position
        cost          = 0.0

        # Execute action
        if action == 1 and self.position != 1:    # buy
            cost           = self.transaction_cost
            self.position  = 1
            self.trades   += 1
        elif action == 2 and self.position != -1: # sell
            cost           = self.transaction_cost
            self.position  = -1
            self.trades   += 1
        # action == 0: hold

        # PnL
        price_ret      = (price_next - price_now) / price_now
        pnl            = self.position * price_ret - cost
        self.portfolio = self.portfolio * (1 + pnl)
        self.returns.append(pnl)

        # Reward: sharpe-like (recent returns)
        recent = self.returns[-20:] if len(self.returns) >= 20 else self.returns
        r_arr  = np.array(recent)
        reward = float(r_arr.mean() / (r_arr.std() + 1e-8)) * 10 - cost * 100

        self.t += 1
        done    = self.t >= len(self.prices) - 1

        info = {"portfolio": self.portfolio, "position": self.position,
                "trades": self.trades, "price": price_now}
        return self._get_state(), reward, done, info
