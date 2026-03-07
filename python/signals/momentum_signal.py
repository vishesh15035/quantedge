import pandas as pd
import numpy as np
from .base_signal import BaseSignal

class MomentumSignal(BaseSignal):
    def __init__(self, lookback=252, skip=21):
        super().__init__("momentum_12_1")
        self.lookback = lookback
        self.skip     = skip

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['Close']
        mom   = close.shift(self.skip) / close.shift(self.lookback) - 1
        return self.normalize(mom)

class ShortTermMomentum(BaseSignal):
    def __init__(self):
        super().__init__("momentum_5d")

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return self.normalize(data['Close'].pct_change(5))

class VWAPMomentum(BaseSignal):
    def __init__(self, window: int = 20):
        super().__init__("vwap_momentum")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        tp   = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (tp * data['Volume']).rolling(self.window).sum() / \
                data['Volume'].rolling(self.window).sum()
        signal = (data['Close'] - vwap) / (vwap + 1e-8)
        return self.normalize(signal)
