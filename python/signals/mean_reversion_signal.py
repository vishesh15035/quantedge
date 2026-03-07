import pandas as pd
import numpy as np
from .base_signal import BaseSignal

class BollingerReversion(BaseSignal):
    def __init__(self, window=20, num_std=2.0):
        super().__init__("bollinger_reversion")
        self.window  = window
        self.num_std = num_std

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['Close']
        sma   = close.rolling(self.window).mean()
        std   = close.rolling(self.window).std()
        return -((close - sma) / (self.num_std * std + 1e-8)).clip(-1, 1).fillna(0)

class RSIReversion(BaseSignal):
    def __init__(self, period=14):
        super().__init__("rsi_reversion")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['Close']
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.period).mean()
        rsi   = 100 - (100 / (1 + gain / (loss + 1e-8)))
        return -((rsi - 50) / 50).clip(-1, 1).fillna(0)

class ZScoreReversion(BaseSignal):
    def __init__(self, window=60):
        super().__init__("zscore_reversion")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        import numpy as np
        log_p  = np.log(data['Close'])
        zscore = (log_p - log_p.rolling(self.window).mean()) / log_p.rolling(self.window).std()
        return (-zscore.clip(-3, 3) / 3).fillna(0)
