from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseSignal(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass

    def normalize(self, series: pd.Series, window: int = 252) -> pd.Series:
        roll_mean = series.rolling(window, min_periods=20).mean()
        roll_std  = series.rolling(window, min_periods=20).std()
        z = (series - roll_mean) / (roll_std + 1e-8)
        return (z.clip(-3, 3) / 3).fillna(0)

    def sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        if returns.std() < 1e-8: return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods))
