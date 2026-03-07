import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.optimize import minimize

class Portfolio:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.returns = None
        self.weights = None

    def load_returns(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = pd.DataFrame({t: prices[t]['Close'] for t in self.tickers if t in prices})
        self.returns = close.pct_change().dropna()
        return self.returns

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()

    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns.cov() * 252  # annualized

    def risk_parity_weights(self) -> pd.Series:
        cov = self.covariance_matrix().values
        n   = len(self.tickers)

        def risk_parity_obj(w):
            w      = np.array(w)
            port_var = w @ cov @ w
            marginal = cov @ w
            rc       = w * marginal / port_var
            return np.sum((rc - 1/n)**2)

        constraints = [{'type':'eq','fun': lambda w: np.sum(w)-1}]
        bounds      = [(0.01, 0.3)] * n
        w0          = np.ones(n) / n
        res         = minimize(risk_parity_obj, w0, method='SLSQP',
                               bounds=bounds, constraints=constraints)
        self.weights = pd.Series(res.x, index=self.tickers)
        return self.weights

    def mean_variance_weights(self, target_return=None) -> pd.Series:
        mu  = self.returns.mean() * 252
        cov = self.covariance_matrix().values
        n   = len(self.tickers)

        def neg_sharpe(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(w @ cov @ w)
            return -(ret - 0.05) / (vol + 1e-8)

        constraints = [{'type':'eq','fun': lambda w: np.sum(w)-1}]
        bounds      = [(0.0, 0.4)] * n
        res         = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP',
                               bounds=bounds, constraints=constraints)
        self.weights = pd.Series(res.x, index=self.tickers)
        return self.weights

    def portfolio_metrics(self, weights: pd.Series = None) -> dict:
        w   = (weights or self.weights).values
        mu  = self.returns.mean().values * 252
        cov = self.covariance_matrix().values
        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        return {
            "expected_return": ret,
            "volatility":      vol,
            "sharpe_ratio":    (ret - 0.05) / (vol + 1e-8),
            "max_weight":      float(np.max(w)),
            "min_weight":      float(np.min(w)),
            "effective_n":     1.0 / np.sum(w**2),
        }

    def sector_exposure(self, sector_map: Dict[str,str]) -> pd.Series:
        if self.weights is None: return pd.Series()
        exposure = {}
        for ticker, weight in self.weights.items():
            sector = sector_map.get(ticker, "Unknown")
            exposure[sector] = exposure.get(sector, 0) + weight
        return pd.Series(exposure).sort_values(ascending=False)
