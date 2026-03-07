import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class FamaFrench5Factor:
    """
    Fama-French 5-Factor Model
    Factors: Mkt-RF, SMB, HML, RMW, CMA
    Each stock's return = alpha + b1*Mkt + b2*SMB + b3*HML + b4*RMW + b5*CMA + error
    """
    FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]

    def __init__(self):
        self.factor_returns = None
        self.betas          = {}
        self.alphas         = {}
        self.r_squared      = {}

    def build_factors(self, prices: Dict[str, pd.DataFrame],
                      market_ticker="SPY") -> pd.DataFrame:
        close = pd.DataFrame({t: p['Close'] for t,p in prices.items()})
        rets  = close.pct_change().dropna()
        mkt_rf = rets[market_ticker] - 0.05/252

        # SMB: small minus big (bottom vs top tercile by avg price)
        avg_price = close.mean()
        small = avg_price.nsmallest(len(avg_price)//3).index.tolist()
        big   = avg_price.nlargest(len(avg_price)//3).index.tolist()
        smb   = rets[small].mean(axis=1) - rets[big].mean(axis=1)

        # HML: high minus low book-to-market proxy (low price = value)
        low_p  = avg_price.nsmallest(len(avg_price)//3).index.tolist()
        high_p = avg_price.nlargest(len(avg_price)//3).index.tolist()
        hml    = rets[low_p].mean(axis=1) - rets[high_p].mean(axis=1)

        # RMW: robust minus weak profitability (momentum proxy)
        mom    = rets.rolling(252).mean().iloc[-1]
        robust = mom.nlargest(len(mom)//3).index.tolist()
        weak   = mom.nsmallest(len(mom)//3).index.tolist()
        rmw    = rets[robust].mean(axis=1) - rets[weak].mean(axis=1)

        # CMA: conservative minus aggressive investment (low vs high vol)
        vol    = rets.std()
        conserv = vol.nsmallest(len(vol)//3).index.tolist()
        aggress = vol.nlargest(len(vol)//3).index.tolist()
        cma     = rets[conserv].mean(axis=1) - rets[aggress].mean(axis=1)

        self.factor_returns = pd.DataFrame({
            "Mkt_RF": mkt_rf, "SMB": smb,
            "HML": hml, "RMW": rmw, "CMA": cma
        }).dropna()
        return self.factor_returns

    def fit(self, stock_returns: pd.Series, ticker: str) -> dict:
        from sklearn.linear_model import LinearRegression
        idx = stock_returns.index.intersection(self.factor_returns.index)
        y   = stock_returns[idx].values - 0.05/252
        X   = self.factor_returns.loc[idx].values
        if len(y) < 50: return {}
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        self.betas[ticker]     = dict(zip(self.FACTORS, model.coef_))
        self.alphas[ticker]    = model.intercept_ * 252
        self.r_squared[ticker] = 1 - ss_res/ss_tot
        return {
            "ticker":    ticker,
            "alpha_ann": self.alphas[ticker],
            "betas":     self.betas[ticker],
            "r_squared": self.r_squared[ticker]
        }

    def fit_universe(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = pd.DataFrame({t: p['Close'] for t,p in prices.items()})
        rets  = close.pct_change().dropna()
        rows  = []
        for ticker in rets.columns:
            if ticker in self.factor_returns.columns: continue
            res = self.fit(rets[ticker], ticker)
            if res: rows.append(res)
        df = pd.DataFrame(rows).set_index("ticker")
        return df

    def factor_exposure_report(self) -> pd.DataFrame:
        rows = []
        for ticker in self.betas:
            row = {"ticker": ticker, "alpha": self.alphas[ticker],
                   "r2": self.r_squared[ticker]}
            row.update(self.betas[ticker])
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker").round(4)
