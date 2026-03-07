import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

@dataclass
class OptionResult:
    option_type: str
    S: float   # spot
    K: float   # strike
    T: float   # time to expiry (years)
    r: float   # risk-free rate
    sigma: float  # implied vol
    price: float
    delta: float
    gamma: float
    theta: float
    vega:  float
    rho:   float

    def summary(self):
        print(f"\n{self.option_type.upper()} | S={self.S} K={self.K} "
              f"T={self.T*365:.0f}d σ={self.sigma*100:.1f}%")
        print(f"  Price : ${self.price:.4f}")
        print(f"  Delta : {self.delta:.4f}  (hedge ratio)")
        print(f"  Gamma : {self.gamma:.6f} (delta sensitivity)")
        print(f"  Theta : {self.theta:.4f}  (time decay/day)")
        print(f"  Vega  : {self.vega:.4f}   (vol sensitivity)")
        print(f"  Rho   : {self.rho:.4f}    (rate sensitivity)")


class BlackScholesPricer:
    def __init__(self, r: float = 0.05):
        self.r = r

    def price(self, S, K, T, sigma,
              option_type: Literal["call","put"] = "call") -> OptionResult:
        if T <= 0: return OptionResult(option_type,S,K,T,self.r,sigma,0,0,0,0,0,0)
        sqrtT = np.sqrt(T)
        d1    = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2    = d1 - sigma*sqrtT
        nd1   = norm.pdf(d1)
        disc  = np.exp(-self.r*T)

        if option_type == "call":
            price = S*norm.cdf(d1) - K*disc*norm.cdf(d2)
            delta = norm.cdf(d1)
            rho   = K*T*disc*norm.cdf(d2) / 100
        else:
            price = K*disc*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            rho   = -K*T*disc*norm.cdf(-d2) / 100

        gamma = nd1 / (S*sigma*sqrtT)
        vega  = S*nd1*sqrtT / 100
        theta = (-S*nd1*sigma/(2*sqrtT) - self.r*K*disc*(
                  norm.cdf(d2) if option_type=="call" else norm.cdf(-d2))) / 365

        return OptionResult(option_type,S,K,T,self.r,sigma,price,delta,gamma,theta,vega,rho)

    def options_chain(self, S: float, T: float, sigma: float,
                      n_strikes: int = 10) -> pd.DataFrame:
        strikes = np.linspace(S*0.85, S*1.15, n_strikes)
        rows = []
        for K in strikes:
            c = self.price(S, K, T, sigma, "call")
            p = self.price(S, K, T, sigma, "put")
            rows.append({
                "strike": round(K,2),
                "call_price": round(c.price,4), "call_delta": round(c.delta,4),
                "call_gamma": round(c.gamma,6), "call_theta": round(c.theta,4),
                "call_vega":  round(c.vega,4),
                "put_price":  round(p.price,4), "put_delta":  round(p.delta,4),
                "put_gamma":  round(p.gamma,6), "put_theta":  round(p.theta,4),
                "put_vega":   round(p.vega,4),
                "put_call_parity_check": round(c.price - p.price - S + K*np.exp(-self.r*T), 6)
            })
        return pd.DataFrame(rows)

    def implied_vol(self, S, K, T, market_price,
                    option_type="call", tol=1e-6, max_iter=100) -> float:
        # Newton-Raphson IV solver
        sigma = 0.20
        for _ in range(max_iter):
            res   = self.price(S, K, T, sigma, option_type)
            diff  = res.price - market_price
            if abs(diff) < tol: break
            if abs(res.vega) < 1e-10: break
            sigma -= diff / (res.vega * 100)
            sigma  = max(0.001, min(sigma, 5.0))
        return sigma
