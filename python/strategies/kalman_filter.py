import numpy as np
import pandas as pd

class KalmanFilterSignal:
    """
    Kalman Filter for dynamic hedge ratio estimation
    Used in pairs trading to adapt to regime changes
    State: [hedge_ratio, intercept]
    """
    def __init__(self, delta=1e-4, vt=1e-3):
        self.delta = delta   # state transition noise
        self.vt    = vt      # observation noise
        self.theta = None    # state estimate
        self.P     = None    # state covariance

    def update(self, x: float, y: float) -> tuple:
        F = np.array([[x, 1.0]])   # observation matrix

        if self.theta is None:
            self.theta = np.zeros(2)
            self.P     = np.eye(2) * 1.0

        # Predict
        Q          = self.delta * np.eye(2)
        P_pred     = self.P + Q

        # Update
        S          = F @ P_pred @ F.T + self.vt
        K          = P_pred @ F.T / S[0,0]
        y_hat      = float(F @ self.theta)
        innovation = y - y_hat

        self.theta = self.theta + K.flatten() * innovation
        self.P     = (np.eye(2) - K @ F) @ P_pred

        hedge_ratio = self.theta[0]
        intercept   = self.theta[1]
        return hedge_ratio, intercept, innovation

    def fit_series(self, x: pd.Series, y: pd.Series) -> pd.DataFrame:
        hedges, intercepts, innovations = [], [], []
        for xi, yi in zip(x.values, y.values):
            h, c, e = self.update(xi, yi)
            hedges.append(h)
            intercepts.append(c)
            innovations.append(e)
        return pd.DataFrame({
            'hedge_ratio': hedges,
            'intercept':   intercepts,
            'innovation':  innovations,
            'spread':      y.values - np.array(hedges)*x.values
        }, index=x.index)
