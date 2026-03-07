import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional

@dataclass
class DriftReport:
    feature:        str
    drift_detected: bool
    pvalue:         float
    statistic:      float
    reference_mean: float
    current_mean:   float
    shift_pct:      float

    def __str__(self):
        status = "⚠️  DRIFT" if self.drift_detected else "✓  OK"
        return (f"{status} | {self.feature:<20} "
                f"ref={self.reference_mean:.4f} "
                f"cur={self.current_mean:.4f} "
                f"shift={self.shift_pct:+.1f}% "
                f"p={self.pvalue:.4f}")


class DriftDetector:
    """
    Detects data drift and signal degradation
    Triggers retraining when drift is detected
    Uses KS test + PSI (Population Stability Index)
    """
    def __init__(self, pvalue_threshold=0.05, psi_threshold=0.2):
        self.pvalue_threshold = pvalue_threshold
        self.psi_threshold    = psi_threshold
        self.reference_data   = None

    def set_reference(self, data: pd.DataFrame):
        self.reference_data = data.copy()
        print(f"[Drift] Reference set: {len(data)} rows, {len(data.columns)} features")

    def detect(self, current_data: pd.DataFrame) -> list:
        if self.reference_data is None:
            raise ValueError("Call set_reference() first")
        reports = []
        for col in self.reference_data.columns:
            if col not in current_data.columns: continue
            ref = self.reference_data[col].dropna().values
            cur = current_data[col].dropna().values
            if len(ref) < 30 or len(cur) < 30: continue
            stat, pval   = stats.ks_2samp(ref, cur)
            ref_mean     = float(ref.mean())
            cur_mean     = float(cur.mean())
            shift_pct    = (cur_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100
            reports.append(DriftReport(
                feature        = col,
                drift_detected = pval < self.pvalue_threshold,
                pvalue         = float(pval),
                statistic      = float(stat),
                reference_mean = ref_mean,
                current_mean   = cur_mean,
                shift_pct      = float(shift_pct),
            ))
        return reports

    def psi(self, ref: np.ndarray, cur: np.ndarray, bins=10) -> float:
        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref) + 1e-8
        cur_pct = np.histogram(cur, bins=bins)[0] / len(cur) + 1e-8
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def should_retrain(self, current_data: pd.DataFrame) -> bool:
        reports = self.detect(current_data)
        drifted = [r for r in reports if r.drift_detected]
        print(f"\n[Drift] {len(drifted)}/{len(reports)} features drifted")
        for r in reports: print(f"  {r}")
        return len(drifted) > len(reports) * 0.3
