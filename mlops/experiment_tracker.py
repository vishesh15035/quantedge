import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class ExperimentTracker:
    """
    Auto-logs every backtest to MLflow
    Tracks: params, metrics, signals, equity curves
    """
    def __init__(self, tracking_uri="./mlflow-artifacts",
                 experiment_name="quantedge"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_backtest(self, result: dict, signal_params: dict = None) -> str:
        with mlflow.start_run(run_name=f"{result['signal_name']}_{result['ticker']}_{datetime.now().strftime('%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params({
                "ticker":      result["ticker"],
                "signal_name": result["signal_name"],
                **(signal_params or {})
            })
            # Log metrics
            mlflow.log_metrics({
                "sharpe_ratio":      result["sharpe_ratio"],
                "sortino_ratio":     result["sortino_ratio"],
                "total_return":      result["total_return"],
                "annualized_return": result["annualized_return"],
                "max_drawdown":      result["max_drawdown"],
                "win_rate":          result["win_rate"],
                "profit_factor":     result["profit_factor"],
                "num_trades":        result["num_trades"],
                "volatility":        result["volatility"],
                "calmar_ratio":      result["calmar_ratio"],
            })
            mlflow.set_tags({
                "strategy_type": self._classify_strategy(result["signal_name"]),
                "version":       "v2.0",
            })
            run_id = run.info.run_id
        print(f"[MLflow] Logged: {result['signal_name']} on {result['ticker']} → run_id={run_id[:8]}")
        return run_id

    def log_rl_training(self, ticker: str, result: dict) -> str:
        with mlflow.start_run(run_name=f"PPO_{ticker}_{datetime.now().strftime('%H%M%S')}") as run:
            mlflow.log_params({"ticker": ticker, "model": "PPO_ES", "episodes": result["episodes"]})
            mlflow.log_metrics({
                "best_return":  result["best_return"],
                "total_return": result.get("eval_return", 0),
                "sharpe_ratio": result.get("eval_sharpe", 0),
            })
            mlflow.set_tag("strategy_type", "reinforcement_learning")
            # Log learning curve
            for i, r in enumerate(result["history"]["episode_returns"]):
                mlflow.log_metric("episode_return", r, step=i)
            run_id = run.info.run_id
        print(f"[MLflow] RL Training logged → run_id={run_id[:8]}")
        return run_id

    def log_factor_model(self, ticker: str, result: dict) -> str:
        with mlflow.start_run(run_name=f"FF5_{ticker}") as run:
            mlflow.log_params({"ticker": ticker, "model": "FamaFrench5"})
            mlflow.log_metrics({
                "alpha_ann": result.get("alpha_ann", 0),
                "r_squared": result.get("r_squared", 0),
                **{f"beta_{k}": v for k,v in result.get("betas", {}).items()}
            })
            mlflow.set_tag("strategy_type", "factor_model")
            run_id = run.info.run_id
        return run_id

    def get_best_runs(self, metric="sharpe_ratio", n=5) -> pd.DataFrame:
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(self.experiment_name)
        if not exp: return pd.DataFrame()
        runs   = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=n
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id":      r.info.run_id[:8],
                "signal":      r.data.params.get("signal_name",""),
                "ticker":      r.data.params.get("ticker",""),
                "sharpe":      r.data.metrics.get("sharpe_ratio",0),
                "return":      r.data.metrics.get("total_return",0),
                "max_dd":      r.data.metrics.get("max_drawdown",0),
            })
        return pd.DataFrame(rows)

    def _classify_strategy(self, name: str) -> str:
        name = name.lower()
        if "momentum" in name: return "momentum"
        if "reversion" in name or "rsi" in name or "bollinger" in name: return "mean_reversion"
        if "pairs" in name: return "stat_arb"
        if "factor" in name or "ff5" in name: return "factor"
        return "other"
