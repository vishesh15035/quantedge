import yfinance as yf
import pandas as pd
from pathlib import Path

class MarketDataFetcher:
    def __init__(self, cache_dir="./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        cache_file = self.cache_dir / f"{ticker}_{period}_{interval}.parquet"
        if cache_file.exists():
            print(f"[Fetcher] {ticker} — loaded from cache")
            return pd.read_parquet(cache_file)
        print(f"[Fetcher] Downloading {ticker}...")
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        data = raw[['Open','High','Low','Close','Volume']].dropna()
        data.to_parquet(cache_file)
        print(f"[Fetcher] {ticker}: {len(data)} rows cached")
        return data

    def fetch_multiple(self, tickers: list, period: str = "5y") -> dict:
        return {t: self.fetch(t, period) for t in tickers}
