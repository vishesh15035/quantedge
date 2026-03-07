import asyncio, json, random, time, sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.parent))

app = FastAPI(title="QuantEdge API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class BacktestRequest(BaseModel):
    ticker:       str   = "SPY"
    signal:       str   = "momentum"
    period:       str   = "5y"
    cost_bps:     float = 5.0
    slippage_bps: float = 2.0

@app.get("/")
def root(): return {"status": "ok", "service": "QuantEdge API"}

@app.get("/health")
def health(): return {"status": "healthy", "timestamp": time.time()}

@app.post("/backtest/run")
def run_backtest(req: BacktestRequest):
    from python.data.fetcher import MarketDataFetcher
    from python.backtest.engine import Backtester
    from python.signals.momentum_signal import MomentumSignal, ShortTermMomentum
    from python.signals.mean_reversion_signal import RSIReversion, BollingerReversion

    data = MarketDataFetcher(cache_dir="./data/cache").fetch(req.ticker, req.period)
    signals = {"momentum": MomentumSignal(), "short_momentum": ShortTermMomentum(),
               "rsi": RSIReversion(), "bollinger": BollingerReversion()}
    sig_obj = signals.get(req.signal, MomentumSignal())
    sig_vals = sig_obj.compute(data)
    bt  = Backtester(transaction_cost_bps=req.cost_bps, slippage_bps=req.slippage_bps)
    res = bt.run(data['Close'], sig_vals, ticker=req.ticker, signal_name=req.signal)

    pos        = pd.Series(0.0, index=data.index) if False else __import__('pandas').Series(0.0, index=data.index)
    import pandas as pd
    pos        = pd.Series(0.0, index=data.index)
    pos[sig_vals >  0.2] =  1.0
    pos[sig_vals < -0.2] = -1.0
    strat_rets = (pos.shift(1) * data['Close'].pct_change()).dropna()
    equity     = (1 + strat_rets).cumprod() * 100_000
    curve      = [{"date": str(d.date()), "value": round(v, 2)}
                  for d, v in zip(equity.index[-252:], equity.values[-252:])]
    return {**res.to_dict(), "equity_curve": curve}

@app.get("/orderbook/snapshot")
def orderbook_snapshot(ticker: str = "SPY"):
    mid = 450.0 + random.uniform(-2, 2)
    bids = [{"price": round(mid - 0.02*(i+1), 2), "quantity": random.randint(100, 2000)} for i in range(5)]
    asks = [{"price": round(mid + 0.02*(i+1), 2), "quantity": random.randint(100, 2000)} for i in range(5)]
    bid_qty = sum(b['quantity'] for b in bids)
    ask_qty = sum(a['quantity'] for a in asks)
    return {"ticker": ticker, "mid_price": round(mid,2), "spread": 0.04,
            "order_imbalance": round((bid_qty-ask_qty)/(bid_qty+ask_qty), 4),
            "bids": bids, "asks": asks, "timestamp_ms": int(time.time()*1000)}

@app.websocket("/ws/prices")
async def websocket_prices(ws: WebSocket):
    await ws.accept()
    price = 450.0
    try:
        while True:
            price += random.gauss(0, 0.05)
            price  = max(400, min(500, price))
            await ws.send_json({"price": round(price,2), "bid": round(price-0.01,2),
                                "ask": round(price+0.01,2), "volume": random.randint(100,5000),
                                "timestamp": int(time.time()*1000)})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
