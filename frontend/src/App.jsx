import { useState, useEffect, useCallback, useRef } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const API = 'http://localhost:8000'
const WS  = 'ws://localhost:8000/ws/prices'
const C = {
  bg:'#0a0e1a', card:'#111827', border:'#1f2937',
  blue:'#3b82f6', green:'#10b981', red:'#ef4444',
  yellow:'#f59e0b', purple:'#8b5cf6', text:'#f9fafb', muted:'#6b7280',
}
const fmt  = (v,d=2) => typeof v==='number' ? v.toFixed(d) : '—'
const fmtp = (v)     => `${v>=0?'+':''}${fmt(v*100,2)}%`
const col  = (v)     => v>0 ? C.green : C.red

const MetricCard = ({label,value,sub,highlight}) => (
  <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,padding:'14px 18px',minWidth:140}}>
    <div style={{color:C.muted,fontSize:11,textTransform:'uppercase',letterSpacing:1,marginBottom:4}}>{label}</div>
    <div style={{color:highlight||C.text,fontSize:22,fontWeight:700,fontFamily:'monospace'}}>{value}</div>
    {sub && <div style={{color:C.muted,fontSize:11,marginTop:2}}>{sub}</div>}
  </div>
)

const OrderBook = ({book}) => {
  if (!book) return <div style={{color:C.muted,padding:20,textAlign:'center'}}>Loading…</div>
  const maxQty = Math.max(...book.bids.map(b=>b.quantity),...book.asks.map(a=>a.quantity))
  const Bar = ({qty,side}) => (
    <div style={{position:'absolute',top:0,bottom:0,
      [side==='bid'?'right':'left']:0,
      width:`${(qty/maxQty)*100}%`,
      background:side==='bid'?'rgba(16,185,129,0.15)':'rgba(239,68,68,0.15)'}}/>
  )
  return (
    <div style={{fontFamily:'monospace',fontSize:12}}>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',color:C.muted,marginBottom:6,padding:'0 4px'}}>
        <span>Qty</span><span style={{textAlign:'center'}}>Price</span><span style={{textAlign:'right'}}>Qty</span>
      </div>
      {book.asks.slice().reverse().map((a,i)=>(
        <div key={i} style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',position:'relative',padding:'2px 4px'}}>
          <Bar qty={a.quantity} side="ask"/>
          <span style={{color:C.muted}}></span>
          <span style={{textAlign:'center',color:C.red,zIndex:1}}>{fmt(a.price)}</span>
          <span style={{textAlign:'right',color:C.muted,zIndex:1}}>{a.quantity}</span>
        </div>
      ))}
      <div style={{borderTop:`1px solid ${C.border}`,borderBottom:`1px solid ${C.border}`,
        padding:'4px',textAlign:'center',color:C.yellow,fontWeight:700,margin:'4px 0'}}>
        Mid ${fmt(book.mid_price)} · Spread ${fmt(book.spread,4)}
      </div>
      {book.bids.map((b,i)=>(
        <div key={i} style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',position:'relative',padding:'2px 4px'}}>
          <Bar qty={b.quantity} side="bid"/>
          <span style={{color:C.muted,zIndex:1}}>{b.quantity}</span>
          <span style={{textAlign:'center',color:C.green,zIndex:1}}>{fmt(b.price)}</span>
          <span style={{textAlign:'right',color:C.muted,zIndex:1}}></span>
        </div>
      ))}
      <div style={{marginTop:8,padding:'4px',color:book.order_imbalance>0?C.green:C.red,fontSize:11}}>
        Order Imbalance: {book.order_imbalance>0?'▲':'▼'} {fmt(book.order_imbalance,4)}
        <span style={{color:C.muted}}> (key HFT signal)</span>
      </div>
    </div>
  )
}

export default function App() {
  const [ticks,     setTicks]     = useState([])
  const [book,      setBook]      = useState(null)
  const [btResult,  setBtResult]  = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [connected, setConnected] = useState(false)
  const [ticker,    setTicker]    = useState('SPY')
  const [signal,    setSignal]    = useState('momentum')
  const wsRef = useRef(null)

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS)
      ws.onopen    = ()  => setConnected(true)
      ws.onclose   = ()  => { setConnected(false); setTimeout(connect,2000) }
      ws.onerror   = ()  => ws.close()
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data)
        setTicks(prev => [...prev,{time:new Date(d.timestamp).toLocaleTimeString(),
          price:d.price,bid:d.bid,ask:d.ask}].slice(-60))
      }
      wsRef.current = ws
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  useEffect(() => {
    const poll = async () => {
      try { const r = await fetch(`${API}/orderbook/snapshot?ticker=${ticker}`); setBook(await r.json()) } catch {}
    }
    poll(); const id = setInterval(poll,1000); return () => clearInterval(id)
  }, [ticker])

  const runBacktest = useCallback(async () => {
    setLoading(true); setBtResult(null)
    try {
      const r = await fetch(`${API}/backtest/run`,{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ticker,signal,period:'5y',cost_bps:5,slippage_bps:2})})
      setBtResult(await r.json())
    } catch(e){ console.error(e) }
    setLoading(false)
  }, [ticker,signal])

  const latestPrice = ticks.length ? ticks[ticks.length-1].price : null

  return (
    <div style={{background:C.bg,minHeight:'100vh',color:C.text,fontFamily:"'Inter',sans-serif",padding:'20px 28px'}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:24}}>
        <div>
          <h1 style={{margin:0,fontSize:26,fontWeight:800,
            background:`linear-gradient(90deg,${C.blue},${C.purple})`,
            WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>
            ⚡ QuantEdge
          </h1>
          <div style={{color:C.muted,fontSize:12,marginTop:2}}>
            Algorithmic Trading Research Platform · HFT Signal Analytics
          </div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <div style={{width:8,height:8,borderRadius:'50%',background:connected?C.green:C.red,
            boxShadow:connected?`0 0 8px ${C.green}`:'none'}}/>
          <span style={{color:C.muted,fontSize:12}}>{connected?'Live Feed':'Connecting…'}</span>
          {latestPrice && <span style={{color:C.yellow,fontFamily:'monospace',fontWeight:700,marginLeft:12,fontSize:18}}>
            {ticker} ${latestPrice.toFixed(2)}
          </span>}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 340px',gap:20,marginBottom:20}}>
        <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
          <div style={{color:C.muted,fontSize:12,marginBottom:14,textTransform:'uppercase',letterSpacing:1}}>
            Live Price Feed (last 60 ticks)
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={ticks}>
              <defs>
                <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.blue} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={C.blue} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
              <XAxis dataKey="time" tick={{fill:C.muted,fontSize:10}} interval={9} stroke={C.border}/>
              <YAxis domain={['auto','auto']} tick={{fill:C.muted,fontSize:10}} stroke={C.border}/>
              <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8}}
                formatter={v=>[`$${v.toFixed(2)}`,'Price']}/>
              <Area type="monotone" dataKey="price" stroke={C.blue} fill="url(#pg)" strokeWidth={2} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
          <div style={{color:C.muted,fontSize:12,marginBottom:14,textTransform:'uppercase',letterSpacing:1}}>
            Order Book — {ticker}
          </div>
          <OrderBook book={book}/>
        </div>
      </div>

      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:24,marginBottom:20}}>
        <div style={{display:'flex',gap:12,alignItems:'center',marginBottom:20,flexWrap:'wrap'}}>
          <div style={{color:C.muted,fontSize:12,textTransform:'uppercase',letterSpacing:1}}>Backtest</div>
          <select value={ticker} onChange={e=>setTicker(e.target.value)}
            style={{background:C.bg,color:C.text,border:`1px solid ${C.border}`,borderRadius:6,padding:'6px 12px',fontSize:13}}>
            {['SPY','AAPL','MSFT','GOOGL','TSLA','NVDA'].map(t=><option key={t}>{t}</option>)}
          </select>
          <select value={signal} onChange={e=>setSignal(e.target.value)}
            style={{background:C.bg,color:C.text,border:`1px solid ${C.border}`,borderRadius:6,padding:'6px 12px',fontSize:13}}>
            <option value="momentum">Momentum 12-1m</option>
            <option value="short_momentum">Short Momentum 5d</option>
            <option value="rsi">RSI Reversion</option>
            <option value="bollinger">Bollinger Reversion</option>
          </select>
          <button onClick={runBacktest} disabled={loading}
            style={{background:loading?C.border:C.blue,color:C.text,border:'none',borderRadius:6,
              padding:'8px 20px',cursor:loading?'not-allowed':'pointer',fontWeight:600,fontSize:13}}>
            {loading?'Running…':'▶ Run Backtest'}
          </button>
          {loading && <div style={{color:C.muted,fontSize:12}}>Fetching 5yr data + computing signals…</div>}
        </div>

        {btResult && (
          <>
            <div style={{display:'flex',gap:12,flexWrap:'wrap',marginBottom:20}}>
              <MetricCard label="Sharpe Ratio" value={fmt(btResult.sharpe_ratio,4)}
                sub="≥1.0 good, ≥2.0 excellent"
                highlight={btResult.sharpe_ratio>=1?C.green:btResult.sharpe_ratio>=0?C.yellow:C.red}/>
              <MetricCard label="Total Return"      value={fmtp(btResult.total_return)}      highlight={col(btResult.total_return)}/>
              <MetricCard label="Ann. Return"       value={fmtp(btResult.annualized_return)} highlight={col(btResult.annualized_return)} sub="5-year period"/>
              <MetricCard label="Max Drawdown"      value={fmtp(btResult.max_drawdown)}      highlight={C.red} sub="Lower is better"/>
              <MetricCard label="Sortino Ratio"     value={fmt(btResult.sortino_ratio,4)}    sub="Downside-adjusted"/>
              <MetricCard label="Win Rate"          value={`${fmt(btResult.win_rate*100,1)}%`}/>
              <MetricCard label="Num Trades"        value={btResult.num_trades}              sub="incl. costs + slippage"/>
              <MetricCard label="Calmar Ratio"      value={fmt(btResult.calmar_ratio,4)}     sub="Return / MaxDD"/>
            </div>
            {btResult.equity_curve?.length > 0 && (
              <>
                <div style={{color:C.muted,fontSize:12,marginBottom:10,textTransform:'uppercase',letterSpacing:1}}>
                  Equity Curve (last 252 days · $100k initial)
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={btResult.equity_curve}>
                    <defs>
                      <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor={C.green} stopOpacity={0.25}/>
                        <stop offset="95%" stopColor={C.green} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="date" tick={{fill:C.muted,fontSize:10}} interval={49} stroke={C.border}/>
                    <YAxis tick={{fill:C.muted,fontSize:10}} stroke={C.border}
                      tickFormatter={v=>`$${(v/1000).toFixed(0)}k`}/>
                    <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8}}
                      formatter={v=>[`$${v.toLocaleString(undefined,{maximumFractionDigits:0})}`,'Portfolio']}/>
                    <ReferenceLine y={100000} stroke={C.muted} strokeDasharray="4 4"/>
                    <Area type="monotone" dataKey="value" stroke={C.green} fill="url(#eg)" strokeWidth={2} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </>
            )}
          </>
        )}
        {!btResult && !loading && (
          <div style={{color:C.muted,textAlign:'center',padding:40,fontSize:14}}>
            Select a ticker + signal and click Run Backtest
          </div>
        )}
      </div>

      <div style={{color:C.muted,fontSize:11,textAlign:'center'}}>
        QuantEdge · C++ Order Book Engine · Python Signal Research · FastAPI · React · MLflow · AWS
      </div>
    </div>
  )
}
