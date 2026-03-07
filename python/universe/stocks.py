# 50+ stocks across all 11 GICS sectors
# Used for sector rotation, factor model, pairs trading

UNIVERSE = {
    "Technology": [
        "AAPL","MSFT","NVDA","AMD","INTC","QCOM","AVGO","TXN","MU","AMAT"
    ],
    "Financials": [
        "JPM","BAC","GS","MS","WFC","BLK","C","AXP","SCHW","COF"
    ],
    "Healthcare": [
        "JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN"
    ],
    "Consumer_Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","CMG"
    ],
    "Communication": [
        "GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS","ATVI","EA"
    ],
    "Industrials": [
        "CAT","BA","HON","UPS","RTX","LMT","GE","MMM","DE","EMR"
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY"
    ],
    "Consumer_Staples": [
        "PG","KO","PEP","WMT","COST","CL","MDLZ","GIS","K","HSY"
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","ED","ETR"
    ],
    "Real_Estate": [
        "AMT","PLD","CCI","EQIX","PSA","O","SPG","WELL","DLR","AVB"
    ],
    "Materials": [
        "LIN","APD","SHW","ECL","DD","NEM","FCX","NUE","VMC","MLM"
    ],
}

# Flat list of all tickers
ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]

# Sector ETFs for rotation strategy
SECTOR_ETFS = {
    "Technology":              "XLK",
    "Financials":              "XLF",
    "Healthcare":              "XLV",
    "Consumer_Discretionary":  "XLY",
    "Communication":           "XLC",
    "Industrials":             "XLI",
    "Energy":                  "XLE",
    "Consumer_Staples":        "XLP",
    "Utilities":               "XLU",
    "Real_Estate":             "XLRE",
    "Materials":               "XLB",
}

BENCHMARK = "SPY"

def get_sector(ticker: str) -> str:
    for sector, tickers in UNIVERSE.items():
        if ticker in tickers:
            return sector
    return "Unknown"

def get_sector_tickers(sector: str) -> list:
    return UNIVERSE.get(sector, [])
