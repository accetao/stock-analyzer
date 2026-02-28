"""
Stock Analyzer - Configuration
Central configuration for all modules.
"""

# ─── Data Defaults ────────────────────────────────────────────────────────────
DEFAULT_PERIOD = "1y"           # Default historical data period
DEFAULT_INTERVAL = "1d"         # Default data interval
CACHE_EXPIRY_MINUTES = 15       # How long cached data stays fresh

# ─── Technical Analysis Defaults ──────────────────────────────────────────────
SMA_SHORT = 20                  # Short-term Simple Moving Average window
SMA_MEDIUM = 50                 # Medium-term SMA window
SMA_LONG = 200                  # Long-term SMA window
EMA_SHORT = 12                  # Short-term Exponential Moving Average
EMA_LONG = 26                   # Long-term EMA
RSI_PERIOD = 14                 # Relative Strength Index period
RSI_OVERBOUGHT = 70             # RSI overbought threshold
RSI_OVERSOLD = 30               # RSI oversold threshold
MACD_FAST = 12                  # MACD fast EMA
MACD_SLOW = 26                  # MACD slow EMA
MACD_SIGNAL = 9                 # MACD signal line
BBANDS_PERIOD = 20              # Bollinger Bands period
BBANDS_STD = 2                  # Bollinger Bands standard deviations
ATR_PERIOD = 14                 # Average True Range period
STOCH_K = 14                    # Stochastic %K period
STOCH_D = 3                     # Stochastic %D smoothing
ADX_PERIOD = 14                 # Average Directional Index period

# ─── Trend Analysis ──────────────────────────────────────────────────────────
TREND_SHORT_WINDOW = 20         # Short-term trend window (days)
TREND_MEDIUM_WINDOW = 50        # Medium-term trend window (days)
TREND_LONG_WINDOW = 200         # Long-term trend window (days)
TREND_STRENGTH_THRESHOLD = 25   # ADX threshold for trend strength

# ─── Stock Screener Defaults ─────────────────────────────────────────────────
SCREENER_MIN_VOLUME = 500_000           # Minimum average daily volume
SCREENER_MIN_MARKET_CAP = 1_000_000_000 # Minimum market cap ($1B)
SCREENER_MAX_PE = 40                    # Maximum P/E ratio
SCREENER_MIN_PE = 0                     # Minimum P/E (exclude negative)

# ─── Scoring Weights (must sum to 1.0) ───────────────────────────────────────
SCORE_WEIGHT_TECHNICAL = 0.35
SCORE_WEIGHT_FUNDAMENTAL = 0.30
SCORE_WEIGHT_TREND = 0.20
SCORE_WEIGHT_MOMENTUM = 0.15

# ─── Visualization ───────────────────────────────────────────────────────────
CHART_STYLE = "seaborn-v0_8-darkgrid"
CHART_WIDTH = 14
CHART_HEIGHT = 8
CHART_DPI = 100

# ─── Popular Stock Universe (S&P 500 subset + notable stocks) ────────────────
DEFAULT_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
    "CRM", "ADBE", "NFLX", "ORCL", "CSCO", "AVGO", "QCOM", "TXN", "IBM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "C",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "DIS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "BA", "HON", "UPS", "GE", "MMM", "LMT", "RTX",
]

# ─── Output Paths ─────────────────────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WATCHLIST_DIR = os.path.join(DATA_DIR, "watchlists")
