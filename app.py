"""
Stock Analyzer - Streamlit Web UI
A professional dashboard for stock analysis powered by Yahoo Finance.

Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import config
from src import (
    data_fetcher,
    technical_analysis as ta,
    fundamental_analysis as fa,
    trend_analyzer,
    stock_screener,
    stock_scorer,
    utils,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (responsive: desktop + tablet + mobile) ──────────────────────
st.markdown("""
<style>
    /* ═══ Base / Desktop Styles ═══ */
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1e88e5, #43a047);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #1e88e5; margin-bottom: 0.5rem;
    }
    .score-strong-buy { color: #1b5e20; font-weight: 700; }
    .score-buy        { color: #4caf50; font-weight: 700; }
    .score-hold       { color: #ff9800; font-weight: 700; }
    .score-sell        { color: #f44336; font-weight: 700; }
    .score-strong-sell { color: #b71c1c; font-weight: 700; }
    .signal-bullish   { color: #4caf50; font-weight: 600; }
    .signal-bearish   { color: #f44336; font-weight: 600; }
    .signal-neutral   { color: #ff9800; font-weight: 600; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 0.8rem;
    }

    /* ═══ Tablet (≤ 1024px): max 2 columns per row ═══ */
    @media (max-width: 1024px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: calc(48%) !important;
            flex: 1 1 calc(48%) !important;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
    }

    /* ═══ Phone (≤ 768px): single-column stack ═══ */
    @media (max-width: 768px) {
        /* Stack every column group vertically */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Tighter page padding */
        .block-container {
            padding: 0.75rem 0.5rem !important;
            max-width: 100% !important;
        }

        /* Smaller branding */
        .main-header { font-size: 1.5rem !important; }
        .sub-header  { font-size: 0.8rem  !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.15rem !important; }

        /* Compact metrics */
        div[data-testid="stMetric"] { padding: 0.5rem !important; }
        div[data-testid="stMetric"] label { font-size: 0.72rem !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }

        /* Narrower sidebar */
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 250px !important;
        }
        [data-testid="stSidebar"] .main-header { font-size: 1.3rem !important; }

        /* Horizontal-scroll for wide tables */
        [data-testid="stDataFrame"] > div {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }

        /* Tabs: smaller labels, horizontal scroll */
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
        }
        [data-testid="stTabs"] button {
            font-size: 0.76rem !important;
            padding: 0.4rem 0.55rem !important;
            white-space: nowrap !important;
        }

        /* Bigger touch targets for buttons */
        .stButton > button {
            min-height: 2.75rem !important;
            font-size: 0.9rem !important;
        }

        /* Prevent iOS auto-zoom on text inputs */
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            font-size: 16px !important;
        }

        /* Constrain Plotly charts to viewport height */
        [data-testid="stPlotlyChart"] > div {
            max-height: 65vh !important;
        }
    }

    /* ═══ Small phones (≤ 480px) ═══ */
    @media (max-width: 480px) {
        .block-container {
            padding: 0.5rem 0.35rem !important;
        }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        div[data-testid="stMetric"] label { font-size: 0.65rem !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
        }

        /* Dashboard stock cards: tighter padding */
        .metric-card { padding: 0.6rem 0.8rem; }
    }

    /* ═══ Dashboard: stock card buttons ═══ */
    .stock-card-btn .stButton > button {
        background: linear-gradient(135deg, #fafbfc 60%, #f0f4f8) !important;
        border: 1px solid #e4e8ec !important;
        border-radius: 12px !important;
        padding: 0.65rem 0.85rem !important;
        min-height: unset !important;
        height: auto !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        width: 100% !important;
    }
    .stock-card-btn .stButton > button:hover {
        border-color: #1e88e5 !important;
        box-shadow: 0 4px 14px rgba(30,136,229,0.18) !important;
        transform: translateY(-2px);
        background: linear-gradient(135deg, #ffffff 60%, #e8f0fe) !important;
    }
    .stock-card-btn .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 4px rgba(30,136,229,0.12) !important;
    }
    .stock-card-btn .stButton > button p {
        margin: 0 !important;
        line-height: 1 !important;
    }
    /* Hide default Streamlit button focus ring */
    .stock-card-btn .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(30,136,229,0.25) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def rating_color(rating: str) -> str:
    colors = {
        "STRONG BUY": "#1b5e20", "BUY": "#4caf50",
        "HOLD": "#ff9800", "SELL": "#f44336", "STRONG SELL": "#b71c1c",
    }
    return colors.get(rating, "#888")


def rating_badge(rating: str) -> str:
    c = rating_color(rating)
    return f'<span style="background:{c};color:white;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.95rem">{rating}</span>'


def signal_badge(text: str) -> str:
    t = str(text).upper()
    if any(w in t for w in ("BULLISH", "ABOVE", "GOLDEN", "OVERSOLD", "STRONG", "UP")):
        cls = "signal-bullish"
    elif any(w in t for w in ("BEARISH", "BELOW", "DEATH", "OVERBOUGHT", "DOWN")):
        cls = "signal-bearish"
    else:
        cls = "signal-neutral"
    return f'<span class="{cls}">{text}</span>'


def fmt_number(val, prefix="", suffix=""):
    if val is None: return "N/A"
    if abs(val) >= 1e12: return f"{prefix}{val/1e12:.2f}T{suffix}"
    if abs(val) >= 1e9:  return f"{prefix}{val/1e9:.2f}B{suffix}"
    if abs(val) >= 1e6:  return f"{prefix}{val/1e6:.2f}M{suffix}"
    return f"{prefix}{val:,.2f}{suffix}"


def fmt_pct(val):
    if val is None: return "N/A"
    return f"{val*100:.1f}%"


def go_to_analysis(symbol: str):
    """Navigate to Stock Analysis page for the given symbol."""
    st.session_state["analyze_symbol"] = symbol.upper().strip()
    st.session_state["nav_to"] = "🔍 Stock Analysis"


def csv_download(df, filename, label="📥 Download CSV"):
    """Render a CSV download button for a DataFrame."""
    csv = df.to_csv(index=False)
    st.download_button(label, csv, file_name=filename,
                       mime="text/csv", use_container_width=True)


# ─── Symbol Search ───────────────────────────────────────────────────────────

_SYMBOLS_PATH = os.path.join(os.path.dirname(__file__), "data", "stock_symbols.json")
_SYMBOLS_REFRESH_URL = ("https://api.nasdaq.com/api/screener/stocks"
                        "?tableType=earnings&limit=25000&offset=0")


@st.cache_data(ttl=86400, show_spinner=False)
def _load_stock_symbols() -> dict:
    """Load the ticker → company-name map.  Tries the bundled JSON first;
    if it is missing or stale (>30 days), fetches a fresh copy from NASDAQ."""
    symbols = {}
    try:
        mtime = os.path.getmtime(_SYMBOLS_PATH)
        age_days = (datetime.now().timestamp() - mtime) / 86400
        with open(_SYMBOLS_PATH, "r") as f:
            symbols = json.load(f)
        if age_days < 30 and len(symbols) > 100:
            return symbols
    except Exception:
        pass

    # Refresh from NASDAQ API
    try:
        import urllib.request as _ul, re as _re
        req = _ul.Request(_SYMBOLS_REFRESH_URL,
                          headers={"User-Agent": "Mozilla/5.0"})
        resp = _ul.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        rows = data.get("data", {}).get("table", {}).get("rows", [])
        fresh = {}
        for r in rows:
            sym = r.get("symbol", "").strip()
            name = r.get("name", "").strip()
            if not sym or not name:
                continue
            # --- Filter out non-common-stock instruments ---
            nl = name.lower()
            # Preferred stocks (^ in ticker like MS^F)
            if "^" in sym:
                continue
            # Warrants, rights, units by suffix
            if any(sym.endswith(s) for s in ("W", "WS", "R", "U")) and len(sym) > 3:
                if "Warrant" in name or "Unit" in name or "Right" in name:
                    continue
            # Preferred / depositary preferred by name
            if "preferred stock" in nl:
                continue
            if "depositary shares" in nl and ("1/1000" in nl or "preferred" in nl or "interest in" in nl):
                continue
            # Warrants / rights by name
            if "warrant" in nl:
                continue
            if "subscription right" in nl or "contingent value right" in nl:
                continue
            # Units
            if nl.endswith(" units"):
                continue
            # Blank-check / SPAC shells
            if "blank check" in nl:
                continue
            if "acquisition corp" in nl or "acquisition inc" in nl:
                continue
            if "merger corp" in nl or "merger sub" in nl:
                continue
            # --- Clean company name ---
            name = _re.sub(r"\s*Common Stock$", "", name)
            name = _re.sub(r"\s*Common Shares$", "", name)
            name = _re.sub(r"\s*Ordinary Shares$", "", name)
            name = _re.sub(r"\s*(Class [A-C])$", "", name)
            fresh[sym] = name.strip()
        if len(fresh) > 100:
            os.makedirs(os.path.dirname(_SYMBOLS_PATH), exist_ok=True)
            with open(_SYMBOLS_PATH, "w") as f:
                json.dump(dict(sorted(fresh.items())), f, indent=2)
            return fresh
    except Exception:
        pass
    return symbols


@st.cache_data(ttl=86400, show_spinner=False)
def _build_symbol_options() -> list[str]:
    """Build searchable dropdown options with BOTH formats so users can
    type either a ticker prefix ('AAPL') or a company name ('Apple').

    Returns a sorted list containing two entries per stock:
      - 'AAPL — Apple Inc.'      (matches when typing ticker)
      - 'Apple Inc. — AAPL'      (matches when typing company name)
    """
    symbols = _load_stock_symbols()
    opts = set()
    for t, n in symbols.items():
        opts.add(f"{t} — {n}")
        opts.add(f"{n} — {t}")
    return sorted(opts)


def _extract_ticker(option: str) -> str:
    """Extract the ticker symbol from a display option like
    'AAPL — Apple Inc.' or 'Apple Inc. — AAPL'."""
    if not option or " — " not in option:
        return option.strip().upper() if option else ""
    left, right = option.split(" — ", 1)
    left, right = left.strip(), right.strip()
    syms = _load_stock_symbols()
    if left.upper() in syms:
        return left.upper()
    if right.upper() in syms:
        return right.upper()
    return left.upper()


def _ticker_option(ticker: str) -> str:
    """Return the ticker-first display string for a known ticker."""
    syms = _load_stock_symbols()
    name = syms.get(ticker.upper(), ticker)
    return f"{ticker.upper()} — {name}"


def symbol_search(label: str = "🔍 Search stock",
                  key: str = "sym",
                  default: str = "AAPL") -> str:
    """Searchable stock picker with live autocomplete.

    Uses a single st.selectbox — the user clicks, types a ticker or company
    name, and sees matching candidates instantly (no Enter key needed).
    Returns the chosen ticker string (e.g. 'AAPL').
    """
    result_key = f"_{key}_result"
    options = _build_symbol_options()

    sel = st.selectbox(
        label, options=options, index=None, key=key,
        placeholder="Type ticker or company name…",
    )

    if sel:
        ticker = _extract_ticker(sel)
        st.session_state[result_key] = ticker
        return ticker

    # Nothing selected yet — use previously stored result or the default
    stored = st.session_state.get(result_key, "")
    if stored:
        return stored
    if default:
        st.session_state[result_key] = default
    return default


# ─── AI / LLM Engine ────────────────────────────────────────────────────────

def _normalize_base_url(url: str) -> str:
    """Ensure the base URL ends with /v1 and uses 127.0.0.1 instead of localhost.

    On Windows, 'localhost' often resolves to ::1 (IPv6) first, but Ollama
    only listens on 127.0.0.1 (IPv4).  Using the IP directly avoids the
    DNS-resolution mismatch that causes 'Connection error' inside Streamlit.
    """
    url = url.rstrip("/")
    # Fix IPv6/IPv4 mismatch: Ollama binds to 127.0.0.1, not ::1
    url = url.replace("://localhost:", "://127.0.0.1:")
    url = url.replace("://localhost/", "://127.0.0.1/")
    if url.endswith("://localhost"):
        url = url.replace("://localhost", "://127.0.0.1")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _get_secret(key: str, default: str = "") -> str:
    """Read a value from Streamlit secrets (secrets.toml) safely."""
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default


_AI_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "data", "ai_settings.json")


def _save_ai_settings(api_key: str, base_url: str, model: str) -> bool:
    """Persist AI settings to a local JSON file so they survive refresh."""
    import base64
    try:
        os.makedirs(os.path.dirname(_AI_SETTINGS_FILE), exist_ok=True)
        data = {
            "api_key": base64.b64encode(api_key.encode()).decode(),
            "base_url": base_url,
            "model": model,
        }
        with open(_AI_SETTINGS_FILE, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def _load_ai_settings() -> dict | None:
    """Load previously saved AI settings. Returns dict or None."""
    import base64
    try:
        with open(_AI_SETTINGS_FILE, "r") as f:
            data = json.load(f)
        return {
            "api_key": base64.b64decode(data["api_key"].encode()).decode(),
            "base_url": data.get("base_url", ""),
            "model": data.get("model", "gpt-4o-mini"),
        }
    except Exception:
        return None


def _delete_ai_settings() -> None:
    """Remove saved AI settings file."""
    try:
        os.remove(_AI_SETTINGS_FILE)
    except Exception:
        pass


def get_ai_client():
    """Return an OpenAI-compatible client if API key is configured.

    Priority order for each setting:
      1. Session-state (sidebar widget)
      2. Streamlit secrets  (secrets.toml / Streamlit Cloud dashboard)
      3. Environment variable
    """
    if not HAS_OPENAI:
        return None
    # --- API key ---
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    # --- Base URL ---
    base_url = st.session_state.get("openai_base_url", "").strip()
    if not base_url:
        base_url = _get_secret("OPENAI_BASE_URL")
    # --- Build client ---
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = _normalize_base_url(base_url)
    # Generous timeout for local models (Ollama can be slow on first call)
    try:
        import httpx
        kwargs["timeout"] = httpx.Timeout(120.0, connect=15.0)
    except ImportError:
        pass
    return OpenAI(**kwargs)


def ai_available() -> bool:
    """Check if the AI engine is configured and ready."""
    return get_ai_client() is not None


def call_llm(system_prompt: str, user_prompt: str,
             model: str = None, temperature: float = 0.7,
             max_tokens: int = 1500) -> str:
    """Call the LLM and return the text response."""
    client = get_ai_client()
    if not client:
        return ""
    mdl = (model
           or st.session_state.get("openai_model", "").strip()
           or _get_secret("OPENAI_MODEL", "gpt-4o-mini"))
    try:
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err_type = type(e).__name__
        hint = ""
        err_lower = str(e).lower()
        if "connection" in err_lower or "connect" in err_lower:
            base = st.session_state.get("openai_base_url", "")
            if "localhost" in base:
                hint = " Try changing 'localhost' to '127.0.0.1' in the Base URL."
            else:
                hint = " Is the AI server running?"
        return f"⚠️ AI Error ({err_type}): {e}{hint}"


def build_stock_context(symbol, info, result, signals, trend_data, metrics, articles=None):
    """Build a rich context string with all stock data for the LLM."""
    name = info.get('shortName', symbol)
    price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    ctx = {
        "symbol": symbol,
        "company_name": name,
        "sector": info.get('sector', 'Unknown'),
        "industry": info.get('industry', 'Unknown'),
        "current_price": price,
        "market_cap": info.get('marketCap'),
        "52w_high": info.get('fiftyTwoWeekHigh'),
        "52w_low": info.get('fiftyTwoWeekLow'),
        "score": {
            "overall": result.get('overall_score'),
            "rating": result.get('rating'),
            "technical": result.get('technical_score'),
            "fundamental": result.get('fundamental_score'),
            "trend": result.get('trend_score'),
            "momentum": result.get('momentum_score'),
        },
        "technical_signals": {
            "rsi": signals.get('rsi'),
            "rsi_signal": signals.get('rsi_signal'),
            "macd_signal": signals.get('macd_signal'),
            "above_sma200": signals.get('above_sma200'),
            "ma_trend": signals.get('ma_trend'),
            "bb_signal": signals.get('bb_signal'),
            "adx": signals.get('adx'),
        },
        "trend": {
            "direction": str(trend_data.get('trend', 'SIDEWAYS')),
            "confidence": trend_data.get('confidence'),
            "bullish_pct": trend_data.get('bullish_pct'),
            "bearish_pct": trend_data.get('bearish_pct'),
        },
        "fundamentals": {},
    }
    if metrics:
        ctx["fundamentals"] = {
            "trailing_pe": metrics.get('trailing_pe'),
            "forward_pe": metrics.get('forward_pe'),
            "peg_ratio": metrics.get('peg_ratio'),
            "profit_margin": metrics.get('profit_margin'),
            "revenue_growth": metrics.get('revenue_growth'),
            "earnings_growth": metrics.get('earnings_growth'),
            "debt_to_equity": metrics.get('debt_to_equity'),
            "roe": metrics.get('roe'),
            "free_cashflow": metrics.get('free_cashflow'),
            "analyst_recommendation": metrics.get('recommendation'),
            "target_mean_price": metrics.get('target_mean_price'),
            "target_low": metrics.get('target_low_price'),
            "target_high": metrics.get('target_high_price'),
        }
    if articles:
        ctx["recent_news"] = [
            {"title": a.get("title", ""), "publisher": a.get("publisher", ""),
             "time": a.get("publish_time", "")}
            for a in articles[:8]
        ]
    return json.dumps(ctx, indent=2, default=str)


AI_SYSTEM_PROMPT = """You are a world-class stock analyst AI assistant. You analyze stocks using 
technical analysis, fundamental analysis, trend data, and recent news. 

Your analysis should be:
- Written in clear, conversational English that retail investors can understand
- Data-driven: reference specific numbers from the data provided
- Balanced: always discuss both bullish and bearish factors
- Actionable: end with a clear recommendation and key levels to watch
- Honest: if data is missing or unclear, say so

Format your response using Markdown with headers, bullet points, and bold for key numbers.
Keep it concise but thorough (300-500 words)."""

NEWS_SENTIMENT_PROMPT = """You are a financial news analyst AI. Analyze the following news headlines 
about a stock and provide:
1. An overall sentiment score from -100 (extremely bearish) to +100 (extremely bullish)
2. A brief 2-3 sentence summary of what the news means for investors
3. For each headline, a quick sentiment tag (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) and a 
   one-line explanation of its market impact

Be specific and reference the actual headlines. Format using Markdown."""


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, period, interval="1d"):
    return data_fetcher.get_history(symbol, period=period, interval=interval)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_info(symbol):
    return data_fetcher.get_info(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def compute_score(symbol, period="1y"):
    return stock_scorer.score_stock(symbol, period=period)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol, max_items=10):
    return data_fetcher.get_news(symbol, max_items=max_items)


# ─── Market Pulse helpers ────────────────────────────────────────────────────

MARKET_INDICES = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
    "Russell 2000": "^RUT", "VIX": "^VIX",
}
SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Energy": "XLE", "Consumer Disc.": "XLY", "Consumer Staples": "XLP",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communications": "XLC",
}


def compute_fear_greed():
    """Compute a simple Fear & Greed index from market data."""
    signals = {}
    score = 50
    try:
        # VIX component
        vix_df = data_fetcher.get_history("^VIX", period="3mo")
        if not vix_df.empty:
            vix_now = vix_df["Close"].iloc[-1]
            vix_avg = vix_df["Close"].mean()
            if vix_now < 15:
                signals["VIX"] = ("Extreme Greed", 90)
            elif vix_now < 20:
                signals["VIX"] = ("Greed", 70)
            elif vix_now < 25:
                signals["VIX"] = ("Neutral", 50)
            elif vix_now < 30:
                signals["VIX"] = ("Fear", 30)
            else:
                signals["VIX"] = ("Extreme Fear", 10)

        # Market momentum (S&P vs 125-day SMA)
        sp_df = data_fetcher.get_history("^GSPC", period="1y")
        if not sp_df.empty and len(sp_df) >= 125:
            sp_close = sp_df["Close"].iloc[-1]
            sp_sma = sp_df["Close"].tail(125).mean()
            pct_above = (sp_close / sp_sma - 1) * 100
            if pct_above > 5:
                signals["Momentum"] = ("Extreme Greed", 85)
            elif pct_above > 2:
                signals["Momentum"] = ("Greed", 70)
            elif pct_above > -2:
                signals["Momentum"] = ("Neutral", 50)
            elif pct_above > -5:
                signals["Momentum"] = ("Fear", 30)
            else:
                signals["Momentum"] = ("Extreme Fear", 15)

        # Market breadth (% of sample above SMA-50)
        breadth_syms = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
                        "NVDA", "JPM", "JNJ", "V", "PG",
                        "UNH", "HD", "BAC", "XOM", "PFE",
                        "KO", "DIS", "CSCO", "NFLX", "INTC"]
        above_count = 0
        total = 0
        for sym in breadth_syms:
            try:
                h = data_fetcher.get_history(sym, period="6mo")
                if not h.empty and len(h) >= 50:
                    total += 1
                    if h["Close"].iloc[-1] > h["Close"].tail(50).mean():
                        above_count += 1
            except Exception:
                pass
        if total > 0:
            breadth_pct = above_count / total * 100
            if breadth_pct > 70:
                signals["Breadth"] = ("Greed", int(breadth_pct))
            elif breadth_pct > 50:
                signals["Breadth"] = ("Neutral", int(breadth_pct))
            else:
                signals["Breadth"] = ("Fear", int(breadth_pct))

        # Safe-haven demand (Gold performance)
        gld_df = data_fetcher.get_history("GLD", period="3mo")
        if not gld_df.empty and len(gld_df) >= 21:
            gld_ret = (gld_df["Close"].iloc[-1] / gld_df["Close"].iloc[-21] - 1) * 100
            if gld_ret > 5:
                signals["Safe Haven"] = ("Extreme Fear", 15)
            elif gld_ret > 2:
                signals["Safe Haven"] = ("Fear", 35)
            elif gld_ret > -1:
                signals["Safe Haven"] = ("Neutral", 50)
            else:
                signals["Safe Haven"] = ("Greed", 75)

        if signals:
            score = sum(v[1] for v in signals.values()) / len(signals)
    except Exception:
        pass
    return round(score), signals


def generate_ai_narrative(symbol, info, result, signals, trend_data, metrics):
    """Generate a plain-English AI narrative summary for a stock."""
    name = info.get('shortName', symbol)
    price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    rating = result.get('rating', 'N/A')
    overall = result.get('overall_score', 0)
    sector = info.get('sector', 'Unknown')
    trend = str(trend_data.get('trend', 'SIDEWAYS')).replace('_', ' ').lower()
    conf = trend_data.get('confidence', 0)

    # Build narrative paragraphs
    paras = []

    # Opening
    if overall >= 70:
        paras.append(f"**{name}** looks like a strong investment opportunity right now. "
                     f"With an overall score of **{overall:.0f}/100** ({rating}), "
                     f"multiple indicators are flashing positive signals.")
    elif overall >= 50:
        paras.append(f"**{name}** presents a mixed picture for investors. "
                     f"At **{overall:.0f}/100** ({rating}), there are both "
                     f"opportunities and risks to consider.")
    else:
        paras.append(f"**{name}** is facing headwinds currently. "
                     f"With a score of **{overall:.0f}/100** ({rating}), "
                     f"caution may be warranted.")

    # Price action
    paras.append(f"The stock is trading at **${price:.2f}** in the **{sector}** sector, "
                 f"currently in a **{trend}** trend with {conf}% confidence.")

    # Technical summary
    rsi_sig = signals.get('rsi_signal', 'NEUTRAL')
    macd_sig = signals.get('macd_signal', 'NEUTRAL')
    tech_parts = []
    if rsi_sig == 'OVERSOLD':
        tech_parts.append("the RSI indicates the stock is oversold — a potential buying opportunity")
    elif rsi_sig == 'OVERBOUGHT':
        tech_parts.append("the RSI shows overbought conditions — the stock may be due for a pullback")
    if 'BULLISH' in str(macd_sig):
        tech_parts.append("MACD confirms bullish momentum")
    elif 'BEARISH' in str(macd_sig):
        tech_parts.append("MACD signals bearish momentum")
    if signals.get('above_sma200'):
        tech_parts.append("it's trading above its 200-day moving average (a positive long-term sign)")
    else:
        tech_parts.append("it's below its 200-day moving average (a cautionary sign)")
    if tech_parts:
        paras.append("**Technical outlook:** " + "; ".join(tech_parts) + ".")

    # Fundamental summary
    if metrics:
        fund_parts = []
        pe = metrics.get('trailing_pe')
        if pe:
            desc = "attractively valued" if pe < 20 else ("fairly valued" if pe < 30 else "richly valued")
            fund_parts.append(f"P/E ratio of {pe:.1f} ({desc})")
        pm = metrics.get('profit_margin')
        if pm:
            fund_parts.append(f"profit margin of {pm*100:.1f}%")
        rg = metrics.get('revenue_growth')
        if rg:
            fund_parts.append(f"revenue growing at {rg*100:.1f}%")
        rec = (metrics.get('recommendation') or '').upper()
        tp = metrics.get('target_mean_price')
        if tp and price:
            upside = (tp - price) / price * 100
            fund_parts.append(f"analysts target ${tp:.0f} ({upside:+.0f}%)")
        if fund_parts:
            paras.append("**Fundamentals:** " + ", ".join(fund_parts) + ".")

    # Bottom-line
    if overall >= 70:
        paras.append("**Bottom line:** The data supports a bullish case. "
                     "Consider this stock for your portfolio, but always "
                     "diversify and manage risk.")
    elif overall >= 50:
        paras.append("**Bottom line:** This is a hold/watch situation. "
                     "Wait for clearer signals before committing new capital.")
    else:
        paras.append("**Bottom line:** The current data suggests caution. "
                     "If you're holding, consider setting stop-losses. "
                     "If you're considering buying, wait for improvement in key metrics.")

    return "\n\n".join(paras)


# ─── Interactive Plotly Charts ───────────────────────────────────────────────

def create_price_chart(df, symbol, show_sma=True, show_bb=True,
                       show_volume=True, show_rsi=True, show_macd=True):
    """Build a multi-panel Plotly chart."""
    if "SMA_20" not in df.columns:
        ta.add_all_indicators(df)

    rows, titles, heights = [1], ["Price"], [0.45]
    cur_row = 1
    if show_volume:
        cur_row += 1; rows.append(cur_row); titles.append("Volume"); heights.append(0.12)
    if show_rsi:
        cur_row += 1; rows.append(cur_row); titles.append("RSI"); heights.append(0.15)
    if show_macd:
        cur_row += 1; rows.append(cur_row); titles.append("MACD"); heights.append(0.15)

    total_rows = cur_row
    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights + [0.13] * (total_rows - len(heights)),
        subplot_titles=titles,
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # ── SMAs ──
    if show_sma:
        for col, color, name in [
            ("SMA_20", "#FF9800", "SMA 20"),
            ("SMA_50", "#4CAF50", "SMA 50"),
            ("SMA_200", "#F44336", "SMA 200"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=name,
                    line=dict(color=color, width=1.2), opacity=0.8,
                ), row=1, col=1)

    # ── Bollinger Bands ──
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="#9C27B0", width=0.8, dash="dot"), opacity=0.4,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="#9C27B0", width=0.8, dash="dot"), opacity=0.4,
            fill="tonexty", fillcolor="rgba(156,39,176,0.06)",
        ), row=1, col=1)

    # ── Volume ──
    panel = 2
    if show_volume:
        colors = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else "#ef5350" for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.6,
        ), row=panel, col=1)
        panel += 1

    # ── RSI ──
    if show_rsi and "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#FF9800", width=1.5),
        ), row=panel, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#F44336",
                      opacity=0.5, row=panel, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50",
                      opacity=0.5, row=panel, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="#F44336", opacity=0.05,
                      row=panel, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="#4CAF50", opacity=0.05,
                      row=panel, col=1)
        panel += 1

    # ── MACD ──
    if show_macd and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#2196F3", width=1.2),
        ), row=panel, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#FF9800", width=1.2),
        ), row=panel, col=1)
        hist_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Histogram",
            marker_color=hist_colors, opacity=0.5,
        ), row=panel, col=1)

    fig.update_layout(
        height=200 + total_rows * 180,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=60, r=30, t=60, b=40),
        title=dict(text=f"{symbol} — Technical Analysis", font_size=18),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def create_comparison_chart(data: dict, normalize=True):
    """Normalized return comparison chart."""
    fig = go.Figure()
    for sym, df in data.items():
        if df.empty:
            continue
        values = df["Close"]
        if normalize:
            values = (values / values.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=values, name=sym, mode="lines", line=dict(width=2),
        ))
    fig.update_layout(
        height=500, template="plotly_white",
        yaxis_title="Return (%)" if normalize else "Price ($)",
        title="Stock Comparison — Normalized Returns",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-header">📈 Stock Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time analysis • Yahoo Finance</div>', unsafe_allow_html=True)
    st.divider()

    nav_options = ["🏠 Dashboard", "🔍 Stock Analysis", "🌡️ Market Pulse",
                   "📊 Screener", "🏆 Rankings", "⚖️ Compare",
                   "⏳ What-If Machine", "� Portfolio Tracker",
                   "📋 Watchlist", "🧓 Buffett Portfolio"]
    if "nav_to" in st.session_state:
        st.session_state["main_nav"] = st.session_state.pop("nav_to")
    page = st.radio(
        "Navigation", nav_options, key="main_nav",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Data cached for {config.CACHE_EXPIRY_MINUTES} min")
    if st.button("🔄 Clear Cache", use_container_width=True):
        data_fetcher.clear_cache()
        st.cache_data.clear()
        st.success("Cache cleared!")

    # ── AI Configuration ──
    st.divider()

    # Auto-detect Ollama BEFORE widgets render so defaults are pre-filled
    _ollama_running = False
    _ollama_models = []
    try:
        import urllib.request as _ul
        _ollama_req = _ul.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        _ollama_running = _ollama_req.status == 200
        import json as _json
        _ollama_models = [m["name"] for m in _json.loads(_ollama_req.read()).get("models", [])]
    except Exception:
        pass

    # Pre-fill defaults BEFORE text_input widgets render
    # Priority: 1) already in session  2) saved file  3) Ollama  4) Secrets
    if not st.session_state.get("openai_api_key"):
        _saved = _load_ai_settings()
        if _saved:
            # Restore from previously saved settings
            st.session_state["openai_api_key"] = _saved["api_key"]
            st.session_state["openai_base_url"] = _saved["base_url"]
            st.session_state["openai_model"] = _saved["model"]
        elif _ollama_running:
            # Local Ollama available — use it
            _pick = "llama3.2"
            for _m in _ollama_models:
                if _m.startswith("llama3.2"):
                    _pick = _m.split(":")[0]
                    break
            st.session_state["openai_api_key"] = "ollama"
            st.session_state["openai_base_url"] = "http://127.0.0.1:11434"
            st.session_state["openai_model"] = _pick
        else:
            # Cloud / no Ollama — try Streamlit Secrets
            _secret_key = _get_secret("OPENAI_API_KEY")
            if _secret_key:
                st.session_state["openai_api_key"] = _secret_key
                st.session_state["openai_base_url"] = _get_secret("OPENAI_BASE_URL")
                st.session_state["openai_model"] = _get_secret("OPENAI_MODEL", "gpt-4o-mini")

    _has_saved_settings = os.path.exists(_AI_SETTINGS_FILE)

    with st.expander("🤖 AI Settings", expanded=False):
        st.caption("Connect any OpenAI-compatible API for live AI insights")

        if _ollama_running:
            st.success(f"🌟 Ollama detected — {len(_ollama_models)} model(s) available")
            if st.button("⚡ Re-configure Ollama", use_container_width=True):
                _pick = "llama3.2"
                for _m in _ollama_models:
                    if _m.startswith("llama3.2"):
                        _pick = _m.split(":")[0]
                        break
                st.session_state["openai_api_key"] = "ollama"
                st.session_state["openai_base_url"] = "http://127.0.0.1:11434"
                st.session_state["openai_model"] = _pick
                st.rerun()
        elif not st.session_state.get("openai_api_key"):
            st.info(
                "☁️ **Cloud mode** — Ollama is not available here. "
                "Enter an API key below, or the app owner can set "
                "secrets in the Streamlit Cloud dashboard.\n\n"
                "**Free options:** [Groq](https://console.groq.com) · "
                "[OpenRouter](https://openrouter.ai)\n\n"
                "**Paid:** [OpenAI](https://platform.openai.com/api-keys)"
            )

        st.text_input(
            "API Key", type="password", key="openai_api_key",
            placeholder="sk-... or gsk-... or any compatible key",
            help="Your key is stored only in this session unless you click Save.",
        )
        st.text_input(
            "Base URL (optional)", key="openai_base_url",
            placeholder="https://api.groq.com/openai  or leave empty for OpenAI",
            help="Leave empty for OpenAI. Groq: https://api.groq.com/openai",
        )
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o-mini"
        st.text_input(
            "Model", key="openai_model",
            help="e.g. gpt-4o-mini, llama-3.3-70b-versatile, llama3.2, etc.",
        )

        if ai_available():
            _src = "Ollama" if _ollama_running else "Cloud API"
            st.success(f"✅ AI engine connected ({_src})")

            # ── Save / Clear / Test buttons ──
            _btn_cols = st.columns(2)
            with _btn_cols[0]:
                if st.button("💾 Save Settings", use_container_width=True,
                             help="Remember these settings for next time"):
                    _ok = _save_ai_settings(
                        st.session_state.get("openai_api_key", ""),
                        st.session_state.get("openai_base_url", ""),
                        st.session_state.get("openai_model", ""),
                    )
                    if _ok:
                        st.success("✅ Settings saved!")
                    else:
                        st.error("Failed to save settings")
            with _btn_cols[1]:
                if _has_saved_settings:
                    if st.button("🗑️ Clear Saved", use_container_width=True,
                                 help="Delete saved settings from disk"):
                        _delete_ai_settings()
                        st.success("Saved settings cleared")
                        st.rerun()
                else:
                    st.button("📡 Test", use_container_width=True,
                              key="_test_conn_btn", disabled=True,
                              help="Save first, then test")

            if _has_saved_settings:
                st.caption("💾 Settings loaded from saved file")

            if st.button("📡 Test Connection", use_container_width=True):
                with st.spinner("Testing AI connection..."):
                    test_result = call_llm(
                        "You are a helpful assistant.",
                        "Respond with exactly: CONNECTION OK",
                        temperature=0.0, max_tokens=20,
                    )
                if test_result and "⚠️" not in test_result:
                    st.success(f"✅ Working! Response: {test_result[:50]}")
                else:
                    st.error(f"❌ {test_result}")
                    base = st.session_state.get('openai_base_url', '')
                    if 'localhost' in base or '127.0.0.1' in base:
                        st.info("💡 You're using a local URL but Ollama isn't "
                                "reachable. If deployed to the cloud, you need "
                                "a cloud API (OpenAI, Groq, etc.) instead.")
                    elif base and '/v1' not in base:
                        st.info("💡 Tip: The URL will be auto-corrected to include /v1")
        elif HAS_OPENAI:
            st.info("Enter an API key to enable AI insights")
        else:
            st.warning("Install `openai` package for AI features")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("## 🏠 Dashboard")
    st.markdown("Quick overview of your watchlist stocks.")

    watchlist = utils.load_watchlist("default")
    cols_per_row = 4

    # Quick symbol entry
    col_a, col_b = st.columns([3, 1])
    with col_a:
        quick_sym = symbol_search("Quick lookup", key="dash_sym", default="")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        quick_go = st.button("Analyze →", use_container_width=True)

    if quick_go and quick_sym:
        go_to_analysis(quick_sym)
        st.rerun()

    st.divider()

    # Watchlist grid
    if watchlist:
        progress = st.progress(0, text="Loading watchlist data...")
        cards_data = []
        for i, sym in enumerate(watchlist[:20]):
            progress.progress((i + 1) / min(len(watchlist), 20),
                              text=f"Loading {sym}...")
            try:
                info = fetch_info(sym)
                if info:
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
                    change = ((price - prev) / prev * 100) if price and prev else None
                    cards_data.append({
                        "symbol": sym,
                        "name": info.get("shortName", sym),
                        "price": price,
                        "change": change,
                        "market_cap": info.get("marketCap"),
                        "pe": info.get("trailingPE"),
                        "sector": info.get("sector", ""),
                    })
            except Exception:
                pass
        progress.empty()

        # Display as grid
        for row_start in range(0, len(cards_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(cards_data):
                    break
                d = cards_data[idx]
                with col:
                    change_str = f"{d['change']:+.2f}%" if d['change'] is not None else "N/A"
                    change_color = "#4caf50" if (d['change'] or 0) >= 0 else "#f44336"
                    arrow = "▲" if (d['change'] or 0) >= 0 else "▼"
                    cap_str = fmt_number(d.get('market_cap'), '$')
                    card_label = (
                        f"**{d['symbol']}** "
                        f"&nbsp;·&nbsp; {d['name'][:20]}\n\n"
                        f"${d['price']:.2f} "
                        f"&ensp; :{('green' if (d['change'] or 0) >= 0 else 'red')}[{arrow} {change_str}]\n\n"
                        f"Cap {cap_str}"
                    )
                    st.markdown('<div class="stock-card-btn">', unsafe_allow_html=True)
                    if st.button(card_label, key=f"dash_{d['symbol']}",
                                 use_container_width=True):
                        go_to_analysis(d['symbol'])
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No watchlist found. Go to 📋 Watchlist to set one up.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Stock Analysis
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Stock Analysis":
    st.markdown("## 🔍 Stock Analysis")

    # Input bar
    auto_run = "analyze_symbol" in st.session_state
    _auto_sym = ""
    if auto_run:
        _auto_sym = st.session_state.pop("analyze_symbol")
        st.session_state["_analysis_active"] = True

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = symbol_search("🔍 Search stock", key="sym_main",
                               default=_auto_sym if auto_run else "AAPL")
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        st.session_state["_analysis_active"] = True

    _should_analyze = symbol and (analyze_btn or auto_run or st.session_state.get("_analysis_active"))

    if _should_analyze:
        with st.spinner(f"Analyzing {symbol}..."):
            df = fetch_history(symbol, period)
            info = fetch_info(symbol)

        if df.empty or not info:
            st.error(f"❌ No data found for **{symbol}**. Please check the symbol.")
        else:
            ta.add_all_indicators(df)
            last = df.iloc[-1]

            # ── Header ──
            price = info.get("currentPrice") or info.get("regularMarketPrice") or last["Close"]
            prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
            change = ((price - prev) / prev * 100) if price and prev else 0

            st.markdown(f"### {info.get('shortName', symbol)} ({symbol})")
            st.caption(f"{info.get('sector', '')} · {info.get('industry', '')}")

            # ── Key Metrics Row ──
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Price", f"${price:.2f}", f"{change:+.2f}%")
            m2.metric("Market Cap", fmt_number(info.get("marketCap"), "$"))
            m3.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
            m4.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
            m5.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

            # ── Compute score once for verdict + score tab ──
            with st.spinner("Computing investment score..."):
                result = compute_score(symbol, period)
            signals = ta.get_latest_signals(df)
            metrics = fa.get_fundamental_metrics(symbol)
            trend_data = trend_analyzer.classify_trend(df)

            # ════════════════════════════════════════════════════════════════
            #  INVESTMENT VERDICT – the most important section
            # ════════════════════════════════════════════════════════════════
            if not result.get("error"):
                rating = result["rating"]
                overall = result["overall_score"]
                r_color = rating_color(rating)
                bg_color = {
                    "STRONG BUY": "#e8f5e9", "BUY": "#f1f8e9",
                    "HOLD": "#fff8e1", "SELL": "#fbe9e7",
                    "STRONG SELL": "#ffebee",
                }.get(rating, "#f5f5f5")
                emoji = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
                         "SELL": "🔴", "STRONG SELL": "🔴🔴"}.get(rating, "⚪")

                # Build plain-English bullet points
                pros, cons = [], []
                # Trend
                trend_dir = trend_data.get("trend", "SIDEWAYS")
                if "UP" in str(trend_dir).upper():
                    pros.append(f"📈 Price is in a **{trend_dir.replace('_', ' ').lower()}** trend")
                elif "DOWN" in str(trend_dir).upper():
                    cons.append(f"📉 Price is in a **{trend_dir.replace('_', ' ').lower()}** trend")
                # RSI
                rsi_sig = signals.get("rsi_signal", "")
                rsi_val = signals.get("rsi", "")
                if rsi_sig == "OVERSOLD":
                    pros.append(f"🔋 RSI at {rsi_val} — oversold, potential bounce ahead")
                elif rsi_sig == "OVERBOUGHT":
                    cons.append(f"⚠️ RSI at {rsi_val} — overbought, may pull back")
                # MACD
                macd_sig = signals.get("macd_signal", "")
                if "BULLISH" in macd_sig:
                    pros.append("✅ MACD shows bullish momentum")
                elif "BEARISH" in macd_sig:
                    cons.append("❌ MACD shows bearish momentum")
                # SMA 200
                if signals.get("above_sma200"):
                    pros.append("✅ Trading above 200-day moving average (long-term uptrend)")
                else:
                    cons.append("❌ Trading below 200-day moving average (long-term weakness)")
                # Fundamental
                if metrics:
                    pe = metrics.get("trailing_pe")
                    if pe is not None:
                        if pe < 15:
                            pros.append(f"💰 P/E ratio of {pe:.1f} — attractively valued")
                        elif pe > 35:
                            cons.append(f"💸 P/E ratio of {pe:.1f} — expensive valuation")
                    pm = metrics.get("profit_margin")
                    if pm is not None and pm > 0.15:
                        pros.append(f"💎 Strong profit margin of {pm*100:.1f}%")
                    rg = metrics.get("revenue_growth")
                    if rg is not None:
                        if rg > 0.1:
                            pros.append(f"🚀 Revenue growing at {rg*100:.1f}%")
                        elif rg < -0.05:
                            cons.append(f"📉 Revenue declining at {rg*100:.1f}%")
                    de = metrics.get("debt_to_equity")
                    if de is not None and de > 200:
                        cons.append(f"⚠️ High debt-to-equity ratio of {de:.0f}")
                    rec = (metrics.get("recommendation") or "").upper()
                    target = metrics.get("target_mean_price")
                    if target and price:
                        upside = (target - price) / price * 100
                        if upside > 5:
                            pros.append(f"🎯 Analyst target ${target:.0f} ({upside:+.0f}% upside)")
                        elif upside < -5:
                            cons.append(f"🎯 Analyst target ${target:.0f} ({upside:+.0f}% downside)")

                # Render verdict card
                st.markdown(
                    f'<div style="background:{bg_color};border:2px solid {r_color};'
                    f'border-radius:14px;padding:1.2rem 1.5rem;margin:0.8rem 0 1rem 0">'
                    f'<div style="display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap">'
                    f'<span style="font-size:2.2rem;font-weight:800;color:{r_color}">'
                    f'{overall:.0f}</span>'
                    f'<span style="font-size:1.4rem;font-weight:700;color:{r_color}">'
                    f'{emoji} {rating}</span>'
                    f'<span style="color:#666;font-size:0.9rem;margin-left:auto">'
                    f'Investment Score out of 100</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                vc1, vc2 = st.columns(2)
                with vc1:
                    if pros:
                        st.markdown("**✅ Bullish Factors**")
                        for p in pros:
                            st.markdown(f"- {p}")
                    else:
                        st.markdown("**✅ Bullish Factors**\n- _No strong bullish signals detected_")
                with vc2:
                    if cons:
                        st.markdown("**⚠️ Risk Factors**")
                        for c in cons:
                            st.markdown(f"- {c}")
                    else:
                        st.markdown("**⚠️ Risk Factors**\n- _No significant risks detected_")

            st.divider()

            # ── Tabs ──
            tab_chart, tab_score, tab_tech, tab_fund, tab_trend, tab_news, tab_ai = st.tabs(
                ["📊 Chart", "⭐ Score Details", "🔧 Technical",
                 "📑 Fundamental", "📈 Trend", "📰 News", "🧠 AI Insights"]
            )

            # ── TAB: Chart ──
            with tab_chart:
                cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                show_sma = cc1.checkbox("SMA", True)
                show_bb = cc2.checkbox("Bollinger", True)
                show_vol = cc3.checkbox("Volume", True)
                show_rsi = cc4.checkbox("RSI", True)
                show_macd = cc5.checkbox("MACD", True)

                fig = create_price_chart(df, symbol, show_sma, show_bb,
                                         show_vol, show_rsi, show_macd)
                st.plotly_chart(fig, use_container_width=True)

            # ── TAB: Score Details ──
            with tab_score:
                if result.get("error"):
                    st.error(result["error"])
                else:
                    sc1, sc2 = st.columns([1, 2])
                    with sc1:
                        st.markdown("### Overall Score")
                        st.markdown(f"# {result['overall_score']:.0f}/100")
                        st.markdown(rating_badge(result["rating"]), unsafe_allow_html=True)

                    with sc2:
                        # Visual progress bars for each dimension
                        dims = [
                            ("🔧 Technical", result.get("technical_score") or 0, 35),
                            ("📑 Fundamental", result.get("fundamental_score") or 0, 30),
                            ("📈 Trend", result.get("trend_score") or 0, 20),
                            ("🚀 Momentum", result.get("momentum_score") or 0, 15),
                        ]
                        for label, sc_val, weight in dims:
                            dcol1, dcol2 = st.columns([3, 1])
                            with dcol1:
                                st.caption(f"{label} (weight {weight}%)")
                                st.progress(min(sc_val / 100, 1.0))
                            with dcol2:
                                st.markdown(f"**{sc_val:.0f}**/100")

                    st.divider()
                    st.markdown("#### What's driving the score?")
                    bd = result.get("breakdown", {})
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("**Technical Signals**")
                        for k, v in bd.get("technical", {}).items():
                            st.markdown(f"- {k.replace('_', ' ').title()}: {v}")
                    with bc2:
                        st.markdown("**Price Momentum**")
                        for k, v in bd.get("momentum", {}).items():
                            st.markdown(f"- {k.replace('_', ' ').title()}: {v}")

            # ── TAB: Technical ──
            with tab_tech:
                st.markdown("#### Signal Dashboard")
                st.caption("Current readings from key technical indicators")

                sg1, sg2, sg3, sg4 = st.columns(4)
                sg1.markdown(f"**RSI**<br>{signal_badge(signals.get('rsi_signal', 'N/A'))}<br>"
                             f"Value: {signals.get('rsi', 'N/A')}", unsafe_allow_html=True)
                sg2.markdown(f"**MACD**<br>{signal_badge(signals.get('macd_signal', 'N/A'))}",
                             unsafe_allow_html=True)
                sg3.markdown(f"**Bollinger**<br>{signal_badge(signals.get('bb_signal', 'N/A'))}",
                             unsafe_allow_html=True)
                sg4.markdown(f"**Stochastic**<br>{signal_badge(signals.get('stoch_signal', 'N/A'))}<br>"
                             f"Value: {signals.get('stoch_k', 'N/A')}", unsafe_allow_html=True)

                st.divider()
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**Trend Indicators**")
                    st.markdown(f"- MA Trend: {signal_badge(signals.get('ma_trend', 'N/A'))}",
                                unsafe_allow_html=True)
                    st.markdown(f"- Above SMA 200: {'✅ Yes' if signals.get('above_sma200') else '❌ No'}")
                    pct_sma = signals.get('pct_from_sma200', 'N/A')
                    st.markdown(f"- Distance from SMA 200: {pct_sma}%")
                with tc2:
                    st.markdown("**Trend Strength**")
                    st.markdown(f"- ADX: {signals.get('adx', 'N/A')}")
                    st.markdown(f"- Strength: {signal_badge(signals.get('trend_strength', 'N/A'))}",
                                unsafe_allow_html=True)
                    st.markdown(f"- Bollinger Width: {signals.get('bb_width', 'N/A')}")

                st.divider()
                st.markdown("#### All Indicator Values")
                ind_cols = [c for c in df.columns
                            if c not in ("Open", "High", "Low", "Close", "Volume")]
                ind_data = {col: [round(last[col], 4) if pd.notna(last[col]) else None]
                            for col in ind_cols}
                st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

            # ── TAB: Fundamental ──
            with tab_fund:
                fund_score = fa.get_fundamental_score(symbol)

                if not metrics:
                    st.warning("No fundamental data available for this stock.")
                else:
                    fs = fund_score.get("overall_score", "N/A")
                    st.markdown(f"#### Fundamental Health Score: **{fs}/100**")
                    st.caption("How strong is this company's financial foundation?")

                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Valuation", f"{fund_score.get('valuation_score', 'N/A')}/100")
                    fc2.metric("Profitability", f"{fund_score.get('profitability_score', 'N/A')}/100")
                    fc3.metric("Growth", f"{fund_score.get('growth_score', 'N/A')}/100")
                    fc4.metric("Health", f"{fund_score.get('health_score', 'N/A')}/100")

                    st.divider()
                    fl1, fl2, fl3 = st.columns(3)
                    with fl1:
                        st.markdown("**📊 Valuation**")
                        st.markdown(f"- P/E (Trailing): {metrics.get('trailing_pe', 'N/A')}")
                        st.markdown(f"- P/E (Forward): {metrics.get('forward_pe', 'N/A')}")
                        st.markdown(f"- PEG Ratio: {metrics.get('peg_ratio', 'N/A')}")
                        st.markdown(f"- P/B: {metrics.get('price_to_book', 'N/A')}")
                        st.markdown(f"- EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}")
                    with fl2:
                        st.markdown("**💰 Profitability**")
                        st.markdown(f"- Profit Margin: {fmt_pct(metrics.get('profit_margin'))}")
                        st.markdown(f"- Operating Margin: {fmt_pct(metrics.get('operating_margin'))}")
                        st.markdown(f"- ROE: {fmt_pct(metrics.get('roe'))}")
                        st.markdown(f"- ROA: {fmt_pct(metrics.get('roa'))}")
                    with fl3:
                        st.markdown("**🏦 Financial Health**")
                        st.markdown(f"- Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}")
                        st.markdown(f"- Current Ratio: {metrics.get('current_ratio', 'N/A')}")
                        st.markdown(f"- Free Cash Flow: {fmt_number(metrics.get('free_cashflow'), '$')}")

                    st.divider()
                    ga1, ga2 = st.columns(2)
                    with ga1:
                        st.markdown("**🚀 Growth**")
                        st.markdown(f"- Revenue Growth: {fmt_pct(metrics.get('revenue_growth'))}")
                        st.markdown(f"- Earnings Growth: {fmt_pct(metrics.get('earnings_growth'))}")
                    with ga2:
                        st.markdown("**🎯 Analyst Consensus**")
                        rec = (metrics.get("recommendation") or "N/A").upper()
                        st.markdown(f"- Recommendation: **{rec}**")
                        tp = metrics.get('target_mean_price')
                        if tp and price:
                            upside = (tp - price) / price * 100
                            st.markdown(
                                f"- Target Price: **${tp:.2f}** "
                                f"({'🟢' if upside >= 0 else '🔴'} {upside:+.1f}%)"
                            )
                        else:
                            st.markdown(f"- Target Price: N/A")
                        st.markdown(f"- Target Range: ${metrics.get('target_low_price', 'N/A')} "
                                    f"– ${metrics.get('target_high_price', 'N/A')}")

            # ── TAB: Trend ──
            with tab_trend:
                sr = trend_analyzer.find_support_resistance(df)

                tr1, tr2 = st.columns([1, 2])
                with tr1:
                    trend_str = trend_data.get("trend", "N/A")
                    conf = trend_data.get("confidence", 0)
                    trend_emoji = "📈" if "UP" in str(trend_str).upper() else (
                        "📉" if "DOWN" in str(trend_str).upper() else "➡️")
                    st.markdown(f"### {trend_emoji} {trend_str.replace('_', ' ').title()}")
                    st.progress(min(conf / 100, 1.0), text=f"Confidence: {conf}%")
                    st.metric("Bullish Signals", f"{trend_data.get('bullish_pct', 0)}%")
                    st.metric("Bearish Signals", f"{trend_data.get('bearish_pct', 0)}%")

                with tr2:
                    st.markdown("**What factors determine the trend?**")
                    factors = trend_data.get("factors", {})
                    factor_df = pd.DataFrame([
                        {"Factor": k.replace("_", " ").title(), "Value": str(v)}
                        for k, v in factors.items()
                    ])
                    if not factor_df.empty:
                        st.dataframe(factor_df, use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("#### Key Price Levels")
                st.caption("Support = price floors where buyers step in · Resistance = ceilings where sellers appear")
                sr1, sr2, sr3 = st.columns(3)
                sr1.metric("Current Price", f"${sr.get('current_price', 'N/A')}")
                with sr2:
                    st.markdown("**🔴 Resistance (sell pressure)**")
                    for r in sr.get("resistance", []):
                        dist = ((r - price) / price * 100) if price else 0
                        st.markdown(f"- ${r:,.2f}  ({dist:+.1f}% away)")
                with sr3:
                    st.markdown("**🟢 Support (buy pressure)**")
                    for s in sr.get("support", []):
                        dist = ((s - price) / price * 100) if price else 0
                        st.markdown(f"- ${s:,.2f}  ({dist:+.1f}% away)")

            # ── TAB: News ──
            with tab_news:
                st.markdown("#### 📰 Latest News")
                st.caption(f"Recent headlines about {info.get('shortName', symbol)} from Yahoo Finance")

                with st.spinner("Fetching latest news..."):
                    articles = fetch_news(symbol, max_items=10)

                if not articles:
                    st.info("No recent news found for this stock.")
                else:
                    # ── AI News Sentiment Analysis ──
                    if ai_available():
                        headlines_text = "\n".join(
                            f"- {a.get('title', 'N/A')} ({a.get('publisher', 'Unknown')}, {a.get('publish_time', '')})"
                            for a in articles
                        )
                        news_prompt = (f"Stock: {symbol} ({info.get('shortName', symbol)})\n"
                                       f"Current price: ${price:.2f}\n\n"
                                       f"Headlines:\n{headlines_text}")

                        with st.spinner("🤖 AI is analyzing news sentiment..."):
                            sentiment_response = call_llm(
                                NEWS_SENTIMENT_PROMPT, news_prompt,
                                temperature=0.4, max_tokens=1200,
                            )

                        if sentiment_response and not sentiment_response.startswith("⚠️"):
                            st.markdown(
                                f'<div style="background:linear-gradient(135deg,#e8f5e9,#e3f2fd);'
                                f'border-radius:14px;padding:1.2rem;border:1px solid #c8e6c9;'
                                f'margin-bottom:1rem">'
                                f'<div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem">'
                                f'🤖 AI News Sentiment Analysis</div>'
                                f'<div style="font-size:0.9rem;line-height:1.7">{sentiment_response}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.divider()
                        elif sentiment_response.startswith("⚠️"):
                            st.warning(sentiment_response)
                    else:
                        st.info("💡 Configure an AI API key in the sidebar to get AI-powered news sentiment analysis")

                    # ── News list ──
                    for i, art in enumerate(articles):
                        title = art.get("title", "No title")
                        link = art.get("link", "")
                        publisher = art.get("publisher", "Unknown")
                        pub_time = art.get("publish_time", "")

                        with st.container(border=True):
                            nc1, nc2 = st.columns([5, 1])
                            with nc1:
                                if link:
                                    st.markdown(f"**[{title}]({link})**")
                                else:
                                    st.markdown(f"**{title}**")
                                meta_parts = []
                                if publisher:
                                    meta_parts.append(f"📰 {publisher}")
                                if pub_time:
                                    meta_parts.append(f"🕐 {pub_time}")
                                if meta_parts:
                                    st.caption(" · ".join(meta_parts))
                            with nc2:
                                thumb = art.get("thumbnail", "")
                                if thumb:
                                    st.image(thumb, width=80)

            # ── TAB: AI Insights ──
            with tab_ai:
                st.markdown("#### 🧠 AI-Powered Analysis")

                # ── Dynamic LLM Analysis ──
                if ai_available():
                    st.caption("🔴 Live AI analysis — powered by your connected LLM")

                    # Fetch news for context
                    ai_articles = fetch_news(symbol, max_items=8)

                    # Build full context
                    stock_ctx = build_stock_context(
                        symbol, info, result, signals, trend_data, metrics, ai_articles
                    )

                    ai_user_prompt = (
                        f"Analyze this stock in detail. Here is the complete data:\n\n"
                        f"{stock_ctx}\n\n"
                        f"Please provide:\n"
                        f"1. **Executive Summary** — a 2-3 sentence overview\n"
                        f"2. **Technical Analysis** — what the charts and indicators are telling us\n"
                        f"3. **Fundamental Assessment** — valuation, growth, financial health\n"
                        f"4. **News & Sentiment** — what recent news means for the stock\n"
                        f"5. **Risk Factors** — key risks investors should know\n"
                        f"6. **Action Plan** — specific entry/exit levels, position sizing advice\n"
                        f"7. **Verdict** — final BUY/HOLD/SELL recommendation with conviction level"
                    )

                    # Cache key for this analysis
                    ai_cache_key = f"ai_analysis_{symbol}_{result.get('overall_score', 0):.0f}"
                    if ai_cache_key not in st.session_state:
                        with st.spinner("🤖 AI is generating a deep analysis..."):
                            st.session_state[ai_cache_key] = call_llm(
                                AI_SYSTEM_PROMPT, ai_user_prompt,
                                temperature=0.6, max_tokens=2000,
                            )

                    ai_response = st.session_state[ai_cache_key]

                    if ai_response and not ai_response.startswith("⚠️"):
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#f3e5f5,#e8f0fe);'
                            f'border-radius:14px;padding:1.5rem;border:1px solid #ce93d8;'
                            f'line-height:1.8;font-size:0.92rem">'
                            f'<div style="display:flex;align-items:center;gap:0.5rem;'
                            f'margin-bottom:0.8rem">'
                            f'<span style="font-size:1.5rem">🤖</span>'
                            f'<span style="font-weight:700;font-size:1.05rem">'
                            f'Live AI Analysis — {info.get("shortName", symbol)}</span>'
                            f'<span style="background:#ce93d8;color:white;padding:2px 10px;'
                            f'border-radius:12px;font-size:0.75rem;font-weight:600">'
                            f'AI-GENERATED</span></div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(ai_response)

                        # Regenerate button
                        if st.button("🔄 Regenerate Analysis", key="regen_ai"):
                            if ai_cache_key in st.session_state:
                                del st.session_state[ai_cache_key]
                            st.rerun()
                    else:
                        st.error(ai_response or "Failed to get AI response")

                    st.divider()

                    # ── Ask AI a follow-up question ──
                    st.markdown("#### 💬 Ask AI About This Stock")
                    user_q = st.text_input(
                        "Ask anything", placeholder=f"Is {symbol} a good buy for retirement?",
                        key="ai_question", label_visibility="collapsed",
                    )
                    if st.button("🚀 Ask", key="ai_ask_btn") and user_q:
                        followup_prompt = (
                            f"Context about {symbol}:\n{stock_ctx}\n\n"
                            f"User question: {user_q}\n\n"
                            f"Answer the question based on the stock data provided. "
                            f"Be specific, reference actual numbers, and be helpful."
                        )
                        with st.spinner("🤖 Thinking..."):
                            answer = call_llm(
                                AI_SYSTEM_PROMPT, followup_prompt,
                                temperature=0.5, max_tokens=800,
                            )
                        if answer:
                            st.markdown(
                                f'<div style="background:#f3e5f5;border-radius:12px;'
                                f'padding:1rem;border:1px solid #ce93d8;margin-top:0.5rem">'
                                f'{answer}</div>',
                                unsafe_allow_html=True,
                            )

                else:
                    # ── Fallback: static analysis (no API key) ──
                    st.caption("📊 Rule-based analysis — configure AI in sidebar for live LLM insights")
                    st.info(
                        "💡 **Upgrade to AI-powered insights!** Enter an OpenAI-compatible API key "
                        "in the sidebar → 🤖 AI Settings to unlock:\n"
                        "- Deep AI narrative analysis\n"
                        "- AI news sentiment scoring\n"
                        "- Interactive Q&A about any stock\n"
                        "- Works with OpenAI, Azure, Ollama, and any compatible API"
                    )

                    narrative = generate_ai_narrative(
                        symbol, info, result, signals, trend_data, metrics
                    )
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#f8f9fa,#e8f0fe);'
                        f'border-radius:14px;padding:1.5rem;border:1px solid #ddd;'
                        f'line-height:1.8;font-size:0.95rem">{narrative}</div>',
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Decision helper (always shown)
                st.markdown("#### ✅ Decision Checklist")
                checklist_items = [
                    ("Score above 60 (Buy zone)", (result.get('overall_score', 0) or 0) >= 60),
                    ("Price in uptrend", "UP" in str(trend_data.get('trend', '')).upper()),
                    ("RSI not overbought (< 70)", (signals.get('rsi', 50) or 50) < 70),
                    ("Above 200-day MA", bool(signals.get('above_sma200'))),
                    ("Positive revenue growth", bool(metrics and (metrics.get('revenue_growth') or 0) > 0)),
                    ("Analyst upside potential", bool(
                        metrics and metrics.get('target_mean_price') and price and
                        metrics['target_mean_price'] > price)),
                    ("Manageable debt (D/E < 150)", bool(
                        metrics and metrics.get('debt_to_equity') is not None and
                        metrics['debt_to_equity'] < 150)),
                ]
                passed = sum(1 for _, ok in checklist_items if ok)
                st.progress(passed / len(checklist_items),
                            text=f"{passed}/{len(checklist_items)} criteria met")
                for label, ok in checklist_items:
                    icon = "✅" if ok else "❌"
                    st.markdown(f"{icon} {label}")

                # Verdict bar
                if passed >= 6:
                    st.success("🚀 **Strong candidate** — Most criteria are met. Worth a deeper look.")
                elif passed >= 4:
                    st.info("🔍 **Mixed signals** — Some positives, some concerns. Monitor closely.")
                else:
                    st.warning("⚠️ **Proceed with caution** — Multiple red flags detected.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Market Pulse
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🌡️ Market Pulse":
    st.markdown("## 🌡️ Market Pulse")
    st.markdown("Real-time market sentiment, sector performance, and fear/greed reading")

    with st.spinner("Reading the market's vital signs..."):
        fg_score, fg_signals = compute_fear_greed()

    # ── Fear & Greed Gauge ──
    if fg_score <= 25:
        fg_label, fg_color, fg_emoji = "Extreme Fear", "#b71c1c", "😨"
    elif fg_score <= 40:
        fg_label, fg_color, fg_emoji = "Fear", "#f44336", "😟"
    elif fg_score <= 60:
        fg_label, fg_color, fg_emoji = "Neutral", "#ff9800", "😐"
    elif fg_score <= 75:
        fg_label, fg_color, fg_emoji = "Greed", "#4caf50", "😏"
    else:
        fg_label, fg_color, fg_emoji = "Extreme Greed", "#1b5e20", "🤑"

    fg1, fg2 = st.columns([1, 2])
    with fg1:
        st.markdown(
            f'<div style="text-align:center;background:linear-gradient(135deg,#f8f9fa,#e8eaf6);'
            f'border-radius:16px;padding:1.5rem;border:2px solid {fg_color}">' 
            f'<div style="font-size:3.5rem;font-weight:900;color:{fg_color}">{fg_score}</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:{fg_color}">{fg_emoji} {fg_label}</div>'
            f'<div style="color:#888;font-size:0.8rem;margin-top:0.3rem">Fear & Greed Index</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("0 = Extreme Fear · 100 = Extreme Greed")

    with fg2:
        st.markdown("#### Signal Breakdown")
        for sig_name, (sig_label, sig_val) in fg_signals.items():
            scol1, scol2, scol3 = st.columns([2, 3, 1])
            with scol1:
                st.markdown(f"**{sig_name}**")
            with scol2:
                bar_color = (
                    "#b71c1c" if sig_val <= 25 else
                    "#f44336" if sig_val <= 40 else
                    "#ff9800" if sig_val <= 60 else
                    "#4caf50" if sig_val <= 75 else "#1b5e20"
                )
                st.progress(sig_val / 100)
            with scol3:
                st.caption(sig_label)

    st.divider()

    # ── Major Indices ──
    st.markdown("#### 🌍 Major Indices")
    idx_cols = st.columns(len(MARKET_INDICES))
    for i, (name, sym) in enumerate(MARKET_INDICES.items()):
        with idx_cols[i]:
            try:
                idf = fetch_info(sym)
                p = idf.get("regularMarketPrice", 0)
                prev = idf.get("regularMarketPreviousClose") or idf.get("previousClose", p)
                chg = ((p - prev) / prev * 100) if prev else 0
                st.metric(name, f"{p:,.2f}", f"{chg:+.2f}%")
            except Exception:
                st.metric(name, "N/A")

    st.divider()

    # ── Sector Heatmap ──
    st.markdown("#### 🟩 Sector Performance Heatmap")
    st.caption("Click any sector to analyze the ETF")
    sector_data = []
    for sec_name, etf in SECTOR_ETFS.items():
        try:
            sinfo = fetch_info(etf)
            p = sinfo.get("regularMarketPrice") or sinfo.get("currentPrice", 0)
            prev = sinfo.get("regularMarketPreviousClose") or sinfo.get("previousClose", p)
            chg = ((p - prev) / prev * 100) if prev else 0
            sector_data.append({"sector": sec_name, "etf": etf, "change": chg})
        except Exception:
            sector_data.append({"sector": sec_name, "etf": etf, "change": 0})

    # Treemap
    if sector_data:
        labels = [f"{d['sector']}\n{d['change']:+.2f}%" for d in sector_data]
        values = [abs(d["change"]) + 0.1 for d in sector_data]
        colors = ["#4caf50" if d["change"] >= 0 else "#f44336" for d in sector_data]

        fig_tm = go.Figure(go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            marker=dict(
                colors=[d["change"] for d in sector_data],
                colorscale=[[0, "#c62828"], [0.5, "#fff9c4"], [1, "#2e7d32"]],
                cmid=0,
            ),
            textinfo="label",
            textfont=dict(size=14),
        ))
        fig_tm.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tm, use_container_width=True)

        # Sector pills
        etf_syms = [d["etf"] for d in sector_data]
        selected_etf = st.pills("🔍 Analyze sector ETF", etf_syms,
                                default=None, key="sector_pill")
        if selected_etf:
            del st.session_state["sector_pill"]
            go_to_analysis(selected_etf)
            st.rerun()

    # ── Market tip ──
    st.divider()
    if fg_score <= 30:
        st.info("💡 **When others are fearful, be greedy.** — Warren Buffett. "
                "Extreme fear can be a buying opportunity for long-term investors.")
    elif fg_score >= 70:
        st.info("💡 **When others are greedy, be fearful.** — Warren Buffett. "
                "Extreme greed can signal a market top. Consider taking profits.")
    else:
        st.info("💡 **Market sentiment is balanced.** Focus on individual stock "
                "fundamentals rather than market-wide emotion.")

elif page == "📊 Screener":
    st.markdown("## 📊 Stock Screener")

    scr_col1, scr_col2 = st.columns([1, 3])
    with scr_col1:
        strategy = st.selectbox("Strategy", ["growth", "value", "momentum", "dividend"])
        descriptions = {
            "growth": "High-growth companies in uptrends with strong revenue growth",
            "value": "Undervalued companies with solid fundamentals and low debt",
            "momentum": "Stocks with strong technical momentum and bullish signals",
            "dividend": "Stable companies with positive cash flow for income investors",
        }
        st.info(descriptions[strategy])
        run_screen = st.button("🚀 Run Screener", type="primary", use_container_width=True)

    with scr_col2:
        if run_screen:
            criteria = stock_screener.SCREENS[strategy]
            symbols = config.DEFAULT_UNIVERSE
            progress = st.progress(0, text="Screening...")

            def screen_progress(sym, i, total):
                progress.progress(i / total, text=f"Screening {sym} ({i}/{total})...")

            with st.spinner("Running screener..."):
                results = stock_screener.screen_stocks(
                    symbols, criteria, progress_callback=screen_progress
                )
            progress.empty()

            if results:
                st.success(f"✅ {len(results)} stocks matched the **{strategy}** criteria")
                rows = []
                for r in results:
                    rows.append({
                        "Symbol": r["symbol"],
                        "Name": r.get("name", ""),
                        "Price": f"${r['price']:.2f}" if r.get("price") else "N/A",
                        "P/E": f"{r['pe']:.1f}" if r.get("pe") else "N/A",
                        "Sector": r.get("sector", ""),
                        "Reasons": ", ".join(r.get("reasons", [])),
                    })
                scr_df = pd.DataFrame(rows)
                st.dataframe(scr_df, use_container_width=True, hide_index=True)
                csv_download(scr_df, "screener_results.csv")

                # Compact clickable symbol pills
                _syms = [r["symbol"] for r in results]
                selected = st.pills("🔍 Quick analyze", _syms,
                                    default=None, key="scr_pill")
                if selected:
                    del st.session_state["scr_pill"]
                    go_to_analysis(selected)
                    st.rerun()
            else:
                st.warning("No stocks matched. Try a different strategy.")
        else:
            st.markdown("Select a strategy and click **Run Screener** to find matching stocks.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Rankings
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Rankings":
    st.markdown("## 🏆 Stock Rankings")

    rk_col1, rk_col2 = st.columns([1, 3])
    with rk_col1:
        source = st.radio("Stock source", ["Watchlist", "Full Universe", "Custom"])
        top_n = st.slider("Top N", 5, 30, 10)
        run_rank = st.button("🏆 Rank Stocks", type="primary", use_container_width=True)

        if source == "Custom":
            custom_syms = st.text_area("Symbols (one per line)",
                                       "AAPL\nMSFT\nGOOGL\nAMZN\nNVDA")

    with rk_col2:
        if run_rank:
            if source == "Watchlist":
                symbols = utils.load_watchlist("default")
            elif source == "Full Universe":
                symbols = config.DEFAULT_UNIVERSE
            else:
                symbols = [s.strip().upper() for s in custom_syms.strip().split("\n") if s.strip()]

            progress = st.progress(0, text="Scoring stocks...")

            def rank_progress(sym, i, total):
                progress.progress(i / total, text=f"Scoring {sym} ({i}/{total})...")

            with st.spinner("Ranking..."):
                ranked = stock_scorer.rank_stocks(
                    symbols, top_n=top_n, progress_callback=rank_progress
                )
            progress.empty()

            if ranked:
                st.success(f"Ranked {len(ranked)} stocks")

                # Bar chart
                chart_df = pd.DataFrame(ranked)
                fig = go.Figure()
                colors = [rating_color(r["rating"]) for r in ranked]
                fig.add_trace(go.Bar(
                    y=[r["symbol"] for r in ranked][::-1],
                    x=[r["overall_score"] for r in ranked][::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{r['overall_score']:.0f} ({r['rating']})" for r in ranked][::-1],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=max(300, len(ranked) * 40),
                    template="plotly_white",
                    xaxis=dict(range=[0, 105], title="Score"),
                    title="Investment Scores",
                    margin=dict(l=80, r=120, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                rows = []
                for i, r in enumerate(ranked, 1):
                    rows.append({
                        "#": i,
                        "Symbol": r["symbol"],
                        "Name": r.get("name", "")[:25],
                        "Score": f"{r['overall_score']:.1f}",
                        "Rating": r["rating"],
                        "Technical": r.get("technical_score", "N/A"),
                        "Fundamental": r.get("fundamental_score", "N/A"),
                        "Trend": r.get("trend_score", "N/A"),
                        "Momentum": r.get("momentum_score", "N/A"),
                        "Direction": r.get("trend_direction", ""),
                    })
                rank_df = pd.DataFrame(rows)
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
                csv_download(rank_df, "stock_rankings.csv")

                # Compact clickable symbol pills
                _rsyms = [r["symbol"] for r in ranked]
                selected = st.pills("🔍 Quick analyze", _rsyms,
                                    default=None, key="rank_pill")
                if selected:
                    del st.session_state["rank_pill"]
                    go_to_analysis(selected)
                    st.rerun()
            else:
                st.warning("No stocks could be scored.")
        else:
            st.markdown("Choose your stock source and click **Rank Stocks** to see the leaderboard.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Compare
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Compare":
    st.markdown("## ⚖️ Stock X-Ray Comparison")
    st.caption("Side-by-side deep analysis — returns, risk, correlation, drawdowns")

    _all_opts = _build_symbol_options()
    _cmp_defaults = [_ticker_option(t) for t in ("AAPL", "MSFT", "GOOGL", "AMZN")
                     if _ticker_option(t) in _all_opts]
    cmp_picks = st.multiselect(
        "🔍 Search & select stocks to compare",
        options=_all_opts,
        default=_cmp_defaults,
        placeholder="Type ticker or company name…",
    )
    _cmp_custom = st.text_input("Add custom symbols (comma-separated)",
                                placeholder="CUSTOM1, CUSTOM2",
                                key="cmp_custom")
    cmp_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    cmp_go = st.button("⚖️ Compare", type="primary")

    if cmp_go:
        symbols = [_extract_ticker(p) for p in cmp_picks]
        if _cmp_custom:
            symbols += [s.strip().upper() for s in _cmp_custom.split(",") if s.strip()]

        if len(symbols) < 2:
            st.warning("Enter at least 2 symbols.")
        else:
            with st.spinner("Fetching data..."):
                data = {}
                for sym in symbols:
                    df = fetch_history(sym, cmp_period)
                    if not df.empty:
                        data[sym] = df

            if len(data) < 2:
                st.error("Could not fetch data for enough symbols.")
            else:
                # Tab layout for comparison views
                cmp_tab1, cmp_tab2, cmp_tab3, cmp_tab4 = st.tabs(
                    ["📈 Returns", "🎲 Risk & Volatility", "🔗 Correlation", "⭐ Head-to-Head"]
                )

                # TAB: Returns
                with cmp_tab1:
                    fig = create_comparison_chart(data)
                    st.plotly_chart(fig, use_container_width=True)

                    # Return stats table
                    ret_rows = []
                    for sym, df in data.items():
                        c = df["Close"]
                        total_ret = (c.iloc[-1] / c.iloc[0] - 1) * 100
                        ret_rows.append({
                            "Symbol": sym,
                            "Start": f"${c.iloc[0]:.2f}",
                            "End": f"${c.iloc[-1]:.2f}",
                            "Total Return": f"{total_ret:+.2f}%",
                            "Best Day": f"{c.pct_change().max()*100:+.2f}%",
                            "Worst Day": f"{c.pct_change().min()*100:+.2f}%",
                        })
                    ret_df = pd.DataFrame(ret_rows)
                    st.dataframe(ret_df, use_container_width=True, hide_index=True)
                    csv_download(ret_df, "comparison_returns.csv")

                # TAB: Risk & Volatility
                with cmp_tab2:
                    risk_rows = []
                    for sym, df in data.items():
                        daily_ret = df["Close"].pct_change().dropna()
                        vol = daily_ret.std() * np.sqrt(252) * 100
                        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                                  if daily_ret.std() > 0 else 0)
                        # Max drawdown
                        cum = (1 + daily_ret).cumprod()
                        peak = cum.cummax()
                        dd = ((cum - peak) / peak * 100).min()
                        risk_rows.append({
                            "Symbol": sym,
                            "Annualized Volatility": f"{vol:.1f}%",
                            "Sharpe Ratio": f"{sharpe:.2f}",
                            "Max Drawdown": f"{dd:.1f}%",
                            "Avg Daily Move": f"{daily_ret.abs().mean()*100:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(risk_rows), use_container_width=True,
                                 hide_index=True)

                    # Drawdown chart
                    st.markdown("#### Drawdown Over Time")
                    fig_dd = go.Figure()
                    for sym, df in data.items():
                        daily_ret = df["Close"].pct_change().dropna()
                        cum = (1 + daily_ret).cumprod()
                        peak = cum.cummax()
                        dd = (cum - peak) / peak * 100
                        fig_dd.add_trace(go.Scatter(
                            x=dd.index, y=dd, name=sym, mode="lines",
                            fill="tozeroy", opacity=0.4,
                        ))
                    fig_dd.update_layout(
                        height=350, template="plotly_white",
                        yaxis_title="Drawdown (%)",
                        margin=dict(l=60, r=30, t=30, b=40),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

                # TAB: Correlation
                with cmp_tab3:
                    st.markdown("#### Return Correlation Matrix")
                    st.caption("Values close to 1.0 = stocks move together · Close to -1.0 = they move opposite")
                    returns_df = pd.DataFrame({
                        sym: df["Close"].pct_change()
                        for sym, df in data.items()
                    }).dropna()
                    corr = returns_df.corr()

                    # Heatmap
                    fig_corr = go.Figure(go.Heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale=[[0, "#c62828"], [0.5, "#fff9c4"], [1, "#2e7d32"]],
                        zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in corr.values],
                        texttemplate="%{text}",
                        textfont=dict(size=14),
                    ))
                    fig_corr.update_layout(
                        height=350, margin=dict(l=80, r=30, t=30, b=80),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Diversification tip
                    avg_corr = corr.where(
                        np.triu(np.ones(corr.shape), k=1).astype(bool)
                    ).stack().mean()
                    if avg_corr > 0.8:
                        st.warning("⚠️ These stocks are highly correlated — "
                                   "they tend to move together. Low diversification benefit.")
                    elif avg_corr < 0.3:
                        st.success("✅ Good diversification — these stocks have "
                                   "low correlation, reducing portfolio risk.")
                    else:
                        st.info(f"Moderate average correlation ({avg_corr:.2f}). "
                                "Some diversification benefit.")

                # TAB: Head-to-Head Scores
                with cmp_tab4:
                    score_cols = st.columns(min(len(symbols), 4))
                    for i, sym in enumerate(symbols[:4]):
                        with score_cols[i]:
                            with st.spinner(f"Scoring {sym}..."):
                                result = compute_score(sym, cmp_period)
                            if result.get("error"):
                                st.error(f"{sym}: {result['error']}")
                            else:
                                st.markdown(f"### {sym}")
                                st.markdown(f"## {result['overall_score']:.0f}")
                                st.markdown(rating_badge(result["rating"]),
                                            unsafe_allow_html=True)
                                st.markdown(f"- Tech: {result.get('technical_score', 'N/A')}")
                                st.markdown(f"- Fund: {result.get('fundamental_score', 'N/A')}")
                                st.markdown(f"- Trend: {result.get('trend_score', 'N/A')}")
                                st.markdown(f"- Mom: {result.get('momentum_score', 'N/A')}")
                                if st.button(f"→ Full Analysis", key=f"cmp_{sym}",
                                             type="tertiary"):
                                    go_to_analysis(sym)
                                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: What-If Time Machine
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⏳ What-If Machine":
    st.markdown("## ⏳ What-If Time Machine")
    st.markdown("*\"What if I had invested $X in Stock Y, Z years ago?\"*")

    wi1, wi2, wi3 = st.columns(3)
    with wi1:
        wi_symbol = symbol_search("Stock symbol", key="wi_sym", default="AAPL")
    with wi2:
        wi_amount = st.number_input("Investment amount ($)", min_value=100,
                                     max_value=10_000_000, value=10_000, step=1000)
    with wi3:
        wi_years = st.selectbox("Years ago", [1, 2, 3, 5, 10, 15, 20], index=3)

    wi_go = st.button("🚀 Calculate", type="primary", use_container_width=True)

    if wi_go and wi_symbol:
        period_map = {1: "1y", 2: "2y", 3: "5y", 5: "5y", 10: "max",
                      15: "max", 20: "max"}
        with st.spinner(f"Traveling back {wi_years} years..."):
            wi_df = fetch_history(wi_symbol, period_map.get(wi_years, "max"))

        if wi_df.empty:
            st.error(f"No data for {wi_symbol}")
        else:
            # Find the target date
            target_date = datetime.now() - timedelta(days=wi_years * 365)
            # Find closest available date
            available = wi_df.index[wi_df.index >= target_date]
            if available.empty:
                available = wi_df.index
            start_idx = available[0]

            start_price = wi_df.loc[start_idx, "Close"]
            end_price = wi_df["Close"].iloc[-1]
            shares_bought = wi_amount / start_price
            current_value = shares_bought * end_price
            total_return = (current_value / wi_amount - 1) * 100
            profit = current_value - wi_amount
            actual_years = (wi_df.index[-1] - start_idx).days / 365.25
            cagr = ((current_value / wi_amount) ** (1 / max(actual_years, 0.1)) - 1) * 100

            # Results
            profit_color = "#4caf50" if profit >= 0 else "#f44336"
            profit_emoji = "📈" if profit >= 0 else "📉"

            st.markdown(
                f'<div style="background:linear-gradient(135deg,#f8f9fa,#e8f0fe);'
                f'border-radius:16px;padding:1.5rem;border:2px solid {profit_color};'
                f'text-align:center;margin:1rem 0">'
                f'<div style="font-size:1rem;color:#666">Your ${wi_amount:,.0f} investment '
                f'in <b>{wi_symbol}</b> {wi_years} years ago would be worth</div>'
                f'<div style="font-size:3rem;font-weight:900;color:{profit_color};'
                f'margin:0.3rem 0">${current_value:,.2f}</div>'
                f'<div style="font-size:1.3rem;color:{profit_color};font-weight:700">'
                f'{profit_emoji} {total_return:+,.1f}% total return '
                f'(${profit:+,.2f})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Shares Bought", f"{shares_bought:,.2f}")
            r2.metric("Buy Price", f"${start_price:.2f}")
            r3.metric("Current Price", f"${end_price:.2f}")
            r4.metric("CAGR", f"{cagr:.1f}%")

            st.divider()

            # Portfolio growth chart
            wi_subset = wi_df.loc[start_idx:].copy()
            wi_subset["Portfolio"] = (wi_subset["Close"] / start_price) * wi_amount

            fig_wi = go.Figure()
            fig_wi.add_trace(go.Scatter(
                x=wi_subset.index, y=wi_subset["Portfolio"],
                name="Portfolio Value", mode="lines",
                fill="tozeroy",
                line=dict(color=profit_color, width=2),
                fillcolor=f"rgba({'76,175,80' if profit >= 0 else '244,67,54'},0.15)",
            ))
            fig_wi.add_hline(y=wi_amount, line_dash="dash",
                             line_color="#888", opacity=0.5,
                             annotation_text=f"Initial ${wi_amount:,.0f}")
            fig_wi.update_layout(
                height=400, template="plotly_white",
                yaxis_title="Portfolio Value ($)",
                title="Growth of Your Investment Over Time",
                margin=dict(l=60, r=30, t=60, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig_wi, use_container_width=True)

            # Compare vs index
            st.divider()
            st.markdown("#### 🏁 vs. S&P 500")
            sp_df = fetch_history("^GSPC", period_map.get(wi_years, "max"))
            if not sp_df.empty:
                sp_avail = sp_df.index[sp_df.index >= start_idx]
                if not sp_avail.empty:
                    sp_start = sp_df.loc[sp_avail[0], "Close"]
                    sp_end = sp_df["Close"].iloc[-1]
                    sp_value = wi_amount * (sp_end / sp_start)
                    sp_return = (sp_end / sp_start - 1) * 100

                    vs1, vs2 = st.columns(2)
                    with vs1:
                        st.metric(f"{wi_symbol}",
                                  f"${current_value:,.0f}",
                                  f"{total_return:+.1f}%")
                    with vs2:
                        st.metric("S&P 500",
                                  f"${sp_value:,.0f}",
                                  f"{sp_return:+.1f}%")

                    if current_value > sp_value:
                        st.success(f"🏆 {wi_symbol} beat the S&P 500 by "
                                   f"**${current_value - sp_value:,.0f}** "
                                   f"({total_return - sp_return:+.1f} percentage points)")
                    else:
                        st.warning(f"The S&P 500 outperformed {wi_symbol} by "
                                   f"**${sp_value - current_value:,.0f}** "
                                   f"({sp_return - total_return:+.1f} percentage points)")

            # Quick analyze link
            if st.button(f"🔍 Full Analysis of {wi_symbol}", type="tertiary"):
                go_to_analysis(wi_symbol)
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Portfolio Tracker
# ═════════════════════════════════════════════════════════════════════════════

elif page == "💼 Portfolio Tracker":
    st.markdown("## 💼 Portfolio Tracker")
    st.markdown("Track your real investments — see live P&L, allocation, and performance")

    # Initialize portfolio in session state
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = [
            {"symbol": "AAPL", "shares": 10, "buy_price": 150.00},
            {"symbol": "MSFT", "shares": 5, "buy_price": 300.00},
            {"symbol": "GOOGL", "shares": 8, "buy_price": 140.00},
        ]

    # ── Add/Remove holdings ──
    with st.expander("✏️ Edit Holdings", expanded=False):
        pe1, pe2, pe3, pe4 = st.columns([2, 1, 1, 1])
        with pe1:
            pf_sym = symbol_search("Symbol", key="pf_add_sym", default="NVDA")
        with pe2:
            pf_shares = st.number_input("Shares", min_value=0.01, value=10.0,
                                         step=1.0, key="pf_add_shares")
        with pe3:
            pf_cost = st.number_input("Avg cost ($)", min_value=0.01, value=100.0,
                                       step=1.0, key="pf_add_cost")
        with pe4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", use_container_width=True, key="pf_add_btn"):
                st.session_state["portfolio"].append({
                    "symbol": pf_sym, "shares": pf_shares, "buy_price": pf_cost
                })
                st.success(f"Added {pf_shares} shares of {pf_sym}")
                st.rerun()

        # Show remove buttons for each holding
        if st.session_state["portfolio"]:
            st.markdown("**Current holdings:**")
            for idx, h in enumerate(st.session_state["portfolio"]):
                rc1, rc2 = st.columns([4, 1])
                with rc1:
                    st.caption(f"{h['symbol']} — {h['shares']} shares @ ${h['buy_price']:.2f}")
                with rc2:
                    if st.button("🗑️", key=f"pf_rm_{idx}"):
                        st.session_state["portfolio"].pop(idx)
                        st.rerun()

    portfolio = st.session_state["portfolio"]

    if not portfolio:
        st.info("Your portfolio is empty. Add some holdings above!")
    else:
        # ── Fetch live data ──
        with st.spinner("Fetching live portfolio data..."):
            pf_rows = []
            total_cost = 0
            total_value = 0
            sector_alloc = {}

            for h in portfolio:
                try:
                    info = fetch_info(h["symbol"])
                    live_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                    cost_basis = h["shares"] * h["buy_price"]
                    market_val = h["shares"] * live_price
                    pnl = market_val - cost_basis
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                    total_cost += cost_basis
                    total_value += market_val
                    sec = info.get("sector", "Other")
                    sector_alloc[sec] = sector_alloc.get(sec, 0) + market_val

                    pf_rows.append({
                        "Symbol": h["symbol"],
                        "Name": info.get("shortName", "")[:25],
                        "Shares": h["shares"],
                        "Avg Cost": f"${h['buy_price']:.2f}",
                        "Price": f"${live_price:.2f}",
                        "Cost Basis": f"${cost_basis:,.2f}",
                        "Market Value": f"${market_val:,.2f}",
                        "P&L": f"${pnl:+,.2f}",
                        "P&L %": f"{pnl_pct:+.1f}%",
                        "Sector": sec,
                        "_pnl": pnl,
                        "_value": market_val,
                    })
                except Exception:
                    pf_rows.append({
                        "Symbol": h["symbol"], "Name": "Error",
                        "Shares": h["shares"],
                        "Avg Cost": f"${h['buy_price']:.2f}",
                        "Price": "N/A", "Cost Basis": "N/A",
                        "Market Value": "N/A", "P&L": "N/A",
                        "P&L %": "N/A", "Sector": "",
                        "_pnl": 0, "_value": 0,
                    })

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        pnl_color = "#4caf50" if total_pnl >= 0 else "#f44336"
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        # ── Hero card ──
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#f8f9fa,#e8f0fe);'
            f'border-radius:16px;padding:1.2rem;border:2px solid {pnl_color};'
            f'text-align:center;margin-bottom:1rem">'
            f'<div style="font-size:0.9rem;color:#666">Total Portfolio Value</div>'
            f'<div style="font-size:2.5rem;font-weight:900">${total_value:,.2f}</div>'
            f'<div style="font-size:1.2rem;color:{pnl_color};font-weight:700">'
            f'{pnl_emoji} {total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Total Invested", f"${total_cost:,.2f}")
        pm2.metric("Current Value", f"${total_value:,.2f}")
        pm3.metric("Total P&L", f"${total_pnl:+,.2f}")
        pm4.metric("Holdings", len(portfolio))

        st.divider()

        # ── Charts row ──
        pc1, pc2 = st.columns([1, 1])

        with pc1:
            st.markdown("#### Allocation by Stock")
            alloc_labels = [r["Symbol"] for r in pf_rows if r["_value"] > 0]
            alloc_values = [r["_value"] for r in pf_rows if r["_value"] > 0]
            if alloc_labels:
                fig_alloc = go.Figure(go.Pie(
                    labels=alloc_labels, values=alloc_values,
                    hole=0.4, textinfo="label+percent",
                    textposition="outside",
                ))
                fig_alloc.update_layout(
                    height=320, margin=dict(l=20, r=20, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

        with pc2:
            st.markdown("#### Allocation by Sector")
            if sector_alloc:
                fig_sec = go.Figure(go.Pie(
                    labels=list(sector_alloc.keys()),
                    values=list(sector_alloc.values()),
                    hole=0.4, textinfo="label+percent",
                    textposition="outside",
                ))
                fig_sec.update_layout(
                    height=320, margin=dict(l=20, r=20, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_sec, use_container_width=True)

        # ── P&L bar chart ──
        st.markdown("#### P&L by Holding")
        pnl_syms = [r["Symbol"] for r in pf_rows]
        pnl_vals = [r["_pnl"] for r in pf_rows]
        pnl_colors = ["#4caf50" if v >= 0 else "#f44336" for v in pnl_vals]
        fig_pnl = go.Figure(go.Bar(
            x=pnl_syms, y=pnl_vals,
            marker_color=pnl_colors,
            text=[f"${v:+,.0f}" for v in pnl_vals],
            textposition="outside",
        ))
        fig_pnl.update_layout(
            height=300, template="plotly_white",
            yaxis_title="P&L ($)",
            margin=dict(l=60, r=30, t=20, b=40),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.divider()

        # ── Holdings table ──
        st.markdown("#### Holdings Detail")
        display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in pf_rows]
        pf_df = pd.DataFrame(display_rows)
        st.dataframe(pf_df, use_container_width=True, hide_index=True)
        csv_download(pf_df, "my_portfolio.csv")

        # Quick-analyze pills
        pf_syms = [h["symbol"] for h in portfolio]
        selected = st.pills("🔍 Analyze holding", pf_syms,
                            default=None, key="pf_pill")
        if selected:
            del st.session_state["pf_pill"]
            go_to_analysis(selected)
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Watchlist
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📋 Watchlist":
    st.markdown("## 📋 Watchlist Manager")

    watchlist = utils.load_watchlist("default")

    wl1, wl2 = st.columns([2, 1])

    with wl2:
        st.markdown("### Actions")
        _wl_opts = _build_symbol_options()
        _wl_add_picks = st.multiselect(
            "🔍 Add stocks", options=_wl_opts,
            placeholder="Type ticker or company name…",
            key="wl_add_picks",
        )
        _wl_add_custom = st.text_input("Add custom symbols", placeholder="CUSTOM1, CUSTOM2",
                                        key="wl_add_custom")
        if st.button("➕ Add", use_container_width=True):
            new = [_extract_ticker(p) for p in _wl_add_picks]
            if _wl_add_custom:
                new += [s.strip().upper() for s in _wl_add_custom.split(",") if s.strip()]
            if new:
                utils.add_to_watchlist(new)
                st.success(f"Added: {', '.join(new)}")
                st.rerun()

        remove_syms = st.text_input("Remove symbols", placeholder="INTC, IBM")
        if st.button("➖ Remove", use_container_width=True) and remove_syms:
            rem = [s.strip().upper() for s in remove_syms.split(",") if s.strip()]
            utils.remove_from_watchlist(rem)
            st.success(f"Removed: {', '.join(rem)}")
            st.rerun()

        st.divider()
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            utils.save_watchlist(config.DEFAULT_UNIVERSE[:20], "default")
            st.success("Reset to default watchlist.")
            st.rerun()

    with wl1:
        st.markdown(f"### Current Watchlist ({len(watchlist)} symbols)")
        if watchlist:
            wl_data = []
            for sym in watchlist:
                try:
                    info = fetch_info(sym)
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    prev = info.get("previousClose")
                    change = ((price - prev) / prev * 100) if price and prev else None
                    wl_data.append({
                        "Symbol": sym,
                        "Name": info.get("shortName", "")[:30],
                        "Price": f"${price:.2f}" if price else "N/A",
                        "Change": f"{change:+.2f}%" if change else "N/A",
                        "Sector": info.get("sector", ""),
                        "Market Cap": fmt_number(info.get("marketCap"), "$"),
                    })
                except Exception:
                    wl_data.append({"Symbol": sym, "Name": "Error", "Price": "N/A",
                                    "Change": "N/A", "Sector": "", "Market Cap": "N/A"})
            wl_df = pd.DataFrame(wl_data)
            st.dataframe(wl_df, use_container_width=True, hide_index=True)
            csv_download(wl_df, "watchlist.csv")

            # Compact clickable symbol pills
            selected = st.pills("🔍 Quick analyze", watchlist,
                                default=None, key="wl_pill")
            if selected:
                del st.session_state["wl_pill"]
                go_to_analysis(selected)
                st.rerun()
        else:
            st.info("Watchlist is empty. Add some symbols!")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Buffett Portfolio
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🧓 Buffett Portfolio":
    st.markdown("## 🧓 Warren Buffett's Portfolio")
    st.markdown(
        "Berkshire Hathaway's equity holdings from the latest SEC 13-F filing. "
        "These are the stocks that Warren Buffett and his team have chosen to hold."
    )

    # ── Berkshire Hathaway 13F holdings ──
    # Source: SEC 13-F filing by Berkshire Hathaway Inc (CIK 0001067983)
    # Data reflects the most recent publicly available quarterly filing.
    BUFFETT_HOLDINGS = [
        {"symbol": "AAPL",  "name": "Apple Inc",                "shares": 300_000_000, "pct_portfolio": 28.1, "sector": "Technology"},
        {"symbol": "AXP",   "name": "American Express",         "shares": 151_610_700, "pct_portfolio": 16.2, "sector": "Financials"},
        {"symbol": "BAC",   "name": "Bank of America",          "shares": 680_233_587, "pct_portfolio": 11.2, "sector": "Financials"},
        {"symbol": "KO",    "name": "Coca-Cola",                "shares": 400_000_000, "pct_portfolio": 10.4, "sector": "Consumer Staples"},
        {"symbol": "CVX",   "name": "Chevron",                  "shares": 118_610_534, "pct_portfolio":  6.5, "sector": "Energy"},
        {"symbol": "OXY",   "name": "Occidental Petroleum",     "shares": 264_224_664, "pct_portfolio":  4.9, "sector": "Energy"},
        {"symbol": "MCO",   "name": "Moody's Corp",             "shares":  24_669_778, "pct_portfolio":  4.5, "sector": "Financials"},
        {"symbol": "KHC",   "name": "Kraft Heinz",              "shares": 325_634_818, "pct_portfolio":  3.7, "sector": "Consumer Staples"},
        {"symbol": "CB",    "name": "Chubb Ltd",                "shares":  25_915_831, "pct_portfolio":  2.7, "sector": "Financials"},
        {"symbol": "DVA",   "name": "DaVita Inc",               "shares":  36_095_570, "pct_portfolio":  2.1, "sector": "Healthcare"},
        {"symbol": "KR",    "name": "Kroger",                   "shares":  50_000_000, "pct_portfolio":  1.1, "sector": "Consumer Staples"},
        {"symbol": "VRSN",  "name": "VeriSign",                 "shares":  12_815_613, "pct_portfolio":  0.9, "sector": "Technology"},
        {"symbol": "V",     "name": "Visa Inc",                 "shares":   8_297_460, "pct_portfolio":  0.9, "sector": "Financials"},
        {"symbol": "MA",    "name": "Mastercard",               "shares":   3_986_648, "pct_portfolio":  0.7, "sector": "Financials"},
        {"symbol": "AMZN",  "name": "Amazon.com",               "shares":  10_000_000, "pct_portfolio":  0.7, "sector": "Consumer Discretionary"},
        {"symbol": "NU",    "name": "Nu Holdings",              "shares":  86_389_383, "pct_portfolio":  0.4, "sector": "Financials"},
        {"symbol": "COF",   "name": "Capital One Financial",    "shares":   2_639_022, "pct_portfolio":  0.2, "sector": "Financials"},
        {"symbol": "SIRI",  "name": "Sirius XM Holdings",       "shares":  34_291_627, "pct_portfolio":  0.1, "sector": "Communication"},
        {"symbol": "ALLY",  "name": "Ally Financial",            "shares":   9_000_000, "pct_portfolio":  0.1, "sector": "Financials"},
        {"symbol": "LSXMK", "name": "Liberty SiriusXM Grp C",   "shares":  43_208_291, "pct_portfolio":  0.1, "sector": "Communication"},
    ]

    # ── Buffett's investment principles ──
    with st.expander("🎓 Buffett's Investment Principles", expanded=False):
        st.markdown("""
        Warren Buffett follows a **value investing** philosophy. Key principles:

        1. **Buy wonderful companies at fair prices** — not fair companies at wonderful prices
        2. **Economic moat** — invest in companies with durable competitive advantages
        3. **Long-term holding** — "Our favorite holding period is forever"
        4. **Margin of safety** — buy when the price is well below intrinsic value
        5. **Understand the business** — invest only in what you understand
        6. **Strong management** — look for honest, competent leaders
        7. **Consistent earnings** — prefer predictable, growing earnings power
        8. **Low debt** — companies that don't need leverage to generate returns
        """)

    st.divider()

    # ── Sector allocation chart ──
    sectors = {}
    for h in BUFFETT_HOLDINGS:
        sectors[h["sector"]] = sectors.get(h["sector"], 0) + h["pct_portfolio"]

    ch1, ch2 = st.columns([1, 2])
    with ch1:
        st.markdown("#### Sector Allocation")
        fig_pie = go.Figure(go.Pie(
            labels=list(sectors.keys()),
            values=list(sectors.values()),
            hole=0.45,
            marker=dict(colors=[
                "#1e88e5", "#43a047", "#fb8c00", "#e53935",
                "#8e24aa", "#00acc1", "#6d4c41",
            ]),
            textinfo="label+percent",
            textposition="outside",
        ))
        fig_pie.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.markdown("#### Top 10 Holdings by Portfolio Weight")
        top10 = BUFFETT_HOLDINGS[:10]
        fig_bar = go.Figure(go.Bar(
            y=[h["symbol"] for h in top10][::-1],
            x=[h["pct_portfolio"] for h in top10][::-1],
            orientation="h",
            marker_color=["#1e88e5"] * 10,
            text=[f"{h['pct_portfolio']:.1f}%" for h in top10][::-1],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=350, template="plotly_white",
            xaxis=dict(title="% of Portfolio", range=[0, max(h['pct_portfolio'] for h in top10) * 1.25]),
            margin=dict(l=60, r=60, t=30, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Live holdings table ──
    st.markdown("#### Full Holdings — Live Prices")

    progress = st.progress(0, text="Fetching live data...")
    table_rows = []
    for i, h in enumerate(BUFFETT_HOLDINGS):
        progress.progress((i + 1) / len(BUFFETT_HOLDINGS),
                          text=f"Loading {h['symbol']}...")
        try:
            info = fetch_info(h["symbol"])
            live_price = info.get("currentPrice") or info.get("regularMarketPrice")
            prev_close = info.get("previousClose")
            day_change = ((live_price - prev_close) / prev_close * 100
                          if live_price and prev_close else None)
            mkt_cap = info.get("marketCap")
            pe = info.get("trailingPE")
            div_yield = info.get("dividendYield")
            value_held = live_price * h["shares"] if live_price else None
        except Exception:
            live_price = day_change = mkt_cap = pe = div_yield = value_held = None

        row = {
            "#": i + 1,
            "Symbol": h["symbol"],
            "Company": h["name"],
            "% Portfolio": f"{h['pct_portfolio']:.1f}%",
            "Shares": f"{h['shares']:,}",
            "Price": f"${live_price:.2f}" if live_price else "N/A",
            "Day Chg": f"{day_change:+.2f}%" if day_change is not None else "N/A",
            "Est. Value": fmt_number(value_held, "$") if value_held else "N/A",
            "P/E": f"{pe:.1f}" if pe else "N/A",
            "Div Yield": f"{div_yield * 100:.2f}%" if div_yield else "N/A",
            "Sector": h["sector"],
        }
        table_rows.append(row)
    progress.empty()

    buff_df = pd.DataFrame(table_rows)

    # Clickable table — select a row to analyze that stock
    st.caption("👆 Click a row in the table below to analyze that stock")
    event = st.dataframe(
        buff_df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key="buff_table_sel",
    )
    _sel_rows = event.selection.rows if event and event.selection else []
    if _sel_rows:
        _picked_sym = buff_df.iloc[_sel_rows[0]]["Symbol"]
        go_to_analysis(_picked_sym)
        st.rerun()

    csv_download(buff_df, "buffett_portfolio.csv")

    # Quick-analyze pills (alternative)
    buff_syms = [h["symbol"] for h in BUFFETT_HOLDINGS]
    st.markdown("**Or pick a stock directly:**")
    selected = st.pills("🔍 Analyze a Buffett stock", buff_syms,
                        default=None, key="buff_pill")
    if selected:
        del st.session_state["buff_pill"]
        go_to_analysis(selected)
        st.rerun()

    # ── Why these stocks? ──
    st.divider()
    with st.expander("💡 Why does Buffett hold these stocks?", expanded=False):
        st.markdown("""
        | Stock | Buffett's Rationale |
        |-------|--------------------|
        | **AAPL** | Massive ecosystem, unmatched brand loyalty, huge buybacks & cash flow |
        | **AXP** | Dominant payment network with affluent customer base & pricing power |
        | **BAC** | Largest U.S. bank by deposits, benefits from higher interest rates |
        | **KO** | Ultimate consumer brand — sold in 200+ countries, 60+ years of dividend growth |
        | **CVX** | Top-tier energy major with disciplined capital allocation |
        | **OXY** | Low-cost U.S. oil producer; Buffett likes the CEO and balance sheet |
        | **MCO** | Duopoly in credit ratings — essential service with recurring revenue |
        | **KHC** | Iconic food brands; Berkshire helped create the company |
        | **CB** | World's largest publicly traded P&C insurer — strong underwriting |
        | **DVA** | Dominant in kidney dialysis — aging population tailwind |
        """)

    st.caption(
        "⚠️ Data based on Berkshire Hathaway's SEC 13-F filings. "
        "Holdings are reported with a ~45-day delay and may not reflect current positions. "
        "This is not investment advice."
    )
