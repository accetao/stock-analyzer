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
from datetime import datetime

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

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
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
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; }
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol, period, interval="1d"):
    return data_fetcher.get_history(symbol, period=period, interval=interval)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_info(symbol):
    return data_fetcher.get_info(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def compute_score(symbol, period="1y"):
    return stock_scorer.score_stock(symbol, period=period)


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

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Stock Analysis", "📊 Screener", "🏆 Rankings",
         "⚖️ Compare", "📋 Watchlist"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Data cached for {config.CACHE_EXPIRY_MINUTES} min")
    if st.button("🔄 Clear Cache", use_container_width=True):
        data_fetcher.clear_cache()
        st.cache_data.clear()
        st.success("Cache cleared!")


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
        quick_sym = st.text_input("Quick lookup", placeholder="Enter symbol e.g. AAPL",
                                  label_visibility="collapsed")
    with col_b:
        quick_go = st.button("Analyze →", use_container_width=True)

    if quick_go and quick_sym:
        st.session_state["analyze_symbol"] = quick_sym.upper().strip()
        st.session_state["page_override"] = "🔍 Stock Analysis"
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
                    st.markdown(f"""
                    <div style="background:#f8f9fa;border-radius:10px;padding:0.8rem 1rem;
                                border-left:4px solid {change_color};margin-bottom:4px">
                        <div style="font-weight:700;font-size:1.05rem">{d['symbol']}</div>
                        <div style="color:#666;font-size:0.78rem;margin-bottom:4px">{d['name'][:28]}</div>
                        <div style="font-size:1.15rem;font-weight:600">${d['price']:.2f}</div>
                        <div style="color:{change_color};font-weight:600">{arrow} {change_str}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info("No watchlist found. Go to 📋 Watchlist to set one up.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Stock Analysis
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Stock Analysis":
    st.markdown("## 🔍 Stock Analysis")

    # Input bar
    col1, col2, col3 = st.columns([2, 1, 1])
    default_sym = st.session_state.pop("analyze_symbol", "AAPL")
    with col1:
        symbol = st.text_input("Symbol", value=default_sym).upper().strip()
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if symbol and (analyze_btn or default_sym != "AAPL"):
        with st.spinner(f"Analyzing {symbol}..."):
            df = fetch_history(symbol, period)
            info = fetch_info(symbol)

        if df.empty or not info:
            st.error(f"❌ No data found for **{symbol}**. Please check the symbol.")
        else:
            ta.add_all_indicators(df)
            last = df.iloc[-1]

            # ── Header metrics ──
            price = info.get("currentPrice") or info.get("regularMarketPrice") or last["Close"]
            prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
            change = ((price - prev) / prev * 100) if price and prev else 0

            st.markdown(f"### {info.get('shortName', symbol)} ({symbol})")
            st.caption(f"{info.get('sector', '')} · {info.get('industry', '')}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Price", f"${price:.2f}", f"{change:+.2f}%")
            m2.metric("Market Cap", fmt_number(info.get("marketCap"), "$"))
            m3.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
            m4.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
            m5.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

            # ── Tabs ──
            tab_chart, tab_score, tab_tech, tab_fund, tab_trend = st.tabs(
                ["📊 Chart", "⭐ Score", "🔧 Technical", "📑 Fundamental", "📈 Trend"]
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

            # ── TAB: Score ──
            with tab_score:
                with st.spinner("Computing investment score..."):
                    result = compute_score(symbol, period)

                if result.get("error"):
                    st.error(result["error"])
                else:
                    sc1, sc2 = st.columns([1, 2])
                    with sc1:
                        overall = result["overall_score"]
                        st.markdown(f"### Overall Score")
                        st.markdown(f"# {overall:.0f}/100")
                        st.markdown(rating_badge(result["rating"]), unsafe_allow_html=True)

                    with sc2:
                        sub_data = {
                            "Dimension": ["Technical", "Fundamental", "Trend", "Momentum"],
                            "Score": [
                                result.get("technical_score") or 0,
                                result.get("fundamental_score") or 0,
                                result.get("trend_score") or 0,
                                result.get("momentum_score") or 0,
                            ],
                            "Weight": ["35%", "30%", "20%", "15%"],
                        }
                        st.dataframe(pd.DataFrame(sub_data), use_container_width=True,
                                     hide_index=True)

                    # Breakdown details
                    st.divider()
                    bd = result.get("breakdown", {})
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("**Technical Breakdown**")
                        for k, v in bd.get("technical", {}).items():
                            st.markdown(f"- `{k}`: {v}")
                    with bc2:
                        st.markdown("**Momentum Details**")
                        for k, v in bd.get("momentum", {}).items():
                            st.markdown(f"- `{k}`: {v}")

            # ── TAB: Technical ──
            with tab_tech:
                signals = ta.get_latest_signals(df)
                st.markdown("#### Current Signals")

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
                    st.markdown(f"- Above SMA 200: {'✅' if signals.get('above_sma200') else '❌'}")
                    st.markdown(f"- % from SMA 200: {signals.get('pct_from_sma200', 'N/A')}%")
                with tc2:
                    st.markdown("**Strength**")
                    st.markdown(f"- ADX: {signals.get('adx', 'N/A')}")
                    st.markdown(f"- Trend Strength: {signal_badge(signals.get('trend_strength', 'N/A'))}",
                                unsafe_allow_html=True)
                    st.markdown(f"- BB Width: {signals.get('bb_width', 'N/A')}")

                st.divider()
                st.markdown("#### Indicator Values (Latest)")
                ind_cols = [c for c in df.columns
                            if c not in ("Open", "High", "Low", "Close", "Volume")]
                ind_data = {col: [round(last[col], 4) if pd.notna(last[col]) else None]
                            for col in ind_cols}
                st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

            # ── TAB: Fundamental ──
            with tab_fund:
                metrics = fa.get_fundamental_metrics(symbol)
                fund_score = fa.get_fundamental_score(symbol)

                if not metrics:
                    st.warning("No fundamental data available.")
                else:
                    # Score header
                    fs = fund_score.get("overall_score", "N/A")
                    st.markdown(f"**Fundamental Score: {fs}/100**")

                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Valuation", f"{fund_score.get('valuation_score', 'N/A')}/100")
                    fc2.metric("Profitability", f"{fund_score.get('profitability_score', 'N/A')}/100")
                    fc3.metric("Growth", f"{fund_score.get('growth_score', 'N/A')}/100")
                    fc4.metric("Health", f"{fund_score.get('health_score', 'N/A')}/100")

                    st.divider()
                    fl1, fl2, fl3 = st.columns(3)
                    with fl1:
                        st.markdown("**Valuation**")
                        st.markdown(f"- P/E (Trailing): {metrics.get('trailing_pe', 'N/A')}")
                        st.markdown(f"- P/E (Forward): {metrics.get('forward_pe', 'N/A')}")
                        st.markdown(f"- PEG Ratio: {metrics.get('peg_ratio', 'N/A')}")
                        st.markdown(f"- P/B: {metrics.get('price_to_book', 'N/A')}")
                        st.markdown(f"- EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}")
                    with fl2:
                        st.markdown("**Profitability**")
                        st.markdown(f"- Profit Margin: {fmt_pct(metrics.get('profit_margin'))}")
                        st.markdown(f"- Operating Margin: {fmt_pct(metrics.get('operating_margin'))}")
                        st.markdown(f"- ROE: {fmt_pct(metrics.get('roe'))}")
                        st.markdown(f"- ROA: {fmt_pct(metrics.get('roa'))}")
                    with fl3:
                        st.markdown("**Financial Health**")
                        st.markdown(f"- Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}")
                        st.markdown(f"- Current Ratio: {metrics.get('current_ratio', 'N/A')}")
                        st.markdown(f"- Free Cash Flow: {fmt_number(metrics.get('free_cashflow'), '$')}")

                    st.divider()
                    ga1, ga2 = st.columns(2)
                    with ga1:
                        st.markdown("**Growth**")
                        st.markdown(f"- Revenue Growth: {fmt_pct(metrics.get('revenue_growth'))}")
                        st.markdown(f"- Earnings Growth: {fmt_pct(metrics.get('earnings_growth'))}")
                    with ga2:
                        st.markdown("**Analyst Consensus**")
                        rec = (metrics.get("recommendation") or "N/A").upper()
                        st.markdown(f"- Recommendation: **{rec}**")
                        st.markdown(f"- Target Price: ${metrics.get('target_mean_price', 'N/A')}")
                        st.markdown(f"- Target Range: ${metrics.get('target_low_price', 'N/A')} "
                                    f"– ${metrics.get('target_high_price', 'N/A')}")

            # ── TAB: Trend ──
            with tab_trend:
                trend_data = trend_analyzer.classify_trend(df)
                sr = trend_analyzer.find_support_resistance(df)

                tr1, tr2 = st.columns([1, 2])
                with tr1:
                    trend_str = trend_data.get("trend", "N/A")
                    conf = trend_data.get("confidence", 0)
                    st.markdown(f"### {trend_str}")
                    st.progress(min(conf / 100, 1.0), text=f"Confidence: {conf}%")
                    st.metric("Bullish %", f"{trend_data.get('bullish_pct', 0)}%")
                    st.metric("Bearish %", f"{trend_data.get('bearish_pct', 0)}%")

                with tr2:
                    st.markdown("**Trend Factors**")
                    factors = trend_data.get("factors", {})
                    factor_df = pd.DataFrame([
                        {"Factor": k.replace("_", " ").title(), "Value": str(v)}
                        for k, v in factors.items()
                    ])
                    if not factor_df.empty:
                        st.dataframe(factor_df, use_container_width=True, hide_index=True)

                st.divider()
                sr1, sr2, sr3 = st.columns(3)
                sr1.metric("Current Price", f"${sr.get('current_price', 'N/A')}")
                with sr2:
                    st.markdown("**Resistance Levels**")
                    for r in sr.get("resistance", []):
                        st.markdown(f"- 🔴 ${r}")
                with sr3:
                    st.markdown("**Support Levels**")
                    for s in sr.get("support", []):
                        st.markdown(f"- 🟢 ${s}")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Screener
# ═════════════════════════════════════════════════════════════════════════════

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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No stocks could be scored.")
        else:
            st.markdown("Choose your stock source and click **Rank Stocks** to see the leaderboard.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Compare
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Compare":
    st.markdown("## ⚖️ Stock Comparison")

    cmp_syms = st.text_input("Symbols (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
    cmp_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    cmp_go = st.button("⚖️ Compare", type="primary")

    if cmp_go and cmp_syms:
        symbols = [s.strip().upper() for s in cmp_syms.split(",") if s.strip()]

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
                # Returns chart
                fig = create_comparison_chart(data)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.markdown("### Head-to-Head Scores")

                score_cols = st.columns(len(symbols))
                for i, sym in enumerate(symbols):
                    with score_cols[i]:
                        with st.spinner(f"Scoring {sym}..."):
                            result = compute_score(sym, cmp_period)
                        if result.get("error"):
                            st.error(f"{sym}: {result['error']}")
                        else:
                            st.markdown(f"### {sym}")
                            st.markdown(f"## {result['overall_score']:.0f}")
                            st.markdown(rating_badge(result["rating"]), unsafe_allow_html=True)
                            st.markdown(f"- Tech: {result.get('technical_score', 'N/A')}")
                            st.markdown(f"- Fund: {result.get('fundamental_score', 'N/A')}")
                            st.markdown(f"- Trend: {result.get('trend_score', 'N/A')}")
                            st.markdown(f"- Mom: {result.get('momentum_score', 'N/A')}")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Watchlist
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📋 Watchlist":
    st.markdown("## 📋 Watchlist Manager")

    watchlist = utils.load_watchlist("default")

    wl1, wl2 = st.columns([2, 1])

    with wl2:
        st.markdown("### Actions")
        add_syms = st.text_input("Add symbols", placeholder="TSLA, AMD, NFLX")
        if st.button("➕ Add", use_container_width=True) and add_syms:
            new = [s.strip().upper() for s in add_syms.split(",") if s.strip()]
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
            st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)
        else:
            st.info("Watchlist is empty. Add some symbols!")

# ─── Handle page override from dashboard quick lookup ────────────────────────
if st.session_state.get("page_override"):
    del st.session_state["page_override"]
