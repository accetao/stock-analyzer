"""
Stock Analyzer - Stock Scorer & Recommender
Combines technical, fundamental, and trend analysis into a unified
investment score and generates actionable recommendations.
"""

import logging
from typing import Optional

import pandas as pd

import config
from src import (
    data_fetcher,
    technical_analysis as ta,
    fundamental_analysis as fa,
    trend_analyzer,
)

logger = logging.getLogger(__name__)


# ─── Score categories ────────────────────────────────────────────────────────

class Rating:
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


def _score_to_rating(score: float) -> str:
    if score >= 75:
        return Rating.STRONG_BUY
    elif score >= 60:
        return Rating.BUY
    elif score >= 40:
        return Rating.HOLD
    elif score >= 25:
        return Rating.SELL
    else:
        return Rating.STRONG_SELL


# ─── Technical Score ─────────────────────────────────────────────────────────

def compute_technical_score(df: pd.DataFrame) -> dict:
    """
    Score 0-100 based on technical indicators.
    """
    if df.empty or len(df) < config.SMA_LONG:
        return {"score": None, "details": {}}

    if "RSI" not in df.columns:
        ta.add_all_indicators(df)

    signals = ta.get_latest_signals(df)
    score = 50.0
    details = {}

    # RSI scoring
    rsi_sig = signals.get("rsi_signal", "NEUTRAL")
    if rsi_sig == "OVERSOLD":
        score += 10  # potential buying opportunity
        details["rsi"] = "+10 (oversold, reversal potential)"
    elif rsi_sig == "OVERBOUGHT":
        score -= 10
        details["rsi"] = "-10 (overbought risk)"
    else:
        rsi_val = signals.get("rsi", 50)
        if 50 < rsi_val < 65:
            score += 5
            details["rsi"] = "+5 (healthy momentum)"

    # MACD scoring
    macd_sig = signals.get("macd_signal", "")
    if macd_sig == "BULLISH_CROSS":
        score += 12
        details["macd"] = "+12 (bullish crossover)"
    elif macd_sig == "BULLISH":
        score += 5
        details["macd"] = "+5 (bullish)"
    elif macd_sig == "BEARISH_CROSS":
        score -= 12
        details["macd"] = "-12 (bearish crossover)"
    elif macd_sig == "BEARISH":
        score -= 5
        details["macd"] = "-5 (bearish)"

    # Moving average trend
    ma_trend = signals.get("ma_trend", "")
    if ma_trend == "GOLDEN_CROSS":
        score += 10
        details["ma_trend"] = "+10 (golden cross)"
    elif ma_trend == "DEATH_CROSS":
        score -= 10
        details["ma_trend"] = "-10 (death cross)"

    # Price vs SMA 200
    if signals.get("above_sma200"):
        score += 5
        details["sma200"] = "+5 (above 200-day)"
    else:
        score -= 5
        details["sma200"] = "-5 (below 200-day)"

    # Bollinger Bands
    bb_sig = signals.get("bb_signal", "")
    if bb_sig == "BELOW_LOWER":
        score += 5  # potentially oversold
        details["bb"] = "+5 (below lower band)"
    elif bb_sig == "ABOVE_UPPER":
        score -= 5
        details["bb"] = "-5 (above upper band)"

    # Stochastic
    stoch_sig = signals.get("stoch_signal", "")
    if stoch_sig == "OVERSOLD":
        score += 5
        details["stochastic"] = "+5 (oversold)"
    elif stoch_sig == "OVERBOUGHT":
        score -= 5
        details["stochastic"] = "-5 (overbought)"

    # ADX trend strength
    if signals.get("trend_strength") == "STRONG":
        score += 3
        details["adx"] = "+3 (strong trend)"

    return {"score": max(0, min(100, round(score, 1))), "details": details}


# ─── Trend Score ─────────────────────────────────────────────────────────────

def compute_trend_score(df: pd.DataFrame) -> dict:
    """Score 0-100 based on trend analysis."""
    td = trend_analyzer.classify_trend(df)
    trend = td.get("trend", trend_analyzer.Trend.SIDEWAYS)

    trend_scores = {
        trend_analyzer.Trend.STRONG_UP: 90,
        trend_analyzer.Trend.UP: 75,
        trend_analyzer.Trend.WEAK_UP: 60,
        trend_analyzer.Trend.SIDEWAYS: 50,
        trend_analyzer.Trend.WEAK_DOWN: 40,
        trend_analyzer.Trend.DOWN: 25,
        trend_analyzer.Trend.STRONG_DOWN: 10,
    }

    score = trend_scores.get(trend, 50)
    return {"score": score, "trend": trend, "confidence": td.get("confidence", 0)}


# ─── Momentum Score ──────────────────────────────────────────────────────────

def compute_momentum_score(df: pd.DataFrame) -> dict:
    """Score 0-100 based on price momentum across timeframes."""
    if df.empty or len(df) < 60:
        return {"score": None, "details": {}}

    close = df["Close"]
    current = close.iloc[-1]
    details = {}
    score = 50.0

    # 1-week return
    if len(df) >= 5:
        ret_1w = (current / close.iloc[-5] - 1) * 100
        details["1w_return"] = f"{ret_1w:+.1f}%"
        if ret_1w > 3:
            score += 5
        elif ret_1w < -3:
            score -= 5

    # 1-month return
    if len(df) >= 21:
        ret_1m = (current / close.iloc[-21] - 1) * 100
        details["1m_return"] = f"{ret_1m:+.1f}%"
        if ret_1m > 5:
            score += 8
        elif ret_1m < -5:
            score -= 8

    # 3-month return
    if len(df) >= 63:
        ret_3m = (current / close.iloc[-63] - 1) * 100
        details["3m_return"] = f"{ret_3m:+.1f}%"
        if ret_3m > 10:
            score += 10
        elif ret_3m < -10:
            score -= 10

    # Relative volume (today vs average)
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
    if avg_vol > 0:
        rel_vol = df["Volume"].iloc[-1] / avg_vol
        details["relative_volume"] = f"{rel_vol:.2f}x"
        if rel_vol > 1.5:
            score += 5  # high interest
        elif rel_vol < 0.5:
            score -= 3

    # Price vs 52-week range
    if len(df) >= 252:
        high_52 = df["High"].tail(252).max()
        low_52 = df["Low"].tail(252).min()
        range_pos = (current - low_52) / (high_52 - low_52) * 100 if high_52 != low_52 else 50
        details["52w_range_position"] = f"{range_pos:.0f}%"
        if range_pos > 80:
            score += 5  # near highs – momentum
        elif range_pos < 20:
            score -= 5

    return {"score": max(0, min(100, round(score, 1))), "details": details}


# ─── Combined Score ──────────────────────────────────────────────────────────

def score_stock(symbol: str, period: str = "1y") -> dict:
    """
    Compute a comprehensive investment score for a stock.

    Returns:
        symbol, name, overall_score, rating,
        technical_score, fundamental_score, trend_score, momentum_score,
        breakdown (detailed sub-scores and reasoning)
    """
    # Fetch data
    df = data_fetcher.get_history(symbol, period=period)
    if df.empty:
        return {"symbol": symbol, "overall_score": None, "error": "No price data"}

    # Compute sub-scores
    tech = compute_technical_score(df)
    fund = fa.get_fundamental_score(symbol)
    trend = compute_trend_score(df)
    momentum = compute_momentum_score(df)

    tech_score = tech.get("score")
    fund_score = fund.get("overall_score")
    trend_score = trend.get("score")
    mom_score = momentum.get("score")

    # Weighted average (skip None components)
    weights = {}
    scores = {}
    if tech_score is not None:
        weights["technical"] = config.SCORE_WEIGHT_TECHNICAL
        scores["technical"] = tech_score
    if fund_score is not None:
        weights["fundamental"] = config.SCORE_WEIGHT_FUNDAMENTAL
        scores["fundamental"] = fund_score
    if trend_score is not None:
        weights["trend"] = config.SCORE_WEIGHT_TREND
        scores["trend"] = trend_score
    if mom_score is not None:
        weights["momentum"] = config.SCORE_WEIGHT_MOMENTUM
        scores["momentum"] = mom_score

    if not weights:
        return {"symbol": symbol, "overall_score": None, "error": "Insufficient data"}

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    overall = sum(scores[k] * (weights[k] / total_weight) for k in scores)
    overall = round(overall, 1)

    info = data_fetcher.get_info(symbol)

    return {
        "symbol": symbol,
        "name": info.get("shortName", ""),
        "sector": info.get("sector", ""),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "overall_score": overall,
        "rating": _score_to_rating(overall),
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "trend_score": trend_score,
        "momentum_score": mom_score,
        "trend_direction": trend.get("trend", ""),
        "breakdown": {
            "technical": tech.get("details", {}),
            "fundamental": {
                "valuation": fund.get("valuation_score"),
                "profitability": fund.get("profitability_score"),
                "growth": fund.get("growth_score"),
                "health": fund.get("health_score"),
            },
            "momentum": momentum.get("details", {}),
        },
    }


def rank_stocks(
    symbols: list[str],
    period: str = "1y",
    top_n: int = 10,
    progress_callback=None,
) -> list[dict]:
    """
    Score and rank multiple stocks. Returns top_n sorted by overall_score desc.
    """
    scored = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(sym, i + 1, total)
        try:
            result = score_stock(sym, period)
            if result.get("overall_score") is not None:
                scored.append(result)
        except Exception as e:
            logger.debug("Scoring failed for %s: %s", sym, e)

    scored.sort(key=lambda x: x["overall_score"], reverse=True)
    return scored[:top_n]


def format_score_report(result: dict) -> str:
    """Format a single stock score as a readable report."""
    if result.get("error"):
        return f"❌ {result['symbol']}: {result['error']}"

    lines = [
        f"═══ Investment Score: {result.get('name', '')} ({result['symbol']}) ═══",
        f"  Sector: {result.get('sector', 'N/A')}",
        f"  Price:  ${result.get('price', 'N/A')}",
        "",
        f"  ★ Overall Score: {result['overall_score']}/100  →  {result['rating']}",
        "",
        f"  ── Sub-Scores ──",
        f"    Technical:     {_bar(result.get('technical_score'))}",
        f"    Fundamental:   {_bar(result.get('fundamental_score'))}",
        f"    Trend:         {_bar(result.get('trend_score'))}  ({result.get('trend_direction', '')})",
        f"    Momentum:      {_bar(result.get('momentum_score'))}",
    ]

    bd = result.get("breakdown", {})

    # Technical details
    tech_det = bd.get("technical", {})
    if tech_det:
        lines.append("")
        lines.append("  ── Technical Breakdown ──")
        for k, v in tech_det.items():
            lines.append(f"    {k:15s} {v}")

    # Momentum details
    mom_det = bd.get("momentum", {})
    if mom_det:
        lines.append("")
        lines.append("  ── Momentum Details ──")
        for k, v in mom_det.items():
            lines.append(f"    {k:25s} {v}")

    return "\n".join(lines)


def format_ranking_table(ranked: list[dict]) -> str:
    """Format ranked stocks as a leaderboard table."""
    if not ranked:
        return "No stocks scored."

    lines = [
        "═══ Stock Rankings ═══",
        "",
        f"  {'#':>3} {'Symbol':<8} {'Name':<22} {'Score':>6} {'Rating':<12} "
        f"{'Tech':>5} {'Fund':>5} {'Trend':>5} {'Mom':>5} {'Trend Dir':<16}",
        f"  {'─'*3} {'─'*8} {'─'*22} {'─'*6} {'─'*12} "
        f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*16}",
    ]

    for i, r in enumerate(ranked, 1):
        name = r.get("name", "")[:22]
        lines.append(
            f"  {i:>3} {r['symbol']:<8} {name:<22} {r['overall_score']:>5.1f} "
            f"{r['rating']:<12} "
            f"{_sc(r.get('technical_score')):>5} "
            f"{_sc(r.get('fundamental_score')):>5} "
            f"{_sc(r.get('trend_score')):>5} "
            f"{_sc(r.get('momentum_score')):>5} "
            f"{r.get('trend_direction', ''):<16}"
        )

    return "\n".join(lines)


def _bar(val, width=20) -> str:
    if val is None:
        return "N/A"
    filled = int(val / 100 * width)
    return f"{'█' * filled}{'░' * (width - filled)} {val:.1f}/100"


def _sc(val) -> str:
    return f"{val:.0f}" if val is not None else "N/A"
