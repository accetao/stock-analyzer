"""
Stock Analyzer - Trend Analyzer
Identifies market trends using multiple methods:
    - Moving average alignment
    - Higher highs / higher lows pattern
    - ADX strength classification
    - Support / Resistance levels
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src import technical_analysis as ta

logger = logging.getLogger(__name__)


# ─── Trend Direction ─────────────────────────────────────────────────────────

class Trend:
    STRONG_UP = "STRONG UPTREND"
    UP = "UPTREND"
    WEAK_UP = "WEAK UPTREND"
    SIDEWAYS = "SIDEWAYS"
    WEAK_DOWN = "WEAK DOWNTREND"
    DOWN = "DOWNTREND"
    STRONG_DOWN = "STRONG DOWNTREND"


def classify_trend(df: pd.DataFrame) -> dict:
    """
    Multi-factor trend classification.

    Returns:
        trend:      Trend string (see Trend class)
        confidence: 0-100 confidence score
        factors:    dict of individual factor assessments
    """
    if df.empty or len(df) < config.SMA_LONG:
        return {"trend": Trend.SIDEWAYS, "confidence": 0, "factors": {}}

    # Ensure indicators are computed
    if "SMA_20" not in df.columns:
        ta.add_all_indicators(df)

    last = df.iloc[-1]
    factors = {}
    bullish_points = 0
    bearish_points = 0
    total_points = 0

    # 1. Price vs Moving Averages
    close = last["Close"]
    for sma_col, weight in [("SMA_20", 1), ("SMA_50", 2), ("SMA_200", 3)]:
        if sma_col in df.columns and pd.notna(last[sma_col]):
            above = close > last[sma_col]
            if above:
                bullish_points += weight
            else:
                bearish_points += weight
            total_points += weight
            factors[f"price_vs_{sma_col}"] = "ABOVE" if above else "BELOW"

    # 2. Moving Average Alignment (golden/death cross)
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        if pd.notna(last["SMA_50"]) and pd.notna(last["SMA_200"]):
            golden = last["SMA_50"] > last["SMA_200"]
            if golden:
                bullish_points += 3
            else:
                bearish_points += 3
            total_points += 3
            factors["ma_cross"] = "GOLDEN" if golden else "DEATH"

    # 3. SMA slopes (direction of trend)
    for sma_col, weight in [("SMA_20", 1), ("SMA_50", 2)]:
        if sma_col in df.columns and len(df) > 5:
            slope = df[sma_col].iloc[-1] - df[sma_col].iloc[-5]
            if slope > 0:
                bullish_points += weight
            else:
                bearish_points += weight
            total_points += weight
            factors[f"{sma_col}_slope"] = "UP" if slope > 0 else "DOWN"

    # 4. RSI positioning
    if "RSI" in df.columns and pd.notna(last["RSI"]):
        rsi = last["RSI"]
        if rsi > 55:
            bullish_points += 1
        elif rsi < 45:
            bearish_points += 1
        total_points += 1
        factors["rsi_position"] = f"{rsi:.1f}"

    # 5. MACD
    if "MACD_Hist" in df.columns and pd.notna(last["MACD_Hist"]):
        if last["MACD_Hist"] > 0:
            bullish_points += 2
        else:
            bearish_points += 2
        total_points += 2
        factors["macd_direction"] = "BULLISH" if last["MACD_Hist"] > 0 else "BEARISH"

    # 6. ADX trend strength
    if "ADX" in df.columns and pd.notna(last["ADX"]):
        adx = last["ADX"]
        factors["adx"] = round(adx, 1)
        factors["trend_strength"] = "STRONG" if adx > 25 else "WEAK"

    # 7. Higher highs / higher lows (last 20 bars)
    hh_hl = _check_higher_highs_lows(df, window=20)
    factors["hh_hl_pattern"] = hh_hl
    if hh_hl == "HIGHER_HIGHS_HIGHER_LOWS":
        bullish_points += 2
    elif hh_hl == "LOWER_HIGHS_LOWER_LOWS":
        bearish_points += 2
    total_points += 2

    # ── Compute final trend ──────────────────────────────────────────────
    if total_points == 0:
        return {"trend": Trend.SIDEWAYS, "confidence": 0, "factors": factors}

    bull_pct = bullish_points / total_points * 100
    bear_pct = bearish_points / total_points * 100
    net = bull_pct - bear_pct

    adx_val = factors.get("adx", 20)
    strong = adx_val > 30

    if net > 60:
        trend = Trend.STRONG_UP if strong else Trend.UP
    elif net > 25:
        trend = Trend.UP if strong else Trend.WEAK_UP
    elif net > -25:
        trend = Trend.SIDEWAYS
    elif net > -60:
        trend = Trend.DOWN if strong else Trend.WEAK_DOWN
    else:
        trend = Trend.STRONG_DOWN if strong else Trend.DOWN

    confidence = min(100, abs(net) + (adx_val / 2))

    return {
        "trend": trend,
        "confidence": round(confidence, 1),
        "bullish_pct": round(bull_pct, 1),
        "bearish_pct": round(bear_pct, 1),
        "factors": factors,
    }


def _check_higher_highs_lows(df: pd.DataFrame, window: int = 20) -> str:
    """Check for higher-highs/higher-lows or lower-highs/lower-lows pattern."""
    recent = df.tail(window)
    if len(recent) < window:
        return "INSUFFICIENT_DATA"

    mid = window // 2
    first_half = recent.iloc[:mid]
    second_half = recent.iloc[mid:]

    hh = second_half["High"].max() > first_half["High"].max()
    hl = second_half["Low"].min() > first_half["Low"].min()
    lh = second_half["High"].max() < first_half["High"].max()
    ll = second_half["Low"].min() < first_half["Low"].min()

    if hh and hl:
        return "HIGHER_HIGHS_HIGHER_LOWS"
    elif lh and ll:
        return "LOWER_HIGHS_LOWER_LOWS"
    elif hh and ll:
        return "EXPANDING"
    elif lh and hl:
        return "CONTRACTING"
    else:
        return "MIXED"


# ─── Support / Resistance ───────────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, num_levels: int = 3) -> dict:
    """
    Identify key support and resistance levels using pivot points
    and price clustering.

    Returns:
        support:    list of price levels
        resistance: list of price levels
    """
    if df.empty or len(df) < 20:
        return {"support": [], "resistance": []}

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    current = close[-1]

    # Find local minima (support) and maxima (resistance)
    pivots_high = []
    pivots_low = []
    window = 5

    for i in range(window, len(df) - window):
        if high[i] == max(high[i - window : i + window + 1]):
            pivots_high.append(high[i])
        if low[i] == min(low[i - window : i + window + 1]):
            pivots_low.append(low[i])

    # Cluster nearby levels (within 1.5% of each other)
    resistance = _cluster_levels([p for p in pivots_high if p > current], num_levels)
    support = _cluster_levels([p for p in pivots_low if p < current], num_levels)

    # Sort: support descending (nearest first), resistance ascending
    support.sort(reverse=True)
    resistance.sort()

    return {
        "support": [round(s, 2) for s in support[:num_levels]],
        "resistance": [round(r, 2) for r in resistance[:num_levels]],
        "current_price": round(current, 2),
    }


def _cluster_levels(levels: list, max_clusters: int, threshold: float = 0.015) -> list:
    """Group nearby price levels into clusters, return cluster averages."""
    if not levels:
        return []

    levels = sorted(levels)
    clusters = [[levels[0]]]

    for lvl in levels[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] < threshold:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    # Sort clusters by frequency (most visited levels first)
    clusters.sort(key=len, reverse=True)

    return [np.mean(c) for c in clusters[:max_clusters]]


# ─── Formatted Report ───────────────────────────────────────────────────────

def format_trend_report(symbol: str, df: pd.DataFrame) -> str:
    """Generate a formatted trend analysis report."""
    trend_data = classify_trend(df)
    sr = find_support_resistance(df)

    lines = [
        f"═══ Trend Analysis: {symbol} ═══",
        "",
        f"  Trend:       {trend_data['trend']}",
        f"  Confidence:  {trend_data['confidence']}%",
        f"  Bullish:     {trend_data.get('bullish_pct', 'N/A')}%",
        f"  Bearish:     {trend_data.get('bearish_pct', 'N/A')}%",
        "",
        "  ── Key Factors ──",
    ]

    for k, v in trend_data["factors"].items():
        lines.append(f"    {k:25s} {v}")

    lines += [
        "",
        f"  ── Support & Resistance ──",
        f"    Current Price: ${sr['current_price']}",
        f"    Resistance:    {', '.join(f'${r}' for r in sr['resistance']) or 'N/A'}",
        f"    Support:       {', '.join(f'${s}' for s in sr['support']) or 'N/A'}",
    ]

    return "\n".join(lines)
