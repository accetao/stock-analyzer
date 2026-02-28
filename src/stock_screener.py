"""
Stock Analyzer - Stock Screener
Filters stocks from a universe based on technical and fundamental criteria.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
from src import data_fetcher, technical_analysis as ta, fundamental_analysis as fa, trend_analyzer

logger = logging.getLogger(__name__)


@dataclass
class ScreenCriteria:
    """Configurable screening criteria."""
    # Volume & Size
    min_avg_volume: int = config.SCREENER_MIN_VOLUME
    min_market_cap: float = config.SCREENER_MIN_MARKET_CAP

    # Valuation
    min_pe: float = config.SCREENER_MIN_PE
    max_pe: float = config.SCREENER_MAX_PE

    # Technical
    above_sma_200: bool = False          # Price above 200-day SMA
    rsi_min: float = 0                    # Minimum RSI
    rsi_max: float = 100                  # Maximum RSI
    macd_bullish: bool = False            # MACD histogram > 0
    trend_up: bool = False                # Uptrend required

    # Fundamental
    min_roe: Optional[float] = None       # e.g. 0.10 for 10%
    min_revenue_growth: Optional[float] = None
    max_debt_to_equity: Optional[float] = None
    positive_free_cashflow: bool = False

    # Sector filter
    sectors: list = field(default_factory=list)  # empty = all sectors


# ─── Pre-built Screen Strategies ────────────────────────────────────────────

SCREEN_GROWTH = ScreenCriteria(
    above_sma_200=True,
    min_revenue_growth=0.10,
    min_roe=0.12,
    trend_up=True,
    max_pe=40,
)

SCREEN_VALUE = ScreenCriteria(
    max_pe=18,
    min_roe=0.10,
    max_debt_to_equity=100,
    positive_free_cashflow=True,
)

SCREEN_MOMENTUM = ScreenCriteria(
    above_sma_200=True,
    rsi_min=50,
    rsi_max=75,
    macd_bullish=True,
    trend_up=True,
)

SCREEN_DIVIDEND = ScreenCriteria(
    max_pe=30,
    max_debt_to_equity=120,
    positive_free_cashflow=True,
)

SCREENS = {
    "growth": SCREEN_GROWTH,
    "value": SCREEN_VALUE,
    "momentum": SCREEN_MOMENTUM,
    "dividend": SCREEN_DIVIDEND,
}


def screen_stocks(
    symbols: list[str],
    criteria: ScreenCriteria,
    progress_callback=None,
) -> list[dict]:
    """
    Screen a list of symbols against the given criteria.

    Returns a list of dicts for stocks that pass all filters, each containing:
        symbol, name, sector, price, pe, market_cap, reasons (why it passed)
    """
    results = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(sym, i + 1, total)

        try:
            passed, details = _evaluate_stock(sym, criteria)
            if passed:
                results.append(details)
        except Exception as e:
            logger.debug("Screener skipped %s: %s", sym, e)

    # Sort by a composite of met criteria count
    results.sort(key=lambda x: x.get("criteria_met", 0), reverse=True)
    return results


def _evaluate_stock(symbol: str, criteria: ScreenCriteria) -> tuple[bool, dict]:
    """Evaluate a single stock against criteria. Returns (passed, details)."""
    info = data_fetcher.get_info(symbol)
    if not info:
        return False, {}

    details = {
        "symbol": symbol,
        "name": info.get("shortName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "reasons": [],
        "criteria_met": 0,
    }

    total_criteria = 0
    met_criteria = 0

    # ── Volume filter ────────────────────────────────────────────────────
    avg_vol = info.get("averageVolume", 0)
    if criteria.min_avg_volume > 0:
        total_criteria += 1
        if avg_vol and avg_vol >= criteria.min_avg_volume:
            met_criteria += 1
        else:
            return False, details

    # ── Market cap filter ────────────────────────────────────────────────
    mcap = info.get("marketCap", 0)
    if criteria.min_market_cap > 0:
        total_criteria += 1
        if mcap and mcap >= criteria.min_market_cap:
            met_criteria += 1
        else:
            return False, details

    # ── Sector filter ────────────────────────────────────────────────────
    if criteria.sectors:
        total_criteria += 1
        if info.get("sector", "") in criteria.sectors:
            met_criteria += 1
        else:
            return False, details

    # ── P/E filter ───────────────────────────────────────────────────────
    pe = info.get("trailingPE")
    if criteria.max_pe < 100 or criteria.min_pe > 0:
        total_criteria += 1
        if pe is not None and criteria.min_pe <= pe <= criteria.max_pe:
            met_criteria += 1
            details["reasons"].append(f"P/E {pe:.1f}")
        else:
            return False, details

    # ── Fundamental filters ──────────────────────────────────────────────
    if criteria.min_roe is not None:
        total_criteria += 1
        roe = info.get("returnOnEquity")
        if roe is not None and roe >= criteria.min_roe:
            met_criteria += 1
            details["reasons"].append(f"ROE {roe*100:.1f}%")
        else:
            return False, details

    if criteria.min_revenue_growth is not None:
        total_criteria += 1
        rg = info.get("revenueGrowth")
        if rg is not None and rg >= criteria.min_revenue_growth:
            met_criteria += 1
            details["reasons"].append(f"RevGrowth {rg*100:.1f}%")
        else:
            return False, details

    if criteria.max_debt_to_equity is not None:
        total_criteria += 1
        de = info.get("debtToEquity")
        if de is not None and de <= criteria.max_debt_to_equity:
            met_criteria += 1
            details["reasons"].append(f"D/E {de:.0f}")
        elif de is None:
            met_criteria += 1  # no debt data = pass
        else:
            return False, details

    if criteria.positive_free_cashflow:
        total_criteria += 1
        fcf = info.get("freeCashflow")
        if fcf is not None and fcf > 0:
            met_criteria += 1
            details["reasons"].append("FCF+")
        else:
            return False, details

    # ── Technical filters (need price data) ──────────────────────────────
    need_technical = (
        criteria.above_sma_200
        or criteria.rsi_min > 0
        or criteria.rsi_max < 100
        or criteria.macd_bullish
        or criteria.trend_up
    )

    if need_technical:
        df = data_fetcher.get_history(symbol)
        if df.empty or len(df) < config.SMA_LONG:
            return False, details

        ta.add_all_indicators(df)
        last = df.iloc[-1]
        close = last["Close"]

        if criteria.above_sma_200:
            total_criteria += 1
            if pd.notna(last.get("SMA_200")) and close > last["SMA_200"]:
                met_criteria += 1
                details["reasons"].append("Above SMA200")
            else:
                return False, details

        if criteria.rsi_min > 0 or criteria.rsi_max < 100:
            total_criteria += 1
            rsi = last.get("RSI")
            if rsi is not None and criteria.rsi_min <= rsi <= criteria.rsi_max:
                met_criteria += 1
                details["reasons"].append(f"RSI {rsi:.0f}")
            else:
                return False, details

        if criteria.macd_bullish:
            total_criteria += 1
            hist = last.get("MACD_Hist")
            if hist is not None and hist > 0:
                met_criteria += 1
                details["reasons"].append("MACD bullish")
            else:
                return False, details

        if criteria.trend_up:
            total_criteria += 1
            td = trend_analyzer.classify_trend(df)
            if "UP" in td["trend"]:
                met_criteria += 1
                details["reasons"].append(td["trend"])
            else:
                return False, details

    details["criteria_met"] = met_criteria
    return True, details


def format_screen_results(results: list[dict], title: str = "Screen Results") -> str:
    """Format screening results as a table."""
    if not results:
        return f"═══ {title} ═══\n  No stocks matched the criteria."

    lines = [
        f"═══ {title} ═══",
        f"  {len(results)} stocks matched",
        "",
        f"  {'Symbol':<8} {'Name':<25} {'Price':>10} {'P/E':>8} {'Sector':<15} Reasons",
        f"  {'─'*8} {'─'*25} {'─'*10} {'─'*8} {'─'*15} {'─'*25}",
    ]

    for r in results:
        price = f"${r['price']:.2f}" if r.get("price") else "N/A"
        pe = f"{r['pe']:.1f}" if r.get("pe") else "N/A"
        reasons = ", ".join(r.get("reasons", []))
        lines.append(
            f"  {r['symbol']:<8} {r.get('name','')[:25]:<25} {price:>10} {pe:>8} "
            f"{r.get('sector','')[:15]:<15} {reasons}"
        )

    return "\n".join(lines)
