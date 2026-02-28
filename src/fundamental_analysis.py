"""
Stock Analyzer - Fundamental Analysis
Evaluates company financials: valuation, profitability, growth, and health.
"""

import logging
from typing import Optional

import pandas as pd

from src import data_fetcher

logger = logging.getLogger(__name__)


def get_fundamental_metrics(symbol: str) -> dict:
    """
    Compute a full set of fundamental metrics for a symbol.

    Categories:
        Valuation   – P/E, Forward P/E, PEG, P/B, P/S, EV/EBITDA
        Profitability – Margins, ROE, ROA
        Growth      – Revenue growth, Earnings growth
        Dividends   – Yield, Payout ratio
        Health      – Debt/Equity, Current ratio, Quick ratio
    """
    info = data_fetcher.get_info(symbol)
    if not info:
        return {}

    metrics = {
        "symbol": symbol,
        "name": info.get("shortName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
    }

    # ── Valuation ─────────────────────────────────────────────────────────
    metrics["market_cap"] = info.get("marketCap")
    metrics["trailing_pe"] = info.get("trailingPE")
    metrics["forward_pe"] = info.get("forwardPE")
    metrics["peg_ratio"] = info.get("pegRatio")
    metrics["price_to_book"] = info.get("priceToBook")
    metrics["price_to_sales"] = info.get("priceToSalesTrailing12Months")
    metrics["ev_to_ebitda"] = info.get("enterpriseToEbitda")
    metrics["ev_to_revenue"] = info.get("enterpriseToRevenue")

    # ── Profitability ─────────────────────────────────────────────────────
    metrics["profit_margin"] = info.get("profitMargins")
    metrics["operating_margin"] = info.get("operatingMargins")
    metrics["gross_margin"] = info.get("grossMargins")
    metrics["roe"] = info.get("returnOnEquity")
    metrics["roa"] = info.get("returnOnAssets")
    metrics["trailing_eps"] = info.get("trailingEps")
    metrics["forward_eps"] = info.get("forwardEps")

    # ── Growth ────────────────────────────────────────────────────────────
    metrics["revenue_growth"] = info.get("revenueGrowth")
    metrics["earnings_growth"] = info.get("earningsGrowth")
    metrics["earnings_quarterly_growth"] = info.get("earningsQuarterlyGrowth")

    # ── Dividends ─────────────────────────────────────────────────────────
    metrics["dividend_yield"] = info.get("dividendYield")
    metrics["payout_ratio"] = info.get("payoutRatio")

    # ── Financial Health ──────────────────────────────────────────────────
    metrics["debt_to_equity"] = info.get("debtToEquity")
    metrics["current_ratio"] = info.get("currentRatio")
    metrics["quick_ratio"] = info.get("quickRatio")
    metrics["total_cash"] = info.get("totalCash")
    metrics["total_debt"] = info.get("totalDebt")
    metrics["free_cashflow"] = info.get("freeCashflow")
    metrics["operating_cashflow"] = info.get("operatingCashflow")

    # ── Price Context ─────────────────────────────────────────────────────
    metrics["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    metrics["52w_high"] = info.get("fiftyTwoWeekHigh")
    metrics["52w_low"] = info.get("fiftyTwoWeekLow")
    metrics["50d_avg"] = info.get("fiftyDayAverage")
    metrics["200d_avg"] = info.get("twoHundredDayAverage")
    metrics["beta"] = info.get("beta")

    # ── Analyst ───────────────────────────────────────────────────────────
    metrics["target_mean_price"] = info.get("targetMeanPrice")
    metrics["target_high_price"] = info.get("targetHighPrice")
    metrics["target_low_price"] = info.get("targetLowPrice")
    metrics["recommendation"] = info.get("recommendationKey")
    metrics["num_analyst_opinions"] = info.get("numberOfAnalystOpinions")

    return metrics


def score_valuation(metrics: dict) -> float:
    """
    Score valuation on 0-100 scale.
    Lower P/E, PEG, P/B → higher score.
    """
    score = 50.0  # neutral start
    adjustments = 0

    pe = metrics.get("trailing_pe")
    if pe is not None and pe > 0:
        if pe < 15:
            score += 15
        elif pe < 20:
            score += 8
        elif pe < 30:
            score -= 5
        else:
            score -= 15
        adjustments += 1

    peg = metrics.get("peg_ratio")
    if peg is not None and peg > 0:
        if peg < 1:
            score += 15
        elif peg < 1.5:
            score += 8
        elif peg < 2:
            score -= 3
        else:
            score -= 12
        adjustments += 1

    pb = metrics.get("price_to_book")
    if pb is not None and pb > 0:
        if pb < 1.5:
            score += 10
        elif pb < 3:
            score += 3
        elif pb > 5:
            score -= 10
        adjustments += 1

    ev_ebitda = metrics.get("ev_to_ebitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 10:
            score += 10
        elif ev_ebitda < 15:
            score += 3
        elif ev_ebitda > 25:
            score -= 10
        adjustments += 1

    return max(0, min(100, score))


def score_profitability(metrics: dict) -> float:
    """Score profitability on 0-100 scale."""
    score = 50.0

    roe = metrics.get("roe")
    if roe is not None:
        if roe > 0.25:
            score += 15
        elif roe > 0.15:
            score += 8
        elif roe > 0.10:
            score += 3
        elif roe < 0:
            score -= 15

    margin = metrics.get("profit_margin")
    if margin is not None:
        if margin > 0.20:
            score += 12
        elif margin > 0.10:
            score += 6
        elif margin > 0:
            score += 2
        else:
            score -= 12

    op_margin = metrics.get("operating_margin")
    if op_margin is not None:
        if op_margin > 0.25:
            score += 10
        elif op_margin > 0.15:
            score += 5
        elif op_margin < 0:
            score -= 10

    return max(0, min(100, score))


def score_growth(metrics: dict) -> float:
    """Score growth on 0-100 scale."""
    score = 50.0

    rev_growth = metrics.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0.25:
            score += 18
        elif rev_growth > 0.10:
            score += 10
        elif rev_growth > 0:
            score += 3
        else:
            score -= 12

    earn_growth = metrics.get("earnings_growth")
    if earn_growth is not None:
        if earn_growth > 0.25:
            score += 15
        elif earn_growth > 0.10:
            score += 8
        elif earn_growth > 0:
            score += 3
        else:
            score -= 10

    return max(0, min(100, score))


def score_financial_health(metrics: dict) -> float:
    """Score financial health on 0-100 scale."""
    score = 50.0

    de = metrics.get("debt_to_equity")
    if de is not None:
        if de < 30:
            score += 15
        elif de < 80:
            score += 5
        elif de > 150:
            score -= 15

    cr = metrics.get("current_ratio")
    if cr is not None:
        if cr > 2:
            score += 10
        elif cr > 1.5:
            score += 5
        elif cr < 1:
            score -= 12

    fcf = metrics.get("free_cashflow")
    if fcf is not None:
        if fcf > 0:
            score += 10
        else:
            score -= 10

    return max(0, min(100, score))


def get_fundamental_score(symbol: str) -> dict:
    """
    Compute an overall fundamental score and sub-scores.

    Returns dict with:
        overall_score, valuation_score, profitability_score,
        growth_score, health_score
    """
    metrics = get_fundamental_metrics(symbol)
    if not metrics:
        return {"symbol": symbol, "overall_score": None, "error": "No data"}

    val = score_valuation(metrics)
    prof = score_profitability(metrics)
    growth = score_growth(metrics)
    health = score_financial_health(metrics)

    overall = 0.30 * val + 0.25 * prof + 0.25 * growth + 0.20 * health

    return {
        "symbol": symbol,
        "name": metrics.get("name", ""),
        "sector": metrics.get("sector", ""),
        "overall_score": round(overall, 1),
        "valuation_score": round(val, 1),
        "profitability_score": round(prof, 1),
        "growth_score": round(growth, 1),
        "health_score": round(health, 1),
        "metrics": metrics,
    }


def format_fundamental_report(symbol: str) -> str:
    """Return a formatted text report of fundamental analysis."""
    result = get_fundamental_score(symbol)
    if result.get("error"):
        return f"❌ {symbol}: {result['error']}"

    m = result["metrics"]
    lines = [
        f"═══ Fundamental Analysis: {result['name']} ({symbol}) ═══",
        f"  Sector: {m.get('sector', 'N/A')}  |  Industry: {m.get('industry', 'N/A')}",
        "",
        f"  Overall Score: {result['overall_score']}/100",
        f"    Valuation:     {result['valuation_score']}/100",
        f"    Profitability: {result['profitability_score']}/100",
        f"    Growth:        {result['growth_score']}/100",
        f"    Health:        {result['health_score']}/100",
        "",
        "  ── Valuation ──",
        f"    P/E (Trailing):  {_fmt(m.get('trailing_pe'))}",
        f"    P/E (Forward):   {_fmt(m.get('forward_pe'))}",
        f"    PEG Ratio:       {_fmt(m.get('peg_ratio'))}",
        f"    P/B:             {_fmt(m.get('price_to_book'))}",
        f"    EV/EBITDA:       {_fmt(m.get('ev_to_ebitda'))}",
        "",
        "  ── Profitability ──",
        f"    Profit Margin:   {_pct(m.get('profit_margin'))}",
        f"    Operating Margin:{_pct(m.get('operating_margin'))}",
        f"    ROE:             {_pct(m.get('roe'))}",
        f"    ROA:             {_pct(m.get('roa'))}",
        "",
        "  ── Growth ──",
        f"    Revenue Growth:  {_pct(m.get('revenue_growth'))}",
        f"    Earnings Growth: {_pct(m.get('earnings_growth'))}",
        "",
        "  ── Financial Health ──",
        f"    Debt/Equity:     {_fmt(m.get('debt_to_equity'))}",
        f"    Current Ratio:   {_fmt(m.get('current_ratio'))}",
        f"    Free Cash Flow:  {_money(m.get('free_cashflow'))}",
        "",
        "  ── Analyst Consensus ──",
        f"    Recommendation:  {m.get('recommendation', 'N/A').upper()}",
        f"    Target Price:    ${_fmt(m.get('target_mean_price'))}",
        f"    Target Range:    ${_fmt(m.get('target_low_price'))} – ${_fmt(m.get('target_high_price'))}",
    ]
    return "\n".join(lines)


# ─── Formatting helpers ─────────────────────────────────────────────────────

def _fmt(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _money(val) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"
