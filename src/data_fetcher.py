"""
Stock Analyzer - Data Fetcher
Handles all communication with Yahoo Finance.
Provides clean, validated market data for other modules.
"""

import datetime as dt
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# ─── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict[str, tuple[dt.datetime, object]] = {}


def _cache_key(symbol: str, kind: str, period: str, interval: str) -> str:
    return f"{symbol}|{kind}|{period}|{interval}"


def _get_cached(key: str):
    if key in _cache:
        ts, data = _cache[key]
        if (dt.datetime.now() - ts).total_seconds() < config.CACHE_EXPIRY_MINUTES * 60:
            return data
        del _cache[key]
    return None


def _set_cached(key: str, data):
    _cache[key] = (dt.datetime.now(), data)


# ─── Core fetchers ───────────────────────────────────────────────────────────

def get_ticker(symbol: str) -> yf.Ticker:
    """Return a yfinance Ticker object."""
    return yf.Ticker(symbol.upper().strip())


def get_history(
    symbol: str,
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a symbol.

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume
    Index is DatetimeIndex.
    """
    key = _cache_key(symbol, "history", period, interval)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    ticker = get_ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        logger.warning("No history data returned for %s", symbol)
        return pd.DataFrame()

    # Keep only essential columns
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df.index.name = "Date"

    _set_cached(key, df)
    return df


def get_info(symbol: str) -> dict:
    """
    Fetch fundamental / summary info for a symbol.

    Returns a dict with keys like:
        marketCap, trailingPE, forwardPE, trailingEps, dividendYield,
        revenueGrowth, profitMargins, returnOnEquity, debtToEquity,
        fiftyTwoWeekHigh, fiftyTwoWeekLow, shortName, sector, industry, ...
    """
    key = _cache_key(symbol, "info", "", "")
    cached = _get_cached(key)
    if cached is not None:
        return cached

    ticker = get_ticker(symbol)
    info = ticker.info

    if not info or info.get("regularMarketPrice") is None:
        logger.warning("No info returned for %s", symbol)
        return {}

    _set_cached(key, info)
    return info


def get_financials(symbol: str) -> dict[str, pd.DataFrame]:
    """
    Fetch key financial statements.

    Returns a dict with keys:
        income_stmt, balance_sheet, cashflow
    Each value is a DataFrame.
    """
    key = _cache_key(symbol, "financials", "", "")
    cached = _get_cached(key)
    if cached is not None:
        return cached

    ticker = get_ticker(symbol)
    result = {
        "income_stmt": ticker.income_stmt,
        "balance_sheet": ticker.balance_sheet,
        "cashflow": ticker.cashflow,
    }
    _set_cached(key, result)
    return result


def get_current_price(symbol: str) -> Optional[float]:
    """Get the latest market price for a symbol."""
    info = get_info(symbol)
    return info.get("currentPrice") or info.get("regularMarketPrice")


def get_multiple_histories(
    symbols: list[str],
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """Fetch history for multiple symbols. Returns dict symbol -> DataFrame."""
    results = {}
    for sym in symbols:
        try:
            df = get_history(sym, period, interval)
            if not df.empty:
                results[sym] = df
        except Exception as e:
            logger.error("Error fetching %s: %s", sym, e)
    return results


def validate_symbol(symbol: str) -> bool:
    """Check if a symbol is valid and has data on Yahoo Finance."""
    try:
        info = get_info(symbol)
        return bool(info)
    except Exception:
        return False


def clear_cache():
    """Clear all cached data."""
    _cache.clear()
    logger.info("Cache cleared.")


def get_news(symbol: str, max_items: int = 10) -> list[dict]:
    """
    Fetch latest news articles for a symbol from Yahoo Finance.

    Returns a list of dicts with keys:
        title, publisher, link, publish_time, thumbnail
    """
    key = _cache_key(symbol, "news", "", "")
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        ticker = get_ticker(symbol)
        raw_news = ticker.news or []
        articles = []
        for item in raw_news[:max_items]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            # yfinance ≥ 0.2.31 nests under "content"; older versions are flat
            if content:
                pub_date = content.get("pubDate", "")
                articles.append({
                    "title": content.get("title", "No title"),
                    "publisher": content.get("provider", {}).get("displayName", "")
                                if isinstance(content.get("provider"), dict)
                                else str(content.get("provider", "")),
                    "link": content.get("canonicalUrl", {}).get("url", "")
                            if isinstance(content.get("canonicalUrl"), dict)
                            else content.get("url", content.get("link", "")),
                    "publish_time": pub_date,
                    "thumbnail": (
                        content.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", "")
                        if isinstance(content.get("thumbnail"), dict)
                        else ""
                    ),
                })
            else:
                # Flat dict fallback (older yfinance)
                ts = item.get("providerPublishTime", 0)
                pub_str = (
                    dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    if ts else ""
                )
                articles.append({
                    "title": item.get("title", "No title"),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "publish_time": pub_str,
                    "thumbnail": (item.get("thumbnail", {})
                                   .get("resolutions", [{}])[0]
                                   .get("url", ""))
                                  if isinstance(item.get("thumbnail"), dict) else "",
                })
        _set_cached(key, articles)
        return articles
    except Exception as e:
        logger.error("Error fetching news for %s: %s", symbol, e)
        return []
