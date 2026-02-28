"""
Stock Analyzer - Technical Analysis
Computes all standard technical indicators on OHLCV data.
Each function takes a DataFrame and returns it with new indicator columns.
"""

import pandas as pd
import numpy as np

import config


# ─── Moving Averages ─────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, window: int, col: str = "Close") -> pd.DataFrame:
    """Add Simple Moving Average column."""
    df[f"SMA_{window}"] = df[col].rolling(window=window).mean()
    return df


def add_ema(df: pd.DataFrame, window: int, col: str = "Close") -> pd.DataFrame:
    """Add Exponential Moving Average column."""
    df[f"EMA_{window}"] = df[col].ewm(span=window, adjust=False).mean()
    return df


def add_all_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add default SMA and EMA set."""
    for w in (config.SMA_SHORT, config.SMA_MEDIUM, config.SMA_LONG):
        add_sma(df, w)
    for w in (config.EMA_SHORT, config.EMA_LONG):
        add_ema(df, w)
    return df


# ─── RSI ──────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.DataFrame:
    """Add Relative Strength Index."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ─── MACD ─────────────────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """Add MACD line, Signal line, and Histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = config.BBANDS_PERIOD,
    std_dev: int = config.BBANDS_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower)."""
    sma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    df["BB_Upper"] = sma + (std_dev * std)
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - (std_dev * std)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


# ─── Average True Range ─────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.DataFrame:
    """Add Average True Range (volatility indicator)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=period).mean()
    return df


# ─── Stochastic Oscillator ──────────────────────────────────────────────────

def add_stochastic(
    df: pd.DataFrame,
    k_period: int = config.STOCH_K,
    d_period: int = config.STOCH_D,
) -> pd.DataFrame:
    """Add Stochastic %K and %D."""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["STOCH_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["STOCH_D"] = df["STOCH_K"].rolling(window=d_period).mean()
    return df


# ─── Average Directional Index (ADX) ─────────────────────────────────────────

def add_adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.DataFrame:
    """Add ADX, +DI, and -DI."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    df["ADX"] = adx
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    return df


# ─── On-Balance Volume ──────────────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume."""
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df


# ─── Volume Weighted Average Price ──────────────────────────────────────────

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add VWAP (cumulative for the visible period)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


# ─── Convenience: apply all indicators ──────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full standard set of technical indicators."""
    if df.empty or len(df) < config.SMA_LONG:
        return df

    add_all_moving_averages(df)
    add_rsi(df)
    add_macd(df)
    add_bollinger_bands(df)
    add_atr(df)
    add_stochastic(df)
    add_adx(df)
    add_obv(df)
    add_vwap(df)
    return df


# ─── Signal helpers ──────────────────────────────────────────────────────────

def get_latest_signals(df: pd.DataFrame) -> dict:
    """
    Extract the latest values of all indicators and generate simple signals.

    Returns dict with keys: rsi_signal, macd_signal, bb_signal, trend_signal, etc.
    """
    if df.empty:
        return {}

    last = df.iloc[-1]
    signals = {}

    # RSI
    if "RSI" in df.columns:
        rsi = last["RSI"]
        signals["rsi"] = round(rsi, 2)
        if rsi > config.RSI_OVERBOUGHT:
            signals["rsi_signal"] = "OVERBOUGHT"
        elif rsi < config.RSI_OVERSOLD:
            signals["rsi_signal"] = "OVERSOLD"
        else:
            signals["rsi_signal"] = "NEUTRAL"

    # MACD
    if "MACD_Hist" in df.columns:
        hist = last["MACD_Hist"]
        prev_hist = df["MACD_Hist"].iloc[-2] if len(df) > 1 else 0
        signals["macd_hist"] = round(hist, 4)
        if hist > 0 and prev_hist <= 0:
            signals["macd_signal"] = "BULLISH_CROSS"
        elif hist < 0 and prev_hist >= 0:
            signals["macd_signal"] = "BEARISH_CROSS"
        elif hist > 0:
            signals["macd_signal"] = "BULLISH"
        else:
            signals["macd_signal"] = "BEARISH"

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        close = last["Close"]
        if close >= last["BB_Upper"]:
            signals["bb_signal"] = "ABOVE_UPPER"
        elif close <= last["BB_Lower"]:
            signals["bb_signal"] = "BELOW_LOWER"
        else:
            signals["bb_signal"] = "WITHIN_BANDS"
        signals["bb_width"] = round(last["BB_Width"], 4)

    # Moving Average trend
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        if last["SMA_50"] > last["SMA_200"]:
            signals["ma_trend"] = "GOLDEN_CROSS"
        else:
            signals["ma_trend"] = "DEATH_CROSS"

    # Price vs SMA 200
    if "SMA_200" in df.columns:
        close = last["Close"]
        sma200 = last["SMA_200"]
        signals["above_sma200"] = close > sma200
        signals["pct_from_sma200"] = round((close - sma200) / sma200 * 100, 2)

    # ADX trend strength
    if "ADX" in df.columns:
        adx = last["ADX"]
        signals["adx"] = round(adx, 2)
        if adx > config.TREND_STRENGTH_THRESHOLD:
            signals["trend_strength"] = "STRONG"
        else:
            signals["trend_strength"] = "WEAK"

    # Stochastic
    if "STOCH_K" in df.columns:
        k = last["STOCH_K"]
        signals["stoch_k"] = round(k, 2)
        if k > 80:
            signals["stoch_signal"] = "OVERBOUGHT"
        elif k < 20:
            signals["stoch_signal"] = "OVERSOLD"
        else:
            signals["stoch_signal"] = "NEUTRAL"

    return signals
