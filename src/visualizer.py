"""
Stock Analyzer - Visualizer
Generates professional charts with technical indicators.
"""

import os
import logging
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import config
from src import technical_analysis as ta

logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def plot_stock_analysis(
    df: pd.DataFrame,
    symbol: str,
    show_volume: bool = True,
    show_sma: bool = True,
    show_bb: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    save: bool = True,
    show: bool = False,
) -> Optional[str]:
    """
    Generate a comprehensive stock analysis chart.

    Layout:
        Panel 1 (large): Price + Moving Averages + Bollinger Bands
        Panel 2:         Volume
        Panel 3:         RSI
        Panel 4:         MACD

    Returns the file path if saved, else None.
    """
    if df.empty:
        logger.warning("Empty DataFrame, cannot plot")
        return None

    # Ensure indicators are present
    if "SMA_20" not in df.columns:
        ta.add_all_indicators(df)

    # Determine panel count
    n_panels = 1
    ratios = [4]
    if show_volume:
        n_panels += 1
        ratios.append(1)
    if show_rsi:
        n_panels += 1
        ratios.append(1.2)
    if show_macd:
        n_panels += 1
        ratios.append(1.2)

    fig = plt.figure(figsize=(config.CHART_WIDTH, config.CHART_HEIGHT + n_panels))
    gs = GridSpec(n_panels, 1, height_ratios=ratios, hspace=0.05)

    axes = []
    panel = 0

    # ── Panel 1: Price ───────────────────────────────────────────────────
    ax_price = fig.add_subplot(gs[panel])
    axes.append(ax_price)
    panel += 1

    ax_price.plot(df.index, df["Close"], linewidth=1.5, color="#2196F3", label="Close")

    if show_sma:
        sma_styles = {
            "SMA_20": ("#FF9800", 1.0, "SMA 20"),
            "SMA_50": ("#4CAF50", 1.0, "SMA 50"),
            "SMA_200": ("#F44336", 1.2, "SMA 200"),
        }
        for col, (color, lw, label) in sma_styles.items():
            if col in df.columns:
                ax_price.plot(df.index, df[col], color=color, linewidth=lw,
                              label=label, alpha=0.8)

    if show_bb and "BB_Upper" in df.columns:
        ax_price.fill_between(
            df.index, df["BB_Upper"], df["BB_Lower"],
            alpha=0.1, color="#9C27B0", label="Bollinger Bands"
        )
        ax_price.plot(df.index, df["BB_Upper"], color="#9C27B0", linewidth=0.5, alpha=0.5)
        ax_price.plot(df.index, df["BB_Lower"], color="#9C27B0", linewidth=0.5, alpha=0.5)

    ax_price.set_title(f"{symbol} - Stock Analysis", fontsize=14, fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_ylabel("Price ($)")
    ax_price.grid(True, alpha=0.3)
    ax_price.tick_params(labelbottom=False)

    # ── Panel 2: Volume ──────────────────────────────────────────────────
    if show_volume:
        ax_vol = fig.add_subplot(gs[panel], sharex=ax_price)
        axes.append(ax_vol)
        panel += 1

        colors = ["#4CAF50" if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]
                   else "#F44336" for i in range(1, len(df))]
        colors.insert(0, "#4CAF50")

        ax_vol.bar(df.index, df["Volume"], color=colors, alpha=0.6, width=0.8)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.3)
        ax_vol.tick_params(labelbottom=False)

        # Format volume axis
        ax_vol.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

    # ── Panel 3: RSI ─────────────────────────────────────────────────────
    if show_rsi and "RSI" in df.columns:
        ax_rsi = fig.add_subplot(gs[panel], sharex=ax_price)
        axes.append(ax_rsi)
        panel += 1

        ax_rsi.plot(df.index, df["RSI"], color="#FF9800", linewidth=1.2)
        ax_rsi.axhline(y=config.RSI_OVERBOUGHT, color="#F44336", linestyle="--",
                       alpha=0.5, linewidth=0.8)
        ax_rsi.axhline(y=config.RSI_OVERSOLD, color="#4CAF50", linestyle="--",
                       alpha=0.5, linewidth=0.8)
        ax_rsi.axhline(y=50, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        ax_rsi.fill_between(df.index, config.RSI_OVERBOUGHT, 100,
                            alpha=0.1, color="#F44336")
        ax_rsi.fill_between(df.index, 0, config.RSI_OVERSOLD,
                            alpha=0.1, color="#4CAF50")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.tick_params(labelbottom=False)

    # ── Panel 4: MACD ────────────────────────────────────────────────────
    if show_macd and "MACD" in df.columns:
        ax_macd = fig.add_subplot(gs[panel], sharex=ax_price)
        axes.append(ax_macd)
        panel += 1

        ax_macd.plot(df.index, df["MACD"], color="#2196F3", linewidth=1.0, label="MACD")
        ax_macd.plot(df.index, df["MACD_Signal"], color="#FF9800", linewidth=1.0,
                     label="Signal")

        colors_hist = ["#4CAF50" if v >= 0 else "#F44336" for v in df["MACD_Hist"]]
        ax_macd.bar(df.index, df["MACD_Hist"], color=colors_hist, alpha=0.5, width=0.8)

        ax_macd.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left", fontsize=8)
        ax_macd.grid(True, alpha=0.3)

    # Format x-axis on bottom panel
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, fontsize=8)

    plt.tight_layout()

    filepath = None
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, f"{symbol}_analysis.png")
        fig.savefig(filepath, dpi=config.CHART_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        logger.info("Chart saved: %s", filepath)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filepath


def plot_comparison(
    data: dict[str, pd.DataFrame],
    title: str = "Stock Comparison",
    normalize: bool = True,
    save: bool = True,
) -> Optional[str]:
    """
    Plot multiple stocks on the same chart for comparison.

    Args:
        data: dict of symbol -> DataFrame
        normalize: If True, normalize to percentage returns from start
    """
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(config.CHART_WIDTH, config.CHART_HEIGHT))

    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    for (symbol, df), color in zip(data.items(), colors):
        if df.empty:
            continue
        values = df["Close"]
        if normalize:
            values = (values / values.iloc[0] - 1) * 100  # percentage change
        ax.plot(df.index, values, label=symbol, linewidth=1.5, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)" if normalize else "Price ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    filepath = None
    if save:
        safe_title = title.replace(" ", "_").lower()
        filepath = os.path.join(config.OUTPUT_DIR, f"{safe_title}.png")
        fig.savefig(filepath, dpi=config.CHART_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        logger.info("Comparison chart saved: %s", filepath)

    plt.close(fig)
    return filepath


def plot_score_dashboard(ranked: list[dict], save: bool = True) -> Optional[str]:
    """
    Create a horizontal bar chart showing top stocks by score.
    """
    if not ranked:
        return None

    symbols = [r["symbol"] for r in ranked]
    scores = [r["overall_score"] for r in ranked]
    ratings = [r["rating"] for r in ranked]

    color_map = {
        "STRONG BUY": "#1B5E20",
        "BUY": "#4CAF50",
        "HOLD": "#FF9800",
        "SELL": "#F44336",
        "STRONG SELL": "#B71C1C",
    }
    colors = [color_map.get(r, "#9E9E9E") for r in ratings]

    fig, ax = plt.subplots(figsize=(config.CHART_WIDTH, max(4, len(ranked) * 0.5)))

    y_pos = range(len(symbols))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=10)
    ax.set_xlabel("Investment Score (0-100)")
    ax.set_title("Stock Investment Scores", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add score labels
    for bar, score, rating in zip(bars, scores, ratings):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f} ({rating})", va="center", fontsize=9)

    plt.tight_layout()

    filepath = None
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, "score_dashboard.png")
        fig.savefig(filepath, dpi=config.CHART_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        logger.info("Dashboard saved: %s", filepath)

    plt.close(fig)
    return filepath
