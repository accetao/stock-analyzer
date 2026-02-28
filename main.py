"""
Stock Analyzer - Main Entry Point
A focused stock analysis tool powered by Yahoo Finance.

Usage:
    python main.py                      Interactive menu
    python main.py analyze AAPL         Full analysis of a stock
    python main.py trend AAPL           Trend analysis
    python main.py screen growth        Screen stocks by strategy
    python main.py rank                 Rank watchlist stocks
    python main.py compare AAPL MSFT    Compare stocks
    python main.py chart AAPL           Generate analysis chart
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src import (
    data_fetcher,
    technical_analysis as ta,
    fundamental_analysis as fa,
    trend_analyzer,
    stock_screener,
    stock_scorer,
    visualizer,
    utils,
)


def print_banner():
    print(r"""
    ╔═══════════════════════════════════════════════════════╗
    ║           📈  STOCK ANALYZER  📉                     ║
    ║      Real-Time Analysis · Yahoo Finance Data         ║
    ║      Technical · Fundamental · Trend · Scoring       ║
    ╚═══════════════════════════════════════════════════════╝
    """)


def interactive_menu():
    """Run the interactive menu loop."""
    print_banner()

    while True:
        print("\n  ┌─────────────── MAIN MENU ───────────────────┐")
        print("  │                                              │")
        print("  │   1.  Analyze a Stock (full report)          │")
        print("  │   2.  Technical Analysis                     │")
        print("  │   3.  Fundamental Analysis                   │")
        print("  │   4.  Trend Analysis                         │")
        print("  │   5.  Stock Screener                         │")
        print("  │   6.  Rank & Recommend Top Stocks            │")
        print("  │   7.  Compare Stocks                         │")
        print("  │   8.  Generate Chart                         │")
        print("  │   9.  Manage Watchlist                       │")
        print("  │   0.  Exit                                   │")
        print("  │                                              │")
        print("  └──────────────────────────────────────────────┘")

        choice = input("\n  Enter choice (0-9): ").strip()

        if choice == "0":
            print("\n  👋 Goodbye! Happy investing.\n")
            break
        elif choice == "1":
            cmd_full_analysis()
        elif choice == "2":
            cmd_technical()
        elif choice == "3":
            cmd_fundamental()
        elif choice == "4":
            cmd_trend()
        elif choice == "5":
            cmd_screen()
        elif choice == "6":
            cmd_rank()
        elif choice == "7":
            cmd_compare()
        elif choice == "8":
            cmd_chart()
        elif choice == "9":
            cmd_watchlist()
        else:
            print("  ⚠️  Invalid choice. Try again.")


# ─── Command: Full Analysis ─────────────────────────────────────────────────

def cmd_full_analysis(symbol: str = None):
    """Run full analysis: technical + fundamental + trend + score."""
    if not symbol:
        symbol = input("  Enter stock symbol: ").strip().upper()
    if not symbol:
        return

    utils.print_header(f"Full Analysis: {symbol}")
    print(f"  Fetching data for {symbol}...")

    # Validate
    if not data_fetcher.validate_symbol(symbol):
        print(f"  ❌ Invalid symbol: {symbol}")
        return

    # Fetch price data
    df = data_fetcher.get_history(symbol)
    if df.empty:
        print(f"  ❌ No price data for {symbol}")
        return

    # Add indicators
    ta.add_all_indicators(df)

    # 1. Score
    print()
    score_result = stock_scorer.score_stock(symbol)
    print(stock_scorer.format_score_report(score_result))

    # 2. Technical signals
    print()
    signals = ta.get_latest_signals(df)
    print("  ── Current Technical Signals ──")
    for key, val in signals.items():
        print(f"    {key:25s} {val}")

    # 3. Trend
    print()
    print(trend_analyzer.format_trend_report(symbol, df))

    # 4. Fundamental
    print()
    print(fa.format_fundamental_report(symbol))

    # 5. Generate chart
    print()
    filepath = visualizer.plot_stock_analysis(df, symbol)
    if filepath:
        print(f"  📊 Chart saved: {filepath}")

    # 6. Save report
    report = "\n".join([
        stock_scorer.format_score_report(score_result),
        "",
        trend_analyzer.format_trend_report(symbol, df),
        "",
        fa.format_fundamental_report(symbol),
    ])
    utils.save_report(report, f"{symbol}_full_analysis.txt")


# ─── Command: Technical Analysis ────────────────────────────────────────────

def cmd_technical(symbol: str = None):
    """Show technical indicators and signals."""
    if not symbol:
        symbol = input("  Enter stock symbol: ").strip().upper()
    if not symbol:
        return

    utils.print_header(f"Technical Analysis: {symbol}")
    df = data_fetcher.get_history(symbol)
    if df.empty:
        print(f"  ❌ No data for {symbol}")
        return

    ta.add_all_indicators(df)
    signals = ta.get_latest_signals(df)

    last = df.iloc[-1]
    print(f"\n  Price: ${last['Close']:.2f}")
    print(f"  Date:  {df.index[-1].strftime('%Y-%m-%d')}")
    print()

    print("  ── Indicator Values ──")
    indicator_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
    for col in indicator_cols:
        val = last[col]
        if not pd.isna(val):
            print(f"    {col:20s} {val:.4f}")

    print()
    print("  ── Signals ──")
    for key, val in signals.items():
        print(f"    {key:25s} {val}")


# ─── Command: Fundamental Analysis ──────────────────────────────────────────

def cmd_fundamental(symbol: str = None):
    """Show fundamental analysis report."""
    if not symbol:
        symbol = input("  Enter stock symbol: ").strip().upper()
    if not symbol:
        return

    utils.print_header(f"Fundamental Analysis: {symbol}")
    print(fa.format_fundamental_report(symbol))


# ─── Command: Trend Analysis ────────────────────────────────────────────────

def cmd_trend(symbol: str = None):
    """Show trend analysis."""
    if not symbol:
        symbol = input("  Enter stock symbol: ").strip().upper()
    if not symbol:
        return

    utils.print_header(f"Trend Analysis: {symbol}")
    df = data_fetcher.get_history(symbol)
    if df.empty:
        print(f"  ❌ No data for {symbol}")
        return

    print(trend_analyzer.format_trend_report(symbol, df))


# ─── Command: Screener ──────────────────────────────────────────────────────

def cmd_screen(strategy: str = None):
    """Run a stock screener."""
    if not strategy:
        print("\n  Available screening strategies:")
        print("    1. growth    – High growth companies in uptrends")
        print("    2. value     – Undervalued companies with solid fundamentals")
        print("    3. momentum  – Stocks with strong technical momentum")
        print("    4. dividend  – Stable dividend-paying stocks")
        strategy = input("\n  Choose strategy (growth/value/momentum/dividend): ").strip().lower()

    if strategy not in stock_screener.SCREENS:
        print(f"  ❌ Unknown strategy: {strategy}")
        return

    criteria = stock_screener.SCREENS[strategy]
    symbols = config.DEFAULT_UNIVERSE

    utils.print_header(f"Stock Screener: {strategy.upper()}")
    print(f"  Screening {len(symbols)} stocks...")

    def progress(sym, i, total):
        utils.progress_bar(i, total, sym)

    results = stock_screener.screen_stocks(symbols, criteria, progress_callback=progress)
    print()
    print(stock_screener.format_screen_results(results, f"{strategy.upper()} Screen"))


# ─── Command: Rank ───────────────────────────────────────────────────────────

def cmd_rank():
    """Score and rank stocks from the watchlist."""
    print("\n  Stock sources:")
    print("    1. Default watchlist")
    print("    2. Full universe (slower)")
    print("    3. Custom symbols")

    choice = input("  Choose (1/2/3): ").strip()

    if choice == "1":
        symbols = utils.load_watchlist("default")
    elif choice == "2":
        symbols = config.DEFAULT_UNIVERSE
    elif choice == "3":
        raw = input("  Enter symbols (comma-separated): ").strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    else:
        return

    top_n = input("  How many top stocks to show? (default 10): ").strip()
    top_n = int(top_n) if top_n.isdigit() else 10

    utils.print_header(f"Ranking {len(symbols)} Stocks")

    def progress(sym, i, total):
        utils.progress_bar(i, total, sym)

    ranked = stock_scorer.rank_stocks(symbols, top_n=top_n, progress_callback=progress)
    print()
    print(stock_scorer.format_ranking_table(ranked))

    # Generate dashboard chart
    filepath = visualizer.plot_score_dashboard(ranked)
    if filepath:
        print(f"\n  📊 Dashboard saved: {filepath}")


# ─── Command: Compare ───────────────────────────────────────────────────────

def cmd_compare(symbols: list[str] = None):
    """Compare multiple stocks side by side."""
    if not symbols:
        raw = input("  Enter symbols to compare (comma-separated): ").strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    if len(symbols) < 2:
        print("  ⚠️  Need at least 2 symbols to compare.")
        return

    utils.print_header(f"Comparing: {', '.join(symbols)}")

    # Fetch data
    data = data_fetcher.get_multiple_histories(symbols)

    # Score each
    print("\n  ── Scores ──")
    results = []
    for sym in symbols:
        score = stock_scorer.score_stock(sym)
        results.append(score)
        overall = score.get("overall_score", "N/A")
        rating = score.get("rating", "N/A")
        print(f"    {sym:8s}  Score: {overall}  Rating: {rating}")

    # Generate comparison chart
    filepath = visualizer.plot_comparison(data, f"Comparison: {' vs '.join(symbols)}")
    if filepath:
        print(f"\n  📊 Comparison chart saved: {filepath}")


# ─── Command: Chart ──────────────────────────────────────────────────────────

def cmd_chart(symbol: str = None):
    """Generate a detailed analysis chart."""
    if not symbol:
        symbol = input("  Enter stock symbol: ").strip().upper()
    if not symbol:
        return

    utils.print_header(f"Generating Chart: {symbol}")
    df = data_fetcher.get_history(symbol)
    if df.empty:
        print(f"  ❌ No data for {symbol}")
        return

    ta.add_all_indicators(df)
    filepath = visualizer.plot_stock_analysis(df, symbol)
    if filepath:
        print(f"  📊 Chart saved: {filepath}")


# ─── Command: Watchlist Management ───────────────────────────────────────────

def cmd_watchlist():
    """Manage watchlists."""
    print("\n  ── Watchlist Manager ──")
    print("    1. View current watchlist")
    print("    2. Add symbols")
    print("    3. Remove symbols")
    print("    4. Reset to defaults")
    print("    5. List all watchlists")

    choice = input("  Choose (1-5): ").strip()

    if choice == "1":
        symbols = utils.load_watchlist("default")
        print(f"\n  Default watchlist ({len(symbols)} symbols):")
        for i in range(0, len(symbols), 8):
            chunk = symbols[i:i + 8]
            print(f"    {', '.join(chunk)}")

    elif choice == "2":
        raw = input("  Symbols to add (comma-separated): ").strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        utils.add_to_watchlist(symbols)
        print(f"  ✅ Added: {', '.join(symbols)}")

    elif choice == "3":
        raw = input("  Symbols to remove (comma-separated): ").strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        utils.remove_from_watchlist(symbols)
        print(f"  ✅ Removed: {', '.join(symbols)}")

    elif choice == "4":
        utils.save_watchlist(config.DEFAULT_UNIVERSE[:20], "default")
        print("  ✅ Watchlist reset to defaults.")

    elif choice == "5":
        wls = utils.list_watchlists()
        print(f"\n  Available watchlists: {', '.join(wls) if wls else 'None'}")


# ─── CLI Dispatcher ──────────────────────────────────────────────────────────

def main():
    utils.setup_logging()

    if len(sys.argv) < 2:
        interactive_menu()
        return

    command = sys.argv[1].lower()

    if command == "analyze" and len(sys.argv) >= 3:
        cmd_full_analysis(sys.argv[2].upper())
    elif command == "technical" and len(sys.argv) >= 3:
        cmd_technical(sys.argv[2].upper())
    elif command == "fundamental" and len(sys.argv) >= 3:
        cmd_fundamental(sys.argv[2].upper())
    elif command == "trend" and len(sys.argv) >= 3:
        cmd_trend(sys.argv[2].upper())
    elif command == "screen":
        strategy = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_screen(strategy)
    elif command == "rank":
        cmd_rank()
    elif command == "compare" and len(sys.argv) >= 4:
        cmd_compare([s.upper() for s in sys.argv[2:]])
    elif command == "chart" and len(sys.argv) >= 3:
        cmd_chart(sys.argv[2].upper())
    else:
        print(__doc__)


if __name__ == "__main__":
    import pandas as pd
    main()
