# 📈 Stock Analyzer

A focused, reliable stock analysis tool powered by **Yahoo Finance** real-time data.

## Features

| Feature | Description |
|---|---|
| **Full Analysis** | Complete stock report combining all analysis modules |
| **Technical Analysis** | SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, VWAP, ATR |
| **Fundamental Analysis** | Valuation, profitability, growth, financial health scoring |
| **Trend Analysis** | Multi-factor trend classification with support/resistance levels |
| **Stock Screener** | Filter stocks by growth, value, momentum, or dividend strategies |
| **Stock Scoring** | Weighted composite score (0-100) with BUY/HOLD/SELL ratings |
| **Comparison** | Side-by-side stock comparison with normalized return charts |
| **Visualization** | Professional multi-panel charts with indicators |
| **Watchlist** | Save and manage custom watchlists |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run interactive mode
python main.py

# 3. Or use CLI commands directly
python main.py analyze AAPL
python main.py trend MSFT
python main.py screen growth
python main.py rank
python main.py compare AAPL MSFT GOOGL
python main.py chart NVDA
```

## CLI Commands

| Command | Example | Description |
|---|---|---|
| `analyze` | `python main.py analyze AAPL` | Full analysis report |
| `technical` | `python main.py technical MSFT` | Technical indicators & signals |
| `fundamental` | `python main.py fundamental GOOGL` | Fundamental metrics & scoring |
| `trend` | `python main.py trend TSLA` | Trend direction & support/resistance |
| `screen` | `python main.py screen growth` | Screen stocks (growth/value/momentum/dividend) |
| `rank` | `python main.py rank` | Score & rank stocks from watchlist |
| `compare` | `python main.py compare AAPL MSFT` | Compare multiple stocks |
| `chart` | `python main.py chart NVDA` | Generate analysis chart |

## Scoring System

The investment score combines four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Technical (35%) | Indicators, signals, chart patterns |
| Fundamental (30%) | Valuation, profitability, growth, health |
| Trend (20%) | Trend direction and strength |
| Momentum (15%) | Price momentum across timeframes |

**Ratings:**
- **≥ 75** → STRONG BUY
- **60–74** → BUY
- **40–59** → HOLD
- **25–39** → SELL
- **< 25** → STRONG SELL

## Screening Strategies

- **Growth** – High revenue growth + uptrend + above SMA200
- **Value** – Low P/E + strong ROE + low debt + positive FCF
- **Momentum** – Technical momentum + bullish MACD + RSI sweet spot
- **Dividend** – Stable companies with positive cash flow

## Project Structure

```
stock-analyzer/
├── main.py                   # CLI entry point & interactive menu
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── src/
│   ├── data_fetcher.py       # Yahoo Finance data with caching
│   ├── technical_analysis.py # 10+ technical indicators
│   ├── fundamental_analysis.py # Valuation & financial scoring
│   ├── trend_analyzer.py     # Multi-factor trend classification
│   ├── stock_screener.py     # Configurable stock screening
│   ├── stock_scorer.py       # Composite investment scoring
│   ├── visualizer.py         # Chart generation
│   └── utils.py              # Helpers, watchlist, formatting
├── data/watchlists/          # Saved watchlists
└── output/                   # Generated charts & reports
```

## Disclaimer

This tool is for **educational and research purposes only**. It is not financial advice. Always do your own research and consult a financial advisor before making investment decisions.
