# 📈 Stock Analyzer

A professional stock analysis web app powered by **Yahoo Finance** real-time data, **Plotly** interactive charts, and optional **AI/LLM** insights. Covers all **6,500+ US-listed stocks** with fuzzy search, portfolio tracking, and creative analysis tools.

🌐 **Live Demo:** [stock-analyzer on Streamlit Cloud](https://accetao-stock-analyzer.streamlit.app)

---

## ✨ Features

### 📊 Core Analysis
| Feature | Description |
|---|---|
| **Stock Analysis** | Deep-dive into any US stock — technicals, fundamentals, trend, scoring |
| **Technical Analysis** | SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, VWAP, ATR |
| **Fundamental Analysis** | Valuation, profitability, growth, financial health scoring |
| **Trend Analysis** | Multi-factor trend classification with support/resistance levels |
| **Investment Score** | Weighted composite score (0–100) with BUY/HOLD/SELL ratings |

### 🧠 AI-Powered Insights (Optional)
| Feature | Description |
|---|---|
| **AI Narrative** | Deep AI-written analysis of any stock using live data context |
| **News Sentiment** | AI-scored sentiment analysis of recent news headlines |
| **Ask AI Q&A** | Ask free-form questions about any stock you're viewing |
| **Multi-Provider** | Works with OpenAI, Groq (free), Ollama (local), OpenRouter, and any OpenAI-compatible API |
| **Persistent Settings** | Save your API key locally — no re-entry on refresh |

### 🎨 Creative Tools
| Feature | Description |
|---|---|
| **🌡️ Market Pulse** | Fear & Greed gauge, sector heatmap, market breadth |
| **⏳ What-If Machine** | "What if I invested $10K in Tesla 5 years ago?" time-travel simulator |
| **⚖️ X-Ray Compare** | Multi-stock comparison — returns, risk, correlation, drawdowns |
| **💼 Portfolio Tracker** | Track holdings with live P&L, allocation charts, total return |
| **📋 Watchlist** | Save and manage custom watchlists |
| **🧓 Buffett Portfolio** | Explore Warren Buffett's 20 largest 13-F holdings |
| **📊 Screener** | Filter stocks by growth, value, momentum, or dividend strategies |
| **🏆 Rankings** | Score and rank stocks from your watchlist |

### 🔍 Smart Symbol Search
| Feature | Description |
|---|---|
| **6,500+ US Stocks** | Full NASDAQ + NYSE + AMEX coverage |
| **Fuzzy Search** | Type ticker prefix or company name (e.g. `tesla`, `semi`, `AA`) |
| **Auto-Refresh** | Stock list refreshes from NASDAQ API every 30 days |
| **Available Everywhere** | Search widget on Dashboard, Analysis, Compare, What-If, Portfolio, Watchlist |

---

## 🚀 Quick Start

### Web UI (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web dashboard
streamlit run app.py
```

### CLI (Alternative)
```bash
python main.py                        # Interactive menu
python main.py analyze AAPL           # Full analysis report
python main.py screen growth          # Screen for growth stocks
python main.py compare AAPL MSFT GOOGL
```

---

## 🤖 AI Setup (Optional)

AI features are optional — the app works fully without them. To enable:

### Option A: Ollama (Free, Local, Private)
```bash
# Install Ollama → https://ollama.com
ollama pull llama3.2
# The app auto-detects Ollama and configures itself
```

### Option B: Groq (Free Cloud API)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. In the app sidebar → 🤖 AI Settings → enter key + set Base URL to `https://api.groq.com/openai`

### Option C: OpenAI
1. Get a key at [platform.openai.com](https://platform.openai.com/api-keys)
2. In the app sidebar → 🤖 AI Settings → enter key (no Base URL needed)

### Cloud Deployment (Streamlit Secrets)
Add to your Streamlit Cloud dashboard → Settings → Secrets:
```toml
OPENAI_API_KEY  = "gsk_your_key_here"
OPENAI_BASE_URL = "https://api.groq.com/openai"
OPENAI_MODEL    = "llama-3.3-70b-versatile"
```

---

## 📐 Scoring System

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

---

## 📁 Project Structure

```
stock-analyzer/
├── app.py                        # Streamlit web UI (main app, ~2800 lines)
├── main.py                       # CLI entry point & interactive menu
├── config.py                     # Central configuration
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── secrets.toml.example      # Template for cloud AI secrets
├── src/
│   ├── data_fetcher.py           # Yahoo Finance data with caching
│   ├── technical_analysis.py     # 10+ technical indicators
│   ├── fundamental_analysis.py   # Valuation & financial scoring
│   ├── trend_analyzer.py         # Multi-factor trend classification
│   ├── stock_screener.py         # Configurable stock screening
│   ├── stock_scorer.py           # Composite investment scoring
│   ├── visualizer.py             # Chart generation (CLI)
│   └── utils.py                  # Helpers, watchlist, formatting
├── data/
│   ├── stock_symbols.json        # 6,500+ US stock tickers (auto-refreshed)
│   └── watchlists/               # Saved watchlists
└── output/                       # Generated charts & reports
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Data** | Yahoo Finance (via yfinance) |
| **Backend** | Python 3.12 |
| **Web UI** | Streamlit |
| **Charts** | Plotly (interactive) + Matplotlib (CLI) |
| **AI** | OpenAI SDK (compatible with Ollama, Groq, OpenRouter) |
| **Stock Database** | NASDAQ API (auto-refreshed) |
| **Deployment** | Streamlit Community Cloud |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not financial advice. Always do your own research and consult a financial advisor before making investment decisions.
