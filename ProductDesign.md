# 📈 Stock Analyzer — Product Design Specification

**Version:** 2.0  
**Last Updated:** February 28, 2026  
**Author:** Stock Analyzer Team  
**Status:** Shipped ✅

---

## 1. Overview

### 1.1 Product Vision
Stock Analyzer is a **free, open-source stock analysis web application** that makes professional-grade investment analysis accessible to retail investors. It combines real-time market data, technical & fundamental analysis, AI-powered insights, and creative financial tools into a single, beautiful dashboard.

### 1.2 Problem Statement
Retail investors face several challenges:
- **Information overload** — data is scattered across dozens of platforms
- **Technical barrier** — most analysis tools require deep financial knowledge
- **Cost** — professional-grade platforms cost $20–300/month
- **AI gap** — modern AI insights are locked behind expensive paywalls

### 1.3 Target Users

| Persona | Description | Key Needs |
|---|---|---|
| **Beginner Investor** | New to stocks, learning the basics | Plain-English analysis, clear BUY/HOLD/SELL ratings, decision checklists |
| **Active Trader** | Trades weekly, follows technicals | RSI, MACD, Bollinger Bands, trend signals, multi-stock comparison |
| **Portfolio Manager** | Manages a personal portfolio | Portfolio tracking, P&L, allocation, sector breakdown |
| **Research Enthusiast** | Enjoys deep analysis, follows Buffett | Buffett portfolio analysis, What-If scenarios, AI Q&A |

### 1.4 Success Metrics

| Metric | Target |
|---|---|
| Stock universe coverage | All US-listed stocks (6,500+) ✅ |
| Data freshness | Real-time via Yahoo Finance ✅ |
| Page load time | < 3 seconds for cached data ✅ |
| AI response time | < 15 seconds (cloud) / < 60 seconds (local Ollama) ✅ |
| Mobile responsive | Usable on phone/tablet ✅ |
| Cost to user | Free (open-source) ✅ |

---

## 2. User Experience

### 2.1 Information Architecture

```
📈 Stock Analyzer
├── 🏠 Dashboard              — Watchlist overview + quick lookup
├── 🔍 Stock Analysis         — Deep single-stock analysis
│   ├── Investment Verdict    — Hero score card
│   ├── Price Chart           — Interactive Plotly candlestick
│   ├── Technical Signals     — RSI, MACD, Bollinger, ADX
│   ├── Fundamentals          — PE, PEG, margins, growth
│   ├── AI Insights           — Narrative + sentiment + Q&A
│   └── Decision Checklist    — Actionable pass/fail checks
├── 🌡️ Market Pulse           — Fear & Greed, sector heatmap
├── 📊 Screener               — Filter by strategy
├── 🏆 Rankings               — Score & rank watchlist
├── ⚖️ Compare (X-Ray)        — Multi-stock deep comparison
├── ⏳ What-If Machine         — Historical investment simulator
├── 💼 Portfolio Tracker       — Live P&L + allocation
├── 📋 Watchlist               — Manage tracked stocks
└── 🧓 Buffett Portfolio       — Warren Buffett's 13-F holdings
```

### 2.2 Navigation Model
- **Sidebar radio navigation** — always visible, single-click page switching
- **Cross-page linking** — click any stock card/pill/button → jumps to Stock Analysis
- **Session state persistence** — analysis stays active across widget interactions (reruns)
- **Responsive layout** — 4 CSS breakpoints: desktop (>1200px), tablet landscape (901–1200px), tablet portrait (601–900px), mobile (<600px)

### 2.3 Key User Flows

#### Flow 1: "I want to analyze a stock"
```
Dashboard → Type "apple" in search → Select "AAPL — Apple Inc." → 
Click "Analyze →" → Stock Analysis page loads with full report
```

#### Flow 2: "Should I buy this stock?"
```
Stock Analysis → View Investment Verdict (score, rating, plain-English factors) → 
Scroll to Decision Checklist → Review pass/fail items → 
Ask AI: "Is AAPL a good buy for retirement?"
```

#### Flow 3: "What if I invested earlier?"
```
What-If Machine → Type "Tesla" → Select TSLA → $10,000 → 5 years ago →
Calculate → See growth chart + final value + CAGR
```

#### Flow 4: "Compare my top picks"
```
Compare → Search & select AAPL, MSFT, GOOGL, NVDA → Compare →
View Returns / Risk / Correlation / Head-to-Head tabs
```

---

## 3. Feature Specifications

### 3.1 Smart Symbol Search

**Purpose:** Let users find any US stock without memorizing ticker symbols.

| Attribute | Spec |
|---|---|
| **Database** | 6,543 US-listed stocks (NASDAQ + NYSE + AMEX) |
| **Data source** | NASDAQ Screener API |
| **Refresh** | Auto-refresh when data is >30 days old |
| **Search modes** | Ticker prefix (`AA` → AAPL, AAL…), company name substring (`tesla` → TSLA) |
| **Ranking** | Exact match → ticker prefix → name-word-boundary → name-contains |
| **Result limit** | 12 matches shown |
| **Fallback** | Unknown symbols passed directly to Yahoo Finance |
| **Placement** | Dashboard, Stock Analysis, What-If, Portfolio, Compare, Watchlist |

### 3.2 Stock Analysis Page

**Purpose:** Comprehensive single-stock analysis in one scrollable page.

#### 3.2.1 Investment Verdict (Hero Section)
- Large score circle with color coding (green ≥60, yellow 40–59, red <40)
- Rating badge (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- 4 sub-scores: Technical, Fundamental, Trend, Momentum
- 4 plain-English factors (top reasons for the rating)

#### 3.2.2 Tabs
| Tab | Content |
|---|---|
| **📈 Chart** | Plotly candlestick + volume + SMA20/50/200, interactive zoom |
| **📊 Technicals** | RSI gauge, MACD signal, Bollinger position, ADX strength, trend MA |
| **💰 Fundamentals** | P/E, PEG, margins, growth rates, debt, ROE, analyst targets |
| **🤖 AI Insights** | AI narrative analysis + news sentiment + interactive Q&A |
| **📰 News** | Recent headlines from Yahoo Finance with links |

#### 3.2.3 Decision Checklist
- Score above 60 (Buy zone)
- Price in uptrend
- RSI not overbought (< 70)
- Above 200-day MA
- Positive revenue growth
- Analyst upside potential

### 3.3 AI/LLM Engine

**Purpose:** Augment data-driven analysis with natural language insights.

#### 3.3.1 Architecture
```
User Action → build_stock_context() → call_llm() → OpenAI SDK → Provider
                                                         ↓
                                              Ollama / Groq / OpenAI / Any compatible
```

#### 3.3.2 Configuration Priority
1. Session state (sidebar widget input)
2. Saved settings file (`data/ai_settings.json`)
3. Streamlit Secrets (`secrets.toml`)
4. Environment variable (`OPENAI_API_KEY`)

#### 3.3.3 Features
| Feature | Prompt | Max Tokens |
|---|---|---|
| **AI Narrative** | System: world-class analyst + User: full stock context JSON | 1500 |
| **News Sentiment** | System: financial news analyst + User: headline list | 1500 |
| **Q&A** | System: analyst + User: stock context + user question | 800 |
| **Test Connection** | "Respond with: CONNECTION OK" | 20 |

#### 3.3.4 Provider Compatibility
| Provider | Base URL | Free? |
|---|---|---|
| Ollama (local) | `http://127.0.0.1:11434` | ✅ |
| Groq | `https://api.groq.com/openai` | ✅ |
| OpenAI | (default) | Paid |
| OpenRouter | `https://openrouter.ai/api` | Free tier |
| Together AI | `https://api.together.xyz` | Free tier |
| Azure OpenAI | Custom endpoint | Paid |

#### 3.3.5 URL Normalization
- Auto-appends `/v1` if missing
- Replaces `localhost` → `127.0.0.1` (Windows IPv6 workaround)

### 3.4 Market Pulse

**Purpose:** At-a-glance market health dashboard.

| Component | Description |
|---|---|
| **Fear & Greed Gauge** | Plotly gauge based on VIX level mapping |
| **Market Breadth** | % of S&P 500 stocks above SMA200 |
| **Sector Heatmap** | 11 sector ETF daily performance heatmap |
| **Sector Bar Chart** | Period returns for all sector ETFs |

### 3.5 X-Ray Compare

**Purpose:** Deep multi-stock comparison beyond simple price charts.

| Tab | Content |
|---|---|
| **Returns** | Normalized cumulative return overlay chart |
| **Risk & Volatility** | Annualized volatility, Sharpe ratio, max drawdown, risk-return scatter |
| **Correlation** | Correlation matrix heatmap |
| **Head-to-Head** | Side-by-side metrics table + download CSV |

### 3.6 What-If Time Machine

**Purpose:** Satisfy "what if I had invested…" curiosity with real data.

| Input | Options |
|---|---|
| Stock | Any US stock (via symbol search) |
| Amount | $100 – $10,000,000 |
| Years ago | 1, 2, 3, 5, 10, 15, 20 |

**Output:** Starting value, ending value, total return %, CAGR, max drawdown, interactive growth chart.

### 3.7 Portfolio Tracker

**Purpose:** Track a personal stock portfolio with live market data.

| Feature | Description |
|---|---|
| **Holdings table** | Symbol, shares, avg cost, current price, market value, P&L, % change |
| **Summary metrics** | Total invested, current value, total gain/loss |
| **Allocation chart** | Pie chart by position weight |
| **Add/remove** | Edit holdings in expandable form |
| **CSV download** | Export portfolio to CSV |

### 3.8 Buffett Portfolio

**Purpose:** Learn from Warren Buffett's investment approach.

| Component | Description |
|---|---|
| **Top 20 holdings** | Latest 13-F filing data with sector, weight, value |
| **Sector breakdown** | Pie chart of Berkshire's portfolio allocation |
| **Investment principles** | Buffett's key investing rules |
| **Quick analyze** | Click any holding → jump to full Stock Analysis |

---

## 4. Technical Architecture

### 4.1 System Diagram
```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                       │
│                     (app.py, ~2800 LOC)                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │Dashboard │  │Analysis  │  │Compare   │  │Portfolio │ │
│  │          │  │          │  │          │  │          │ │
│  │  + 6     │  │  AI      │  │  X-Ray   │  │  Tracker │ │
│  │  more    │  │  Engine   │  │          │  │          │ │
│  │  pages   │  │          │  │          │  │          │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│       ↓              ↓             ↓             ↓       │
│  ┌───────────────────────────────────────────────────┐   │
│  │              src/ modules (analysis engine)       │   │
│  │  data_fetcher · technical · fundamental · trend   │   │
│  │  screener · scorer · visualizer · utils           │   │
│  └───────────────────────────────────────────────────┘   │
│       ↓              ↓                                   │
│  ┌──────────┐  ┌──────────────┐                          │
│  │  Yahoo   │  │  AI Provider │                          │
│  │  Finance │  │  (optional)  │                          │
│  │  API     │  │  Ollama/Groq │                          │
│  └──────────┘  └──────────────┘                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow
1. **Data Fetcher** — Pulls price history + company info from Yahoo Finance with `@st.cache_data` (15-minute TTL)
2. **Analysis Modules** — Compute technical indicators, fundamental metrics, trend signals, composite score
3. **AI Engine** (optional) — Builds rich JSON context from analysis data, sends to LLM, returns Markdown narrative
4. **UI Layer** — Renders analysis results as Plotly charts, metrics, tables, and styled HTML cards

### 4.3 Caching Strategy
| Layer | TTL | Purpose |
|---|---|---|
| `@st.cache_data` (history) | 15 min | Avoid redundant Yahoo Finance API calls |
| `@st.cache_data` (info) | 15 min | Company info caching |
| `@st.cache_data` (symbols) | 24 hours | Stock symbol database |
| Session state | Per session | AI responses, user inputs, navigation state |
| `ai_settings.json` | Persistent | Saved API key/URL/model across sessions |

### 4.4 State Management
- **`st.session_state`** for all transient UI state (current symbol, active analysis, AI cache, portfolio holdings)
- **`_analysis_active` flag** — keeps the Stock Analysis page rendered across button-triggered reruns
- **`nav_to` + `analyze_symbol`** — cross-page navigation pattern for clickable stock cards
- **File-based persistence** for watchlists (`data/watchlists/`), AI settings (`data/ai_settings.json`)

---

## 5. Responsive Design

### 5.1 Breakpoints
| Breakpoint | Target | Layout Adaptation |
|---|---|---|
| **> 1200px** | Desktop | Full multi-column layout, large charts |
| **901–1200px** | Tablet landscape | 2-column grid, compressed metrics |
| **601–900px** | Tablet portrait | Single column, stacked cards |
| **< 600px** | Mobile | Full-width cards, hidden sidebar by default |

### 5.2 Mobile Optimizations
- Touch-friendly button sizes (min 44px)
- Stacked metric cards instead of 5-column rows
- Readable font sizes (min 14px body)
- Plotly charts auto-resize to container width

---

## 6. Security & Privacy

| Concern | Approach |
|---|---|
| **API keys** | Stored in session state (ephemeral) or local file (base64-encoded); never sent to our servers |
| **Secrets in repo** | `.gitignore` excludes `secrets.toml` and `ai_settings.json` |
| **User data** | No user accounts, no telemetry, no cookies beyond Streamlit defaults |
| **Data source** | Yahoo Finance public API — no premium data leakage concerns |

---

## 7. Deployment

### 7.1 Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 7.2 Streamlit Community Cloud
1. Push repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set `app.py` as main file
4. (Optional) Add AI secrets in dashboard → Settings → Secrets

### 7.3 Environment Requirements
| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Streamlit | 1.30+ |
| yfinance | 0.2.31+ |
| plotly | 5.18+ |
| openai (optional) | 1.0+ |

---

## 8. Known Limitations & Future Roadmap

### 8.1 Current Limitations
- US stocks only (by design — focused scope)
- Yahoo Finance rate limits may affect bulk operations (screener, rankings)
- AI features require user-provided API key (no built-in key)
- Portfolio holdings are session-based (reset on close unless page is refreshed)
- What-If Machine limited to Yahoo Finance history depth (~20 years max)

### 8.2 Potential Future Enhancements
| Priority | Feature | Description |
|---|---|---|
| P1 | **Real-time alerts** | Price/indicator alerts via email or push |
| P1 | **Persistent portfolio** | Save portfolio holdings to file/database |
| P2 | **Options analysis** | Options chain viewer with implied volatility |
| P2 | **Earnings calendar** | Upcoming earnings dates for watchlist stocks |
| P2 | **Backtesting engine** | Test trading strategies on historical data |
| P3 | **Social sentiment** | Reddit/X sentiment analysis integration |
| P3 | **International stocks** | Expand beyond US markets |
| P3 | **Dark mode** | User-selectable dark theme |

---

## 9. Appendix

### 9.1 Scoring Algorithm Weights
```
Overall Score = (Technical × 0.35) + (Fundamental × 0.30) + (Trend × 0.20) + (Momentum × 0.15)
```

### 9.2 Technical Indicators Used
RSI (14), MACD (12/26/9), SMA (20/50/200), EMA (12/26), Bollinger Bands (20,2), Stochastic (14,3), ADX (14), OBV, VWAP, ATR (14)

### 9.3 Screening Strategies
| Strategy | Criteria |
|---|---|
| **Growth** | Revenue growth > 10%, uptrend, above SMA200 |
| **Value** | P/E < 20, ROE > 15%, D/E < 1, positive FCF |
| **Momentum** | RSI 40–70, bullish MACD, strong ADX |
| **Dividend** | Stable sector, positive cash flow, consistent earnings |
