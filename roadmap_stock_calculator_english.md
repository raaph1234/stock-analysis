# Customized Roadmap for the Stock Calculator

## V12.3 — Reliable News Filter + No-News-Neutral Scoring

**Goal:** only use news that is truly about the company, and make sure that no news does not lower the score.

### Key Rules

- No news = no impact on `FinalScore`.
- Weak or uncertain news = little or no impact.
- Only reliable and relevant news may influence the score.
- Generic clickbait articles are filtered out or heavily penalized.
- Short tickers such as `V`, `MA`, `GE`, `CAT`, and `KO` are checked extra strictly.

### Add/Fix

- Add `NewsConfidence`.
- Add `EffectiveNewsWeight`.
- Add `NewsStatus`.
- No news: `FinalScore = Total`.
- Poor news match: news does not count.
- Reliable news: news counts only in a limited way through `EffectiveNewsWeight`.
- Improve source quality scoring.
- Remove duplicates better.
- Score generic “top stocks to buy” articles lower.
- Score ETF/market overview articles lower.
- Skip articles without a clear company link.
- Use `ClickbaitPenalty`.
- Make `CompanyMatch` requirement more important.
- Add source quality:
  - Reuters / AP / Bloomberg / CNBC / company press release = higher score.
  - Random blog / generic stock-picking site = lower score.

### Code Cleanup Within V12.3

- Remove duplicate `make_excel_safe()`.
- Simplify `compute_news_confidence()`.
- Clearly label the backtest as “quant-only backtest”.
- Log `risk_parity` + long/short fallback.
- Make news-summary columns consistent.
- Expand the Settings sheet with an explanation of news scoring.
- Make logging a bit calmer: small missing data issues = info/debug, not always warning.

## V12.4 — Cleanly Integrate News Score per Mode

**Goal:** news counts toward the ranking, but intelligently and differently per strategy.

### Add

- Mode-specific news weights.
- Aggressive:
  - News and momentum count more heavily.
- Balanced:
  - News counts normally.
- Conservative:
  - Negative news counts more heavily than positive news.
- Dividend:
  - Dividend, cash-flow, and earnings news counts more heavily.
- Extra columns:
  - `QuantRank`
  - `NewsRank`
  - `FinalRank`
  - `RankReason`
- Option to choose:
  - Ranking by `FinalScore`
  - Ranking by `Total`
  - Ranking by `NewsScore`

### Important

The backtest does not fully use news for now, because free news data is not reliably point-in-time historical.

## V12.5 — News Cache and Performance

**Goal:** fetch news faster and make fewer API/RSS calls.

### Add

- News cache per ticker.
- Cache key based on ticker + lookback days.
- `--no-cache`
- `--clear-cache`
- `--cache-ttl-hours`
- Max cache files or cache cleanup.
- Parallel news fetching with `ThreadPoolExecutor`.
- Rate-limit protection.
- Better error messages when Google RSS or yfinance news fails.
- V12.5 — Universe scanner for many tickers to determine which one is best.

### Why

For 40+ tickers, fetching news sequentially becomes slow. With parallel fetching and caching, the program becomes much faster.

## V13 — AI Analysis of News

**Goal:** not only score news, but also explain what it means.

## V13.1 — AI Summary per Stock

Per ticker:

- Positive news
- Negative news
- Neutral news
- Risks
- Short conclusion
- `AIScore`

Excel sheet:

- `ai_summary`

Columns:

- `Ticker`
- `PositiveSummary`
- `NegativeSummary`
- `NeutralSummary`
- `RiskSummary`
- `Conclusion`
- `AIScore`

## V13.2 — Smarter Sentiment/Relevance

**Goal:** become less dependent on simple keywords.

### Possibilities

- Keep using VADER as a lightweight base.
- Add FinBERT for financial sentiment analysis.
- Add a relevance model: is this article really about this company?
- Possibly use a local AI model if your laptop can handle it.

Optional install later:

```bash
pip install transformers torch
```

But only do this after V12 is stable.

## V13.3 — Expand Impact Classification

**Goal:** classify news better.

### Categories

- Earnings
- Guidance
- Contract
- Lawsuit
- Regulation
- Analyst upgrade
- Analyst downgrade
- Insider selling
- Dividend
- Buyback
- Merger/acquisition
- Geopolitical risk
- Product launch
- AI demand
- Valuation concern

## V14 — Better Data and Factor Scoring

**Goal:** make the quant score more accurate.

### Add

- FCF Yield
- ROIC
- Free Cash Flow Margin
- Operating Margin
- EBITDA Margin
- Gross Margin
- Cash Conversion
- Debt Coverage
- Interest Coverage
- Revenue Growth Acceleration
- Earnings Growth

### Improve

- Handle negative P/E better.
- Automatically penalize PEG above an extreme level.
- Handle debt/equity outliers better.
- Improve sector-neutral scoring.
- Add market-cap bucket comparison.
- Compare mega-cap tech with mega-cap tech.
- Compare defense companies with defense companies.
- Compare dividend companies with dividend companies.

## V15 — Database, Snapshots, and Better Backtest

**Goal:** reduce look-ahead bias and improve performance measurement.

### Add

- SQLite database.
- Save every run as a snapshot.
- Save historical quant scores.
- Save historical news scores.
- Save historical rankings.
- Save historical portfolio weights.
- Save historical news items.

Database tables:

- `runs`
- `ticker_scores`
- `news_items`
- `portfolio_weights`
- `backtest_results`
- `cache_metadata`

### Improve Backtest

- CAGR
- Sortino ratio
- Calmar ratio
- Drawdown duration
- Rolling Sharpe
- Best month
- Worst month
- Capture ratio
- Information ratio
- Beta vs SPY
- Comparison with QQQ
- Comparison with sector ETF

### Important

Yfinance fundamentals are current snapshots. This means historical backtests can contain look-ahead bias. Your own snapshots help build fairer historical data from now on.

## V16 — Alerts

**Goal:** automatically receive warnings when something important changes.

### Alerts

- Total score above 70.
- `FinalScore` rises sharply.
- News sentiment becomes strongly negative.
- Stock drops more than 10%.
- Volatility suddenly rises.
- Earnings within 7 days.
- Large contract reported.
- Analyst downgrade.
- Lawsuit / investigation.
- `NewsConfidence` is high and `NewsScore` is strongly negative.

### Channels

- Email
- Discord webhook
- Telegram bot
- Windows desktop notification
- Excel alert sheet

## V17 — Portfolio Monitor

**Goal:** monitor your own positions.

### Input

- Ticker
- Number of shares
- Average purchase price
- Target weight

### Output

- Portfolio return
- Sector exposure
- Risk exposure
- Drawdown
- Correlation
- News risk
- Rebalance advice

## V18 — Visual Interface

**Goal:** create a dashboard.

Recommended:

```bash
pip install streamlit plotly
```

### Tabs

- Dashboard
- Stocks
- News
- Portfolio
- Backtest
- Settings
- Alerts
- Database

## V19 — ML Model

**Goal:** predict whether a stock will outperform SPY over the next 30 days.

### Features

- Quant score
- News score
- Momentum
- Volatility
- Drawdown
- PE
- PEG
- ROE
- Revenue growth
- Sector
- Market cap
- News sentiment
- Regime

### Models

- LogisticRegression
- RandomForest
- XGBoost
- LightGBM

Only do this after V15 has saved enough historical snapshots.
