#!/usr/bin/env python3
"""
stock_screener_v11_fixed.py

Hedge Fund Lite V12.3 - single-file quant screener/backtester.

Features:
- Interactieve tickers als je geen --tickers opgeeft.
- CSV-watchlist support.
- yfinance data fetching + cache.
- Vectorized momentum, volatility, beta, drawdown.
- Multi-factor scoring:
    Value
    Quality
    Growth
    Risk
    Stability
    Dividend
    Momentum
    Drawdown
- Confidence-adjusted score.
- Volume filter + market cap filter.
- Sector caps in optimizer.
- Regime filter:
    bull_market
    neutral_market
    correction
    crash_risk
- Objectives:
    sharpe
    minvar
    risk_parity
- Optional GPU covariance via CuPy, only imported if --gpu is used.
- Excel output with multiple sheets:
    settings
    universe_snapshot
    top_snapshot
    raw_data
    warnings
    equal_metrics
    optimized_metrics
    benchmark_metrics
    comparison
    equal_equity_curve
    opt_equity_curve
    benchmark_equity_curve
    weights

Install:
    pip install yfinance pandas numpy scipy scikit-learn openpyxl

Optional GPU:
    pip install cupy-cuda12x

Examples:
    python stock_screener_v11_fixed.py
    python stock_screener_v11_fixed.py --tickers ASML,KTOS,LHX,PLTR,AVAV
    python stock_screener_v11_fixed.py --watchlist watchlist.csv
    python stock_screener_v11_fixed.py --tickers AAPL,MSFT,NVDA --mode balanced --objective sharpe --top-n 3
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
import re
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.optimize import minimize  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import feedparser  # type: ignore
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

TRADING_DAYS_PER_YEAR = 252.0

WEIGHTS = {
    "aggressive": {
        "value": 0.12,
        "quality": 0.16,
        "growth": 0.24,
        "risk": 0.12,
        "stability": 0.06,
        "dividend": 0.04,
        "momentum": 0.18,
        "drawdown": 0.08,
    },
    "balanced": {
        "value": 0.17,
        "quality": 0.18,
        "growth": 0.14,
        "risk": 0.14,
        "stability": 0.10,
        "dividend": 0.08,
        "momentum": 0.11,
        "drawdown": 0.08,
    },
    "conservative": {
        "value": 0.20,
        "quality": 0.22,
        "growth": 0.05,
        "risk": 0.18,
        "stability": 0.15,
        "dividend": 0.07,
        "momentum": 0.03,
        "drawdown": 0.10,
    },
    "dividend": {
        "value": 0.12,
        "quality": 0.14,
        "growth": 0.04,
        "risk": 0.10,
        "stability": 0.14,
        "dividend": 0.34,
        "momentum": 0.04,
        "drawdown": 0.08,
    },
}

FIELD_CONFIG = {
    "PE": {
        "scale": 1,
        "anchors": ([0, 10, 15, 20, 25, 35, 50, 100], [100, 85, 70, 55, 40, 20, 5, 0]),
    },
    "FPE": {
        "scale": 1,
        "anchors": ([0, 10, 15, 20, 25, 35, 50, 100], [100, 85, 70, 55, 40, 20, 5, 0]),
    },
    "PEG": {
        "scale": 1,
        "anchors": ([0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], [100, 90, 70, 50, 30, 10, 0]),
    },
    "ROE": {
        "scale": 100,
        "anchors": ([0, 5, 10, 15, 20, 30, 40], [20, 35, 50, 65, 80, 95, 100]),
    },
    "ROA": {
        "scale": 100,
        "anchors": ([0, 2, 5, 10, 15, 20], [20, 35, 50, 70, 85, 100]),
    },
    "Beta": {
        "scale": 1,
        "anchors": ([0, 0.5, 1.0, 1.5, 2.0, 3.0], [100, 85, 65, 45, 25, 0]),
    },
    "DebtEquity": {
        "scale": 1,
        "anchors": ([-100, -50, 0, 50, 100, 150, 300], [100, 95, 80, 60, 40, 20, 0]),
    },
    "RevenueGrowth": {
        "scale": 100,
        "anchors": ([-50, -10, 0, 5, 10, 20, 40, 60], [0, 10, 30, 50, 65, 80, 95, 100]),
    },
    "ProfitMargin": {
        "scale": 100,
        "anchors": ([-50, -10, 0, 5, 10, 20, 30], [0, 10, 30, 50, 65, 85, 100]),
    },
    "DividendYield": {
        "scale": 100,
        "anchors": ([0, 1, 2, 3, 4, 6, 8], [10, 30, 50, 65, 80, 95, 100]),
    },
    "Momentum": {
        "scale": 100,
        "anchors": ([-30, -10, 0, 10, 20, 40, 60], [0, 30, 50, 65, 80, 95, 100]),
    },
    "Volatility": {
        "scale": 1,
        "anchors": ([5, 10, 20, 30, 40, 60, 80, 100], [100, 90, 80, 60, 40, 20, 10, 0]),
    },
    "MaxDrawdown": {
        "scale": 1,
        "anchors": ([0, 5, 10, 20, 35, 50, 70, 90], [100, 95, 85, 65, 40, 20, 5, 0]),
    },
    "AverageVolume": {
        "scale": 1,
        "anchors": ([0, 100_000, 500_000, 1_000_000, 5_000_000, 20_000_000], [0, 20, 50, 70, 90, 100]),
    },
}

FACTOR_FIELDS = {
    "Value": ["PE", "FPE", "PEG"],
    "Quality": ["ROE", "ROA"],
    "Growth": ["RevenueGrowth", "ProfitMargin"],
    "Risk": ["Beta", "Volatility"],
    "Stability": ["DebtEquity"],
    "Dividend": ["DividendYield"],
    "Momentum": ["Momentum"],
    "Drawdown": ["MaxDrawdown"],
}

LOWER_IS_BETTER = {
    "PE",
    "FPE",
    "PEG",
    "Beta",
    "DebtEquity",
    "Volatility",
    "MaxDrawdown",
}

BENCHMARK_TICKER = "SPY"
CACHE_DIR = "cache"
CACHE_VERSION = "v12_3_schema_1"
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_HISTORY_PERIOD = "3y"
REQUEST_DELAY = 0.3
WINSOR_PCT = (0.01, 0.99)

# ============================================================
# V12 NEWS CONFIG
# ============================================================

NEWS_DEFAULT_LOOKBACK_DAYS = 14
NEWS_DEFAULT_MAX_ITEMS = 12
NEWS_DEFAULT_WEIGHT = 0.15
NEWS_MIN_RELEVANCE = 50

COMPANY_ALIASES = {
    "ASML": ["ASML", "ASML Holding"],
    "KTOS": ["KTOS", "Kratos", "Kratos Defense", "Kratos Defense & Security"],
    "LHX": ["LHX", "L3Harris", "L3Harris Technologies"],
    "PLTR": ["PLTR", "Palantir", "Palantir Technologies"],
    "AVAV": ["AVAV", "AeroVironment", "AeroVironment Inc"],
}

NEWS_POSITIVE_KEYWORDS = {
    "beat", "beats", "upgrade", "upgraded", "outperform", "buy rating",
    "contract", "award", "awarded", "order", "partnership", "raises guidance",
    "record revenue", "profit rises", "revenue rises", "earnings beat",
    "expands", "wins", "approval", "strong demand", "dividend increase",
    "buyback", "share repurchase",
}

NEWS_NEGATIVE_KEYWORDS = {
    "miss", "misses", "downgrade", "downgraded", "underperform", "sell rating",
    "lawsuit", "investigation", "probe", "fraud", "cuts guidance",
    "lower guidance", "warning", "recall", "sanction", "export restriction",
    "delay", "delayed", "declines", "falls", "loss widens", "short seller",
    "bankruptcy", "layoffs",
}

NEWS_IMPORTANT_KEYWORDS = {
    "earnings", "guidance", "contract", "order", "merger", "acquisition",
    "lawsuit", "investigation", "regulation", "export", "sanction",
    "analyst", "upgrade", "downgrade", "dividend", "buyback", "revenue",
    "profit", "forecast", "deal", "partnership", "defense", "ai",
}

NEWS_JUNK_KEYWORDS = {
    "best stocks to buy", "top stocks", "watchlist", "market today",
    "stock market news", "why stocks are moving", "premarket",
    "after hours", "etf", "mutual fund", "crypto", "forex",
    "options trading", "technical analysis", "zacks rank",
}

# ============================================================
# V12.2 NEWS QUALITY CONFIG
# ============================================================

NEWS_HIGH_QUALITY_SOURCES = {
    "reuters",
    "associated press",
    "ap news",
    "bloomberg",
    "cnbc",
    "marketwatch",
    "wall street journal",
    "wsj",
    "financial times",
    "ft",
    "barron's",
    "barrons",
    "seeking alpha",
    "the motley fool",
    "investopedia",
    "yahoo finance",
    "globenewswire",
    "business wire",
    "pr newswire",
    "company press release",
}

NEWS_LOW_QUALITY_SOURCES = {
    "weex",
    "simply wall st",
    "zacks",
    "benzinga",
    "insider monkey",
    "stocknews.com",
    "gurufocus",
    "marketbeat",
}

NEWS_CLICKBAIT_PATTERNS = {
    "is a good stock to buy",
    "is it too late to buy",
    "should you buy",
    "should you sell",
    "best stocks to buy",
    "top stocks to buy",
    "stock to watch",
    "stocks to watch",
    "why shares are",
    "why stock is",
    "what you should know",
    "trending stock",
    "millionaire maker",
    "could make you rich",
    "forget nvidia",
    "forget tesla",
    "buy now",
}

NEWS_SHORT_TICKERS_REQUIRE_COMPANY = {
    "V",
    "MA",
    "GE",
    "CAT",
    "KO",
    "PG",
    "HD",
    "T",
    "F",
    "C",
}

NEWS_IMPACT_KEYWORDS_BY_CATEGORY = {
    "earnings": {
        "earnings", "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
        "eps", "revenue beat", "earnings beat", "missed estimates", "guidance",
    },
    "contract": {
        "contract", "award", "awarded", "deal", "order", "partnership", "supplier",
        "selected by", "wins", "framework agreement",
    },
    "analyst": {
        "upgrade", "upgraded", "downgrade", "downgraded", "price target",
        "buy rating", "sell rating", "outperform", "underperform", "initiates coverage",
    },
    "regulation": {
        "regulation", "regulatory", "export restriction", "sanction", "antitrust",
        "ftc", "sec", "doj", "eu commission", "china export",
    },
    "legal": {
        "lawsuit", "sues", "investigation", "probe", "fraud", "settlement",
        "class action", "legal challenge",
    },
    "capital_return": {
        "dividend", "buyback", "share repurchase", "capital return", "raises dividend",
    },
    "macro_geopolitical": {
        "war", "conflict", "tariff", "oil price", "interest rates", "inflation",
        "geopolitical", "defense spending",
    },
    "product_ai": {
        "ai", "artificial intelligence", "chip", "semiconductor", "data center",
        "product launch", "cloud", "gpu", "model", "software platform",
    },
}

# ============================================================
# V12.3 NEWS FILTER CONFIG
# ============================================================

NEWS_MIN_ITEMS_FOR_FULL_WEIGHT = 3
NEWS_MIN_CONFIDENCE_FOR_EFFECT = 0.25
NEWS_DEFAULT_NO_NEWS_SCORE = 50.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger("stock_screener_v12_3")

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def clamp_float(value, low: float = 0.0, high: float = 1.0) -> float:
    val = safe_float(value)

    if np.isnan(val):
        return low

    return float(max(low, min(high, val)))


def sanitize_filename_part(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s).upper())


def normalize_price_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)

    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    out.index = pd.DatetimeIndex(idx).normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def standardize_history(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = normalize_price_index(hist)

    if "Adj Close" not in hist.columns and "Close" in hist.columns:
        hist = hist.rename(columns={"Close": "Adj Close"})

    if "Adj Close" not in hist.columns:
        return pd.DataFrame()

    out = hist[["Adj Close"]].copy()
    out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    out = out.dropna(how="all")

    return out


def winsorize_series(s: pd.Series, low_pct: float = 0.01, high_pct: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(s, errors="coerce")
    if clean.dropna().empty:
        return clean
    lo = clean.quantile(low_pct)
    hi = clean.quantile(high_pct)
    return clean.clip(lo, hi)


def interp_score(value, field: str) -> float:
    try:
        val = safe_float(value)
        if np.isnan(val):
            return np.nan

        cfg = FIELD_CONFIG[field]
        xp, fp = cfg["anchors"]

        return float(
            np.interp(
                val * cfg["scale"],
                xp,
                fp,
                left=fp[0],
                right=fp[-1],
            )
        )
    except Exception:
        return np.nan


def load_watchlist_csv(path: str) -> List[str]:
    try:
        df = pd.read_csv(path)

        if "Ticker" in df.columns:
            vals = df["Ticker"].dropna().astype(str).tolist()
        elif "ticker" in df.columns:
            vals = df["ticker"].dropna().astype(str).tolist()
        else:
            vals = df.iloc[:, 0].dropna().astype(str).tolist()

        return [x.strip().upper() for x in vals if x.strip()]

    except Exception as e:
        logger.error("Could not read watchlist CSV: %s", e)
        return []


# ============================================================
# CACHE
# ============================================================

def cache_path(ticker: str, period: str) -> str:
    safe_ticker = sanitize_filename_part(ticker)
    safe_period = sanitize_filename_part(period)
    return os.path.join(CACHE_DIR, f"{safe_ticker}_{safe_period}_{CACHE_VERSION}.pkl")


def cache_get(ticker: str, period: str, ttl_hours: int):
    path = cache_path(ticker, period)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if obj.get("_schema") != CACHE_VERSION:
            return None

        ts = obj.get("_cached_at")

        if ts is None:
            return None

        now = datetime.now(timezone.utc)

        # Backwards compatible: oude cache-bestanden kunnen nog timezone-naive zijn.
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if (now - ts) > timedelta(hours=ttl_hours):
            return None

        return obj.get("data")

    except Exception:
        return None


def cache_set(ticker: str, period: str, data):
    path = cache_path(ticker, period)

    try:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "_schema": CACHE_VERSION,
                    "_cached_at": datetime.now(timezone.utc),
                    "data": data,
                },
                f,
            )
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", ticker, e)


# ============================================================
# PRICE METRICS
# ============================================================

def compute_max_drawdown_from_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Returns max drawdown magnitude in percent.
    Example: -35% drawdown becomes 35.
    """
    if prices.empty:
        return pd.Series(dtype=float)

    p = prices.replace([np.inf, -np.inf], np.nan)
    rolling_max = p.cummax()
    dd = p / rolling_max - 1.0
    max_dd = dd.min(skipna=True).abs() * 100.0
    return max_dd


def compute_price_metrics_matrix(
    price_window: pd.DataFrame,
    spy_window: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Vectorized:
    - Momentum
    - Volatility
    - Beta
    - MaxDrawdown
    """
    if price_window is None or price_window.empty:
        return empty_price_metrics_df()

    prices = price_window.copy()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(axis=1, how="all")

    tickers = prices.columns
    result = pd.DataFrame(index=tickers)

    empty = pd.Series(np.nan, index=tickers, dtype=float)

    m3 = prices.pct_change(periods=63).iloc[-1] if len(prices) > 63 else empty.copy()
    m6 = prices.pct_change(periods=126).iloc[-1] if len(prices) > 126 else empty.copy()
    m12 = prices.pct_change(periods=252).iloc[-1] if len(prices) > 252 else empty.copy()

    m_df = pd.concat(
        [
            m3.rename("m3"),
            m6.rename("m6"),
            m12.rename("m12"),
        ],
        axis=1,
    )

    momentum_weights = pd.Series(
        {
            "m3": 0.50,
            "m6": 0.30,
            "m12": 0.20,
        }
    )

    available_weights = m_df.notna().astype(float).mul(momentum_weights, axis=1)
    denom = available_weights.sum(axis=1).replace(0, np.nan)
    result["Momentum"] = m_df.fillna(0).mul(momentum_weights, axis=1).sum(axis=1) / denom

    safe_prices = prices.replace(0, np.nan)
    logrets = np.log(safe_prices).diff().replace([np.inf, -np.inf], np.nan)
    result["Volatility"] = logrets.std(skipna=True) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

    result["MaxDrawdown"] = compute_max_drawdown_from_prices(prices)

    result["Beta"] = np.nan

    if spy_window is not None and not spy_window.empty and "Adj Close" in spy_window.columns:
        spy = spy_window[["Adj Close"]].copy()
        spy = normalize_price_index(spy)
        spy_ret = spy["Adj Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        stock_rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        aligned = stock_rets.join(spy_ret.rename("_MARKET"), how="inner")

        if not aligned.empty and "_MARKET" in aligned.columns:
            y = aligned["_MARKET"]
            X = aligned[[c for c in tickers if c in aligned.columns]]

            if len(X) >= 20:
                valid = X.notna() & y.notna().values[:, None]
                n = valid.sum(axis=0)

                y_matrix = pd.DataFrame(
                    np.repeat(y.to_numpy()[:, None], X.shape[1], axis=1),
                    index=X.index,
                    columns=X.columns,
                )

                x_sum = X.where(valid).sum(axis=0)
                y_sum = y_matrix.where(valid).sum(axis=0)

                x_mean = x_sum / n.replace(0, np.nan)
                y_mean = y_sum / n.replace(0, np.nan)

                x_centered = X.sub(x_mean, axis=1)
                y_centered = y_matrix.sub(y_mean, axis=1)

                cov_xy = (x_centered.where(valid) * y_centered.where(valid)).sum(axis=0) / (n - 1)
                var_y = (y_centered.where(valid) ** 2).sum(axis=0) / (n - 1)

                beta = cov_xy / var_y.replace(0, np.nan)
                beta[n < 20] = np.nan

                result["Beta"] = beta.reindex(result.index)

    return result


# ============================================================
# DATA FETCHING
# ============================================================

def get_data(ticker: str, spy_hist: pd.DataFrame, period: str) -> Optional[Dict]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        hist_raw = stock.history(period=period, auto_adjust=True)
        hist = standardize_history(hist_raw)

        if hist.empty:
            return None

        ticker_upper = ticker.upper()

        metrics = compute_price_metrics_matrix(
            hist.rename(columns={"Adj Close": ticker_upper}),
            spy_hist,
        )

        pm = metrics.loc[ticker_upper] if ticker_upper in metrics.index else pd.Series(dtype=float)

        beta = safe_float(info.get("beta"))

        if np.isnan(beta):
            beta = safe_float(pm.get("Beta", np.nan))

        return {
            "Ticker": ticker_upper,
            "ShortName": info.get("shortName") or "",
            "LongName": info.get("longName") or "",
            "Sector": info.get("sector") or "Unknown",
            "Industry": info.get("industry") or "Unknown",
            "MarketCap": safe_float(info.get("marketCap")),
            "AverageVolume": safe_float(info.get("averageVolume")),
            "PE": safe_float(info.get("trailingPE")),
            "FPE": safe_float(info.get("forwardPE")),
            "PEG": safe_float(info.get("pegRatio")),
            "ROE": safe_float(info.get("returnOnEquity")),
            "ROA": safe_float(info.get("returnOnAssets")),
            "Beta": beta,
            "DebtEquity": safe_float(info.get("debtToEquity")),
            "RevenueGrowth": safe_float(info.get("revenueGrowth")),
            "ProfitMargin": safe_float(info.get("profitMargins")),
            "DividendYield": safe_float(info.get("dividendYield")),
            "Momentum": safe_float(pm.get("Momentum", np.nan)),
            "Volatility": safe_float(pm.get("Volatility", np.nan)),
            "MaxDrawdown": safe_float(pm.get("MaxDrawdown", np.nan)),
            "PriceHistory": hist.copy(),
        }

    except Exception as e:
        logger.debug("get_data error for %s: %s", ticker, e)
        return None


def get_data_with_retry(
    ticker: str,
    spy_hist: pd.DataFrame,
    period: str,
    max_retries: int,
    cache_ttl_hours: int,
):
    cached = cache_get(ticker, period, cache_ttl_hours)

    if cached is not None:
        logger.debug("Cache hit: %s", ticker)
        return cached

    for attempt in range(1, max_retries + 1):
        data = get_data(ticker, spy_hist, period)

        if data is not None:
            cache_set(ticker, period, data)
            return data

        sleep_time = REQUEST_DELAY * attempt
        logger.debug("Retry %d for %s after %.2fs", attempt, ticker, sleep_time)
        time.sleep(sleep_time)

    logger.warning("No data for %s after %d attempts", ticker, max_retries)
    return None


def fetch_all_data(
    tickers: List[str],
    spy_hist: pd.DataFrame,
    period: str,
    workers: int,
    max_retries: int,
    cache_ttl_hours: int,
) -> List[Dict]:
    results: List[Dict] = []

    if workers <= 1:
        for t in tickers:
            logger.info("Fetching %s", t)
            d = get_data_with_retry(t, spy_hist, period, max_retries, cache_ttl_hours)
            if d is not None:
                results.append(d)
            time.sleep(REQUEST_DELAY)
        return results

    logger.info("Fetching with %d workers", workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(get_data_with_retry, t, spy_hist, period, max_retries, cache_ttl_hours): t
            for t in tickers
        }

        for fut in as_completed(futures):
            t = futures[fut]
            try:
                d = fut.result()
                if d is not None:
                    results.append(d)
                else:
                    logger.warning("Skipping %s: no data", t)
            except Exception as e:
                logger.warning("Skipping %s: %s", t, e)

    return results

# ============================================================
# V12 NEWS MODULE
# ============================================================

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_news_datetime(value):
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
    except Exception:
        pass

    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def get_company_terms(ticker: str, row: pd.Series) -> List[str]:
    terms = []

    ticker = str(ticker).upper().strip()

    if ticker:
        terms.append(ticker)

    for alias in COMPANY_ALIASES.get(ticker, []):
        if alias and alias not in terms:
            terms.append(alias)

    for col in ["ShortName", "LongName"]:
        val = clean_text(row.get(col, ""))

        if val and val not in terms:
            terms.append(val)

            simplified = re.sub(
                r"\b(inc|inc\.|corp|corp\.|corporation|plc|nv|n\.v\.|holding|holdings|technologies|company|co\.|ltd|limited)\b",
                "",
                val,
                flags=re.IGNORECASE,
            )
            simplified = re.sub(r"\s+", " ", simplified).strip()

            if len(simplified) >= 3 and simplified not in terms:
                terms.append(simplified)

    # Remove very short generic terms, except ticker.
    cleaned = []

    for term in terms:
        term = clean_text(term)

        if not term:
            continue

        if term.upper() == ticker or len(term) >= 4:
            if term not in cleaned:
                cleaned.append(term)

    return cleaned


def normalize_yfinance_news_item(item: Dict, ticker: str) -> Dict:
    """
    yfinance news format can differ by version.
    This handles both older flat dicts and newer nested content dicts.
    """
    content = item.get("content", {}) if isinstance(item, dict) else {}

    title = (
        content.get("title")
        or item.get("title")
        or ""
    )

    summary = (
        content.get("summary")
        or content.get("description")
        or item.get("summary")
        or ""
    )

    publisher = (
        item.get("publisher")
        or content.get("provider", {}).get("displayName")
        or content.get("publisher")
        or "Yahoo Finance"
    )

    link = (
        item.get("link")
        or item.get("url")
        or content.get("canonicalUrl", {}).get("url")
        or content.get("clickThroughUrl", {}).get("url")
        or ""
    )

    published = (
        item.get("providerPublishTime")
        or item.get("pubDate")
        or content.get("pubDate")
        or content.get("displayTime")
    )

    published_dt = parse_news_datetime(published)

    return {
        "Ticker": ticker,
        "SourceType": "yfinance",
        "Title": clean_text(title),
        "Summary": clean_text(summary),
        "Publisher": clean_text(publisher),
        "Link": link,
        "PublishedAt": published_dt,
    }


def fetch_yfinance_news(ticker: str, max_items: int = 20) -> List[Dict]:
    items = []

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []

        for item in raw_news[:max_items]:
            normalized = normalize_yfinance_news_item(item, ticker)

            if normalized["Title"]:
                items.append(normalized)

    except Exception as e:
        logger.debug("yfinance news failed for %s: %s", ticker, e)

    return items


def build_google_news_query(ticker: str, terms: List[str]) -> str:
    ticker = str(ticker).upper().strip()

    company_terms = [
        clean_text(t) for t in terms
        if clean_text(t) and clean_text(t).upper() != ticker and len(clean_text(t)) >= 4
    ]

    company = company_terms[0] if company_terms else ticker

    # Voor korte tickers niet los op ticker zoeken.
    if ticker in NEWS_SHORT_TICKERS_REQUIRE_COMPANY:
        return f'"{company}" stock'

    return f'("{company}" OR "{ticker}") stock'


def fetch_google_news_rss(ticker: str, terms: List[str], max_items: int = 20) -> List[Dict]:
    if not FEEDPARSER_AVAILABLE:
        return []

    items = []

    query = build_google_news_query(ticker, terms)

    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)

        for entry in feed.entries[:max_items]:
            title = clean_text(getattr(entry, "title", ""))
            summary = clean_text(getattr(entry, "summary", ""))
            link = getattr(entry, "link", "")

            published = (
                getattr(entry, "published", None)
                or getattr(entry, "updated", None)
            )

            published_dt = parse_news_datetime(published)

            publisher = "Google News RSS"

            if " - " in title:
                possible_title, possible_source = title.rsplit(" - ", 1)
                title = clean_text(possible_title)
                publisher = clean_text(possible_source)

            if title:
                items.append(
                    {
                        "Ticker": ticker,
                        "SourceType": "google_rss",
                        "Title": title,
                        "Summary": summary,
                        "Publisher": publisher,
                        "Link": link,
                        "PublishedAt": published_dt,
                        "SearchQuery": query,
                    }
                )

    except Exception as e:
        logger.debug("Google RSS failed for %s: %s", ticker, e)

    return items


def news_fingerprint(title: str) -> str:
    title = clean_text(title).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:140]

def contains_exact_term(text: str, term: str) -> bool:
    """
    Zoekt een term redelijk exact in tekst.
    Voor tickers gebruiken we woordgrenzen, zodat V niet overal matcht.
    """
    text = clean_text(text).lower()
    term = clean_text(term).lower()

    if not text or not term:
        return False

    if len(term) <= 3 or re.fullmatch(r"[a-z0-9.\-=]+", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

    return term in text


def score_company_match(ticker: str, terms: List[str], title: str, summary: str) -> Tuple[float, str]:
    """
    Bepaalt hoe zeker we zijn dat het artikel echt over dit bedrijf gaat.
    """
    ticker = str(ticker).upper().strip()
    title_text = clean_text(title)
    full_text = clean_text(f"{title} {summary}")

    score = 0.0
    reasons = []

    company_terms = [
        t for t in terms
        if clean_text(t).upper() != ticker and len(clean_text(t)) >= 4
    ]

    if contains_exact_term(title_text, ticker):
        score += 30
        reasons.append("ticker_in_title")

    elif contains_exact_term(full_text, ticker):
        score += 15
        reasons.append("ticker_in_text")

    company_title_hit = False
    company_text_hit = False

    for term in company_terms:
        if contains_exact_term(title_text, term):
            score += 45
            company_title_hit = True
            reasons.append(f"company_in_title:{term}")
            break

    if not company_title_hit:
        for term in company_terms:
            if contains_exact_term(full_text, term):
                score += 25
                company_text_hit = True
                reasons.append(f"company_in_text:{term}")
                break

    # Korte tickers zijn gevaarlijk: V, MA, GE, CAT, KO enz.
    # Daar willen we liever bedrijfsnaam-match zien.
    if ticker in NEWS_SHORT_TICKERS_REQUIRE_COMPANY and not (company_title_hit or company_text_hit):
        score = min(score, 20)
        reasons.append("short_ticker_without_company_name")

    return float(max(0, min(100, score))), "; ".join(reasons)


def score_news_source_quality(publisher: str) -> float:
    """
    Geeft bronkwaliteit een score.
    """
    p = clean_text(publisher).lower()

    if not p:
        return 50.0

    for src in NEWS_HIGH_QUALITY_SOURCES:
        if src in p:
            return 85.0

    for src in NEWS_LOW_QUALITY_SOURCES:
        if src in p:
            return 40.0

    if "google news" in p:
        return 55.0

    return 60.0


def score_news_clickbait_penalty(title: str, summary: str) -> float:
    """
    Straft algemene/clickbait stock-artikelen.
    """
    text = clean_text(f"{title} {summary}").lower()
    penalty = 0.0

    for pattern in NEWS_CLICKBAIT_PATTERNS:
        if pattern in text:
            penalty += 18.0

    for junk in NEWS_JUNK_KEYWORDS:
        if junk in text:
            penalty += 15.0

    # Generieke artikelen zonder concrete bedrijfsgebeurtenis.
    generic_words = [
        "good stock to buy",
        "stock to buy now",
        "investment analysis",
        "stock valuation",
        "stock forecast",
        "trending",
    ]

    for word in generic_words:
        if word in text:
            penalty += 10.0

    return float(min(60.0, penalty))


def score_news_impact(title: str, summary: str) -> Tuple[float, str]:
    """
    Herkent wat voor soort nieuws het is.
    """
    text = clean_text(f"{title} {summary}").lower()

    categories = []
    score = 20.0

    for category, keywords in NEWS_IMPACT_KEYWORDS_BY_CATEGORY.items():
        hits = [kw for kw in keywords if kw in text]

        if hits:
            categories.append(category)
            score += min(25.0, 8.0 * len(hits))

    if not categories:
        return 25.0, "general"

    return float(min(100.0, score)), ", ".join(categories)


def score_news_relevance(ticker: str, terms: List[str], title: str, summary: str) -> float:
    """
    V12.2:
    Betere relevantiescore op basis van:
    - duidelijke ticker/bedrijfsnaam-match
    - impacttype van nieuws
    - clickbait/generic penalty
    """
    company_match, _ = score_company_match(ticker, terms, title, summary)
    impact_score, _ = score_news_impact(title, summary)
    clickbait_penalty = score_news_clickbait_penalty(title, summary)

    text = clean_text(f"{title} {summary}").lower()

    important_bonus = 0.0

    for kw in NEWS_IMPORTANT_KEYWORDS:
        if kw in text:
            important_bonus += 3.0

    score = (
        0.70 * company_match
        + 0.30 * impact_score
        + important_bonus
        - clickbait_penalty
    )

    # Als de match met het bedrijf zwak is, nooit hoog laten scoren.
    if company_match < 35:
        score = min(score, 35)

    return float(max(0, min(100, score)))


def score_news_recency(published_at, lookback_days: int) -> float:
    if published_at is None:
        return 40.0

    now = datetime.now(timezone.utc)

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - published_at).total_seconds() / 86400)

    if age_days > lookback_days:
        return 0.0

    # Vandaag = 100, einde lookback = ongeveer 20.
    return float(max(20, 100 - (age_days / max(1, lookback_days)) * 80))


def score_news_sentiment(title: str, summary: str) -> float:
    text = f"{title}. {summary}"
    text_l = text.lower()

    base = 50.0

    if VADER_AVAILABLE:
        try:
            analyzer = SentimentIntensityAnalyzer()
            compound = analyzer.polarity_scores(text)["compound"]
            base = 50.0 + compound * 50.0
        except Exception:
            base = 50.0

    adjustment = 0.0

    for kw in NEWS_POSITIVE_KEYWORDS:
        if kw in text_l:
            adjustment += 7

    for kw in NEWS_NEGATIVE_KEYWORDS:
        if kw in text_l:
            adjustment -= 9

    return float(max(0, min(100, base + adjustment)))


def classify_news_label(sentiment_score: float) -> str:
    if sentiment_score >= 65:
        return "Positive"
    if sentiment_score <= 35:
        return "Negative"
    return "Neutral"


def score_news_items(
    ticker: str,
    row: pd.Series,
    items: List[Dict],
    lookback_days: int,
    min_relevance: float,
) -> pd.DataFrame:
    terms = get_company_terms(ticker, row)

    rows = []
    seen = set()

    for item in items:
        title = clean_text(item.get("Title", ""))
        summary = clean_text(item.get("Summary", ""))
        publisher = clean_text(item.get("Publisher", ""))

        if not title:
            continue

        fp = news_fingerprint(title)

        if fp in seen:
            continue

        seen.add(fp)

        published_at = item.get("PublishedAt")

        recency = score_news_recency(published_at, lookback_days)

        if recency <= 0:
            continue

        company_match, match_reason = score_company_match(ticker, terms, title, summary)
        relevance = score_news_relevance(ticker, terms, title, summary)
        source_quality = score_news_source_quality(publisher)
        impact_score, impact_category = score_news_impact(title, summary)
        clickbait_penalty = score_news_clickbait_penalty(title, summary)
        sentiment = score_news_sentiment(title, summary)
        label = classify_news_label(sentiment)

        # Hard filter:
        # Als het artikel nauwelijks over het bedrijf lijkt te gaan, overslaan.
        if company_match < 25:
            continue

        # Clickbait mag alleen door als relevance alsnog sterk is.
        if clickbait_penalty >= 35 and relevance < 70:
            continue

        if relevance < min_relevance:
            continue

        # V12.2 NewsItemScore:
        # Relevance en company match zijn het belangrijkst.
        # Sentiment telt mee, maar minder zwaar.
        item_score = (
            0.30 * relevance
            + 0.20 * company_match
            + 0.15 * recency
            + 0.15 * sentiment
            + 0.10 * source_quality
            + 0.10 * impact_score
        )

        rows.append(
            {
                "Ticker": ticker,
                "Title": title,
                "Summary": summary,
                "Publisher": publisher,
                "SourceType": item.get("SourceType", ""),
                "PublishedAt": published_at,
                "Link": item.get("Link", ""),
                "SearchQuery": item.get("SearchQuery", ""),
                "CompanyMatch": round(company_match, 2),
                "MatchReason": match_reason,
                "Relevance": round(relevance, 2),
                "RecencyScore": round(recency, 2),
                "SentimentScore": round(sentiment, 2),
                "SentimentLabel": label,
                "SourceQuality": round(source_quality, 2),
                "ImpactScore": round(impact_score, 2),
                "ImpactCategory": impact_category,
                "ClickbaitPenalty": round(clickbait_penalty, 2),
                "NewsItemScore": round(float(item_score), 2),
            }
        )

    if not rows:
        return empty_news_items_df()

    df_news = pd.DataFrame(rows)
    df_news = df_news.sort_values(
        ["NewsItemScore", "Relevance", "CompanyMatch", "RecencyScore"],
        ascending=False,
    )

    return df_news


def build_news_for_universe(
    df: pd.DataFrame,
    max_items_per_source: int,
    lookback_days: int,
    min_relevance: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_news = []
    summary_rows = []

    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).upper().strip()

        if not ticker:
            continue

        logger.info("Fetching news for %s", ticker)

        terms = get_company_terms(ticker, row)

        raw_items = []
        raw_items.extend(fetch_yfinance_news(ticker, max_items=max_items_per_source))
        raw_items.extend(fetch_google_news_rss(ticker, terms, max_items=max_items_per_source))

        scored = score_news_items(
            ticker=ticker,
            row=row,
            items=raw_items,
            lookback_days=lookback_days,
            min_relevance=min_relevance,
        )

        if not scored.empty:
            top_news = scored.head(max_items_per_source).copy()
            all_news.append(top_news)

            # V12.3: relevantie, bedrijfs-match, bronkwaliteit en impact bepalen
            # hoe zwaar een artikel meetelt in de ticker-news-score.
            weights = (
                top_news["Relevance"].fillna(0)
                + top_news["CompanyMatch"].fillna(0)
                + top_news["SourceQuality"].fillna(0) * 0.5
                + top_news["ImpactScore"].fillna(0) * 0.5
            ).replace(0, 1)

            news_score = float(np.average(top_news["NewsItemScore"], weights=weights))
            news_sentiment = float(np.average(top_news["SentimentScore"], weights=weights))

            positive_count = int((top_news["SentimentLabel"] == "Positive").sum())
            negative_count = int((top_news["SentimentLabel"] == "Negative").sum())
            neutral_count = int((top_news["SentimentLabel"] == "Neutral").sum())

            top_headline = str(top_news.iloc[0]["Title"])
            top_publisher = str(top_news.iloc[0].get("Publisher", ""))
            top_impact = str(top_news.iloc[0].get("ImpactCategory", ""))

            avg_relevance = float(top_news["Relevance"].mean()) if "Relevance" in top_news else np.nan
            avg_company_match = float(top_news["CompanyMatch"].mean()) if "CompanyMatch" in top_news else np.nan
            avg_source_quality = float(top_news["SourceQuality"].mean()) if "SourceQuality" in top_news else np.nan
            avg_impact_score = float(top_news["ImpactScore"].mean()) if "ImpactScore" in top_news else np.nan
            avg_clickbait_penalty = float(top_news["ClickbaitPenalty"].mean()) if "ClickbaitPenalty" in top_news else np.nan

            summary_rows.append(
                {
                    "Ticker": ticker,
                    "NewsScore": round(news_score, 2),
                    "NewsSentiment": round(news_sentiment, 2),
                    "NewsCount": int(len(top_news)),
                    "PositiveNewsCount": positive_count,
                    "NegativeNewsCount": negative_count,
                    "NeutralNewsCount": neutral_count,
                    "TopHeadline": top_headline,
                    "TopPublisher": top_publisher,
                    "TopImpactCategory": top_impact,
                    "AvgNewsRelevance": round(avg_relevance, 2) if not np.isnan(avg_relevance) else np.nan,
                    "AvgCompanyMatch": round(avg_company_match, 2) if not np.isnan(avg_company_match) else np.nan,
                    "AvgSourceQuality": round(avg_source_quality, 2) if not np.isnan(avg_source_quality) else np.nan,
                    "AvgImpactScore": round(avg_impact_score, 2) if not np.isnan(avg_impact_score) else np.nan,
                    "AvgClickbaitPenalty": round(avg_clickbait_penalty, 2) if not np.isnan(avg_clickbait_penalty) else np.nan,
                }
            )
        else:
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "NewsScore": NEWS_DEFAULT_NO_NEWS_SCORE,
                    "NewsSentiment": 50.0,
                    "NewsCount": 0,
                    "PositiveNewsCount": 0,
                    "NegativeNewsCount": 0,
                    "NeutralNewsCount": 0,
                    "TopHeadline": "",
                    "TopPublisher": "",
                    "TopImpactCategory": "",
                    "AvgNewsRelevance": np.nan,
                    "AvgCompanyMatch": np.nan,
                    "AvgSourceQuality": np.nan,
                    "AvgImpactScore": np.nan,
                    "AvgClickbaitPenalty": np.nan,
                }
            )

    news_df = pd.concat(all_news, ignore_index=True) if all_news else empty_news_items_df()
    news_summary_df = pd.DataFrame(summary_rows) if summary_rows else empty_news_summary_df()

    return news_df, news_summary_df


def compute_news_confidence(row: pd.Series) -> float:
    """
    Bepaalt hoeveel vertrouwen we hebben in de NewsScore.

    0.0 = nieuws telt niet mee.
    1.0 = nieuws telt volledig mee volgens args.news_weight.
    """
    news_count = int(safe_float(row.get("NewsCount", 0)) if not np.isnan(safe_float(row.get("NewsCount", 0))) else 0)

    if news_count <= 0:
        return 0.0

    avg_relevance = safe_float(row.get("AvgNewsRelevance", np.nan))
    avg_company_match = safe_float(row.get("AvgCompanyMatch", np.nan))
    avg_source_quality = safe_float(row.get("AvgSourceQuality", np.nan))
    avg_impact_score = safe_float(row.get("AvgImpactScore", np.nan))
    avg_clickbait_penalty = safe_float(row.get("AvgClickbaitPenalty", 0.0))

    if np.isnan(avg_relevance):
        avg_relevance = 50.0

    if np.isnan(avg_company_match):
        avg_company_match = 50.0

    if np.isnan(avg_source_quality):
        avg_source_quality = 60.0

    if np.isnan(avg_impact_score):
        avg_impact_score = 25.0

    if np.isnan(avg_clickbait_penalty):
        avg_clickbait_penalty = 0.0

    count_score = min(1.0, news_count / NEWS_MIN_ITEMS_FOR_FULL_WEIGHT)

    quality_score = (
        0.40 * avg_relevance
        + 0.35 * avg_company_match
        + 0.15 * avg_source_quality
        + 0.10 * avg_impact_score
        - 0.15 * avg_clickbait_penalty
    ) / 100.0

    confidence = count_score * quality_score

    if confidence < NEWS_MIN_CONFIDENCE_FOR_EFFECT:
        return 0.0

    return clamp_float(confidence, 0.0, 1.0)


def classify_news_status(row: pd.Series) -> str:
    news_count_raw = safe_float(row.get("NewsCount", 0))
    news_count = int(news_count_raw) if not np.isnan(news_count_raw) else 0
    confidence = safe_float(row.get("NewsConfidence", 0))

    if news_count <= 0:
        return "No news - neutral"

    if confidence <= 0:
        return "Ignored - low confidence"

    if confidence < 0.5:
        return "Low confidence news"

    if confidence < 0.8:
        return "Medium confidence news"

    return "High confidence news"

# ============================================================
# SCORING
# ============================================================

def percentile_score(s: pd.Series, lower_better: bool) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)

    valid = s.notna()
    n = int(valid.sum())

    if n == 0:
        return out

    if n == 1:
        out.loc[valid] = 50.0
        return out

    ranks = s.rank(method="average")
    pct01 = (ranks - 1.0) / (n - 1.0)

    if lower_better:
        score = (1.0 - pct01) * 100.0
    else:
        score = pct01 * 100.0

    out.loc[valid] = score.loc[valid]

    return out


def compute_factor_scores(df: pd.DataFrame, blend_absolute: float = 0.70) -> pd.DataFrame:
    df = df.copy()

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    numeric_metrics = sorted({m for fields in FACTOR_FIELDS.values() for m in fields})

    # Add AverageVolume as extra metric score for reporting, but not in Total directly.
    extra_metrics = ["AverageVolume"]
    all_metrics = sorted(set(numeric_metrics + extra_metrics))

    raw_available = pd.DataFrame(index=df.index)

    for col in all_metrics:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")
        raw_available[col] = df[col].notna()
        df[col] = winsorize_series(df[col], *WINSOR_PCT)

    for col in all_metrics:
        df[col + "_abs_score"] = df[col].apply(lambda v: interp_score(v, col) if col in FIELD_CONFIG else np.nan)

    global_pct_scores = {}

    for col in all_metrics:
        global_pct_scores[col] = percentile_score(
            df[col],
            lower_better=(col in LOWER_IS_BETTER),
        )

    for col in all_metrics:
        lower_better = col in LOWER_IS_BETTER

        sector_scores = df.groupby("Sector", group_keys=False)[col].transform(
            lambda s: percentile_score(s, lower_better=lower_better)
        )

        df[col + "_sector_pct"] = sector_scores.fillna(global_pct_scores[col])

    for col in all_metrics:
        df[col + "_score_metric"] = (
            blend_absolute * df[col + "_abs_score"].fillna(50.0)
            + (1.0 - blend_absolute) * df[col + "_sector_pct"].fillna(50.0)
        ).clip(0, 100)

    for factor, fields in FACTOR_FIELDS.items():
        metric_cols = [f + "_score_metric" for f in fields]
        df[factor + "_score"] = df[metric_cols].mean(axis=1, skipna=True).fillna(50.0).clip(0, 100)

    total_fields = sum(len(fields) for fields in FACTOR_FIELDS.values())
    factor_metric_names = sorted({m for fields in FACTOR_FIELDS.values() for m in fields})
    df["Confidence"] = (raw_available[factor_metric_names].sum(axis=1) / total_fields * 100.0).round(1)

    return df


def score_total(df: pd.DataFrame, mode: str, confidence_adjust: bool = True) -> pd.DataFrame:
    df = df.copy()

    if mode not in WEIGHTS:
        raise ValueError(f"Unknown mode: {mode}")

    w = WEIGHTS[mode]

    factor_cols = [
        "Value_score",
        "Quality_score",
        "Growth_score",
        "Risk_score",
        "Stability_score",
        "Dividend_score",
        "Momentum_score",
        "Drawdown_score",
    ]

    for col in factor_cols:
        if col not in df.columns:
            logger.debug("Missing factor column %s; filling with neutral 50.", col)
            df[col] = 50.0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50.0)

    df["Raw_Total"] = (
        df["Value_score"] * w["value"]
        + df["Quality_score"] * w["quality"]
        + df["Growth_score"] * w["growth"]
        + df["Risk_score"] * w["risk"]
        + df["Stability_score"] * w["stability"]
        + df["Dividend_score"] * w["dividend"]
        + df["Momentum_score"] * w["momentum"]
        + df["Drawdown_score"] * w["drawdown"]
    ).fillna(50.0).clip(0, 100)

    if confidence_adjust:
        # Confidence penalty:
        # Confidence 100% => factor 1.00
        # Confidence 50%  => factor 0.75
        # Confidence 0%   => factor 0.50
        conf = pd.to_numeric(df.get("Confidence", 100.0), errors="coerce").fillna(100.0)
        penalty_factor = 0.5 + 0.5 * (conf / 100.0)
        df["Total"] = (df["Raw_Total"] * penalty_factor).clip(0, 100)
    else:
        df["Total"] = df["Raw_Total"]

    return df


def add_rating(df: pd.DataFrame, score_col: str = "Total") -> pd.DataFrame:
    """
    Geeft rating op basis van een gekozen scorekolom.
    Voor V12.1 gebruiken we FinalScore, zodat nieuws ook meetelt.
    """
    out = df.copy()

    def rating_from_score(score):
        score = safe_float(score)

        if np.isnan(score):
            return "⚪ Unknown"

        if score >= 70:
            return "🟢 Strong"

        if score >= 55:
            return "🟡 OK"

        if score >= 40:
            return "🟠 Weak"

        return "🔴 Bad"

    if score_col not in out.columns:
        logger.warning("Rating score column %s not found. Falling back to Total.", score_col)
        score_col = "Total"

    out["Rating"] = out[score_col].apply(rating_from_score)

    return out


def create_warnings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    fields_to_check = [
        "PE",
        "FPE",
        "PEG",
        "ROE",
        "ROA",
        "Beta",
        "DebtEquity",
        "RevenueGrowth",
        "ProfitMargin",
        "DividendYield",
        "Momentum",
        "Volatility",
        "MaxDrawdown",
        "AverageVolume",
    ]

    for _, row in df.iterrows():
        ticker = row.get("Ticker", "Unknown")

        missing = []
        for f in fields_to_check:
            if f in row.index and pd.isna(row.get(f)):
                missing.append(f)

        if missing:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "Missing data",
                    "Details": ", ".join(missing),
                }
            )

        pe = safe_float(row.get("PE", np.nan))
        peg = safe_float(row.get("PEG", np.nan))
        conf = safe_float(row.get("Confidence", np.nan))
        vol = safe_float(row.get("Volatility", np.nan))
        dd = safe_float(row.get("MaxDrawdown", np.nan))
        volume = safe_float(row.get("AverageVolume", np.nan))

        if not np.isnan(pe) and pe < 0:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "High",
                    "Warning": "Negative P/E",
                    "Details": "Company may be loss-making. P/E score can be misleading.",
                }
            )

        if not np.isnan(pe) and pe > 80:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "Very high P/E",
                    "Details": f"P/E = {pe:.2f}",
                }
            )

        if not np.isnan(peg) and peg > 3:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "High PEG",
                    "Details": f"PEG = {peg:.2f}",
                }
            )

        if not np.isnan(conf) and conf < 70:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "Low confidence",
                    "Details": f"Confidence = {conf:.1f}%",
                }
            )

        if not np.isnan(vol) and vol > 60:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "High volatility",
                    "Details": f"Volatility = {vol:.1f}%",
                }
            )

        if not np.isnan(dd) and dd > 50:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "High",
                    "Warning": "Large drawdown",
                    "Details": f"Max drawdown = {dd:.1f}%",
                }
            )

        if not np.isnan(volume) and volume < 100_000:
            rows.append(
                {
                    "Ticker": ticker,
                    "Severity": "Medium",
                    "Warning": "Low liquidity",
                    "Details": f"Average volume = {volume:,.0f}",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Severity", "Warning", "Details"])

    return pd.DataFrame(rows)


# ============================================================
# COVARIANCE / GPU
# ============================================================

def estimate_covariance(
    returns: pd.DataFrame,
    use_gpu: bool = False,
    shrink_alpha: float = 0.80,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()

    R = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    if R.shape[1] == 0:
        return pd.DataFrame()

    if use_gpu:
        try:
            import cupy as cp  # type: ignore

            R_fill = R.ffill().bfill().fillna(0.0)
            X_cpu = R_fill.to_numpy(dtype=np.float64)

            if X_cpu.shape[0] >= 5 and X_cpu.shape[1] >= 2:
                X = cp.asarray(X_cpu)
                cov_daily = cp.cov(X, rowvar=False)
                diag = cp.diag(cp.diag(cov_daily))
                shrunk = shrink_alpha * cov_daily + (1.0 - shrink_alpha) * diag
                cov = cp.asnumpy(shrunk) * TRADING_DAYS_PER_YEAR
                cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

                return pd.DataFrame(cov, index=R_fill.columns, columns=R_fill.columns)

        except Exception as e:
            logger.debug("GPU covariance failed, falling back to CPU: %s", e)

    R_clean = R.dropna(axis=0, how="any")

    if SKLEARN_AVAILABLE and R_clean.shape[0] >= 20 and R_clean.shape[1] >= 2:
        try:
            lw = LedoitWolf().fit(R_clean.to_numpy(dtype=np.float64))
            cov = lw.covariance_ * TRADING_DAYS_PER_YEAR
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

            return pd.DataFrame(cov, index=R_clean.columns, columns=R_clean.columns)

        except Exception as e:
            logger.debug("LedoitWolf failed, falling back: %s", e)

    R_fill = R.ffill().bfill().fillna(0.0)

    if R_fill.shape[1] == 1:
        var = float(R_fill.iloc[:, 0].var() * TRADING_DAYS_PER_YEAR)
        return pd.DataFrame([[var]], index=R_fill.columns, columns=R_fill.columns)

    S = R_fill.cov().to_numpy()
    F = np.diag(np.diag(S))
    shrunk = shrink_alpha * S + (1.0 - shrink_alpha) * F
    cov = np.nan_to_num(shrunk * TRADING_DAYS_PER_YEAR, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(cov, index=R_fill.columns, columns=R_fill.columns)


# ============================================================
# OPTIMIZATION
# ============================================================

def make_feasible_bounds(n: int, max_weight: float, long_only: bool) -> List[Tuple[float, float]]:
    if n <= 0:
        return []

    min_required = 1.0 / n

    if long_only and max_weight < min_required:
        logger.warning(
            "max_weight %.4f is infeasible for %d assets. Adjusting to %.4f.",
            max_weight,
            n,
            min_required,
        )
        max_weight = min_required

    if long_only:
        return [(0.0, max_weight)] * n

    return [(-max_weight, max_weight)] * n


def normalize_weights(w: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    w = np.clip(w, lows, highs)

    total = w.sum()

    if abs(total) < 1e-12:
        return np.ones_like(w) / len(w)

    return w / total


def build_sector_constraints(
    tickers: List[str],
    sectors: List[str],
    max_sector_weight: float,
    long_only: bool,
):
    if not long_only:
        return []

    if max_sector_weight <= 0 or max_sector_weight >= 1:
        return []

    unique_sectors = sorted(set(sectors))
    sector_count = len(unique_sectors)

    if sector_count == 0:
        return []

    # Fix:
    # Als je bijvoorbeeld maar 2 sectoren hebt en max_sector_weight = 0.35,
    # dan kan de optimizer maximaal 70% beleggen. Dat is onmogelijk.
    # Daarom verhogen we de cap automatisch naar minimaal 1 / aantal_sectoren.
    min_feasible_sector_cap = 1.0 / sector_count

    if max_sector_weight < min_feasible_sector_cap:
        logger.warning(
            "max_sector_weight %.2f is infeasible with only %d sectors. Adjusting to %.2f.",
            max_sector_weight,
            sector_count,
            min_feasible_sector_cap,
        )
        max_sector_weight = min_feasible_sector_cap

    constraints = []

    for sec in unique_sectors:
        idxs = [i for i, s in enumerate(sectors) if s == sec]

        if not idxs:
            continue

        def sector_constraint(w, idxs=idxs):
            return max_sector_weight - np.sum(w[idxs])

        constraints.append(
            {
                "type": "ineq",
                "fun": sector_constraint,
            }
        )

    return constraints
def optimize_weights(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    objective: str,
    bounds: List[Tuple[float, float]],
    tickers: List[str],
    sectors: List[str],
    max_sector_weight: float,
    long_only: bool,
    initial: Optional[np.ndarray] = None,
    rf: float = 0.0,
) -> np.ndarray:
    n = len(expected_returns)

    if n == 0:
        return np.array([])

    if n == 1:
        return np.array([1.0])

    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not installed. Falling back to equal weights.")
        return np.ones(n) / n

    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    expected_returns = np.nan_to_num(expected_returns, nan=0.0, posinf=0.0, neginf=0.0)
    cov = cov + np.eye(n) * 1e-8

    init = np.ones(n) / n if initial is None else initial
    init = normalize_weights(init, bounds)

    def port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, w @ cov @ w)))

    def port_ret(w: np.ndarray) -> float:
        return float(w @ expected_returns)

    if objective == "sharpe":
        def fun(w):
            return -((port_ret(w) - rf) / (port_vol(w) + 1e-9))

    elif objective == "minvar":
        def fun(w):
            return float(w @ cov @ w)

    elif objective == "risk_parity":
        def fun(w):
            vol = port_vol(w)
            if vol <= 1e-12:
                return 1e6

            marginal = cov @ w / vol
            contribution = w * marginal
            target = vol / n

            return float(np.sum((contribution - target) ** 2))

    else:
        raise ValueError("objective must be sharpe, minvar, or risk_parity")

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
    ]

    constraints.extend(
        build_sector_constraints(
            tickers=tickers,
            sectors=sectors,
            max_sector_weight=max_sector_weight,
            long_only=long_only,
        )
    )

    try:
        res = minimize(
            fun,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 700,
                "ftol": 1e-9,
            },
        )

        if not res.success:
            logger.warning("Optimization did not converge: %s", res.message)
            return normalize_weights(res.x if res.x is not None else init, bounds)

        return normalize_weights(res.x, bounds)

    except Exception as e:
        logger.warning("Optimization crashed: %s. Falling back to equal weights.", e)
        return np.ones(n) / n


# ============================================================
# REGIME FILTER
# ============================================================

def detect_regime(spy_window: pd.DataFrame) -> str:
    if spy_window is None or spy_window.empty or "Adj Close" not in spy_window.columns:
        return "neutral_market"

    close = spy_window["Adj Close"].dropna()

    if len(close) < 120:
        return "neutral_market"

    last = float(close.iloc[-1])
    ma100 = float(close.rolling(100).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma100

    rets = np.log(close / close.shift(1)).dropna()
    vol20 = float(rets.tail(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0) if len(rets) >= 20 else np.nan

    peak = close.cummax()
    drawdown = float((close / peak - 1.0).iloc[-1] * 100.0)

    mom3 = float(close.iloc[-1] / close.iloc[-63] - 1.0) if len(close) > 63 else 0.0

    if last < ma200 and drawdown < -15:
        return "crash_risk"

    if last < ma100 or drawdown < -10 or (not np.isnan(vol20) and vol20 > 32):
        return "correction"

    if last > ma100 and last > ma200 and mom3 > 0:
        return "bull_market"

    return "neutral_market"


# ============================================================
# BACKTEST HELPERS
# ============================================================

def make_rebalance_dates(price_index: pd.DatetimeIndex, rebalance: str) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(sorted(pd.unique(price_index)))

    if len(idx) < 2:
        return list(idx)

    freq = "MS" if rebalance == "monthly" else "QS"
    anchors = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)

    dates: List[pd.Timestamp] = []

    for a in anchors:
        pos = idx.searchsorted(a)

        if pos < len(idx):
            dates.append(idx[pos])

    if idx[-1] not in dates:
        dates.append(idx[-1])

    dates = sorted(pd.unique(pd.DatetimeIndex(dates)))

    return list(dates)


def turnover(new_weights: Dict[str, float], old_weights: Dict[str, float]) -> float:
    all_names = set(new_weights.keys()) | set(old_weights.keys())
    return float(sum(abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in all_names))


def portfolio_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict:
    if returns is None or len(returns) == 0:
        return {}

    r = returns.dropna().astype(float)

    if r.empty:
        return {}

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    ann_return = float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann_return / (ann_vol + 1e-9))

    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_drawdown = float(drawdown.min())

    win_rate = float((r > 0).mean())
    avg_daily_return = float(r.mean())

    result = {
        "cum_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_daily_return": avg_daily_return,
        "days": int(len(r)),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        aligned = pd.concat([r.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()

        if len(aligned) > 5:
            excess = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = float(excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            excess_ann = float(excess.mean() * TRADING_DAYS_PER_YEAR)
            info_ratio = float(excess_ann / (tracking_error + 1e-9))

            bench_var = aligned["benchmark"].var()
            beta_to_bench = float(aligned["portfolio"].cov(aligned["benchmark"]) / bench_var) if bench_var != 0 else np.nan
            corr_to_bench = float(aligned["portfolio"].corr(aligned["benchmark"]))

            result.update(
                {
                    "excess_ann_return": excess_ann,
                    "tracking_error": tracking_error,
                    "information_ratio": info_ratio,
                    "beta_to_benchmark": beta_to_bench,
                    "corr_to_benchmark": corr_to_bench,
                }
            )

    return result

def format_number_or_missing(value, decimals: int = 0) -> str:
    """
    Format getallen netjes voor logging.
    Geeft 'missing' terug bij NaN of lege waarden.
    """
    val = safe_float(value)

    if np.isnan(val):
        return "missing"

    if decimals <= 0:
        return f"{val:,.0f}"

    return f"{val:,.{decimals}f}"


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    prices: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    universe_df: pd.DataFrame,
    spy_prices: pd.DataFrame,
    top_n: int,
    mode: str,
    lookback_days: int,
    objective: str,
    max_weight: float,
    max_sector_weight: float,
    transaction_cost: float,
    long_only: bool,
    blend_absolute: float,
    confidence_adjust: bool,
    use_gpu: bool,
    regime_filter: bool,
) -> Dict:
    prices = prices.copy().sort_index().replace([np.inf, -np.inf], np.nan)
    spy_prices = spy_prices.copy().sort_index()
    dates = sorted(rebalance_dates)

    if len(dates) < 2:
        return {
            "equal_metrics": {},
            "optimized_metrics": {},
            "benchmark_metrics": {},
            "equal_returns": pd.Series(dtype=float),
            "optimized_returns": pd.Series(dtype=float),
            "benchmark_returns": pd.Series(dtype=float),
            "weights": pd.DataFrame(),
        }

    eq_parts: List[pd.Series] = []
    opt_parts: List[pd.Series] = []
    bench_parts: List[pd.Series] = []
    weight_records: List[Dict] = []

    prev_eq_weights: Dict[str, float] = {}
    prev_opt_weights: Dict[str, float] = {}

    base_universe = universe_df.copy()

    if "Ticker" not in base_universe.columns:
        raise ValueError("universe_df must contain Ticker column")

    base_universe = base_universe.drop_duplicates(subset=["Ticker"], keep="first")

    for i in range(len(dates) - 1):
        t0 = dates[i]
        t1 = dates[i + 1]

        if t0 not in prices.index:
            continue

        pos = prices.index.get_loc(t0)

        if isinstance(pos, slice):
            pos = pos.stop - 1

        if isinstance(pos, np.ndarray):
            if len(pos) == 0:
                continue
            pos = int(pos[-1])

        start_pos = max(0, int(pos) - lookback_days + 1)
        window = prices.iloc[start_pos:int(pos) + 1].dropna(axis=1, how="all")

        if len(window) < max(30, min(lookback_days // 2, 60)):
            continue

        spy_window = spy_prices.loc[spy_prices.index <= t0].tail(lookback_days)

        price_metrics = compute_price_metrics_matrix(window, spy_window)

        df_t = base_universe.set_index("Ticker").copy()

        for col in ["Momentum", "Volatility", "Beta", "MaxDrawdown"]:
            df_t[col] = np.nan

        shared = df_t.index.intersection(price_metrics.index)

        if len(shared) > 0:
            for col in ["Momentum", "Volatility", "Beta", "MaxDrawdown"]:
                df_t.loc[shared, col] = price_metrics.loc[shared, col]

        df_t = df_t.reset_index()

        regime = detect_regime(spy_window)

        scoring_mode = mode
        period_objective = objective

        if regime_filter:
            if regime == "crash_risk":
                scoring_mode = "conservative"
                period_objective = "minvar"
            elif regime == "correction" and objective == "sharpe":
                scoring_mode = "conservative"
                period_objective = "minvar"

        df_t = compute_factor_scores(df_t, blend_absolute=blend_absolute)
        df_t = score_total(df_t, scoring_mode, confidence_adjust=confidence_adjust)
        df_t = add_rating(df_t)

        selected = (
            df_t.sort_values("Total", ascending=False)
            .head(min(top_n, len(df_t)))["Ticker"]
            .tolist()
        )

        selected = [t for t in selected if t in window.columns]

        if len(selected) == 0:
            continue

        ret_window = window[selected].pct_change().replace([np.inf, -np.inf], np.nan)

        min_obs = max(20, min(lookback_days // 3, 60))
        valid_cols = ret_window.count() >= min_obs
        common = [c for c in selected if c in valid_cols.index and bool(valid_cols[c])]

        if len(common) == 0:
            continue

        if len(common) == 1:
            w_opt = np.array([1.0])
            w_eq = np.array([1.0])
        else:
            ret_for_cov = ret_window[common].dropna(how="all")

            if ret_for_cov.empty or ret_for_cov.shape[0] < min_obs:
                w_opt = np.ones(len(common)) / len(common)
                w_eq = np.ones(len(common)) / len(common)
            else:
                asset_mu = ret_for_cov.mean() * TRADING_DAYS_PER_YEAR

                spy_ret_window = (
                    spy_window["Adj Close"].pct_change().dropna()
                    if spy_window is not None and not spy_window.empty and "Adj Close" in spy_window.columns
                    else pd.Series(dtype=float)
                )

                market_mu = float(spy_ret_window.mean() * TRADING_DAYS_PER_YEAR) if not spy_ret_window.empty else 0.0

                scores = df_t.set_index("Ticker").reindex(common)["Total"].fillna(50.0)
                score_alpha = ((scores - 50.0) / 50.0) * 0.07

                momentum_raw = df_t.set_index("Ticker").reindex(common)["Momentum"].fillna(0.0)
                momentum_alpha = momentum_raw.clip(-0.5, 0.8) * 0.25

                sector_map = df_t.set_index("Ticker").reindex(common)["Sector"].fillna("Unknown")
                sector_mu = pd.Series(index=common, dtype=float)

                for sec in sector_map.unique():
                    sec_tickers = sector_map[sector_map == sec].index.tolist()
                    if sec_tickers:
                        sector_mu.loc[sec_tickers] = asset_mu.reindex(sec_tickers).mean()

                sector_mu = sector_mu.fillna(market_mu)

                exp = (
                    0.35 * asset_mu.reindex(common).fillna(0.0)
                    + 0.20 * market_mu
                    + 0.20 * sector_mu.reindex(common).fillna(market_mu)
                    + 0.15 * score_alpha.reindex(common).fillna(0.0)
                    + 0.10 * momentum_alpha.reindex(common).fillna(0.0)
                )

                if regime_filter and regime == "crash_risk":
                    exp = exp * 0.35
                elif regime_filter and regime == "correction":
                    exp = exp * 0.60

                cov_df = estimate_covariance(ret_for_cov[common], use_gpu=use_gpu)
                cov_df = cov_df.reindex(index=common, columns=common).fillna(0.0)
                cov = cov_df.to_numpy(dtype=float)
                exp_arr = exp.reindex(common).fillna(0.0).to_numpy(dtype=float)

                bounds = make_feasible_bounds(len(common), max_weight, long_only)
                initial = np.ones(len(common)) / len(common)

                sectors = (
                    df_t.set_index("Ticker")
                    .reindex(common)["Sector"]
                    .fillna("Unknown")
                    .astype(str)
                    .tolist()
                )

                if period_objective == "risk_parity" and not long_only:
                    logger.info("risk_parity is not supported for long/short mode in this script; using minvar for this rebalance.")
                    period_objective = "minvar"

                w_opt = optimize_weights(
                    expected_returns=exp_arr,
                    cov=cov,
                    objective=period_objective,
                    bounds=bounds,
                    tickers=common,
                    sectors=sectors,
                    max_sector_weight=max_sector_weight,
                    long_only=long_only,
                    initial=initial,
                )

                w_eq = np.ones(len(common)) / len(common)

        seg_prices = prices.loc[(prices.index >= t0) & (prices.index <= t1), common]

        if seg_prices.shape[0] < 2:
            continue

        seg_rets = seg_prices.pct_change().dropna(how="all")

        if seg_rets.empty:
            continue

        seg_rets = seg_rets.fillna(0.0)

        new_eq_weights = {c: float(w_eq[j]) for j, c in enumerate(common)}
        new_opt_weights = {c: float(w_opt[j]) for j, c in enumerate(common)}

        eq_turnover = turnover(new_eq_weights, prev_eq_weights)
        opt_turnover = turnover(new_opt_weights, prev_opt_weights)

        eq_cost = eq_turnover * transaction_cost
        opt_cost = opt_turnover * transaction_cost

        daily_eq = pd.Series(
            seg_rets.to_numpy(dtype=float).dot(w_eq),
            index=seg_rets.index,
        )

        daily_opt = pd.Series(
            seg_rets.to_numpy(dtype=float).dot(w_opt),
            index=seg_rets.index,
        )

        if len(daily_eq) > 0:
            daily_eq.iloc[0] -= eq_cost
            daily_opt.iloc[0] -= opt_cost

        eq_parts.append(daily_eq)
        opt_parts.append(daily_opt)

        # Benchmark return over same segment
        if spy_prices is not None and not spy_prices.empty and "Adj Close" in spy_prices.columns:
            bench_seg = spy_prices.loc[(spy_prices.index >= t0) & (spy_prices.index <= t1), "Adj Close"]
            if len(bench_seg) >= 2:
                bench_rets = bench_seg.pct_change().dropna()
                bench_parts.append(bench_rets)

        for c, w in new_opt_weights.items():
            weight_records.append(
                {
                    "Date": t0,
                    "Portfolio": "optimized",
                    "Ticker": c,
                    "Weight": w,
                    "Regime": regime,
                    "Mode": scoring_mode,
                    "Objective": period_objective,
                }
            )

        for c, w in new_eq_weights.items():
            weight_records.append(
                {
                    "Date": t0,
                    "Portfolio": "equal",
                    "Ticker": c,
                    "Weight": w,
                    "Regime": regime,
                    "Mode": scoring_mode,
                    "Objective": "equal",
                }
            )

        prev_eq_weights = new_eq_weights
        prev_opt_weights = new_opt_weights

    equal_returns = pd.concat(eq_parts).sort_index() if eq_parts else pd.Series(dtype=float)
    optimized_returns = pd.concat(opt_parts).sort_index() if opt_parts else pd.Series(dtype=float)
    benchmark_returns = pd.concat(bench_parts).sort_index() if bench_parts else pd.Series(dtype=float)

    equal_returns = equal_returns[~equal_returns.index.duplicated(keep="first")]
    optimized_returns = optimized_returns[~optimized_returns.index.duplicated(keep="first")]
    benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep="first")]

    weights_df = pd.DataFrame(weight_records)

    benchmark_metrics = portfolio_metrics(benchmark_returns)

    return {
        "equal_metrics": portfolio_metrics(equal_returns, benchmark_returns),
        "optimized_metrics": portfolio_metrics(optimized_returns, benchmark_returns),
        "benchmark_metrics": benchmark_metrics,
        "equal_returns": equal_returns,
        "optimized_returns": optimized_returns,
        "benchmark_returns": benchmark_returns,
        "weights": weights_df,
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================

def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maakt een DataFrame veilig voor Excel.
    Excel ondersteunt geen timezone-aware datetimes.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    for col in out.columns:
        try:
            if pd.api.types.is_datetime64tz_dtype(out[col]):
                out[col] = out[col].dt.tz_convert(None)

            elif out[col].dtype == "object":
                def fix_value(v):
                    if isinstance(v, pd.Timestamp):
                        if v.tzinfo is not None:
                            return v.tz_convert(None).to_pydatetime()
                        return v.to_pydatetime()

                    if isinstance(v, datetime):
                        if v.tzinfo is not None:
                            return v.replace(tzinfo=None)
                        return v

                    return v

                out[col] = out[col].apply(fix_value)

        except Exception:
            # Als een kolom niet geconverteerd kan worden, laat hem met rust.
            pass

    return out

def make_export_frames_excel_safe(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Maakt meerdere export-DataFrames veilig voor Excel.
    """
    safe_frames = {}

    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame):
            safe_frames[name] = make_excel_safe(frame)
        else:
            safe_frames[name] = frame

    return safe_frames


def empty_price_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Momentum",
            "Volatility",
            "Beta",
            "MaxDrawdown",
        ]
    )


def empty_news_items_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Ticker",
            "Title",
            "Summary",
            "Publisher",
            "SourceType",
            "PublishedAt",
            "Link",
            "SearchQuery",
            "CompanyMatch",
            "MatchReason",
            "Relevance",
            "RecencyScore",
            "SentimentScore",
            "SentimentLabel",
            "SourceQuality",
            "ImpactScore",
            "ImpactCategory",
            "ClickbaitPenalty",
            "NewsItemScore",
        ]
    )


def empty_news_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Ticker",
            "NewsScore",
            "NewsSentiment",
            "NewsCount",
            "PositiveNewsCount",
            "NegativeNewsCount",
            "NeutralNewsCount",
            "TopHeadline",
            "TopPublisher",
            "TopImpactCategory",
            "AvgNewsRelevance",
            "AvgCompanyMatch",
            "AvgSourceQuality",
            "AvgImpactScore",
            "AvgClickbaitPenalty",
        ]
    )


def ensure_results_output_path(output_name: str) -> str:
    """
    Stuurt alle .xlsx en fallback .csv output naar de /results map naast dit script.
    Ook als je per ongeluk een absoluut pad meegeeft, gebruiken we alleen de bestandsnaam.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if output_name is None or str(output_name).strip() == "":
        output_name = "hedge_fund_v12_3_results.xlsx"

    output_name = os.path.basename(str(output_name))

    if not output_name.lower().endswith(".xlsx"):
        output_name += ".xlsx"

    return os.path.join(results_dir, output_name)


def is_valid_ticker_format(ticker: str) -> bool:
    """
    Basis-validatie voor yfinance tickers.
    Staat normale tickers toe zoals:
    AAPL, MSFT, BRK-B, ASML.AS, 2222.SR, BTC-USD
    """
    ticker = str(ticker).strip().upper()

    if not ticker:
        return False

    if len(ticker) > 18:
        return False

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=")

    return all(c in allowed for c in ticker)


def clean_ticker_list(raw_tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Verwijdert lege/ongeldige tickers en geeft rejected tickers terug.
    """
    clean = []
    rejected = []

    for t in raw_tickers:
        ticker = str(t).strip().upper()

        if not ticker:
            continue

        if is_valid_ticker_format(ticker):
            clean.append(ticker)
        else:
            rejected.append(ticker)

    clean = sorted(set(clean))
    rejected = sorted(set(rejected))

    return clean, rejected


def validate_data_item(d: Dict) -> Tuple[bool, List[str]]:
    """
    Controleert of een opgehaald aandeel bruikbaar genoeg is.
    Niet alles met missende fundamentals wordt weggegooid, want yfinance mist soms data.
    Maar tickers zonder prijsdata of met extreem verdachte basisdata krijgen waarschuwingen.
    """
    reasons = []

    ticker = d.get("Ticker", "UNKNOWN")

    ph = d.get("PriceHistory")

    if ph is None or not isinstance(ph, pd.DataFrame) or ph.empty:
        reasons.append("geen geldige prijsdata")

    elif len(ph) < 60:
        reasons.append(f"te weinig prijsdata ({len(ph)} dagen)")

    market_cap = safe_float(d.get("MarketCap", np.nan))
    avg_volume = safe_float(d.get("AverageVolume", np.nan))
    sector = str(d.get("Sector", "Unknown"))

    if np.isnan(market_cap):
        reasons.append("market cap ontbreekt")

    if np.isnan(avg_volume):
        reasons.append("average volume ontbreekt")

    if sector == "Unknown":
        reasons.append("sector onbekend")

    # Niet automatisch skippen bij fundamentals-missing, maar wel signaleren.
    hard_fail = False

    if ph is None or not isinstance(ph, pd.DataFrame) or ph.empty:
        hard_fail = True

    elif len(ph) < 60:
        hard_fail = True

    return (not hard_fail), reasons


def print_invalid_ticker_report(
    requested_tickers: List[str],
    returned_data: List[Dict],
    rejected_format: List[str],
) -> None:
    """
    Print alleen een rapport als er echt iets mis is.
    """
    returned = {str(d.get("Ticker", "")).upper() for d in returned_data if d.get("Ticker")}
    requested = {str(t).upper() for t in requested_tickers}

    missing = sorted(requested - returned)

    if not rejected_format and not missing:
        return

    if rejected_format:
        logger.warning(
            "Ongeldige ticker-format overgeslagen: %s",
            ", ".join(rejected_format),
        )

    if missing:
        logger.warning(
            "Deze tickers leverden geen bruikbare data op en zijn overgeslagen: %s",
            ", ".join(missing),
        )

def build_settings_df(args) -> pd.DataFrame:
    settings = {
        "version": "V12.3",
        "mode": args.mode,
        "top_n": args.top_n,
        "objective": args.objective,
        "max_weight": args.max_weight,
        "max_sector_weight": args.max_sector_weight,
        "rebalance": args.rebalance,
        "history_period": args.history_period,
        "lookback_days": args.lookback_days,
        "cache_ttl": args.cache_ttl,
        "min_marketcap": args.min_marketcap,
        "min_volume": args.min_volume,
        "transaction_cost": args.transaction_cost,
        "blend_absolute": args.blend_absolute,
        "confidence_adjust": not args.no_confidence_adjust,
        "long_short": args.long_short,
        "gpu": args.gpu,
        "regime_filter": not args.no_regime_filter,
        "workers": args.workers,
        "benchmark": BENCHMARK_TICKER,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "news_enabled": not args.no_news,
        "news_days": args.news_days,
        "news_max": args.news_max,
        "news_weight_max": args.news_weight,
        "news_min_relevance": args.news_min_relevance,
        "v12_3_no_news_neutral": True,
        "v12_3_news_confidence_enabled": True,
        "news_min_items_for_full_weight": NEWS_MIN_ITEMS_FOR_FULL_WEIGHT,
        "news_min_confidence_for_effect": NEWS_MIN_CONFIDENCE_FOR_EFFECT,
        "news_default_no_news_score": NEWS_DEFAULT_NO_NEWS_SCORE,
        "ranking_score": "FinalScore",
        "rating_score": "FinalScore",
        "backtest_score_scope": "quant-only; news is not used historically because free news is not reliable point-in-time data",
        "important_limitation": "Fundamentals from yfinance are current snapshots, not true point-in-time historical fundamentals.",
    }

    return pd.DataFrame(list(settings.items()), columns=["Setting", "Value"])


def build_comparison_df(bt: Dict) -> pd.DataFrame:
    rows = []

    for name, key in [
        ("Equal Weight", "equal_metrics"),
        ("Optimized", "optimized_metrics"),
        ("Benchmark", "benchmark_metrics"),
    ]:
        metrics = bt.get(key, {})
        row = {"Portfolio": name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Lite V11 - single-file quant screener/backtester")

    parser.add_argument("--tickers", "-t", default=None, help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--watchlist", default=None, help="CSV file with tickers. Column can be Ticker or first column.")

    parser.add_argument("--mode", "-m", choices=WEIGHTS.keys(), default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--objective", choices=["sharpe", "minvar", "risk_parity"], default="sharpe")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--max-sector-weight", type=float, default=0.35)

    parser.add_argument("--rebalance", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--history-period", default=DEFAULT_HISTORY_PERIOD, help="yfinance period, e.g. 1y, 2y, 3y, 5y")
    parser.add_argument("--lookback-days", type=int, default=126)

    parser.add_argument("--cache-ttl", type=int, default=DEFAULT_CACHE_TTL_HOURS)
    parser.add_argument("--max-retries", type=int, default=3)

    parser.add_argument("--min-marketcap", type=float, default=0.0)
    parser.add_argument("--min-volume", type=float, default=0.0)

    parser.add_argument("--transaction-cost", type=float, default=0.001, help="0.001 = 0.10 percent per 100 percent turnover")
    parser.add_argument("--blend-absolute", type=float, default=0.70)

    parser.add_argument("--no-confidence-adjust", action="store_true")
    parser.add_argument("--long-short", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use optional CuPy GPU covariance if available")
    parser.add_argument("--no-regime-filter", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-news", action="store_true", help="Disable V12 news fetching")
    parser.add_argument("--news-days", type=int, default=NEWS_DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--news-max", type=int, default=NEWS_DEFAULT_MAX_ITEMS)
    parser.add_argument("--news-weight", type=float, default=NEWS_DEFAULT_WEIGHT)
    parser.add_argument("--news-min-relevance", type=float, default=NEWS_MIN_RELEVANCE)
    parser.add_argument("--output", "-o", default="hedge_fund_v12_3_results.xlsx")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

        # ============================================================
    # INTERACTIVE TICKER INPUT
    # ============================================================
    if args.watchlist:
        raw_tickers = load_watchlist_csv(args.watchlist)

    elif args.tickers:
        raw_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    else:
        print("=" * 70)
        print("HEDGE FUND LITE V12.3")
        print("=" * 70)
        typed = input("Voer tickers in, gescheiden door komma: ").strip()
        raw_tickers = [t.strip().upper() for t in typed.split(",") if t.strip()]

    tickers, rejected_format = clean_ticker_list(raw_tickers)

    if rejected_format:
        logger.warning("Deze tickers hebben een ongeldig format en worden overgeslagen: %s", ", ".join(rejected_format))

    if not tickers:
        logger.error("Geen geldige tickers opgegeven.")
        return

    # ============================================================
    # INTERACTIVE MODE SELECTION
    # ============================================================
    if args.mode is None:
        valid_modes = list(WEIGHTS.keys())

        print()
        print("=" * 70)
        print("KIES STRATEGIE-MODE")
        print("=" * 70)

        for i, mode_name in enumerate(valid_modes, start=1):
            print(f"{i}. {mode_name}")

        while True:
            choice = input("Kies mode met nummer of naam [balanced]: ").strip().lower()

            if choice == "":
                args.mode = "balanced"
                break

            if choice.isdigit():
                idx = int(choice)

                if 1 <= idx <= len(valid_modes):
                    args.mode = valid_modes[idx - 1]
                    break

            if choice in valid_modes:
                args.mode = choice
                break

            print("Ongeldige keuze. Kies bijvoorbeeld 1, 2, 3, 4 of balanced.")

        print(f"Gekozen mode: {args.mode}")
        print()

    tickers = sorted(set(tickers))

    if not tickers:
        logger.error("Geen tickers opgegeven.")
        return

    if args.top_n <= 0:
        logger.warning("top-n moet groter zijn dan 0. Ik zet hem op 5.")
        args.top_n = 5

    if args.lookback_days < 30:
        logger.warning("lookback-days is erg laag. Ik zet hem op 30.")
        args.lookback_days = 30

    if not 0 <= args.blend_absolute <= 1:
        logger.warning("blend-absolute moet tussen 0 en 1 liggen. Ik zet hem op 0.70.")
        args.blend_absolute = 0.70

    if args.max_weight <= 0:
        logger.warning("max-weight moet groter zijn dan 0. Ik zet hem op 0.25.")
        args.max_weight = 0.25

    if not 0 < args.max_sector_weight <= 1:
        logger.warning("max-sector-weight moet tussen 0 en 1 liggen. Ik zet hem op 0.35.")
        args.max_sector_weight = 0.35

    # ============================================================
    # OUTPUT ALTIJD NAAR RESULTS MAP
    # ============================================================
    args.output = ensure_results_output_path(args.output)
    logger.info("Output wordt opgeslagen in: %s", args.output)



    if args.gpu:
        try:
            import cupy  # noqa: F401
            logger.info("GPU requested and CuPy seems importable.")
        except Exception:
            logger.warning("GPU requested, but CuPy is not available. Falling back to CPU.")
            args.gpu = False

    logger.info("Fetching benchmark %s", BENCHMARK_TICKER)

    spy_raw = yf.Ticker(BENCHMARK_TICKER).history(period=args.history_period, auto_adjust=True)
    spy_hist = standardize_history(spy_raw)

    if spy_hist.empty:
        logger.warning("No SPY data found. Benchmark/beta/regime calculations may be weaker.")

    logger.info("Fetching data for %d tickers", len(tickers))

    data = fetch_all_data(
        tickers=tickers,
        spy_hist=spy_hist,
        period=args.history_period,
        workers=max(1, args.workers),
        max_retries=args.max_retries,
        cache_ttl_hours=args.cache_ttl,
    )

    print_invalid_ticker_report(tickers, data, rejected_format)

    if not data:
        logger.error("Geen data opgehaald. Controleer je tickers of internetverbinding.")
        return

    filtered = []

    for d in data:
        ticker = d.get("Ticker", "UNKNOWN")

        ok_data, data_warnings = validate_data_item(d)

        if data_warnings:
            if ok_data:
                logger.info("%s data-opmerking: %s", ticker, "; ".join(data_warnings))
            else:
                logger.warning("%s data-waarschuwing: %s", ticker, "; ".join(data_warnings))

        if not ok_data:
            logger.warning("%s wordt overgeslagen door onvoldoende prijsdata.", ticker)
            continue

        mc = safe_float(d.get("MarketCap", np.nan))
        av = safe_float(d.get("AverageVolume", np.nan))

        if args.min_marketcap > 0 and (np.isnan(mc) or mc < args.min_marketcap):
            logger.warning(
                "%s overgeslagen door market cap filter. MarketCap=%s, min=%s",
                ticker,
                format_number_or_missing(mc),
                format_number_or_missing(args.min_marketcap),
            )
            continue

        if args.min_volume > 0 and (np.isnan(av) or av < args.min_volume):
            logger.warning(
                "%s overgeslagen door volume filter. AverageVolume=%s, min=%s",
                ticker,
                format_number_or_missing(av),
                format_number_or_missing(args.min_volume),
            )
            continue

        filtered.append(d)

    data = filtered

    if not data:
        logger.error("No data left after filters.")
        return

    df = pd.DataFrame(data)

    df_scored = compute_factor_scores(df, blend_absolute=args.blend_absolute)
    df_scored = score_total(df_scored, args.mode, confidence_adjust=not args.no_confidence_adjust)

    # ============================================================
    # V12 NEWS SCORING
    # ============================================================

    news_df = pd.DataFrame()
    news_summary_df = empty_news_summary_df()

    if not args.no_news:
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser is not installed. Google News RSS will be skipped.")
        if not VADER_AVAILABLE:
            logger.warning("vaderSentiment is not installed. Sentiment will use keyword fallback.")

        news_df, news_summary_df = build_news_for_universe(
            df=df_scored,
            max_items_per_source=args.news_max,
            lookback_days=args.news_days,
            min_relevance=args.news_min_relevance,
        )

        if not news_summary_df.empty:
            df_scored = df_scored.merge(news_summary_df, on="Ticker", how="left")

    # ============================================================
    # V12.3 NEWS DEFAULTS + FINAL SCORE
    # ============================================================
    # Geen nieuws = geen invloed. Zwak/onzeker nieuws krijgt weinig of geen gewicht.

    news_defaults = {
        "NewsScore": NEWS_DEFAULT_NO_NEWS_SCORE,
        "NewsSentiment": 50.0,
        "NewsCount": 0,
        "PositiveNewsCount": 0,
        "NegativeNewsCount": 0,
        "NeutralNewsCount": 0,
        "TopHeadline": "",
        "TopPublisher": "",
        "TopImpactCategory": "",
        "AvgNewsRelevance": np.nan,
        "AvgCompanyMatch": np.nan,
        "AvgSourceQuality": np.nan,
        "AvgImpactScore": np.nan,
        "AvgClickbaitPenalty": np.nan,
    }

    for col, default in news_defaults.items():
        if col not in df_scored.columns:
            df_scored[col] = default

    numeric_news_cols = [
        "NewsScore",
        "NewsSentiment",
        "NewsCount",
        "PositiveNewsCount",
        "NegativeNewsCount",
        "NeutralNewsCount",
        "AvgNewsRelevance",
        "AvgCompanyMatch",
        "AvgSourceQuality",
        "AvgImpactScore",
        "AvgClickbaitPenalty",
    ]

    for col in numeric_news_cols:
        df_scored[col] = pd.to_numeric(df_scored[col], errors="coerce")

    df_scored["NewsScore"] = df_scored["NewsScore"].fillna(NEWS_DEFAULT_NO_NEWS_SCORE)
    df_scored["NewsSentiment"] = df_scored["NewsSentiment"].fillna(50.0)
    df_scored["NewsCount"] = df_scored["NewsCount"].fillna(0).astype(int)
    df_scored["PositiveNewsCount"] = df_scored["PositiveNewsCount"].fillna(0).astype(int)
    df_scored["NegativeNewsCount"] = df_scored["NegativeNewsCount"].fillna(0).astype(int)
    df_scored["NeutralNewsCount"] = df_scored["NeutralNewsCount"].fillna(0).astype(int)

    if not 0 <= args.news_weight <= 1:
        logger.warning("news-weight must be between 0 and 1. Setting to %.2f", NEWS_DEFAULT_WEIGHT)
        args.news_weight = NEWS_DEFAULT_WEIGHT

    df_scored["NewsConfidence"] = df_scored.apply(compute_news_confidence, axis=1)
    df_scored["EffectiveNewsWeight"] = (args.news_weight * df_scored["NewsConfidence"]).clip(0, args.news_weight)
    df_scored["NewsStatus"] = df_scored.apply(classify_news_status, axis=1)

    df_scored["FinalScore"] = (
        df_scored["Total"]
        + df_scored["EffectiveNewsWeight"] * (df_scored["NewsScore"] - df_scored["Total"])
    ).clip(0, 100)

    # Rating en ranking gebruiken FinalScore. Backtest blijft quant-only.
    df_scored = add_rating(df_scored, score_col="FinalScore")
    df_scored = df_scored.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    df_scored["Rank"] = df_scored.index + 1

    warnings_df = create_warnings(df_scored)

    price_frames = []

    for row in data:
        ph = row.get("PriceHistory")
        ticker = row.get("Ticker")

        if ph is not None and isinstance(ph, pd.DataFrame) and not ph.empty and ticker:
            p = ph.rename(columns={"Adj Close": ticker})
            price_frames.append(p[[ticker]])

    if not price_frames:
        logger.error("No price histories available.")
        return

    prices = pd.concat(price_frames, axis=1, sort=False).sort_index()
    prices = prices.ffill().dropna(how="all")
    prices = prices.loc[:, ~prices.columns.duplicated()]

    rebalance_dates = make_rebalance_dates(prices.index, args.rebalance)

    logger.info("Running walk-forward backtest with %d rebalance dates.", len(rebalance_dates))

    bt = backtest(
        prices=prices,
        rebalance_dates=rebalance_dates,
        universe_df=df_scored,
        spy_prices=spy_hist,
        top_n=args.top_n,
        mode=args.mode,
        lookback_days=args.lookback_days,
        objective=args.objective,
        max_weight=args.max_weight,
        max_sector_weight=args.max_sector_weight,
        transaction_cost=args.transaction_cost,
        long_only=not args.long_short,
        blend_absolute=args.blend_absolute,
        confidence_adjust=not args.no_confidence_adjust,
        use_gpu=args.gpu,
        regime_filter=not args.no_regime_filter,
    )

    cols_out = [
        "Rank",
        "Ticker",
        "Rating",
        "FinalScore",
        "Total",
        "Raw_Total",
        "NewsScore",
        "NewsConfidence",
        "EffectiveNewsWeight",
        "NewsStatus",
        "NewsSentiment",
        "NewsCount",
        "PositiveNewsCount",
        "NegativeNewsCount",
        "NeutralNewsCount",
        "TopHeadline",
        "TopPublisher",
        "TopImpactCategory",
        "AvgNewsRelevance",
        "AvgCompanyMatch",
        "AvgSourceQuality",
        "AvgImpactScore",
        "AvgClickbaitPenalty",
        "Sector",
        "Industry",
        "MarketCap",
        "AverageVolume",
        "Confidence",
        "Value_score",
        "Quality_score",
        "Growth_score",
        "Risk_score",
        "Stability_score",
        "Dividend_score",
        "Momentum_score",
        "Drawdown_score",
        "PE",
        "FPE",
        "PEG",
        "ROE",
        "ROA",
        "Beta",
        "DebtEquity",
        "RevenueGrowth",
        "ProfitMargin",
        "DividendYield",
        "Momentum",
        "Volatility",
        "MaxDrawdown",
    ]

    available_cols = [c for c in cols_out if c in df_scored.columns]
    display_df = df_scored[available_cols].head(args.top_n)

    logger.info("Top %d static snapshot:", args.top_n)
    logger.info("\n%s", display_df.to_string(index=False, max_rows=100))

    logger.info("Backtest equal-weight: %s", bt["equal_metrics"])
    logger.info("Backtest optimized:   %s", bt["optimized_metrics"])
    logger.info("Benchmark:            %s", bt["benchmark_metrics"])

    export_df = df_scored.drop(columns=["PriceHistory"], errors="ignore").copy()
    raw_df = df.drop(columns=["PriceHistory"], errors="ignore").copy()

    settings_df = build_settings_df(args)
    comparison_df = build_comparison_df(bt)

    # Maak basis-DataFrames veilig voor Excel.
    export_df = export_df.copy()
    raw_df = raw_df.copy()
    settings_df = settings_df.copy()
    comparison_df = comparison_df.copy()
    warnings_df = warnings_df.copy()

    safe_frames = make_export_frames_excel_safe(
        {
            "export_df": export_df,
            "raw_df": raw_df,
            "settings_df": settings_df,
            "comparison_df": comparison_df,
            "warnings_df": warnings_df,
            "news_summary_df": news_summary_df if "news_summary_df" in locals() else empty_news_summary_df(),
            "news_df": news_df if "news_df" in locals() else empty_news_items_df(),
        }
    )

    export_df = safe_frames["export_df"]
    raw_df = safe_frames["raw_df"]
    settings_df = safe_frames["settings_df"]
    comparison_df = safe_frames["comparison_df"]
    warnings_df = safe_frames["warnings_df"]
    news_summary_df = safe_frames["news_summary_df"]
    news_df = safe_frames["news_df"]

    equal_metrics_df = pd.DataFrame([bt["equal_metrics"]])
    optimized_metrics_df = pd.DataFrame([bt["optimized_metrics"]])
    benchmark_metrics_df = pd.DataFrame([bt["benchmark_metrics"]])

    equal_curve = pd.DataFrame(
        {
            "Return": bt["equal_returns"],
            "Equity": (1.0 + bt["equal_returns"]).cumprod()
            if len(bt["equal_returns"])
            else pd.Series(dtype=float),
        }
    )

    opt_curve = pd.DataFrame(
        {
            "Return": bt["optimized_returns"],
            "Equity": (1.0 + bt["optimized_returns"]).cumprod()
            if len(bt["optimized_returns"])
            else pd.Series(dtype=float),
        }
    )

    benchmark_curve = pd.DataFrame(
        {
            "Return": bt["benchmark_returns"],
            "Equity": (1.0 + bt["benchmark_returns"]).cumprod()
            if len(bt["benchmark_returns"])
            else pd.Series(dtype=float),
        }
    )

    weights_df = bt["weights"]

    safe_bt_frames = make_export_frames_excel_safe(
        {
            "equal_metrics_df": equal_metrics_df,
            "optimized_metrics_df": optimized_metrics_df,
            "benchmark_metrics_df": benchmark_metrics_df,
            "equal_curve": equal_curve,
            "opt_curve": opt_curve,
            "benchmark_curve": benchmark_curve,
            "weights_df": weights_df,
        }
    )

    equal_metrics_df = safe_bt_frames["equal_metrics_df"]
    optimized_metrics_df = safe_bt_frames["optimized_metrics_df"]
    benchmark_metrics_df = safe_bt_frames["benchmark_metrics_df"]
    equal_curve = safe_bt_frames["equal_curve"]
    opt_curve = safe_bt_frames["opt_curve"]
    benchmark_curve = safe_bt_frames["benchmark_curve"]
    weights_df = safe_bt_frames["weights_df"]

    try:
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            settings_df.to_excel(writer, sheet_name="settings", index=False)
            export_df.to_excel(writer, sheet_name="universe_snapshot", index=False)
            display_df.to_excel(writer, sheet_name="top_snapshot", index=False)
            raw_df.to_excel(writer, sheet_name="raw_data", index=False)
            warnings_df.to_excel(writer, sheet_name="warnings", index=False)

            if isinstance(news_summary_df, pd.DataFrame) and not news_summary_df.empty:
                news_summary_df.to_excel(writer, sheet_name="news_summary", index=False)

            if isinstance(news_df, pd.DataFrame) and not news_df.empty:
                news_df.to_excel(writer, sheet_name="news_items", index=False)

            comparison_df.to_excel(writer, sheet_name="comparison", index=False)
            equal_metrics_df.to_excel(writer, sheet_name="equal_metrics", index=False)
            optimized_metrics_df.to_excel(writer, sheet_name="optimized_metrics", index=False)
            benchmark_metrics_df.to_excel(writer, sheet_name="benchmark_metrics", index=False)
            equal_curve.to_excel(writer, sheet_name="equal_equity_curve")
            opt_curve.to_excel(writer, sheet_name="opt_equity_curve")
            benchmark_curve.to_excel(writer, sheet_name="benchmark_curve")

            if isinstance(weights_df, pd.DataFrame) and not weights_df.empty:
                weights_df.to_excel(writer, sheet_name="weights", index=False)

        logger.info("Saved results to %s", args.output)

    except Exception as e:
        logger.error("Failed to write Excel: %s", e)

        fallback = args.output.replace(".xlsx", ".csv")

        try:
            export_df.to_csv(fallback, index=False)
            logger.info("Saved CSV fallback to %s", fallback)
        except Exception as e2:
            logger.error("Failed to write CSV fallback: %s", e2)


if __name__ == "__main__":
    main()
