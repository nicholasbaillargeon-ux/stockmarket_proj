"""Market data fetching with in-memory TTL cache."""

import time
from typing import Optional
import pandas as pd
import yfinance as yf

_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def _cached(key: str, fetch_fn, ttl: int = CACHE_TTL):
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < ttl:
        return _cache[key]["data"]
    data = fetch_fn()
    _cache[key] = {"data": data, "ts": now}
    return data


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Return daily adjusted close prices as a DataFrame (columns = tickers)."""
    key = f"prices_{'_'.join(sorted(tickers))}_{period}"

    def fetch():
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
            if len(tickers) == 1:
                closes.columns = tickers
        closes = closes.dropna(how="all")
        return closes

    return _cached(key, fetch)


def fetch_info(ticker: str) -> dict:
    """Return key info fields for a single ticker."""
    key = f"info_{ticker}"

    def fetch():
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "—"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
        }

    return _cached(key, fetch, ttl=600)


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Return OHLCV data for a single ticker (for candlestick charts)."""
    key = f"ohlcv_{ticker}_{period}"

    def fetch():
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    return _cached(key, fetch)


def clear_cache():
    _cache.clear()
