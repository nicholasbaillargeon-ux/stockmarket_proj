"""Market data fetching with a Redis-backed TTL cache.

The cache is shared across gunicorn workers. If Redis is unreachable the module
degrades to a per-process in-memory cache so the app still serves.

Values are serialized as JSON, never pickle, so a corrupted or tampered cache
entry cannot execute code when it is read back.
"""

import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from io import StringIO

import pandas as pd
import redis
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_TTL = 300  # 5 minutes
INFO_TTL = 600
KEY_PREFIX = "pa:"
MEMORY_CACHE_MAX = 256  # bound the fallback dict so a dead Redis can't leak memory

_memory_cache: "OrderedDict[str, dict]" = OrderedDict()
_client: redis.Redis | None = None
_client_init = False
_client_lock = threading.Lock()


def _redis() -> redis.Redis | None:
    """Lazily build the Redis client. None when REDIS_URL is unset."""
    global _client, _client_init
    if _client_init:
        return _client
    with _client_lock:
        # Double-checked: another thread may have finished init while we waited.
        if not _client_init:
            url = os.environ.get("REDIS_URL")
            client = None
            if url:
                client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                )
            _client = client
            _client_init = True  # set last: the flag must never be True before _client
    return _client


# ── Serialization (JSON only — no pickle) ─────────────────────────────────────
def _encode(value) -> str:
    if isinstance(value, pd.DataFrame):
        body = value.to_json(orient="split", date_format="iso", double_precision=15)
        # read_json re-infers dtypes, so a float column of whole numbers (a $150.00
        # close) would come back int64. Carry the dtypes and restore them verbatim.
        dtypes = {str(c): str(t) for c, t in value.dtypes.items()}
        return json.dumps({"t": "df", "v": body, "d": dtypes})
    return json.dumps({"t": "json", "v": value})


def _decode(raw: str):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "t" not in payload:
        raise ValueError("malformed cache payload")
    if payload["t"] == "df":
        # convert_axes=False: read_json would otherwise re-type an all-numeric
        # column axis (e.g. Tokyo tickers like "7203"), voiding the dtypes map.
        df = pd.read_json(StringIO(payload["v"]), orient="split", convert_axes=False)
        df.index = pd.to_datetime(df.index)
        dtypes = {c: t for c, t in (payload.get("d") or {}).items() if c in df.columns}
        return df.astype(dtypes) if dtypes else df
    return payload["v"]


def _memory_get(key: str, ttl: int):
    """Read the in-process fallback, dropping the entry if it has expired."""
    hit = _memory_cache.get(key)
    if not hit:
        return None
    if time.time() - hit["ts"] >= ttl:
        del _memory_cache[key]  # evict on read so dead keys don't linger
        return None
    _memory_cache.move_to_end(key)  # LRU touch
    return hit["data"]


def _memory_put(key: str, data) -> None:
    """Write the in-process fallback, evicting the oldest entries past the cap."""
    _memory_cache[key] = {"data": data, "ts": time.time()}
    _memory_cache.move_to_end(key)
    while len(_memory_cache) > MEMORY_CACHE_MAX:
        _memory_cache.popitem(last=False)


# ── Cache core ────────────────────────────────────────────────────────────────
def _cached(key: str, fetch_fn, ttl: int = CACHE_TTL):
    key = KEY_PREFIX + key
    client = _redis()
    # Track liveness locally: a Redis op that fails must degrade to the in-memory
    # fallback for BOTH read and write, not leave caching silently disabled.
    redis_ok = client is not None

    if redis_ok:
        try:
            raw = client.get(key)
            if raw is not None:
                return _decode(raw)
        except (redis.RedisError, ValueError, KeyError, TypeError) as exc:
            # A corrupt/hostile entry (TypeError/ValueError) or an outage
            # (RedisError) both fall through to a refetch that overwrites it.
            log.warning("Redis read failed for %s (%s); refetching", key, exc)
            if isinstance(exc, redis.RedisError):
                redis_ok = False

    if not redis_ok:
        cached = _memory_get(key, ttl)
        if cached is not None:
            return cached

    data = fetch_fn()

    if redis_ok:
        try:
            client.set(key, _encode(data), ex=ttl)
        except (redis.RedisError, TypeError, ValueError) as exc:
            log.warning("Redis write failed for %s (%s)", key, exc)
            if isinstance(exc, redis.RedisError):
                redis_ok = False
    if not redis_ok:
        _memory_put(key, data)

    return data


def clear_cache() -> None:
    """Drop every cache entry this app owns, in both backends."""
    _memory_cache.clear()
    client = _redis()
    if client is None:
        return
    try:
        for key in client.scan_iter(match=f"{KEY_PREFIX}*", count=500):
            client.delete(key)
    except redis.RedisError as exc:
        log.warning("Redis clear failed (%s)", exc)


# ── Fetchers ──────────────────────────────────────────────────────────────────
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

    return _cached(key, fetch, ttl=INFO_TTL)


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Return OHLCV data for a single ticker (for candlestick charts)."""
    key = f"ohlcv_{ticker}_{period}"

    def fetch():
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    return _cached(key, fetch)
