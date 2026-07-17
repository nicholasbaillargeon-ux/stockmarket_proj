"""Cache-layer tests. Fully offline: no Redis server and no network required.

The core cache-behaviour tests run against BOTH backends: the in-process
fallback dict and a fakeredis-backed real Redis client. fakeredis exercises the
actual get/setex/scan_iter/delete paths without a server.
"""

import json

import fakeredis
import numpy as np
import pandas as pd
import pytest
import redis

import data as dt


@pytest.fixture(autouse=True, params=["memory", "redis"])
def _isolate_cache(request, monkeypatch):
    """Run each test once per backend, always from a clean slate."""
    if request.param == "memory":
        monkeypatch.setattr(dt, "_client", None)
    else:
        monkeypatch.setattr(dt, "_client", fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setattr(dt, "_client_init", True)  # skip REDIS_URL lookup
    dt._memory_cache.clear()
    yield request.param
    dt._memory_cache.clear()


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "AAPL": [1.5, 2.5, np.nan, 3.5],       # float with a gap
            "MSFT": [4.0, 5.0, 6.0, 7.0],          # whole-number floats
            "VOL": [1000, 2000, 3000, 4000],       # genuine ints
            "PREC": [100.123456789012, 1.0000000001, 2.2, 3.3],
        },
        # Explicit dates, not date_range: real market indexes skip weekends and
        # therefore carry no freq. JSON can't round-trip a freq attribute anyway.
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
    )


# ── Serialization ─────────────────────────────────────────────────────────────
def test_dataframe_roundtrip_is_faithful(frame):
    back = dt._decode(dt._encode(frame))
    pd.testing.assert_frame_equal(back, frame)


def test_whole_number_floats_stay_float(frame):
    """Regression: read_json used to infer int64 for [4.0, 5.0, 6.0]."""
    back = dt._decode(dt._encode(frame))
    assert back["MSFT"].dtype == np.float64
    assert back["VOL"].dtype == np.int64  # real ints are still ints


def test_roundtrip_preserves_nan_and_precision(frame):
    back = dt._decode(dt._encode(frame))
    assert back["AAPL"].isna().iloc[2]
    assert back["PREC"].iloc[0] == frame["PREC"].iloc[0]


def test_empty_dataframe_roundtrip():
    assert dt._decode(dt._encode(pd.DataFrame())).empty


def test_dict_roundtrip():
    info = {"name": "Apple", "pe": 31.2, "dividend_yield": None}
    assert dt._decode(dt._encode(info)) == info


def test_decode_tolerates_entry_without_dtypes(frame):
    """Entries cached before dtype-carrying was added must still decode."""
    import json

    legacy = json.dumps({"t": "df", "v": frame.to_json(orient="split", date_format="iso")})
    assert not dt._decode(legacy).empty


def test_encode_never_uses_pickle(frame):
    """Payload must be plain JSON — a tampered entry can't execute code."""
    import json

    assert json.loads(dt._encode(frame))["t"] == "df"


# ── Cache behaviour (runs against both backends) ──────────────────────────────
def test_second_call_hits_cache(frame):
    """A repeat call is served from cache — true for Redis and the memory dict."""
    calls = []

    def fetch():
        calls.append(1)
        return frame

    first = dt._cached("k", fetch, ttl=60)
    second = dt._cached("k", fetch, ttl=60)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_should_cache_gate_skips_write_but_returns_data(_isolate_cache):
    """A result the gate rejects is returned, but never stored — a throttled
    ticker search comes back [] rather than raising, and pinning that for the
    full TTL would leave the search bar dead for that query."""
    results = [[], [{"symbol": "NVDA"}]]
    fetched = []

    def fetch():
        fetched.append(1)
        return results.pop(0)

    assert dt._cached("s", fetch, ttl=60, should_cache=bool) == []
    # The empty result was not cached, so the retry refetches and now gets a hit.
    assert dt._cached("s", fetch, ttl=60, should_cache=bool) == [{"symbol": "NVDA"}]
    assert len(fetched) == 2
    # That hit passes the gate, so it *is* cached.
    assert dt._cached("s", fetch, ttl=60, should_cache=bool) == [{"symbol": "NVDA"}]
    assert len(fetched) == 2


def test_clear_cache_drops_entries(_isolate_cache, frame):
    dt._cached("k", lambda: frame, ttl=60)
    dt.clear_cache()
    calls = []
    dt._cached("k", lambda: (calls.append(1), frame)[1], ttl=60)
    assert len(calls) == 1  # entry was gone, so it refetched


def test_clear_cache_spares_foreign_keys(_isolate_cache):
    """clear_cache must only delete this app's pa:* keys, not co-tenant data."""
    if _isolate_cache != "redis":
        pytest.skip("foreign-key isolation only applies to the shared Redis backend")
    dt._client.set("other_app:keep", "1")
    dt._cached("mine", lambda: {"x": 1}, ttl=60)
    dt.clear_cache()
    assert dt._client.get("other_app:keep") == "1"
    assert dt._client.get("pa:mine") is None


def test_keys_are_namespaced(_isolate_cache, frame):
    dt._cached("prices_AAPL_1y", lambda: frame, ttl=60)
    if _isolate_cache == "redis":
        assert all(k.startswith(dt.KEY_PREFIX) for k in dt._client.keys("*"))
    else:
        assert all(k.startswith(dt.KEY_PREFIX) for k in dt._memory_cache)


def test_redis_entry_carries_ttl(_isolate_cache, frame):
    """A Redis write must set an expiry so stale market data can't live forever."""
    if _isolate_cache != "redis":
        pytest.skip("TTL is a Redis-key property")
    dt._cached("k", lambda: frame, ttl=123)
    assert 0 < dt._client.ttl("pa:k") <= 123


def test_expired_memory_entry_refetches(frame, monkeypatch):
    """In-memory fallback honours ttl and evicts the stale entry on read."""
    monkeypatch.setattr(dt, "_client", None)
    calls = []

    def fetch():
        calls.append(1)
        return frame

    now = [1_000.0]
    monkeypatch.setattr(dt.time, "time", lambda: now[0])

    dt._cached("k", fetch, ttl=60)
    now[0] += 61
    dt._cached("k", fetch, ttl=60)
    assert len(calls) == 2


def test_memory_cache_is_bounded(monkeypatch):
    """The fallback dict must not grow without bound when Redis is absent."""
    monkeypatch.setattr(dt, "_client", None)
    for i in range(dt.MEMORY_CACHE_MAX + 50):
        dt._cached(f"k{i}", lambda i=i: {"v": i}, ttl=600)
    assert len(dt._memory_cache) <= dt.MEMORY_CACHE_MAX


# ── Numeric column names (regression for read_json axis inference) ────────────
def test_numeric_column_name_survives_roundtrip():
    """A Tokyo-style all-numeric column name must keep its str type and dtype."""
    df = pd.DataFrame(
        {"7203": [150.0, 151.5, 152.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    back = dt._decode(dt._encode(df))
    assert list(back.columns) == ["7203"]
    assert back["7203"].dtype == np.float64
    pd.testing.assert_frame_equal(back, df)


# ── Degradation when Redis misbehaves ─────────────────────────────────────────
class _BrokenRedis:
    """A client that is reachable-looking but every op raises (outage/flapping)."""

    def get(self, key):
        raise redis.ConnectionError("redis is down")

    def setex(self, key, ttl, value):
        raise redis.ConnectionError("redis is down")


def test_app_still_serves_when_redis_is_down(frame, monkeypatch):
    """A dead Redis must degrade to the in-memory cache, not disable caching."""
    monkeypatch.setattr(dt, "_client", _BrokenRedis())
    dt._memory_cache.clear()
    calls = []

    def fetch():
        calls.append(1)
        return frame

    first = dt._cached("k", fetch, ttl=60)
    second = dt._cached("k", fetch, ttl=60)  # must hit the in-memory fallback
    pd.testing.assert_frame_equal(first, frame)
    pd.testing.assert_frame_equal(second, frame)
    assert len(calls) == 1                    # not refetched every time
    assert dt._memory_cache                    # fallback actually populated


def test_corrupt_entry_heals_instead_of_raising(_isolate_cache, monkeypatch):
    """A garbage/hostile cache entry must be refetched, never raise to the caller."""
    if _isolate_cache != "redis":
        pytest.skip("seeding a corrupt entry requires the Redis backend")
    for bad in ["not json at all", json.dumps({"t": "df", "v": "garbage"}),
                json.dumps({"no_t_key": 1})]:
        dt._client.set("pa:k", bad)
        calls = []
        got = dt._cached("k", lambda: (calls.append(1), {"ok": True})[1], ttl=60)
        assert got == {"ok": True}
        assert len(calls) == 1
