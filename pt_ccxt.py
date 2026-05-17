#!/usr/bin/env python3
"""
PowerTrader AI — CCXT Multi-Exchange Data Adapter (Ph4)

Provides a drop-in replacement for the KuCoin market-data client used in
pt_thinker.py and pt_trader.py.  KuCoin remains the default; CCXT is
activated when CCXT_EXCHANGE is set.

Configuration (env vars):
    CCXT_EXCHANGE     — exchange id, e.g. "binance", "bybit", "okx"
                        Leave unset (or "kucoin") to keep the KuCoin path.
    CCXT_API_KEY      — optional API key (public data works without it)
    CCXT_API_SECRET   — optional API secret
    CCXT_TESTNET      — set to "1" to use exchange sandbox/testnet

Output format
-------------
get_kline() returns candles as a list of dicts (named fields), matching
the shape produced by pt_replay.py fetch_historical_klines().  When called
from the raw thinker/trainer code that expects integer-indexed lists, use
get_kline_raw() which returns the KuCoin-style list format:
    [ts, open, close, high, low, vol, turnover]
    index 2 = close, 3 = high, 4 = low  (non-standard, matches KuCoin)

get_ticker() returns {"price": str} to match kucoin-python's interface.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

# ── KuCoin fallback ───────────────────────────────────────────────────────────
try:
    from kucoin.client import Market as _KucoinMarket
    _kucoin_available = True
except (ImportError, Exception):
    _kucoin_available = False
    _KucoinMarket = None  # type: ignore[assignment,misc]

# ── CCXT optional import ──────────────────────────────────────────────────────
try:
    import ccxt
    _ccxt_available = True
except ImportError:
    _ccxt_available = False

# ── timeframe map (KuCoin string → CCXT string) ───────────────────────────────
_TF_MAP: Dict[str, str] = {
    "1hour":  "1h",
    "2hour":  "2h",
    "4hour":  "4h",
    "8hour":  "8h",
    "12hour": "12h",
    "1day":   "1d",
    "1week":  "1w",
    "1min":   "1m",
    "3min":   "3m",
    "5min":   "5m",
    "15min":  "15m",
    "30min":  "30m",
}

# ── per-coin exchange override  ───────────────────────────────────────────────
# Loaded from ccxt_config.json if present.
# Format: {"BTC": "binance", "ETH": "bybit"}
_coin_exchange_map: Dict[str, str] = {}
_coin_config_mtime: Optional[float] = None
_COIN_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccxt_config.json")


def _load_coin_config() -> Dict[str, str]:
    global _coin_exchange_map, _coin_config_mtime
    try:
        if not os.path.isfile(_COIN_CONFIG_PATH):
            return {}
        mtime = os.path.getmtime(_COIN_CONFIG_PATH)
        if mtime == _coin_config_mtime:
            return dict(_coin_exchange_map)
        import json
        with open(_COIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _coin_exchange_map = {str(k).upper(): str(v).lower() for k, v in data.items()}
            _coin_config_mtime = mtime
    except Exception:
        pass
    return dict(_coin_exchange_map)


def _exchange_for_coin(coin: str) -> str:
    cfg = _load_coin_config()
    if coin.upper() in cfg:
        return cfg[coin.upper()]
    val = os.environ.get("CCXT_EXCHANGE", "").lower().strip()
    return val if val else "kucoin"


# ── CCXT exchange instance cache ──────────────────────────────────────────────
_exchange_instances: Dict[str, Any] = {}


def _get_ccxt_exchange(exchange_id: str) -> Any:
    if exchange_id in _exchange_instances:
        return _exchange_instances[exchange_id]
    if not _ccxt_available:
        raise RuntimeError("ccxt is not installed. Run: pip install ccxt")
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"CCXT exchange '{exchange_id}' not found.")
    params: Dict[str, Any] = {
        "enableRateLimit": True,
    }
    api_key = os.environ.get("CCXT_API_KEY", "").strip()
    api_secret = os.environ.get("CCXT_API_SECRET", "").strip()
    if api_key:
        params["apiKey"] = api_key
    if api_secret:
        params["secret"] = api_secret
    if os.environ.get("CCXT_TESTNET", "").strip() == "1":
        params["sandbox"] = True
    ex = cls(params)
    _exchange_instances[exchange_id] = ex
    return ex


# ── helpers ────────────────────────────────────────────────────────────────────

def _symbol_to_ccxt(kucoin_pair: str) -> str:
    """Convert 'BTC-USDT' or 'BTC-USD' → 'BTC/USDT'."""
    return kucoin_pair.replace("-", "/")


def _candles_to_named(raw_ccxt: List[List]) -> List[Dict]:
    """Convert CCXT OHLCV list → named-dict candles (replay format)."""
    out = []
    for c in raw_ccxt:
        out.append({
            "time":     int(c[0] / 1000),
            "open":     float(c[1]),
            "close":    float(c[4]),
            "high":     float(c[2]),
            "low":      float(c[3]),
            "volume":   float(c[5]),
        })
    return out


def _candles_to_kucoin_raw(named: List[Dict]) -> List[List]:
    """Convert named-dict candles → KuCoin-style indexed lists.
    KuCoin format: [ts, open, close, high, low, vol, turnover]
    """
    out = []
    for c in named:
        ts  = int(c.get("time", 0))
        op  = c.get("open", 0.0)
        cl  = c.get("close", 0.0)
        hi  = c.get("high", 0.0)
        lo  = c.get("low", 0.0)
        vol = c.get("volume", 0.0)
        out.append([str(ts), str(op), str(cl), str(hi), str(lo), str(vol), "0"])
    return out


# ── KuCoin fallback client ────────────────────────────────────────────────────

class _KuCoinAdapter:
    """Thin adapter so KuCoin path looks identical to CcxtAdapter."""

    def __init__(self) -> None:
        if not _kucoin_available or _KucoinMarket is None:
            raise RuntimeError("kucoin-python is not installed or failed to import.")
        self._client = _KucoinMarket(url="https://api.kucoin.com")

    def get_kline(self, pair: str, timeframe: str, limit: int = 200) -> List[List]:
        """Return KuCoin-style raw candle lists (for thinker/trainer compat)."""
        return self._client.get_kline(pair, timeframe, limit=limit) or []

    def get_ticker(self, pair: str) -> Dict[str, str]:
        return self._client.get_ticker(pair) or {}

    def get_kline_named(self, pair: str, timeframe: str, limit: int = 200) -> List[Dict]:
        raw = self.get_kline(pair, timeframe, limit=limit)
        named = []
        for c in raw:
            named.append({
                "time":   int(c[0]),
                "open":   float(c[1]),
                "close":  float(c[2]),
                "high":   float(c[3]),
                "low":    float(c[4]),
                "volume": float(c[5]),
            })
        return named


class CcxtAdapter:
    """
    CCXT-backed data adapter.  Implements the same interface as _KuCoinAdapter.
    """

    def __init__(self, exchange_id: str = "binance") -> None:
        self._exchange_id = exchange_id
        self._ex = _get_ccxt_exchange(exchange_id)

    def get_kline(self, pair: str, timeframe: str, limit: int = 200) -> List[List]:
        """Return KuCoin-style raw candle lists for backward compat with thinker/trainer."""
        named = self.get_kline_named(pair, timeframe, limit)
        return _candles_to_kucoin_raw(named)

    def get_kline_named(self, pair: str, timeframe: str, limit: int = 200) -> List[Dict]:
        ccxt_tf = _TF_MAP.get(timeframe, timeframe)
        ccxt_sym = _symbol_to_ccxt(pair)
        raw = self._ex.fetch_ohlcv(ccxt_sym, ccxt_tf, limit=limit)
        return _candles_to_named(raw or [])

    def get_ticker(self, pair: str) -> Dict[str, str]:
        ccxt_sym = _symbol_to_ccxt(pair)
        try:
            t = self._ex.fetch_ticker(ccxt_sym)
            return {"price": str(t.get("last", 0.0))}
        except Exception:
            return {}


# ── public factory ─────────────────────────────────────────────────────────────

def get_market_client(coin: str = "") -> Any:
    """
    Return a market data client for the given coin.

    Selection priority:
    1. Per-coin ccxt_config.json entry
    2. CCXT_EXCHANGE env var
    3. KuCoin (default, no env var needed)

    Returns either a CcxtAdapter or a _KuCoinAdapter instance.
    Both expose .get_kline(), .get_kline_named(), .get_ticker().
    """
    exchange_id = _exchange_for_coin(coin)
    if exchange_id in ("kucoin", "", None):
        return _KuCoinAdapter()
    return CcxtAdapter(exchange_id)


# ── module-level singleton (default KuCoin, compatible with existing code) ────
# Usage:  from pt_ccxt import market_client
#         candles = market_client.get_kline("BTC-USDT", "1hour", limit=200)
try:
    market_client = get_market_client()
except Exception:
    market_client = None  # type: ignore[assignment]


if __name__ == "__main__":
    import json
    print("Testing market client…")
    client = get_market_client("BTC")
    candles = client.get_kline_named("BTC-USDT", "1hour", limit=5)
    print(f"Got {len(candles)} candles:")
    for c in candles:
        print(f"  {c}")
    ticker = client.get_ticker("BTC-USDT")
    print(f"Ticker: {ticker}")
