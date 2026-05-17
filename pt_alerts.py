#!/usr/bin/env python3
"""
PowerTrader AI — Alert Dispatcher (Ph3)

Sends trade, signal, PnL, and health alerts to Telegram and/or Discord.
Silently skips if env vars are not configured.

Configuration (env vars):
    TELEGRAM_TOKEN      — Telegram bot token
    TELEGRAM_CHAT_ID    — Chat/channel ID to post to
    DISCORD_WEBHOOK_URL — Full Discord webhook URL

Alert types: SIGNAL, TRADE, PNL, HEALTH, ERROR
Rate limits (per type, per coin): see _RATE_LIMITS
"""

from __future__ import annotations

import os
import time
import threading
import requests
from typing import Optional

# ── rate limits (seconds between same alert type, per coin) ──────────────────
_RATE_LIMITS = {
    "SIGNAL": 120,   # 2 min between same signal level alerts
    "TRADE":  10,    # 10 s — nearly immediate; each trade is unique
    "PNL":    300,   # 5 min between PnL summaries
    "HEALTH": 3600,  # 1 hour between health pings
    "ERROR":  600,   # 10 min between repeated error alerts
}

_lock = threading.Lock()
_last_sent: dict[str, float] = {}   # key = f"{alert_type}:{coin}" → unix timestamp


def _throttle_key(alert_type: str, coin: str = "") -> str:
    return f"{alert_type}:{coin.upper()}"


def _is_throttled(alert_type: str, coin: str = "") -> bool:
    key = _throttle_key(alert_type, coin)
    limit = _RATE_LIMITS.get(alert_type, 60)
    with _lock:
        last = _last_sent.get(key, 0.0)
        return (time.time() - last) < limit


def _mark_sent(alert_type: str, coin: str = "") -> None:
    key = _throttle_key(alert_type, coin)
    with _lock:
        _last_sent[key] = time.time()


# ── Telegram ──────────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Discord ───────────────────────────────────────────────────────────────────

def _send_discord(text: str) -> bool:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    try:
        r = requests.post(webhook, json={"content": text}, timeout=8)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── public API ────────────────────────────────────────────────────────────────

def send(
    alert_type: str,
    message: str,
    coin: str = "",
    force: bool = False,
) -> bool:
    """
    Dispatch an alert to all configured channels.

    Parameters
    ----------
    alert_type : str
        One of SIGNAL, TRADE, PNL, HEALTH, ERROR.
    message : str
        Human-readable alert text (plain text; HTML tags OK for Telegram).
    coin : str
        Coin ticker used for rate-limit bucketing (e.g. "BTC").
    force : bool
        Bypass rate limiting (e.g. for critical errors).

    Returns True if at least one channel accepted the message.
    """
    alert_type = alert_type.upper()
    if not force and _is_throttled(alert_type, coin):
        return False

    prefix = {
        "SIGNAL": "📡",
        "TRADE":  "💱",
        "PNL":    "💰",
        "HEALTH": "🩺",
        "ERROR":  "🚨",
    }.get(alert_type, "🔔")

    full_text = f"{prefix} <b>[PowerTrader {alert_type}]</b>\n{message}"

    sent = _send_telegram(full_text) | _send_discord(f"{prefix} **[PowerTrader {alert_type}]**\n{message}")

    if sent:
        _mark_sent(alert_type, coin)
    return sent


# ── convenience wrappers ──────────────────────────────────────────────────────

def signal_alert(coin: str, long_sig: int, short_sig: int, threshold: int) -> bool:
    msg = (
        f"Coin: <b>{coin}</b>\n"
        f"Long signal: {long_sig}/7  Short signal: {short_sig}/7\n"
        f"Entry threshold: {threshold}"
    )
    return send("SIGNAL", msg, coin=coin)


def trade_alert(
    side: str,
    coin: str,
    qty: float,
    price: float,
    tag: str = "",
    pnl_pct: Optional[float] = None,
) -> bool:
    side_icon = "🟢 BUY" if side.lower() == "buy" else "🔴 SELL"
    pnl_str = f"  PnL: {pnl_pct:+.2f}%" if pnl_pct is not None else ""
    msg = (
        f"{side_icon}  <b>{coin}</b>\n"
        f"Qty: {qty:.6f}  @ ${price:,.4f}{pnl_str}\n"
        f"Tag: {tag or 'ENTRY'}"
    )
    return send("TRADE", msg, coin=coin, force=True)


def pnl_alert(total_realized: float, win_rate: float, trade_count: int) -> bool:
    direction = "▲" if total_realized >= 0 else "▼"
    msg = (
        f"{direction} Realized PnL: <b>${total_realized:+,.2f}</b>\n"
        f"Win rate: {win_rate:.1f}%  ({trade_count} closed trades)"
    )
    return send("PNL", msg)


def health_alert(status: str, uptime_h: float, mode: str = "live") -> bool:
    msg = (
        f"Status: {status}  Mode: {mode.upper()}\n"
        f"Uptime: {uptime_h:.1f} h"
    )
    return send("HEALTH", msg, force=False)


def error_alert(error_msg: str, coin: str = "") -> bool:
    return send("ERROR", error_msg, coin=coin)


# ── test util (run standalone to verify config) ───────────────────────────────

if __name__ == "__main__":
    print("Sending test alert…")
    ok = send("HEALTH", "Alert system online — test message.", force=True)
    print("Delivered" if ok else "No channels configured (set TELEGRAM_TOKEN/CHAT_ID or DISCORD_WEBHOOK_URL).")
