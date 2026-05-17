#!/usr/bin/env python3
"""Unit tests for pt_alerts.py — all network calls mocked."""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pt_alerts


class TestRateLimiting(unittest.TestCase):

    def setUp(self):
        # Clear rate-limit state between tests
        with pt_alerts._lock:
            pt_alerts._last_sent.clear()

    def test_throttle_blocks_duplicate(self):
        with pt_alerts._lock:
            pt_alerts._last_sent["SIGNAL:BTC"] = time.time()
        self.assertTrue(pt_alerts._is_throttled("SIGNAL", "BTC"))

    def test_throttle_allows_after_window(self):
        with pt_alerts._lock:
            pt_alerts._last_sent["SIGNAL:BTC"] = time.time() - 9999
        self.assertFalse(pt_alerts._is_throttled("SIGNAL", "BTC"))

    def test_mark_sent_records_timestamp(self):
        pt_alerts._mark_sent("TRADE", "ETH")
        with pt_alerts._lock:
            self.assertIn("TRADE:ETH", pt_alerts._last_sent)

    def test_force_bypasses_throttle(self):
        with pt_alerts._lock:
            pt_alerts._last_sent["TRADE:BTC"] = time.time()
        # force=True should always call through (we just check it doesn't short-circuit)
        with patch("pt_alerts._send_telegram", return_value=False) as mock_tg, \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.send("TRADE", "test", coin="BTC", force=True)
            mock_tg.assert_called_once()


class TestTelegramDispatch(unittest.TestCase):

    def setUp(self):
        with pt_alerts._lock:
            pt_alerts._last_sent.clear()

    def test_skips_when_no_token(self):
        env = {"TELEGRAM_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
        with patch.dict(os.environ, env, clear=False):
            result = pt_alerts._send_telegram("hello")
        self.assertFalse(result)

    def test_sends_when_configured(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        env = {"TELEGRAM_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "456"}
        with patch.dict(os.environ, env, clear=False), \
             patch("requests.post", return_value=mock_resp) as mock_post:
            result = pt_alerts._send_telegram("test msg")
        self.assertTrue(result)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        self.assertIn("tok123", call_kwargs[0][0])

    def test_returns_false_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        env = {"TELEGRAM_TOKEN": "tok", "TELEGRAM_CHAT_ID": "1"}
        with patch.dict(os.environ, env, clear=False), \
             patch("requests.post", return_value=mock_resp):
            result = pt_alerts._send_telegram("bad")
        self.assertFalse(result)

    def test_returns_false_on_network_exception(self):
        env = {"TELEGRAM_TOKEN": "tok", "TELEGRAM_CHAT_ID": "1"}
        with patch.dict(os.environ, env, clear=False), \
             patch("requests.post", side_effect=ConnectionError("offline")):
            result = pt_alerts._send_telegram("fail")
        self.assertFalse(result)


class TestDiscordDispatch(unittest.TestCase):

    def setUp(self):
        with pt_alerts._lock:
            pt_alerts._last_sent.clear()

    def test_skips_when_no_webhook(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": ""}, clear=False):
            self.assertFalse(pt_alerts._send_discord("hi"))

    def test_sends_when_configured(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.example/wh"}, clear=False), \
             patch("requests.post", return_value=mock_resp):
            self.assertTrue(pt_alerts._send_discord("hello discord"))


class TestConvenienceWrappers(unittest.TestCase):

    def setUp(self):
        with pt_alerts._lock:
            pt_alerts._last_sent.clear()

    def _mock_send(self):
        return patch("pt_alerts._send_telegram", return_value=False), \
               patch("pt_alerts._send_discord", return_value=False)

    def test_signal_alert(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.signal_alert("BTC", 5, 0, 3)

    def test_trade_alert_buy(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.trade_alert("buy", "ETH", 0.5, 3000.0, tag="ENTRY")

    def test_trade_alert_sell_with_pnl(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.trade_alert("sell", "BTC", 0.001, 95000.0, tag="TRAIL_SELL", pnl_pct=5.3)

    def test_pnl_alert(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.pnl_alert(250.0, 62.5, 8)

    def test_health_alert(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.health_alert("OK", 12.5, "paper")

    def test_error_alert(self):
        with patch("pt_alerts._send_telegram", return_value=False), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.error_alert("KuCoin API timeout", coin="BTC")


class TestAlertPrefix(unittest.TestCase):

    def setUp(self):
        with pt_alerts._lock:
            pt_alerts._last_sent.clear()

    def test_unknown_type_gets_bell_prefix(self):
        captured = {}
        def fake_telegram(text):
            captured["text"] = text
            return False
        with patch("pt_alerts._send_telegram", side_effect=fake_telegram), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.send("UNKNOWN", "hello", force=True)
        self.assertIn("🔔", captured.get("text", ""))

    def test_trade_prefix(self):
        captured = {}
        def fake_telegram(text):
            captured["text"] = text
            return False
        with patch("pt_alerts._send_telegram", side_effect=fake_telegram), \
             patch("pt_alerts._send_discord", return_value=False):
            pt_alerts.send("TRADE", "buy 1 BTC", force=True)
        self.assertIn("💱", captured.get("text", ""))


if __name__ == "__main__":
    unittest.main()
