#!/usr/bin/env python3
"""
Unit tests for pt_ccxt.py — all exchange calls mocked.
No live network access required.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub out kucoin before importing pt_ccxt so Python 3.14 pkg_resources bug is bypassed
_mock_kucoin_market = MagicMock()
_kucoin_module = MagicMock()
_kucoin_module.Market = _mock_kucoin_market
sys.modules.setdefault("kucoin", _kucoin_module)
sys.modules.setdefault("kucoin.client", _kucoin_module)

import pt_ccxt
# Patch in _kucoin_available = True so adapter can be instantiated
pt_ccxt._kucoin_available = True
pt_ccxt._KucoinMarket = _mock_kucoin_market


class TestSymbolConversion(unittest.TestCase):

    def test_dash_to_slash(self):
        self.assertEqual(pt_ccxt._symbol_to_ccxt("BTC-USDT"), "BTC/USDT")

    def test_usd_pair(self):
        self.assertEqual(pt_ccxt._symbol_to_ccxt("ETH-USD"), "ETH/USD")


class TestCandleConversions(unittest.TestCase):

    # CCXT format: [ts_ms, open, high, low, close, volume]
    CCXT_RAW = [
        [1_700_000_000_000, 30000.0, 30500.0, 29800.0, 30200.0, 10.5],
        [1_700_003_600_000, 30200.0, 30700.0, 30100.0, 30600.0, 11.2],
    ]

    def test_ccxt_to_named(self):
        named = pt_ccxt._candles_to_named(self.CCXT_RAW)
        self.assertEqual(len(named), 2)
        c = named[0]
        # Timestamp should be seconds
        self.assertEqual(c["time"], 1_700_000_000)
        self.assertAlmostEqual(c["open"],   30000.0)
        self.assertAlmostEqual(c["high"],   30500.0)
        self.assertAlmostEqual(c["low"],    29800.0)
        self.assertAlmostEqual(c["close"],  30200.0)
        self.assertAlmostEqual(c["volume"], 10.5)

    def test_named_to_kucoin_raw(self):
        named = pt_ccxt._candles_to_named(self.CCXT_RAW)
        raw = pt_ccxt._candles_to_kucoin_raw(named)
        self.assertEqual(len(raw), 2)
        row = raw[0]
        # KuCoin format: [ts, open, close, high, low, vol, turnover]
        self.assertEqual(row[0], str(1_700_000_000))   # ts
        self.assertEqual(row[1], str(30000.0))          # open
        self.assertEqual(row[2], str(30200.0))          # close (index 2)
        self.assertEqual(row[3], str(30500.0))          # high  (index 3)
        self.assertEqual(row[4], str(29800.0))          # low   (index 4)
        self.assertEqual(row[5], str(10.5))             # volume
        self.assertEqual(row[6], "0")                   # turnover placeholder


class TestKuCoinAdapter(unittest.TestCase):

    def _make_kucoin_raw(self):
        # KuCoin-style: [ts, open, close, high, low, vol, turnover]
        return [
            ["1700000000", "30000", "30200", "30500", "29800", "10.5", "315000"],
            ["1700003600", "30200", "30600", "30700", "30100", "11.2", "339520"],
        ]

    @patch("pt_ccxt._KucoinMarket")
    def test_get_kline_returns_raw_list(self, mock_cls):
        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_kucoin_raw()
        mock_cls.return_value = mock_market

        adapter = pt_ccxt._KuCoinAdapter()
        result = adapter.get_kline("BTC-USDT", "1hour", limit=200)

        self.assertEqual(len(result), 2)
        # First element is ts string
        self.assertEqual(result[0][0], "1700000000")

    @patch("pt_ccxt._KucoinMarket")
    def test_get_ticker(self, mock_cls):
        mock_market = MagicMock()
        mock_market.get_ticker.return_value = {"price": "30250.0"}
        mock_cls.return_value = mock_market

        adapter = pt_ccxt._KuCoinAdapter()
        ticker = adapter.get_ticker("BTC-USDT")
        self.assertEqual(ticker["price"], "30250.0")

    @patch("pt_ccxt._KucoinMarket")
    def test_get_kline_named(self, mock_cls):
        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_kucoin_raw()
        mock_cls.return_value = mock_market

        adapter = pt_ccxt._KuCoinAdapter()
        named = adapter.get_kline_named("BTC-USDT", "1hour", limit=2)
        self.assertEqual(len(named), 2)
        self.assertIn("close", named[0])
        self.assertAlmostEqual(named[0]["close"], 30200.0)


class TestCcxtAdapter(unittest.TestCase):

    CCXT_OHLCV = [
        [1_700_000_000_000, 30000.0, 30500.0, 29800.0, 30200.0, 10.5],
    ]

    def _make_mock_exchange(self):
        ex = MagicMock()
        ex.fetch_ohlcv.return_value = self.CCXT_OHLCV
        ex.fetch_ticker.return_value = {"last": 30250.0}
        return ex

    @patch("pt_ccxt._get_ccxt_exchange")
    def test_get_kline_named(self, mock_get_ex):
        mock_get_ex.return_value = self._make_mock_exchange()
        adapter = pt_ccxt.CcxtAdapter("binance")
        named = adapter.get_kline_named("BTC-USDT", "1hour", limit=100)
        self.assertEqual(len(named), 1)
        self.assertAlmostEqual(named[0]["close"], 30200.0)

    @patch("pt_ccxt._get_ccxt_exchange")
    def test_get_kline_returns_kucoin_format(self, mock_get_ex):
        mock_get_ex.return_value = self._make_mock_exchange()
        adapter = pt_ccxt.CcxtAdapter("binance")
        raw = adapter.get_kline("BTC-USDT", "1hour", limit=100)
        self.assertEqual(len(raw), 1)
        # close is index 2 in KuCoin format
        self.assertEqual(raw[0][2], str(30200.0))

    @patch("pt_ccxt._get_ccxt_exchange")
    def test_get_ticker(self, mock_get_ex):
        mock_get_ex.return_value = self._make_mock_exchange()
        adapter = pt_ccxt.CcxtAdapter("binance")
        ticker = adapter.get_ticker("BTC-USDT")
        self.assertEqual(ticker["price"], "30250.0")

    @patch("pt_ccxt._get_ccxt_exchange")
    def test_get_ticker_on_exception(self, mock_get_ex):
        ex = MagicMock()
        ex.fetch_ticker.side_effect = RuntimeError("network fail")
        mock_get_ex.return_value = ex
        adapter = pt_ccxt.CcxtAdapter("binance")
        ticker = adapter.get_ticker("BTC-USDT")
        self.assertEqual(ticker, {})


class TestTimeframeMapping(unittest.TestCase):

    def test_standard_timeframes(self):
        self.assertEqual(pt_ccxt._TF_MAP["1hour"], "1h")
        self.assertEqual(pt_ccxt._TF_MAP["4hour"], "4h")
        self.assertEqual(pt_ccxt._TF_MAP["1day"],  "1d")
        self.assertEqual(pt_ccxt._TF_MAP["1week"], "1w")


class TestPerCoinConfig(unittest.TestCase):

    def test_default_is_kucoin(self):
        with patch.dict(os.environ, {"CCXT_EXCHANGE": ""}, clear=False):
            ex_id = pt_ccxt._exchange_for_coin("BTC")
        self.assertEqual(ex_id, "kucoin")

    def test_env_var_override(self):
        with patch.dict(os.environ, {"CCXT_EXCHANGE": "binance"}, clear=False):
            ex_id = pt_ccxt._exchange_for_coin("BTC")
        self.assertEqual(ex_id, "binance")

    def test_per_coin_file_override(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"BTC": "bybit", "ETH": "okx"}, f)
            cfg_path = f.name
        try:
            with patch("pt_ccxt._COIN_CONFIG_PATH", cfg_path), \
                 patch("pt_ccxt._coin_config_mtime", None), \
                 patch("pt_ccxt._coin_exchange_map", {}):
                bt = pt_ccxt._exchange_for_coin("BTC")
                eth = pt_ccxt._exchange_for_coin("ETH")
                sol = pt_ccxt._exchange_for_coin("SOL")
        finally:
            os.unlink(cfg_path)
        self.assertEqual(bt, "bybit")
        self.assertEqual(eth, "okx")


class TestGetMarketClient(unittest.TestCase):

    def test_kucoin_returns_kucoin_adapter(self):
        with patch.dict(os.environ, {"CCXT_EXCHANGE": ""}, clear=False), \
             patch("pt_ccxt._KucoinMarket"):
            client = pt_ccxt.get_market_client("BTC")
        self.assertIsInstance(client, pt_ccxt._KuCoinAdapter)

    def test_ccxt_returns_ccxt_adapter(self):
        mock_ex = MagicMock()
        with patch.dict(os.environ, {"CCXT_EXCHANGE": "binance"}, clear=False), \
             patch("pt_ccxt._get_ccxt_exchange", return_value=mock_ex):
            client = pt_ccxt.get_market_client("BTC")
        self.assertIsInstance(client, pt_ccxt.CcxtAdapter)


if __name__ == "__main__":
    unittest.main()
