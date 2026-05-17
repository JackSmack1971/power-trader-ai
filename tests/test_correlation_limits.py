"""Tests for correlation-aware portfolio limits in pt_trader.py."""
import math
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# Must be set before pt_trader is imported — guards the module-level credential check
if "--paper" not in sys.argv:
    sys.argv.append("--paper")

# Stub heavy dependencies before import
for _mod in ("nacl", "nacl.signing", "cryptography",
             "cryptography.hazmat", "cryptography.hazmat.primitives",
             "cryptography.hazmat.primitives.asymmetric",
             "cryptography.hazmat.primitives.asymmetric.ed25519",
             "cryptography.hazmat.primitives.serialization"):
    sys.modules.setdefault(_mod, MagicMock())

_kucoin_pkg = types.ModuleType("kucoin")
_kucoin_client = types.ModuleType("kucoin.client")
_kucoin_client.Market = MagicMock()
sys.modules.setdefault("kucoin", _kucoin_pkg)
sys.modules.setdefault("kucoin.client", _kucoin_client)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pt_trader as _pt_module

_pt_module.PAPER_MODE = True


def _make_trader():
    """Return a minimal CryptoAPITrading instance with paper mode."""
    with patch.object(_pt_module, "_kucoin_market", MagicMock()), \
         patch("builtins.open", unittest.mock.mock_open(read_data="")), \
         patch("os.path.isfile", return_value=False), \
         patch.object(_pt_module.CryptoAPITrading, "_apply_optimizer_config"), \
         patch.object(_pt_module.CryptoAPITrading, "_init_paper_account"), \
         patch.object(_pt_module.CryptoAPITrading, "_load_pnl_ledger", return_value=[]), \
         patch.object(_pt_module.CryptoAPITrading, "initialize_dca_levels"), \
         patch.object(_pt_module.CryptoAPITrading, "_seed_dca_window_from_history"):
        trader = _pt_module.CryptoAPITrading.__new__(_pt_module.CryptoAPITrading)
        # Minimal init attributes
        trader.correlation_limit_enabled = True
        trader.max_correlated_positions  = 2
        trader.correlation_threshold     = 0.85
        trader.correlation_window        = 10
        trader.atr_sizing_enabled        = False
        trader.kelly_sizing_enabled      = False
        trader.stop_loss_enabled         = False
        trader._stop_levels              = {}
        trader._paper                    = {"holdings": {}, "cash": 10000.0}
        return trader


class TestCalcCorrelationMatrix(unittest.TestCase):
    def _make_candles(self, prices):
        """Convert a list of close prices to KuCoin-format candles (reversed order)."""
        candles = []
        for i, p in enumerate(prices):
            candles.append([str(i * 3600), "0", str(p), str(p * 1.01), str(p * 0.99), "100", "0"])
        return list(reversed(candles))

    def test_perfect_positive_correlation(self):
        trader = _make_trader()
        prices = [100.0 + i for i in range(15)]
        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_candles(prices)
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC", "ETH"])
        self.assertAlmostEqual(matrix["BTC"]["ETH"], 1.0, places=5)

    def test_negative_correlation(self):
        # Zigzag prices: BTC alternates high/low, ETH alternates low/high → returns are sign-flipped
        trader = _make_trader()
        prices_a = [110.0 if i % 2 == 0 else 90.0 for i in range(16)]
        prices_b = [90.0  if i % 2 == 0 else 110.0 for i in range(16)]
        mock_market = MagicMock()

        def side_effect(pair, tf, limit):
            if pair.startswith("BTC"):
                return self._make_candles(prices_a)
            return self._make_candles(prices_b)

        mock_market.get_kline.side_effect = side_effect
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC", "ETH"])
        self.assertLess(matrix["BTC"]["ETH"], -0.9)

    def test_self_correlation_is_one(self):
        trader = _make_trader()
        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_candles([100, 102, 101, 103])
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC", "ETH"])
        self.assertEqual(matrix["BTC"]["BTC"], 1.0)
        self.assertEqual(matrix["ETH"]["ETH"], 1.0)

    def test_api_failure_defaults_to_zero(self):
        trader = _make_trader()
        mock_market = MagicMock()
        mock_market.get_kline.side_effect = Exception("api down")
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC", "ETH"])
        self.assertEqual(matrix["BTC"].get("ETH", 0.0), 0.0)

    def test_single_symbol_returns_empty_peers(self):
        trader = _make_trader()
        mock_market = MagicMock()
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC"])
        self.assertEqual(matrix, {"BTC": {}})

    def test_matrix_is_symmetric(self):
        trader = _make_trader()
        prices = [100.0 + math.sin(i * 0.5) * 5 for i in range(20)]
        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_candles(prices)
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            matrix = trader._calc_correlation_matrix(["BTC", "ETH", "XRP"])
        for a in ["BTC", "ETH", "XRP"]:
            for b in ["BTC", "ETH", "XRP"]:
                self.assertAlmostEqual(matrix[a][b], matrix[b][a], places=10)


class TestIsCorrelatedEntryBlocked(unittest.TestCase):
    def _make_candles_sync(self, prices):
        candles = []
        for i, p in enumerate(prices):
            candles.append([str(i * 3600), "0", str(p), str(p * 1.01), str(p * 0.99), "100", "0"])
        return list(reversed(candles))

    def test_feature_disabled_never_blocks(self):
        trader = _make_trader()
        trader.correlation_limit_enabled = False
        result = trader._is_correlated_entry_blocked("SOL", ["BTC", "ETH"])
        self.assertFalse(result)

    def test_fewer_holdings_than_limit_never_blocks(self):
        trader = _make_trader()
        trader.max_correlated_positions = 3
        # Only 1 holding — below limit of 3
        result = trader._is_correlated_entry_blocked("SOL", ["BTC"])
        self.assertFalse(result)

    def test_high_correlation_blocks_at_limit(self):
        trader = _make_trader()
        trader.max_correlated_positions = 2
        trader.correlation_threshold = 0.85
        prices = [100.0 + i for i in range(15)]

        mock_market = MagicMock()
        mock_market.get_kline.return_value = self._make_candles_sync(prices)
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            blocked = trader._is_correlated_entry_blocked("SOL", ["BTC", "ETH"])
        # All three perfectly correlated; would push correlated_count to 3 > limit 2
        self.assertTrue(blocked)

    def test_low_correlation_does_not_block(self):
        # SOL and BTC have same zigzag (r=1.0); ETH is opposite zigzag (r=-1.0 with SOL/BTC).
        # correlated_count for SOL: 1(self) + 1(BTC) = 2; ETH is negative → not counted.
        # 2 > max_correlated_positions(2) is False → not blocked.
        trader = _make_trader()
        trader.max_correlated_positions = 2
        trader.correlation_threshold = 0.85

        hi_lo = [110.0 if i % 2 == 0 else 90.0 for i in range(16)]
        lo_hi = [90.0  if i % 2 == 0 else 110.0 for i in range(16)]

        def side_effect(pair, tf, limit):
            if pair.startswith("SOL") or pair.startswith("BTC"):
                return self._make_candles_sync(hi_lo)
            return self._make_candles_sync(lo_hi)

        mock_market = MagicMock()
        mock_market.get_kline.side_effect = side_effect
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            blocked = trader._is_correlated_entry_blocked("SOL", ["BTC", "ETH"])
        self.assertFalse(blocked)

    def test_api_exception_does_not_block(self):
        trader = _make_trader()
        trader.max_correlated_positions = 2
        mock_market = MagicMock()
        mock_market.get_kline.side_effect = Exception("network error")
        with patch.object(_pt_module, "_kucoin_market", mock_market):
            blocked = trader._is_correlated_entry_blocked("SOL", ["BTC", "ETH"])
        self.assertFalse(blocked)

    def test_empty_holdings_never_blocks(self):
        trader = _make_trader()
        result = trader._is_correlated_entry_blocked("BTC", [])
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
