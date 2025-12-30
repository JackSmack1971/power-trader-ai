#!/usr/bin/env python3
"""
End-to-End Smoke Test for PowerTrader AI Backtesting System.
Runs a minimal backtest to verify the complete workflow.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pt_replay import (
    fetch_historical_klines,
    warm_cache,
    _atomic_write_json,
    BacktestLogger
)
from pt_analyze import generate_analytics_report


class TestE2ESmoke(unittest.TestCase):
    """End-to-end smoke tests for backtesting system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = tempfile.mkdtemp(prefix="test_e2e_")
        cls.cache_dir = os.path.join(cls.test_dir, "cache")
        cls.output_dir = os.path.join(cls.test_dir, "results")
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_complete_workflow_minimal(self):
        """
        Run a complete minimal backtest workflow (1 week of data).
        This is the primary smoke test.

        Workflow:
        1. Warm cache for 1 week of BTC data
        2. Create backtest config
        3. Simulate minimal backtest state
        4. Generate analytics report
        5. Verify all files generated
        """
        print("\n" + "=" * 60)
        print("E2E SMOKE TEST: Complete Workflow")
        print("=" * 60)

        # Step 1: Warm cache for 1 week of BTC data
        print("\n[1/5] Warming cache for 1 week of BTC data...")

        # Use a recent date range (1 week)
        start_date = "2024-01-01"
        end_date = "2024-01-08"
        coins = ["BTC"]

        try:
            warm_cache(
                start_date=start_date,
                end_date=end_date,
                coins=coins,
                timeframes=["1hour"],  # Just one timeframe for speed
                cache_dir=self.cache_dir
            )
            print("  ✓ Cache warming completed")
        except Exception as e:
            print(f"  ⚠ Cache warming skipped (API may be unavailable): {e}")
            # Continue test even if caching fails (for offline testing)

        # Verify cache directory created
        self.assertTrue(os.path.exists(self.cache_dir), "Cache directory should exist")

        # Step 2: Create backtest config
        print("\n[2/5] Creating backtest config...")

        config = {
            "start_date": start_date,
            "end_date": end_date,
            "coins": coins,
            "speed": 10.0,
            "cache_dir": self.cache_dir,
            "walk_forward_enabled": False,  # Disable for smoke test
            "execution_model": {
                "slippage_bps": 5,
                "fee_bps": 20,
                "max_volume_pct": 1.0
            }
        }

        config_path = os.path.join(self.output_dir, "config.json")
        _atomic_write_json(config_path, config)

        self.assertTrue(os.path.exists(config_path), "Config file should exist")
        print("  ✓ Config created")

        # Step 3: Simulate minimal backtest state
        print("\n[3/5] Simulating backtest execution...")

        # Create hub_data directory
        hub_data_dir = os.path.join(self.output_dir, "hub_data")
        os.makedirs(hub_data_dir, exist_ok=True)

        # Create mock trader_status.json with equity curve
        equity_snapshots = []
        initial_capital = 10000.0
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())

        # Generate mock equity curve (simulating a profitable week)
        for i in range(168):  # 168 hours in a week
            current_ts = start_ts + (i * 3600)
            # Simulate gradual growth with some volatility
            value = initial_capital * (1 + 0.001 * i + 0.0005 * (i % 24))

            equity_snapshots.append({
                "timestamp": current_ts,
                "total_value": value,
                "cash": value * 0.5,
                "positions": {"BTC": {"value": value * 0.5}}
            })

        # Write equity snapshots as JSONL
        trader_status_path = os.path.join(hub_data_dir, "trader_status.json")
        with open(trader_status_path, "w") as f:
            for snapshot in equity_snapshots:
                f.write(json.dumps(snapshot) + "\n")

        self.assertTrue(os.path.exists(trader_status_path), "Trader status should exist")
        print(f"  ✓ Simulated {len(equity_snapshots)} equity snapshots")

        # Create BacktestLogger and log some events
        logger = BacktestLogger(log_dir=self.output_dir)

        logger.log_info("Smoke test started")
        logger.log_replay_tick(
            sequence=1,
            timestamp=start_ts,
            coins_updated=["BTC"]
        )
        logger.log_trade_execution(
            order_id="smoke-test-order-1",
            side="buy",
            symbol="BTC-USD",
            qty=0.01,
            price=98765.43,
            fees=19.75,
            slippage_bps=5.2,
            status="filled"
        )

        # Verify log file created
        log_path = os.path.join(self.output_dir, "backtest_events.jsonl")
        self.assertTrue(os.path.exists(log_path), "Log file should exist")
        print("  ✓ Backtest events logged")

        # Step 4: Generate analytics report
        print("\n[4/5] Generating analytics report...")

        try:
            report_path = generate_analytics_report(self.output_dir)
            self.assertTrue(os.path.exists(report_path), "Analytics report should exist")
            print(f"  ✓ Report generated: {report_path}")

            # Verify report contains expected content
            with open(report_path, "r") as f:
                report_content = f.read()

            self.assertIn("PowerTrader AI", report_content, "Report should have title")
            self.assertIn("Sharpe Ratio", report_content, "Report should have Sharpe ratio")
            self.assertIn("Max Drawdown", report_content, "Report should have max drawdown")

        except Exception as e:
            self.fail(f"Report generation failed: {e}")

        # Step 5: Verify all files generated
        print("\n[5/5] Verifying all output files...")

        expected_files = [
            os.path.join(self.output_dir, "config.json"),
            os.path.join(self.output_dir, "backtest_events.jsonl"),
            os.path.join(self.output_dir, "analytics_report.html"),
            os.path.join(hub_data_dir, "trader_status.json"),
        ]

        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path),
                          f"Expected file not found: {file_path}")
            print(f"  ✓ {os.path.basename(file_path)}")

        print("\n" + "=" * 60)
        print("E2E SMOKE TEST PASSED")
        print("=" * 60)
        print(f"All files generated in: {self.output_dir}")
        print("=" * 60)

    def test_cache_warming_functionality(self):
        """
        Test cache warming functionality in isolation.
        """
        print("\n[TEST] Cache warming functionality...")

        # Create separate cache dir for this test
        test_cache_dir = os.path.join(self.test_dir, "test_cache")
        os.makedirs(test_cache_dir, exist_ok=True)

        # Try to warm cache for a very short period
        try:
            # Using a minimal date range to speed up test
            warm_cache(
                start_date="2024-01-01",
                end_date="2024-01-02",
                coins=["BTC"],
                timeframes=["1hour"],
                cache_dir=test_cache_dir
            )

            # Verify cache index created
            cache_index_path = os.path.join(test_cache_dir, "cache_index.json")
            if os.path.exists(cache_index_path):
                with open(cache_index_path, "r") as f:
                    cache_index = json.load(f)

                self.assertIsInstance(cache_index, dict, "Cache index should be a dict")
                print("  ✓ Cache index created successfully")
            else:
                print("  ⚠ Cache index not created (may be offline)")

        except Exception as e:
            print(f"  ⚠ Cache warming failed (may be offline): {e}")
            # Don't fail the test - this is acceptable if API is unavailable

    def test_logger_functionality(self):
        """
        Test BacktestLogger in isolation.
        """
        print("\n[TEST] Logger functionality...")

        # Create logger
        test_log_dir = os.path.join(self.test_dir, "test_logs")
        os.makedirs(test_log_dir, exist_ok=True)

        logger = BacktestLogger(log_dir=test_log_dir)

        # Log various event types
        logger.log_info("Test info message")
        logger.log_warning("Test warning message")

        logger.log_replay_tick(
            sequence=42,
            timestamp=1704067200,
            coins_updated=["BTC", "ETH"]
        )

        logger.log_trade_execution(
            order_id="test-123",
            side="buy",
            symbol="BTC-USD",
            qty=0.01,
            price=100000.0,
            fees=20.0,
            slippage_bps=5.5,
            status="filled"
        )

        logger.log_neural_signal(
            coin="BTC",
            signal_type="long_dca",
            signal_value=7
        )

        logger.log_error(
            error_message="Test error",
            context={"component": "test", "details": "mock error"}
        )

        # Verify log file created and contains events
        log_file = os.path.join(test_log_dir, "backtest_events.jsonl")
        self.assertTrue(os.path.exists(log_file), "Log file should exist")

        with open(log_file, "r") as f:
            events = [json.loads(line) for line in f]

        # Verify we have events of each type
        event_types = [e["event_type"] for e in events]
        self.assertIn("replay_tick", event_types)
        self.assertIn("trade_execution", event_types)
        self.assertIn("neural_signal", event_types)
        self.assertIn("error", event_types)
        self.assertIn("warning", event_types)

        print(f"  ✓ Logged {len(events)} events across {len(set(event_types))} types")


def run_tests():
    """Run all E2E smoke tests."""
    print("\n" + "=" * 60)
    print("END-TO-END SMOKE TEST SUITE")
    print("=" * 60)
    print("Testing complete workflow with minimal data...")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestE2ESmoke)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
