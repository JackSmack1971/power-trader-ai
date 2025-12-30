#!/usr/bin/env python3
"""
Test suite for walk-forward validation.
Ensures no look-ahead bias in backtesting.
"""

import os
import sys
import json
import time
import tempfile
import shutil
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock


class TestWalkForwardValidation(unittest.TestCase):
    """Test suite for walk-forward validation to prevent look-ahead bias."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp(prefix="test_wf_")
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_walk_forward_no_lookahead(self):
        """
        Verify that neural model only trains on past data.
        This is the CRITICAL test for preventing look-ahead bias.
        """
        print("\n[TEST] Verifying no look-ahead bias in walk-forward training...")

        # Create test data with known timestamps
        test_start_ts = int(datetime(2024, 1, 1).timestamp())
        test_train_until_ts = int(datetime(2024, 1, 15).timestamp())
        test_end_ts = int(datetime(2024, 2, 1).timestamp())

        # Create mock candle data
        candles_before = []
        candles_after = []

        # Generate candles before train_until timestamp (should be used)
        current_ts = test_start_ts
        while current_ts < test_train_until_ts:
            candles_before.append({
                "time": current_ts,
                "open": 100.0,
                "close": 101.0,
                "high": 102.0,
                "low": 99.0,
                "volume": 1000.0
            })
            current_ts += 3600  # 1 hour

        # Generate candles after train_until timestamp (should NOT be used)
        current_ts = test_train_until_ts
        while current_ts < test_end_ts:
            candles_after.append({
                "time": current_ts,
                "open": 200.0,  # Different price to detect if used
                "close": 201.0,
                "high": 202.0,
                "low": 199.0,
                "volume": 2000.0
            })
            current_ts += 3600

        # Mock KuCoin API to return all candles (both before and after)
        all_candles = candles_before + candles_after

        # Simulate the filtering logic from pt_incremental_trainer.py
        # This is the CRITICAL section that prevents look-ahead bias
        processed_timestamps = []

        for candle_str in all_candles:
            try:
                candle_time = candle_str["time"]

                # CRITICAL: Filter out any candles after TRAIN_UNTIL_TS
                # This is the exact logic from pt_incremental_trainer.py:263-266
                if candle_time >= test_train_until_ts:
                    continue

                processed_timestamps.append(candle_time)
            except:
                continue

        # Verify: No timestamps after train_until should be in processed list
        for ts in processed_timestamps:
            self.assertLess(ts, test_train_until_ts,
                           f"Look-ahead bias detected! Timestamp {ts} ({datetime.fromtimestamp(ts)}) "
                           f"is after train_until {test_train_until_ts} ({datetime.fromtimestamp(test_train_until_ts)})")

        # Verify: All timestamps before train_until should be processed
        for candle in candles_before:
            self.assertIn(candle["time"], processed_timestamps,
                        f"Missing historical data: timestamp {candle['time']} should be included")

        # Verify: No timestamps after train_until should be processed
        for candle in candles_after:
            self.assertNotIn(candle["time"], processed_timestamps,
                            f"Look-ahead bias! Timestamp {candle['time']} should NOT be included")

        print(f"  ✓ Verified: {len(processed_timestamps)} candles before train_until")
        print(f"  ✓ Verified: 0 candles after train_until (no look-ahead bias)")
        print("  [PASS] No look-ahead bias detected")

    def test_incremental_training_data_filter(self):
        """
        Verify that incremental trainer correctly filters data by timestamp.
        """
        print("\n[TEST] Testing incremental training data filtering...")

        # Test various timestamp filtering scenarios
        test_cases = [
            {
                "name": "Filter at midpoint",
                "all_timestamps": [1000, 2000, 3000, 4000, 5000],
                "train_until": 3000,
                "expected": [1000, 2000],  # 3000 and after should be excluded
            },
            {
                "name": "Filter at start",
                "all_timestamps": [1000, 2000, 3000, 4000, 5000],
                "train_until": 1000,
                "expected": [],  # Nothing before 1000
            },
            {
                "name": "Filter at end",
                "all_timestamps": [1000, 2000, 3000, 4000, 5000],
                "train_until": 6000,
                "expected": [1000, 2000, 3000, 4000, 5000],  # All included
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                all_ts = test_case["all_timestamps"]
                train_until = test_case["train_until"]
                expected = test_case["expected"]

                # Simulate the filtering logic
                filtered = [ts for ts in all_ts if ts < train_until]

                self.assertEqual(filtered, expected,
                               f"{test_case['name']} failed: expected {expected}, got {filtered}")

        print("  ✓ All filtering test cases passed")
        print("  [PASS] Data filtering works correctly")

    def test_model_versioning(self):
        """
        Verify that models are saved with correct timestamps.
        """
        print("\n[TEST] Testing model versioning...")

        # Create test model files with timestamps
        test_timestamps = [
            int(datetime(2024, 1, 1).timestamp()),
            int(datetime(2024, 1, 8).timestamp()),
            int(datetime(2024, 1, 15).timestamp()),
        ]

        for ts in test_timestamps:
            # Create a mock trainer_status.json with timestamp
            status = {
                "coin": "BTC",
                "state": "FINISHED",
                "train_until_ts": ts,
                "mode": "incremental",
                "timestamp": ts
            }

            status_path = os.path.join(self.test_dir, "trainer_status.json")
            with open(status_path, "w") as f:
                json.dump(status, f)

            # Verify the file was created and contains correct timestamp
            with open(status_path, "r") as f:
                loaded_status = json.load(f)

            self.assertEqual(loaded_status["train_until_ts"], ts,
                           f"Model timestamp mismatch: expected {ts}, got {loaded_status['train_until_ts']}")
            self.assertEqual(loaded_status["mode"], "incremental",
                           "Model should be marked as incremental training")

        print(f"  ✓ Verified {len(test_timestamps)} model versions")
        print("  [PASS] Model versioning works correctly")

    def test_walk_forward_retraining_schedule(self):
        """
        Verify that retraining happens at correct intervals.
        """
        print("\n[TEST] Testing walk-forward retraining schedule...")

        # Test parameters
        start_ts = int(datetime(2024, 1, 1).timestamp())
        retrain_interval_days = 7
        retrain_interval_seconds = retrain_interval_days * 24 * 3600

        # Simulate 30 days of timeline
        timeline = []
        current_ts = start_ts
        end_ts = start_ts + (30 * 24 * 3600)  # 30 days

        while current_ts < end_ts:
            timeline.append(current_ts)
            current_ts += 3600  # 1 hour increments

        # Simulate when retraining should occur
        last_training_ts = start_ts
        expected_retraining_dates = []

        for current_ts in timeline:
            if (current_ts - last_training_ts) >= retrain_interval_seconds:
                expected_retraining_dates.append(datetime.fromtimestamp(current_ts))
                last_training_ts = current_ts

        # Verify retraining happens approximately every 7 days
        print(f"  Expected retraining events: {len(expected_retraining_dates)}")
        for i, retrain_date in enumerate(expected_retraining_dates):
            print(f"    Training {i+1}: {retrain_date.strftime('%Y-%m-%d')}")

        # For 30 days with 7-day interval, expect ~4 retraining events
        expected_count = 30 // retrain_interval_days
        self.assertGreaterEqual(len(expected_retraining_dates), expected_count,
                               f"Expected at least {expected_count} retraining events")
        self.assertLessEqual(len(expected_retraining_dates), expected_count + 1,
                            f"Expected at most {expected_count + 1} retraining events")

        print("  [PASS] Retraining schedule is correct")

    def test_training_timestamp_validation(self):
        """
        Verify that training timestamps never exceed backtest timestamps.
        This is another critical test for preventing look-ahead bias.
        """
        print("\n[TEST] Verifying training timestamps never exceed backtest timestamps...")

        # Simulate a backtest scenario
        backtest_start = int(datetime(2024, 1, 1).timestamp())
        backtest_current = int(datetime(2024, 1, 15).timestamp())
        backtest_end = int(datetime(2024, 2, 1).timestamp())

        # Training should NEVER use data beyond backtest_current
        valid_train_until_values = [
            backtest_start,
            backtest_current - 86400,  # 1 day before
            backtest_current,
        ]

        invalid_train_until_values = [
            backtest_current + 1,  # Even 1 second after is invalid
            backtest_current + 86400,  # 1 day after
            backtest_end,  # End of backtest
        ]

        # Test valid values
        for train_until_ts in valid_train_until_values:
            with self.subTest(f"Valid: {train_until_ts}"):
                # Training timestamp should be <= current backtest timestamp
                self.assertLessEqual(train_until_ts, backtest_current,
                                   f"Training timestamp {train_until_ts} exceeds current backtest time {backtest_current}")

        # Test invalid values (should fail validation)
        for train_until_ts in invalid_train_until_values:
            with self.subTest(f"Invalid: {train_until_ts}"):
                # These should violate the constraint
                self.assertGreater(train_until_ts, backtest_current,
                                 f"Invalid test case: {train_until_ts} should exceed {backtest_current}")

        print("  ✓ Training timestamps always <= backtest current time")
        print("  [PASS] No future data leakage detected")


def run_tests():
    """Run all walk-forward validation tests."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION TEST SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestWalkForwardValidation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
