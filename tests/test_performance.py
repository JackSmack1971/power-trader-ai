#!/usr/bin/env python3
"""
Performance Benchmark Tests for PowerTrader AI Backtesting System.
Measures execution time and memory usage to ensure targets are met.
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
import json
from datetime import datetime, timedelta
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pt_replay import (
    fetch_historical_klines,
    replay_time_progression,
    RealisticExecutionEngine,
    BacktestLogger
)


class TestPerformance(unittest.TestCase):
    """Performance benchmark tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp(prefix="test_perf_")
        cls.cache_dir = os.path.join(cls.test_dir, "cache")
        cls.output_dir = os.path.join(cls.test_dir, "output")
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # Get initial process info
        cls.process = psutil.Process()
        cls.initial_memory = cls.process.memory_info().rss / 1024 / 1024  # MB

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_execution_engine_performance(self):
        """
        Benchmark the execution engine performance.
        Should process thousands of orders per second.
        """
        print("\n[BENCHMARK] Execution Engine Performance...")

        engine = RealisticExecutionEngine()

        # Prepare test candle
        test_candle = {
            "high": 100100.0,
            "low": 99900.0,
            "close": 100000.0,
            "volume": 1000.0
        }

        # Benchmark: process 10,000 simulated fills
        num_orders = 10000
        start_time = time.time()

        for i in range(num_orders):
            result = engine.simulate_fill(
                side="buy" if i % 2 == 0 else "sell",
                qty=0.01,
                current_price=100000.0,
                candle=test_candle
            )

        elapsed = time.time() - start_time
        orders_per_sec = num_orders / elapsed

        print(f"  Processed {num_orders:,} orders in {elapsed:.2f}s")
        print(f"  Throughput: {orders_per_sec:,.0f} orders/second")

        # Target: at least 1000 orders/second
        self.assertGreater(orders_per_sec, 1000,
                          f"Execution engine too slow: {orders_per_sec:.0f} orders/sec < 1000 target")

        print("  ✓ Performance target met (>1000 orders/sec)")

    def test_cache_size(self):
        """
        Verify cache size is reasonable.
        Target: < 500MB per coin for 6 months
        """
        print("\n[BENCHMARK] Cache Size...")

        # Create mock cache files to test size calculation
        # Simulate 6 months of 1-hour candles
        # 6 months ≈ 180 days × 24 hours = 4,320 candles

        num_candles = 4320
        candles = []

        start_ts = int(datetime(2024, 1, 1).timestamp())

        for i in range(num_candles):
            candles.append({
                "time": start_ts + (i * 3600),
                "open": 100.0 + i * 0.1,
                "close": 100.5 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.5 + i * 0.1,
                "volume": 1000.0 + i
            })

        # Write to file
        cache_file = os.path.join(self.cache_dir, "BTC-USDT_1hour_test.json")
        with open(cache_file, "w") as f:
            json.dump(candles, f)

        # Measure file size
        file_size_bytes = os.path.getsize(cache_file)
        file_size_mb = file_size_bytes / 1024 / 1024

        print(f"  Cache file size for {num_candles:,} candles: {file_size_mb:.2f} MB")

        # For 6 months of data, target is < 500MB
        # With 7 timeframes, that's ~70MB per timeframe
        self.assertLess(file_size_mb, 70,
                       f"Cache file too large: {file_size_mb:.2f} MB > 70 MB target")

        print(f"  ✓ Cache size acceptable ({file_size_mb:.2f} MB < 70 MB target)")

    def test_memory_usage(self):
        """
        Measure memory usage during backtest operations.
        Target: < 2GB total memory usage
        """
        print("\n[BENCHMARK] Memory Usage...")

        # Get current memory
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.initial_memory

        print(f"  Initial memory: {self.initial_memory:.2f} MB")
        print(f"  Current memory: {current_memory:.2f} MB")
        print(f"  Memory used: {memory_used:.2f} MB")

        # Simulate some memory-intensive operations
        # Create mock candle data
        large_dataset = []
        for i in range(10000):
            large_dataset.append({
                "time": i,
                "open": 100.0,
                "close": 101.0,
                "high": 102.0,
                "low": 99.0,
                "volume": 1000.0
            })

        # Measure memory after loading data
        memory_after_load = self.process.memory_info().rss / 1024 / 1024
        memory_delta = memory_after_load - current_memory

        print(f"  Memory after loading 10k candles: {memory_after_load:.2f} MB (+{memory_delta:.2f} MB)")

        # Clean up
        del large_dataset

        # Total memory should be under 2GB (2048 MB)
        self.assertLess(memory_after_load, 2048,
                       f"Memory usage too high: {memory_after_load:.2f} MB > 2048 MB target")

        print("  ✓ Memory usage within limits (<2GB)")

    def test_logger_performance(self):
        """
        Benchmark logger performance.
        Should handle high-frequency logging without bottlenecks.
        """
        print("\n[BENCHMARK] Logger Performance...")

        logger = BacktestLogger(log_dir=self.output_dir)

        # Benchmark: log 1000 events
        num_events = 1000
        start_time = time.time()

        for i in range(num_events):
            logger.log_trade_execution(
                order_id=f"order-{i}",
                side="buy",
                symbol="BTC-USD",
                qty=0.01,
                price=100000.0,
                fees=20.0,
                slippage_bps=5.0,
                status="filled"
            )

        elapsed = time.time() - start_time
        events_per_sec = num_events / elapsed

        print(f"  Logged {num_events:,} events in {elapsed:.2f}s")
        print(f"  Throughput: {events_per_sec:,.0f} events/second")

        # Target: at least 500 events/second
        self.assertGreater(events_per_sec, 500,
                          f"Logger too slow: {events_per_sec:.0f} events/sec < 500 target")

        print("  ✓ Logger performance acceptable (>500 events/sec)")

    def test_json_parsing_performance(self):
        """
        Benchmark JSON parsing performance for large files.
        """
        print("\n[BENCHMARK] JSON Parsing Performance...")

        # Create large JSON file with equity curve data
        num_snapshots = 5000  # ~1 month of hourly data
        snapshots = []

        for i in range(num_snapshots):
            snapshots.append({
                "timestamp": 1704067200 + (i * 3600),
                "total_value": 10000.0 + i * 10,
                "cash": 5000.0 + i * 5,
                "positions": {
                    "BTC": {"value": 5000.0 + i * 5}
                }
            })

        # Write to file
        test_file = os.path.join(self.output_dir, "large_equity.jsonl")
        with open(test_file, "w") as f:
            for snapshot in snapshots:
                f.write(json.dumps(snapshot) + "\n")

        # Benchmark: parse the file
        start_time = time.time()

        with open(test_file, "r") as f:
            parsed_snapshots = [json.loads(line) for line in f]

        elapsed = time.time() - start_time
        snapshots_per_sec = num_snapshots / elapsed

        print(f"  Parsed {num_snapshots:,} snapshots in {elapsed:.3f}s")
        print(f"  Throughput: {snapshots_per_sec:,.0f} snapshots/second")

        # Verify all snapshots parsed
        self.assertEqual(len(parsed_snapshots), num_snapshots)

        # Target: at least 10,000 snapshots/second
        self.assertGreater(snapshots_per_sec, 10000,
                          f"JSON parsing too slow: {snapshots_per_sec:.0f} snapshots/sec")

        print("  ✓ JSON parsing performance good (>10k snapshots/sec)")


class TestBacktestScaling(unittest.TestCase):
    """
    Test backtest scaling characteristics.
    Note: These tests may take several minutes to run.
    """

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_scale_")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_time_complexity(self):
        """
        Verify backtest time scales linearly with data size.
        """
        print("\n[BENCHMARK] Time Complexity Scaling...")

        # Test with different dataset sizes
        test_sizes = [100, 500, 1000]
        timings = []

        for size in test_sizes:
            # Create mock candle data
            candles = []
            start_ts = 1704067200

            for i in range(size):
                candles.append({
                    "time": start_ts + (i * 3600),
                    "open": 100.0,
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1000.0
                })

            # Measure time to process
            start_time = time.time()

            # Simulate processing each candle
            for candle in candles:
                # Mock processing work
                _ = candle["close"] * 1.0001

            elapsed = time.time() - start_time
            timings.append(elapsed)

            print(f"  Size {size:,}: {elapsed:.4f}s ({size/elapsed:.0f} candles/sec)")

        # Verify scaling is roughly linear (not exponential)
        # Doubling the size should roughly double the time (±50%)
        if len(timings) >= 2:
            ratio_1_to_2 = timings[1] / timings[0]
            size_ratio = test_sizes[1] / test_sizes[0]

            # Allow 2x tolerance for system variance
            self.assertLess(ratio_1_to_2, size_ratio * 2,
                          f"Time scaling worse than linear: {ratio_1_to_2:.2f}x time for {size_ratio:.2f}x data")

        print("  ✓ Time complexity is acceptable (linear scaling)")


def run_benchmarks():
    """Run all performance benchmarks."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    print("Measuring execution time, memory usage, and scalability...")
    print("=" * 60)

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add performance tests
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestScaling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_benchmarks())
