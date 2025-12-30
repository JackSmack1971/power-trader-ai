#!/usr/bin/env python3
"""
PowerTrader AI - Backtesting Replay System
Implements historical data replay with realistic execution simulation.
"""

import os
import json
import time
import argparse
from datetime import datetime
from kucoin.client import Market


def _atomic_write_json(path, data):
    """
    Atomic write pattern from CLAUDE.md:42-47.
    Prevents race conditions in file-based IPC.

    Args:
        path: Target file path
        data: Dictionary to write as JSON
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _get_timeframe_minutes(timeframe):
    """Convert timeframe string to minutes."""
    timeframe_map = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1hour": 60,
        "2hour": 120,
        "4hour": 240,
        "8hour": 480,
        "12hour": 720,
        "1day": 1440,
        "1week": 10080,
    }
    return timeframe_map.get(timeframe, 60)


def fetch_historical_klines(symbol, timeframe, start_ts, end_ts, cache_dir="backtest_cache"):
    """
    Fetch OHLCV candles from KuCoin with rate limiting and caching.

    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
        timeframe: Candle interval (e.g., "1hour", "1day")
        start_ts: Unix timestamp start
        end_ts: Unix timestamp end
        cache_dir: Directory for caching data

    Returns:
        List[dict]: [{"time": ts, "open": o, "close": c, "high": h, "low": l, "volume": v}, ...]
    """
    # Generate cache filename
    cache_filename = f"{symbol}_{timeframe}_{start_ts}_{end_ts}.json"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Check if data is already cached
    if os.path.exists(cache_path):
        print(f"✓ Loading from cache: {cache_filename}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"Fetching {symbol} {timeframe} data from {start_ts} to {end_ts}...")

    # Initialize KuCoin client
    market = Market(url='https://api.kucoin.com')

    all_candles = []
    timeframe_minutes = _get_timeframe_minutes(timeframe)
    max_candles_per_request = 1500

    # Calculate time window for each request (in seconds)
    window_seconds = max_candles_per_request * timeframe_minutes * 60

    # Start from end_ts and work backwards
    current_end = end_ts
    current_start = max(start_ts, current_end - window_seconds)

    retry_count = 0
    max_retries = 3

    while current_start >= start_ts:
        try:
            # Fetch data for current window
            raw_data = market.get_kline(
                symbol,
                timeframe,
                startAt=current_start,
                endAt=current_end
            )

            # Parse KuCoin response
            # Format: [['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'], ...]
            if raw_data:
                for candle in raw_data:
                    if len(candle) >= 6:
                        all_candles.append({
                            "time": int(candle[0]),
                            "open": float(candle[1]),
                            "close": float(candle[2]),
                            "high": float(candle[3]),
                            "low": float(candle[4]),
                            "volume": float(candle[5])
                        })

            # Rate limiting: 0.1s between requests
            time.sleep(0.1)

            # Move to next window
            current_end = current_start - 1
            current_start = max(start_ts, current_end - window_seconds)

            # Reset retry count on success
            retry_count = 0

            # Progress indicator
            progress = ((end_ts - current_end) / (end_ts - start_ts)) * 100
            print(f"  Progress: {progress:.1f}% ({len(all_candles)} candles)")

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"✗ Error fetching data after {max_retries} retries: {e}")
                raise
            else:
                print(f"  Retry {retry_count}/{max_retries} after error: {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff

    # Sort candles by timestamp (ascending)
    all_candles.sort(key=lambda x: x["time"])

    # Cache the results
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_write_json(cache_path, all_candles)

    # Update cache index
    update_cache_index(symbol, timeframe, start_ts, end_ts, cache_filename, len(all_candles), cache_dir)

    print(f"✓ Fetched {len(all_candles)} candles, cached to {cache_filename}")

    return all_candles


def load_cache_index(cache_dir="backtest_cache"):
    """
    Load the cache index for fast lookups.

    Args:
        cache_dir: Directory containing cache files

    Returns:
        dict: Cache index mapping symbol_timeframe to list of cached ranges
    """
    index_path = os.path.join(cache_dir, "cache_index.json")

    if not os.path.exists(index_path):
        return {}

    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache index: {e}")
        return {}


def update_cache_index(symbol, timeframe, start_ts, end_ts, cache_filename, candle_count, cache_dir="backtest_cache"):
    """
    Update the cache index with a new cache entry.

    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
        timeframe: Candle interval (e.g., "1hour")
        start_ts: Unix timestamp start
        end_ts: Unix timestamp end
        cache_filename: Name of the cache file
        candle_count: Number of candles in the cache
        cache_dir: Directory containing cache files
    """
    index_path = os.path.join(cache_dir, "cache_index.json")

    # Load existing index
    index = load_cache_index(cache_dir)

    # Create key for this symbol/timeframe combination
    key = f"{symbol}_{timeframe}"

    # Initialize list if key doesn't exist
    if key not in index:
        index[key] = []

    # Create new entry
    new_entry = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "file": cache_filename,
        "candle_count": candle_count,
        "created_at": int(time.time())
    }

    # Check if this range already exists in index
    existing_entry = None
    for i, entry in enumerate(index[key]):
        if entry["start_ts"] == start_ts and entry["end_ts"] == end_ts:
            existing_entry = i
            break

    if existing_entry is not None:
        # Update existing entry
        index[key][existing_entry] = new_entry
    else:
        # Add new entry
        index[key].append(new_entry)

    # Sort entries by start_ts for easier lookup
    index[key].sort(key=lambda x: x["start_ts"])

    # Save index atomically
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_write_json(index_path, index)


def find_cached_ranges(symbol, timeframe, start_ts, end_ts, cache_dir="backtest_cache"):
    """
    Find cached data ranges that overlap with the requested range.

    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
        timeframe: Candle interval (e.g., "1hour")
        start_ts: Unix timestamp start
        end_ts: Unix timestamp end
        cache_dir: Directory containing cache files

    Returns:
        list: List of cache entries that overlap with requested range
    """
    index = load_cache_index(cache_dir)
    key = f"{symbol}_{timeframe}"

    if key not in index:
        return []

    overlapping = []
    for entry in index[key]:
        # Check if ranges overlap
        if entry["end_ts"] >= start_ts and entry["start_ts"] <= end_ts:
            overlapping.append(entry)

    return overlapping


def warm_cache(start_date, end_date, coins, timeframes=None, cache_dir="backtest_cache"):
    """
    Pre-fetch and cache historical data for multiple coins and timeframes.

    Args:
        start_date: Start date string "YYYY-MM-DD"
        end_date: End date string "YYYY-MM-DD"
        coins: List of coin symbols (e.g., ['BTC', 'ETH', 'DOGE'])
        timeframes: List of timeframes to cache (defaults to all supported)
        cache_dir: Directory for caching data
    """
    # Default timeframes match those used by pt_thinker.py
    if timeframes is None:
        timeframes = ["1hour", "2hour", "4hour", "8hour", "12hour", "1day", "1week"]

    # Convert dates to timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print("=" * 60)
    print("CACHE WARMING")
    print("=" * 60)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Coins: {', '.join(coins)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Cache Directory: {cache_dir}")
    print("=" * 60)

    total_tasks = len(coins) * len(timeframes)
    completed_tasks = 0

    for coin in coins:
        symbol = f"{coin}-USDT"
        print(f"\n[{coin}]")

        for timeframe in timeframes:
            completed_tasks += 1
            progress_pct = (completed_tasks / total_tasks) * 100

            print(f"\n  [{completed_tasks}/{total_tasks}] {timeframe} ({progress_pct:.1f}% complete)")

            try:
                candles = fetch_historical_klines(symbol, timeframe, start_ts, end_ts, cache_dir)
                print(f"    ✓ {len(candles)} candles cached")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

    print("\n" + "=" * 60)
    print("CACHE WARMING COMPLETE")
    print("=" * 60)

    # Display cache index summary
    index = load_cache_index(cache_dir)
    total_entries = sum(len(entries) for entries in index.values())
    print(f"Total cache entries: {total_entries}")
    print(f"Cached symbols: {len(index)}")


def run_tests():
    """Run internal tests for development."""
    import sys
    import shutil

    # Test atomic write
    test_data = {"test": "data", "timestamp": time.time()}
    _atomic_write_json("test.json", test_data)
    print("✓ Atomic write test successful")

    # Verify file exists and cleanup
    if os.path.exists("test.json"):
        with open("test.json", "r") as f:
            loaded = json.load(f)
            assert loaded["test"] == "data"
        os.remove("test.json")
        print("✓ Test file verified and cleaned up")

    # Test cache index management
    print("\nTesting cache index management...")
    test_cache_dir = "test_cache"
    os.makedirs(test_cache_dir, exist_ok=True)

    # Add entries to index
    update_cache_index("BTC-USDT", "1hour", 1704067200, 1704153600, "test_file1.json", 100, test_cache_dir)
    update_cache_index("BTC-USDT", "1hour", 1704153600, 1704240000, "test_file2.json", 100, test_cache_dir)
    update_cache_index("ETH-USDT", "1day", 1704067200, 1706745600, "test_file3.json", 30, test_cache_dir)

    # Load and verify index
    index = load_cache_index(test_cache_dir)
    assert "BTC-USDT_1hour" in index
    assert len(index["BTC-USDT_1hour"]) == 2
    assert "ETH-USDT_1day" in index
    print("✓ Cache index updates working correctly")

    # Test finding cached ranges
    found = find_cached_ranges("BTC-USDT", "1hour", 1704100000, 1704200000, test_cache_dir)
    assert len(found) == 2  # Should find both overlapping ranges
    print("✓ Cache range lookup working correctly")

    # Cleanup test cache
    shutil.rmtree(test_cache_dir)
    print("✓ Test cache cleaned up")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PowerTrader AI Backtesting Replay System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests
  python pt_replay.py --test

  # Warm cache for one week of BTC data
  python pt_replay.py --warm-cache --start-date 2024-01-01 --end-date 2024-01-08 --coins BTC

  # Warm cache for multiple coins
  python pt_replay.py --warm-cache --start-date 2024-01-01 --end-date 2024-02-01 --coins BTC,ETH,DOGE
        """
    )

    parser.add_argument("--test", action="store_true", help="Run internal tests")
    parser.add_argument("--warm-cache", action="store_true", help="Pre-fetch and cache historical data")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--coins", help="Comma-separated list of coins (e.g., BTC,ETH,DOGE)")
    parser.add_argument("--timeframes", help="Comma-separated list of timeframes (default: all)")
    parser.add_argument("--cache-dir", default="backtest_cache", help="Cache directory (default: backtest_cache)")

    args = parser.parse_args()

    if args.test:
        run_tests()

    elif args.warm_cache:
        if not args.start_date or not args.end_date or not args.coins:
            parser.error("--warm-cache requires --start-date, --end-date, and --coins")

        coins = [c.strip() for c in args.coins.split(",")]
        timeframes = None
        if args.timeframes:
            timeframes = [tf.strip() for tf in args.timeframes.split(",")]

        warm_cache(args.start_date, args.end_date, coins, timeframes, args.cache_dir)

    else:
        parser.print_help()
