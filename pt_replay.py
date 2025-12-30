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


class RealisticExecutionEngine:
    """
    Realistic order execution simulator with slippage, fees, and liquidity constraints.

    Simulates real market conditions including:
    - Slippage based on volatility
    - Transaction fees (maker/taker)
    - Liquidity constraints (partial fills)
    - Network latency
    """

    def __init__(self, slippage_bps=5, fee_bps=20, max_volume_pct=1.0):
        """
        Initialize execution engine with realistic market conditions.

        Args:
            slippage_bps: Base slippage in basis points (default: 5 = 0.05%)
            fee_bps: Transaction fees in basis points (default: 20 = 0.20%)
            max_volume_pct: Max order size as % of candle volume (default: 1.0%)
        """
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.max_volume_pct = max_volume_pct

    def _calculate_volatility_multiplier(self, candle):
        """
        Calculate volatility multiplier based on candle's high-low range.

        High volatility = wider spread = more slippage

        Args:
            candle: Dict with "high", "low", "close" keys

        Returns:
            float: Multiplier (1.0 = normal, 2.0 = high volatility)
        """
        if not candle or "high" not in candle or "low" not in candle or "close" not in candle:
            return 1.0

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        # Calculate range as percentage of close price
        range_pct = ((high - low) / close) * 100

        # Typical crypto range: 1-3%
        # Low volatility: < 1.5% → 1.0x multiplier
        # Normal volatility: 1.5-3% → 1.0-1.5x multiplier
        # High volatility: > 3% → 1.5-2.5x multiplier

        if range_pct < 1.5:
            return 1.0
        elif range_pct < 3.0:
            # Linear interpolation: 1.5% → 1.0x, 3% → 1.5x
            return 1.0 + ((range_pct - 1.5) / 1.5) * 0.5
        else:
            # Linear interpolation: 3% → 1.5x, 6% → 2.5x (capped at 2.5x)
            multiplier = 1.5 + ((range_pct - 3.0) / 3.0)
            return min(multiplier, 2.5)

    def _calculate_latency_slippage(self):
        """
        Simulate random network latency impact on slippage.

        Returns:
            float: Additional slippage in basis points (0-2 bps)
        """
        import random
        # Random latency between 50-500ms
        # Higher latency = more price movement during order submission
        latency_ms = random.uniform(50, 500)

        # Map latency to additional slippage (0-2 bps)
        # 50ms → 0 bps, 500ms → 2 bps
        latency_slippage_bps = ((latency_ms - 50) / 450) * 2

        return latency_slippage_bps

    def _check_liquidity_constraint(self, qty, candle, price):
        """
        Check if order size exceeds liquidity constraints.

        Args:
            qty: Order quantity in base currency (e.g., BTC)
            candle: Dict with "volume" key
            price: Current price

        Returns:
            tuple: (fill_qty, is_partial)
        """
        if not candle or "volume" not in candle:
            # No volume data, assume full fill
            return qty, False

        candle_volume = float(candle["volume"])

        # Calculate max allowable order size
        max_order_volume = candle_volume * (self.max_volume_pct / 100.0)

        if qty > max_order_volume:
            # Partial fill
            fill_qty = max_order_volume
            return fill_qty, True
        else:
            # Full fill
            return qty, False

    def simulate_fill(self, side, qty, current_price, candle):
        """
        Simulate realistic order execution with slippage, fees, and partial fills.

        Args:
            side: "buy" or "sell"
            qty: Order quantity in base currency (e.g., BTC)
            current_price: Current market price
            candle: Dict with candle data {"high": h, "low": l, "close": c, "volume": v}

        Returns:
            dict: {
                "fill_price": float,
                "fill_qty": float,
                "fees": float,
                "status": "filled|partial",
                "slippage_bps": float
            }
        """
        # Check liquidity constraints
        fill_qty, is_partial = self._check_liquidity_constraint(qty, candle, current_price)

        # Calculate volatility multiplier
        vol_multiplier = self._calculate_volatility_multiplier(candle)

        # Calculate latency slippage
        latency_slippage = self._calculate_latency_slippage()

        # Total slippage in basis points
        total_slippage_bps = (self.slippage_bps * vol_multiplier) + latency_slippage

        # Apply slippage to price
        # Buy orders: slippage increases price (you pay more)
        # Sell orders: slippage decreases price (you receive less)
        slippage_multiplier = 1.0 + (total_slippage_bps / 10000.0)

        if side.lower() == "buy":
            fill_price = current_price * slippage_multiplier
        else:  # sell
            fill_price = current_price / slippage_multiplier

        # Calculate fees (percentage of notional value)
        notional_value = fill_qty * fill_price
        fees = notional_value * (self.fee_bps / 10000.0)

        # Determine status
        status = "partial" if is_partial else "filled"

        return {
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "fees": fees,
            "status": status,
            "slippage_bps": total_slippage_bps
        }


def load_execution_config(config_path="backtest_config.json"):
    """
    Load execution model configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Execution configuration parameters
    """
    if not os.path.exists(config_path):
        # Return default configuration
        return {
            "slippage_bps": 5,
            "fee_bps": 20,
            "max_volume_pct": 1.0,
            "latency_ms": [50, 500],
            "partial_fill_threshold": 0.01
        }

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("execution_model", {
                "slippage_bps": 5,
                "fee_bps": 20,
                "max_volume_pct": 1.0,
                "latency_ms": [50, 500],
                "partial_fill_threshold": 0.01
            })
    except Exception as e:
        print(f"Warning: Could not load execution config: {e}")
        return {
            "slippage_bps": 5,
            "fee_bps": 20,
            "max_volume_pct": 1.0,
            "latency_ms": [50, 500],
            "partial_fill_threshold": 0.01
        }


def validate_execution_config(config):
    """
    Validate execution configuration parameters.

    Args:
        config: Dict with execution model parameters

    Returns:
        tuple: (is_valid, error_message)
    """
    required_keys = ["slippage_bps", "fee_bps", "max_volume_pct"]

    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"

    # Validate ranges
    if config["slippage_bps"] < 0 or config["slippage_bps"] > 100:
        return False, "slippage_bps must be between 0 and 100"

    if config["fee_bps"] < 0 or config["fee_bps"] > 100:
        return False, "fee_bps must be between 0 and 100"

    if config["max_volume_pct"] <= 0 or config["max_volume_pct"] > 100:
        return False, "max_volume_pct must be between 0 and 100"

    return True, ""


def process_order(order, current_candle, execution_engine, output_dir="replay_data"):
    """
    Process a single order using the realistic execution engine.

    Args:
        order: Dict with order details from sim_orders.jsonl
        current_candle: Dict with current candle data
        execution_engine: RealisticExecutionEngine instance
        output_dir: Directory to write fills

    Returns:
        dict: Fill result
    """
    # Extract order details
    side = order["side"]
    qty = order["qty"]
    symbol = order["symbol"]
    client_order_id = order["client_order_id"]
    order_ts = order["ts"]

    # Get current price from candle (use close price)
    current_price = float(current_candle["close"])

    # Simulate fill
    fill_result = execution_engine.simulate_fill(side, qty, current_price, current_candle)

    # Create fill record
    fill = {
        "ts": order_ts + 0.5,  # Fill happens slightly after order (500ms delay)
        "client_order_id": client_order_id,
        "side": side,
        "symbol": symbol,
        "qty": fill_result["fill_qty"],
        "fill_price": fill_result["fill_price"],
        "fees": fill_result["fees"],
        "slippage_bps": fill_result["slippage_bps"],
        "state": fill_result["status"]
    }

    # Write fill to sim_fills.jsonl
    os.makedirs(output_dir, exist_ok=True)
    fills_path = os.path.join(output_dir, "sim_fills.jsonl")

    with open(fills_path, "a") as f:
        f.write(json.dumps(fill) + "\n")

    return fill


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

    # Test RealisticExecutionEngine
    print("\nTesting RealisticExecutionEngine...")

    # Create engine with default config
    engine = RealisticExecutionEngine(slippage_bps=5, fee_bps=20, max_volume_pct=1.0)
    print("✓ Engine initialized")

    # Test low volatility candle
    low_vol_candle = {
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
        "volume": 1000.0
    }
    result = engine.simulate_fill("buy", 0.1, 100.0, low_vol_candle)
    assert result["status"] == "filled"
    assert result["fill_qty"] == 0.1
    assert result["fill_price"] > 100.0  # Buy with slippage
    assert result["fees"] > 0
    assert result["slippage_bps"] > 0
    print(f"✓ Low volatility test: slippage={result['slippage_bps']:.2f}bps, price=${result['fill_price']:.2f}")

    # Test high volatility candle (should have higher slippage)
    high_vol_candle = {
        "high": 105.0,
        "low": 95.0,
        "close": 100.0,
        "volume": 1000.0
    }
    result2 = engine.simulate_fill("buy", 0.1, 100.0, high_vol_candle)
    assert result2["slippage_bps"] > result["slippage_bps"]  # Higher volatility = more slippage
    print(f"✓ High volatility test: slippage={result2['slippage_bps']:.2f}bps (higher than low vol)")

    # Test large order (partial fill)
    large_order_candle = {
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
        "volume": 5.0  # Small volume
    }
    result3 = engine.simulate_fill("buy", 1.0, 100.0, large_order_candle)
    assert result3["status"] == "partial"
    assert result3["fill_qty"] < 1.0  # Partial fill
    print(f"✓ Liquidity constraint test: requested=1.0, filled={result3['fill_qty']:.4f} (partial)")

    # Test sell order
    result4 = engine.simulate_fill("sell", 0.1, 100.0, low_vol_candle)
    assert result4["fill_price"] < 100.0  # Sell with slippage (receive less)
    print(f"✓ Sell order test: price=${result4['fill_price']:.2f} (lower than market)")

    # Test execution config
    print("\nTesting execution configuration...")

    # Test config validation
    valid_config = {
        "slippage_bps": 5,
        "fee_bps": 20,
        "max_volume_pct": 1.0
    }
    is_valid, error = validate_execution_config(valid_config)
    assert is_valid
    print("✓ Valid config accepted")

    invalid_config = {
        "slippage_bps": 150,  # Too high
        "fee_bps": 20,
        "max_volume_pct": 1.0
    }
    is_valid, error = validate_execution_config(invalid_config)
    assert not is_valid
    print(f"✓ Invalid config rejected: {error}")

    # Test order processing
    print("\nTesting order processing...")
    test_replay_dir = "test_replay_data"
    os.makedirs(test_replay_dir, exist_ok=True)

    test_order = {
        "ts": time.time(),
        "client_order_id": "test-order-123",
        "side": "buy",
        "symbol": "BTC-USD",
        "qty": 0.01,
        "order_type": "market"
    }

    test_candle = {
        "high": 100100.0,
        "low": 99900.0,
        "close": 100000.0,
        "volume": 100.0
    }

    fill = process_order(test_order, test_candle, engine, test_replay_dir)
    assert fill["client_order_id"] == "test-order-123"
    assert fill["side"] == "buy"
    assert fill["state"] == "filled"
    print(f"✓ Order processed: {fill['qty']} @ ${fill['fill_price']:.2f}, fees=${fill['fees']:.2f}")

    # Verify fill written to file
    fills_path = os.path.join(test_replay_dir, "sim_fills.jsonl")
    assert os.path.exists(fills_path)
    with open(fills_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        fill_data = json.loads(lines[0])
        assert fill_data["client_order_id"] == "test-order-123"
    print("✓ Fill written to sim_fills.jsonl")

    # Cleanup
    shutil.rmtree(test_replay_dir)
    print("✓ Test replay data cleaned up")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


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
