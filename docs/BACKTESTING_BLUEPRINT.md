# PowerTrader AI Backtesting Implementation Blueprint

**Version:** 1.0
**Created:** 2025-12-30
**Status:** Ready for Implementation
**Source:** Based on `docs/proposals/backtesting-feature.md` (v2.0)
**Estimated Timeline:** 21-28 days

---

## Overview

This blueprint breaks down the backtesting feature implementation into **sequential atomic tasks**. Each task is self-contained, testable, and builds upon previous tasks. The implementation follows 5 major phases with clear success criteria for each task.

### Success Criteria
- ✅ Run 6-month backtest on BTC in under 10 minutes
- ✅ Generate reports with corrected Sharpe, Sortino, Calmar ratios
- ✅ Match live trading results within ±2% (accounting for slippage/fees)
- ✅ Eliminate look-ahead bias via walk-forward validation
- ✅ Calculate true max drawdown including unrealized losses

---

## Phase 1: Data Caching Infrastructure (Days 1-4)

### Task 1.1: Create Directory Structure
**Estimated Time:** 15 minutes
**Priority:** CRITICAL
**Dependencies:** None

**Actions:**
1. Create `backtest_cache/` directory
2. Create `replay_data/` directory
3. Create `backtest_results/` directory
4. Create `.gitignore` entries for cache directories

**Commands:**
```bash
mkdir -p backtest_cache replay_data backtest_results
echo "backtest_cache/" >> .gitignore
echo "backtest_results/" >> .gitignore
```

**Validation:**
- Verify all directories exist
- Verify `.gitignore` contains new entries
- Run `ls -la` to confirm structure

---

### Task 1.2: Implement Atomic JSON Write Helper
**Estimated Time:** 30 minutes
**Priority:** CRITICAL
**Dependencies:** Task 1.1
**File:** `pt_replay.py` (NEW)

**Actions:**
1. Create new file `pt_replay.py`
2. Implement `_atomic_write_json()` function (per CLAUDE.md pattern)
3. Add module imports: `os`, `json`, `time`

**Code Implementation:**
```python
import os
import json
import time
from kucoin.client import Market

def _atomic_write_json(path, data):
    """
    Atomic write pattern from CLAUDE.md:42-47.
    Prevents race conditions in file-based IPC.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
```

**Validation:**
- Create test file with `_atomic_write_json("test.json", {"test": "data"})`
- Verify `test.json` exists with correct content
- Verify no `.tmp` file remains
- Delete test file

---

### Task 1.3: Implement KuCoin Data Fetcher
**Estimated Time:** 2 hours
**Priority:** CRITICAL
**Dependencies:** Task 1.2
**File:** `pt_replay.py`
**Reference:** `pt_trainer.py:403-441`

**Actions:**
1. Implement `fetch_historical_klines()` function
2. Add rate limiting (0.1s between requests)
3. Add error handling and retry logic
4. Implement caching to avoid duplicate API calls
5. Support pagination for large date ranges (KuCoin max 1500 candles/request)

**Code Location:** Add to `pt_replay.py` after `_atomic_write_json()`

**Function Signature:**
```python
def fetch_historical_klines(symbol, timeframe, start_ts, end_ts):
    """
    Fetch OHLCV candles from KuCoin with rate limiting.

    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
        timeframe: Candle interval (e.g., "1hour", "1day")
        start_ts: Unix timestamp start
        end_ts: Unix timestamp end

    Returns:
        List[dict]: [{"time": ts, "open": o, "close": c, "high": h, "low": l, "volume": v}, ...]
    """
```

**Validation:**
- Fetch 100 candles for BTC-USDT 1hour
- Verify cache file created in `backtest_cache/`
- Second fetch should read from cache (no API call)
- Verify candle format matches specification

---

### Task 1.4: Implement Cache Index Management
**Estimated Time:** 1.5 hours
**Priority:** HIGH
**Dependencies:** Task 1.3
**File:** `pt_replay.py`

**Actions:**
1. Implement `update_cache_index()` function
2. Implement `load_cache_index()` function
3. Create index schema for fast lookups
4. Add atomic writes for index updates

**Index Schema:**
```json
{
  "BTC-USDT_1hour": [
    {
      "start_ts": 1704067200,
      "end_ts": 1706745600,
      "file": "BTC-USDT_1hour_1704067200_1706745600.json",
      "candle_count": 720,
      "created_at": 1704800000
    }
  ]
}
```

**Validation:**
- Update index after fetching data
- Load index and verify structure
- Verify atomic writes (no corruption during concurrent access)

---

### Task 1.5: Implement Cache Warming CLI
**Estimated Time:** 1 hour
**Priority:** MEDIUM
**Dependencies:** Task 1.4
**File:** `pt_replay.py`

**Actions:**
1. Add argparse CLI argument parsing
2. Implement `warm_cache()` function
3. Support multiple coins and timeframes
4. Add progress indicator

**CLI Interface:**
```bash
python pt_replay.py --warm-cache \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC,ETH,DOGE
```

**Validation:**
- Run cache warming for 1 week of BTC data
- Verify all timeframes cached (1hour, 2hour, 4hour, 8hour, 12hour, 1day, 1week)
- Verify cache index updated correctly
- Check progress output

---

## Phase 2: Realistic Execution Engine (Days 5-8)

### Task 2.1: Implement RealisticExecutionEngine Class
**Estimated Time:** 3 hours
**Priority:** CRITICAL
**Dependencies:** Phase 1 complete
**File:** `pt_replay.py`

**Actions:**
1. Create `RealisticExecutionEngine` class
2. Implement configurable slippage calculation
3. Implement transaction cost modeling (maker/taker fees)
4. Implement liquidity constraints (partial fills)
5. Implement network latency simulation

**Class Structure:**
```python
class RealisticExecutionEngine:
    def __init__(self, slippage_bps=5, fee_bps=20, max_volume_pct=1.0):
        """
        Initialize execution engine with realistic market conditions.

        Args:
            slippage_bps: Base slippage in basis points (default: 5 = 0.05%)
            fee_bps: Transaction fees in basis points (default: 20 = 0.20%)
            max_volume_pct: Max order size as % of candle volume (default: 1.0%)
        """

    def simulate_fill(self, side, qty, current_price, candle):
        """
        Simulate realistic order execution with slippage, fees, and partial fills.

        Returns:
            {
                "fill_price": float,
                "fill_qty": float,
                "fees": float,
                "status": "filled|partial",
                "slippage_bps": float
            }
        """
```

**Slippage Model:**
- Base slippage scaled by volatility
- Volatile candles (high range) = 2x slippage
- Random latency component (50-500ms)

**Validation:**
- Test low volatility candle → minimal slippage
- Test high volatility candle → increased slippage
- Test large order → partial fill
- Test small order → complete fill
- Verify fees calculated correctly

---

### Task 2.2: Create Execution Configuration Schema
**Estimated Time:** 45 minutes
**Priority:** MEDIUM
**Dependencies:** Task 2.1
**File:** `pt_replay.py`

**Actions:**
1. Define execution model configuration
2. Add to backtest config.json
3. Implement config validation

**Config Schema:**
```json
{
  "execution_model": {
    "slippage_bps": 5,
    "fee_bps": 20,
    "max_volume_pct": 1.0,
    "latency_ms": [50, 500],
    "partial_fill_threshold": 0.01
  }
}
```

**Validation:**
- Load config from JSON
- Pass to RealisticExecutionEngine
- Verify all parameters applied

---

### Task 2.3: Implement Mock Order Execution
**Estimated Time:** 2 hours
**Priority:** HIGH
**Dependencies:** Task 2.1
**File:** `pt_replay.py`

**Actions:**
1. Implement `process_order()` function
2. Read orders from `replay_data/sim_orders.jsonl`
3. Use RealisticExecutionEngine for fills
4. Write fills to `replay_data/sim_fills.jsonl`
5. Add order ID tracking

**Order Schema:**
```json
{
  "ts": 1704067200.5,
  "client_order_id": "uuid-...",
  "side": "buy|sell",
  "symbol": "BTC-USD",
  "qty": 0.001,
  "order_type": "market"
}
```

**Fill Schema:**
```json
{
  "ts": 1704067201.0,
  "client_order_id": "uuid-...",
  "side": "buy",
  "symbol": "BTC-USD",
  "qty": 0.001,
  "fill_price": 98800.12,
  "fees": 19.76,
  "slippage_bps": 4.8,
  "state": "filled|partial"
}
```

**Validation:**
- Create mock order in sim_orders.jsonl
- Process order
- Verify fill written to sim_fills.jsonl
- Verify realistic slippage and fees applied

---

## Phase 3: Replay Mode Support (Days 9-14)

### Task 3.1: Add Replay Mode to pt_thinker.py
**Estimated Time:** 2.5 hours
**Priority:** CRITICAL
**Dependencies:** Phase 2 complete
**File:** `pt_thinker.py`
**Reference:** `pt_thinker.py:477-1091`

**Actions:**
1. Add argparse for `--replay` flag
2. Add global `REPLAY_MODE` variable
3. Implement `_read_cached_kline_for_replay()` helper
4. Modify `step_coin()` to use cached data in replay mode
5. Replace `time.time()` with replay timestamp reads

**Code Modifications:**

**Location: ~Line 1095-1115 (before main loop)**
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--replay', action='store_true', help='Run in replay mode')
parser.add_argument('--replay-cache-dir', default='backtest_cache', help='Cache directory')
args = parser.parse_args()

REPLAY_MODE = args.replay
REPLAY_CACHE_DIR = args.replay_cache_dir

if REPLAY_MODE:
    print("=" * 60)
    print("REPLAY MODE ACTIVE - Using cached historical data")
    print("=" * 60)
```

**Location: ~Line 556 (in `step_coin` function)**
```python
# BEFORE:
history = str(market.get_kline(coin, tf_choices[index]))

# AFTER:
if REPLAY_MODE:
    with open('replay_data/current_timestamp.txt', 'r') as f:
        current_ts = int(f.read().strip())
    history = _read_cached_kline_for_replay(coin, tf_choices[index], current_ts)
else:
    history = str(market.get_kline(coin, tf_choices[index]))
```

**Location: ~Line 430 (add new helper function)**
```python
def _read_cached_kline_for_replay(coin, timeframe, current_ts):
    """
    Read cached historical candles for replay mode.
    Returns same format as KuCoin API for compatibility.

    Args:
        coin: Coin symbol (e.g., "BTC")
        timeframe: Timeframe (e.g., "1hour")
        current_ts: Current replay timestamp

    Returns:
        str: KuCoin-formatted candle data
    """
    # Implementation details in proposal section
```

**Validation:**
- Run pt_thinker.py with --replay flag
- Verify reads from cache instead of API
- Verify signal generation works correctly
- Verify no API calls made in replay mode

---

### Task 3.2: Add Replay Mode to pt_trader.py
**Estimated Time:** 3 hours
**Priority:** CRITICAL
**Dependencies:** Task 3.1
**File:** `pt_trader.py`
**Reference:** `pt_trader.py:692-852`

**Actions:**
1. Add argparse for `--replay` flag and `--replay-output-dir`
2. Redirect hub_data writes to replay output directory
3. Modify `get_price()` to read from replay IPC files
4. Modify `place_buy_order()` to write to sim_orders.jsonl
5. Modify `place_sell_order()` to write to sim_orders.jsonl

**Code Modifications:**

**Location: ~Line 1420-1435 (in `if __name__ == "__main__"` block)**
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--replay', action='store_true', help='Run in replay mode')
parser.add_argument('--replay-output-dir', default='backtest_results/latest', help='Output directory')
args = parser.parse_args()

REPLAY_MODE = args.replay
REPLAY_OUTPUT_DIR = args.replay_output_dir

if REPLAY_MODE:
    HUB_DATA_DIR = os.path.join(REPLAY_OUTPUT_DIR, "hub_data")
    os.makedirs(HUB_DATA_DIR, exist_ok=True)
    print("=" * 60)
    print(f"REPLAY MODE ACTIVE - Output to {HUB_DATA_DIR}")
    print("=" * 60)
```

**Location: ~Line 692-734 (`get_price` method)**
```python
def get_price(self, symbol, side='buy'):
    """Fetch current buy or sell price (live or simulated)."""
    if REPLAY_MODE:
        coin = symbol.replace('-USD', '')
        price_file = f"replay_data/{coin}_current_price.txt"
        try:
            with open(price_file, 'r') as f:
                price = float(f.read().strip())
            return price
        except FileNotFoundError:
            print(f"ERROR: No price file for {coin} in replay mode")
            return 0.0
    else:
        # Existing Robinhood API logic
        response = self.make_api_request("GET", f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}")
        # ... existing code ...
```

**Location: ~Line 737-810 (`place_buy_order`)**
```python
def place_buy_order(self, symbol, qty):
    """Place buy order (live or simulated)."""
    if REPLAY_MODE:
        import uuid
        order = {
            "ts": time.time(),
            "client_order_id": str(uuid.uuid4()),
            "side": "buy",
            "symbol": symbol,
            "qty": qty,
            "order_type": "market"
        }

        with open("replay_data/sim_orders.jsonl", "a") as f:
            f.write(json.dumps(order) + "\n")

        # Return instant fill for simplicity
        return {
            "id": order["client_order_id"],
            "state": "filled",
            "side": "buy",
            "symbol": symbol,
            "executed_notional": qty * self.get_price(symbol, 'buy'),
            "cumulative_quantity": qty
        }
    else:
        # Existing Robinhood API logic
        # ... existing code ...
```

**Validation:**
- Run pt_trader.py with --replay flag
- Verify reads prices from replay_data/*.txt files
- Verify orders written to sim_orders.jsonl
- Verify output written to replay output directory
- Verify no Robinhood API calls made

---

### Task 3.3: Implement Atomic State File Protocol
**Estimated Time:** 2 hours
**Priority:** CRITICAL
**Dependencies:** Task 3.2
**File:** `pt_replay.py`

**Actions:**
1. Replace multiple IPC files with single atomic state file
2. Implement sequence numbering to prevent stale reads
3. Create `backtest_state.json` schema
4. Add component handshake protocol via `component_ready.jsonl`

**State File Schema:**
```json
{
  "sequence": 12345,
  "timestamp": 1704067200,
  "prices": {
    "BTC": {"close": 98765.43, "high": 99000.00, "low": 98500.00, "volume": 1234.56},
    "ETH": {"close": 3421.12, "high": 3450.00, "low": 3400.00, "volume": 5678.90}
  },
  "status": "ready",
  "orchestrator_pid": 12345
}
```

**Functions to Implement:**
- `advance_time_atomic(current_ts, price_data)` - Orchestrator writes state
- `read_state_atomic()` - Subprocesses read state
- `wait_for_components(sequence_number, timeout=30.0)` - Orchestrator waits

**Validation:**
- Write state with advancing sequences
- Verify stale reads rejected
- Test component synchronization
- Verify no race conditions under load

---

### Task 3.4: Implement Time Progression Engine
**Estimated Time:** 3 hours
**Priority:** HIGH
**Dependencies:** Task 3.3
**File:** `pt_replay.py`

**Actions:**
1. Implement `replay_time_progression()` function
2. Load cached candles for all coins/timeframes
3. Create unified timeline (sorted timestamps)
4. Iterate timeline, updating atomic state file
5. Add progress tracking
6. Implement replay speed control

**Function Signature:**
```python
def replay_time_progression(start_ts, end_ts, coins, speed=1.0):
    """
    Iterate through historical timeline, updating IPC files.

    Args:
        start_ts: Unix timestamp start
        end_ts: Unix timestamp end
        coins: List of coin symbols (e.g., ['BTC', 'ETH'])
        speed: Replay speed multiplier (1.0 = real-time, 10.0 = 10x faster)
    """
```

**Validation:**
- Run replay for 24 hours of data
- Verify timestamp advances correctly
- Verify prices update for all coins
- Verify progress tracking accurate
- Test speed multiplier (10x should be 10x faster)

---

### Task 3.5: Implement Subprocess Management
**Estimated Time:** 2.5 hours
**Priority:** HIGH
**Dependencies:** Task 3.4
**File:** `pt_replay.py`
**Reference:** `pt_hub.py:1515-1675`

**Actions:**
1. Implement `run_backtest()` function
2. Launch pt_thinker.py and pt_trader.py as subprocesses
3. Add signal handler for graceful shutdown (Ctrl+C)
4. Implement subprocess monitoring
5. Add subprocess output logging
6. Add error handling and cleanup

**Function Signature:**
```python
def run_backtest(start_date, end_date, coins, output_dir, speed=1.0):
    """
    Launch pt_thinker.py and pt_trader.py in replay mode.

    Args:
        start_date: Date string "YYYY-MM-DD"
        end_date: Date string "YYYY-MM-DD"
        coins: List of coin symbols
        output_dir: Output directory for results
        speed: Replay speed multiplier
    """
```

**Validation:**
- Launch backtest with both subprocesses
- Verify both subprocesses start successfully
- Test Ctrl+C graceful shutdown
- Verify subprocess output captured
- Test error handling (subprocess crash)

---

### Task 3.6: Implement Main CLI Entry Point
**Estimated Time:** 1.5 hours
**Priority:** HIGH
**Dependencies:** Task 3.5
**File:** `pt_replay.py`

**Actions:**
1. Implement `main()` function with argparse
2. Add all CLI arguments (start-date, end-date, coins, speed, etc.)
3. Implement auto-generated output directory naming
4. Add cache warming mode
5. Add config file generation

**CLI Interface:**
```bash
python pt_replay.py \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC,ETH,DOGE \
  --speed 10.0 \
  --output-dir backtest_results/my_backtest
```

**Validation:**
- Test all CLI arguments
- Verify output directory created
- Verify config.json generated correctly
- Test --warm-cache mode (cache only, no backtest)

---

## Phase 4: Walk-Forward Validation (Days 15-18)

### Task 4.1: Create pt_incremental_trainer.py
**Estimated Time:** 3 hours
**Priority:** CRITICAL (Prevents look-ahead bias)
**Dependencies:** Phase 3 complete
**File:** `pt_incremental_trainer.py` (NEW)
**Reference:** `pt_trainer.py`

**Actions:**
1. Create new file based on existing pt_trainer.py
2. Add `--train-until` parameter (only use data up to timestamp)
3. Modify data loading to filter by max timestamp
4. Implement incremental training mode
5. Add model versioning (save timestamped models)

**Key Difference from pt_trainer.py:**
- Normal trainer: Uses ALL available data
- Incremental trainer: Uses ONLY data before specified timestamp

**Function Signature:**
```python
def train_incremental(coin, train_until_ts, output_model_path):
    """
    Train neural model using only data up to train_until_ts.
    Prevents look-ahead bias in backtesting.

    Args:
        coin: Coin symbol (e.g., "BTC")
        train_until_ts: Unix timestamp - only use data before this
        output_model_path: Path to save trained model
    """
```

**Validation:**
- Train model with train_until_ts = Jan 1, 2024
- Verify model uses ONLY data before Jan 1
- Train again with train_until_ts = Feb 1, 2024
- Verify model updated with additional data
- Verify no future data leakage

---

### Task 4.2: Integrate Incremental Training into Replay
**Estimated Time:** 2 hours
**Priority:** CRITICAL
**Dependencies:** Task 4.1
**File:** `pt_replay.py`

**Actions:**
1. Add training schedule to replay loop (every 7 days)
2. Launch pt_incremental_trainer.py at intervals
3. Track last training timestamp
4. Pass current timestamp as train_until parameter
5. Wait for training to complete before continuing replay

**Validation:**
- Run 30-day backtest
- Verify model retrains at day 7, 14, 21, 28
- Verify each training uses correct data window
- Verify no look-ahead bias (inspect training logs)

---

### Task 4.3: Add Walk-Forward Test Suite
**Estimated Time:** 2 hours
**Priority:** HIGH
**Dependencies:** Task 4.2
**File:** `tests/test_walk_forward.py` (NEW)

**Actions:**
1. Create test suite for walk-forward validation
2. Test that training timestamps never exceed backtest timestamps
3. Test incremental training data filtering
4. Test model versioning
5. Mock test with known data to verify no future leakage

**Test Cases:**
```python
def test_walk_forward_no_lookahead():
    """Verify neural model only trains on past data."""

def test_incremental_training_data_filter():
    """Verify data filtering by timestamp."""

def test_model_versioning():
    """Verify models saved with correct timestamps."""
```

**Validation:**
- All tests pass
- Coverage > 80% for walk-forward code paths

---

## Phase 5: Analytics & Metrics (Days 19-24)

### Task 5.1: Implement Corrected Sharpe Ratio
**Estimated Time:** 2 hours
**Priority:** CRITICAL
**Dependencies:** Phase 4 complete
**File:** `pt_analyze.py` (NEW)

**Actions:**
1. Create new file pt_analyze.py
2. Implement `build_equity_curve()` from trader_status snapshots
3. Implement `calculate_sharpe_ratio_corrected()` with:
   - 365-day annualization (crypto 24/7)
   - Time-series equity returns (not per-trade)
   - Risk-free rate adjustment
4. Add data validation

**Function Signature:**
```python
def calculate_sharpe_ratio_corrected(equity_curve, timestamps, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio using equity curve time series.

    Args:
        equity_curve: List of account values over time
        timestamps: Corresponding unix timestamps
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        float: Annualized Sharpe ratio
    """
```

**Validation:**
- Test with known equity curve
- Verify uses sqrt(365) not sqrt(252)
- Verify uses equity returns not trade returns
- Compare against manual calculation

---

### Task 5.2: Implement True Max Drawdown
**Estimated Time:** 1.5 hours
**Priority:** CRITICAL
**Dependencies:** Task 5.1
**File:** `pt_analyze.py`

**Actions:**
1. Implement `calculate_max_drawdown_corrected()` function
2. Calculate from total equity (cash + unrealized positions)
3. Add drawdown duration tracking
4. Identify peak and trough timestamps

**Function Signature:**
```python
def calculate_max_drawdown_corrected(equity_curve):
    """
    Calculate max drawdown from total account equity.

    Args:
        equity_curve: List of total account values (cash + unrealized positions)

    Returns:
        {
            "max_drawdown_pct": float,
            "max_drawdown_value": float,
            "peak_idx": int,
            "trough_idx": int,
            "duration_periods": int
        }
    """
```

**Validation:**
- Test scenario: Buy at 100, drop to 80, recover to 102
- Expected: Max DD ~20% (not 0%)
- Verify unrealized losses counted
- Verify peak/trough detection correct

---

### Task 5.3: Implement Sortino & Calmar Ratios
**Estimated Time:** 1.5 hours
**Priority:** MEDIUM
**Dependencies:** Task 5.2
**File:** `pt_analyze.py`

**Actions:**
1. Implement `calculate_sortino_ratio()` (downside-only volatility)
2. Implement `calculate_calmar_ratio()` (return / max drawdown)
3. Add to analytics report

**Function Signatures:**
```python
def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Only penalize downside volatility."""

def calculate_calmar_ratio(total_return, max_drawdown, years):
    """Return / Max Drawdown (annualized)."""
```

**Validation:**
- Test with known return distribution
- Verify Sortino only uses negative returns for volatility
- Verify Calmar calculation correct
- Compare against manual calculation

---

### Task 5.4: Implement Buy-and-Hold Benchmark
**Estimated Time:** 2 hours
**Priority:** HIGH
**Dependencies:** Task 5.3
**File:** `pt_analyze.py`

**Actions:**
1. Implement `calculate_buy_and_hold_benchmark()` function
2. Allocate initial capital equally across coins
3. Calculate final value without rebalancing
4. Add to analytics report with outperformance metric

**Function Signature:**
```python
def calculate_buy_and_hold_benchmark(start_ts, end_ts, initial_capital, coins):
    """
    Calculate buy-and-hold return for comparison.

    Strategy: At start, allocate capital equally across all coins.
              Hold until end without rebalancing.
    """
```

**Validation:**
- Test with known price data
- Verify equal allocation at start
- Verify no rebalancing
- Compare strategy vs. buy-and-hold

---

### Task 5.5: Implement Market Regime Detection
**Estimated Time:** 3 hours
**Priority:** MEDIUM
**Dependencies:** Task 5.4
**File:** `pt_analyze.py`

**Actions:**
1. Create `MarketRegimeDetector` class
2. Implement trend detection (bull/bear via SMA 50/200)
3. Implement volatility classification (low/mid/high)
4. Implement `analyze_performance_by_regime()`
5. Add regime breakdown to analytics report

**Class Methods:**
```python
class MarketRegimeDetector:
    def detect_regime(self, prices):
        """Classify: bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol, sideways"""

    def analyze_performance_by_regime(self, equity_curve, prices, timestamps):
        """Break down performance by regime."""
```

**Validation:**
- Test with known bull market → classifies as "bull"
- Test with known bear market → classifies as "bear"
- Test with sideways market → classifies as "sideways"
- Verify performance breakdown by regime

---

### Task 5.6: Implement Data Source Validation
**Estimated Time:** 2 hours
**Priority:** HIGH
**Dependencies:** Task 5.5
**File:** `pt_analyze.py`

**Actions:**
1. Implement `validate_data_sources()` function
2. Compare KuCoin vs Robinhood prices
3. Calculate divergence statistics
4. Calculate correlation
5. Calculate adjustment factor
6. Add validation report

**Function Signature:**
```python
def validate_data_sources(kucoin_prices, robinhood_prices, tolerance_pct=0.5):
    """
    Compare KuCoin and Robinhood prices to measure divergence.

    Returns:
        {
            "mean_divergence_pct": float,
            "max_divergence_pct": float,
            "correlation": float,
            "adjustment_factor": float
        }
    """
```

**Validation:**
- Fetch sample data from both sources
- Verify divergence calculation
- Verify correlation calculation
- Test adjustment factor application

---

### Task 5.7: Implement Analytics Report Generator
**Estimated Time:** 3 hours
**Priority:** HIGH
**Dependencies:** Tasks 5.1-5.6
**File:** `pt_analyze.py`

**Actions:**
1. Implement `generate_analytics_report()` function
2. Load trade history and config
3. Calculate all metrics (Sharpe, Sortino, Calmar, max DD, etc.)
4. Generate HTML report with tables and charts
5. Add executive summary section
6. Add regime performance breakdown
7. Add buy-and-hold comparison

**Report Sections:**
- Executive Summary (total return, Sharpe, max DD)
- Trade Statistics (count, win rate, profit factor)
- Risk Metrics (Sharpe, Sortino, Calmar, max DD)
- Performance by Regime
- Buy-and-Hold Comparison
- Configuration Details

**Validation:**
- Generate report from test backtest
- Verify all metrics calculated
- Verify HTML renders correctly
- Verify charts display

---

### Task 5.8: Implement CLI for Analytics
**Estimated Time:** 1 hour
**Priority:** MEDIUM
**Dependencies:** Task 5.7
**File:** `pt_analyze.py`

**Actions:**
1. Add `main()` function with argparse
2. Add CLI argument for backtest directory
3. Generate report and save to output directory

**CLI Interface:**
```bash
python pt_analyze.py backtest_results/backtest_2025-01-15_143022
```

**Validation:**
- Run analyzer on test backtest
- Verify report generated
- Verify report saved to correct location

---

## Phase 6: Observability & Testing (Days 25-28)

### Task 6.1: Implement Structured Logging
**Estimated Time:** 2.5 hours
**Priority:** MEDIUM
**Dependencies:** Phase 5 complete
**File:** `pt_replay.py`, `pt_thinker.py`, `pt_trader.py`

**Actions:**
1. Create `BacktestLogger` class
2. Add JSON structured logging
3. Add log events: replay_tick, trade_execution, neural_signal, error
4. Add console and file handlers
5. Integrate into all components

**Logger Events:**
- `replay_tick`: Each timestamp advance
- `trade_execution`: Order fills with slippage/fees
- `neural_signal`: Signal generation
- `error`: All errors with context

**Validation:**
- Run backtest with logging
- Verify JSON logs written
- Verify console output readable
- Test log analysis queries (grep + jq)

---

### Task 6.2: Create Integration Test Suite
**Estimated Time:** 4 hours
**Priority:** HIGH
**Dependencies:** Task 6.1
**File:** `tests/test_backtest_integration.py` (NEW)

**Actions:**
1. Create test file structure
2. Implement `test_subprocess_synchronization()`
3. Implement `test_missing_candle_data()`
4. Implement `test_realistic_execution_slippage()`
5. Implement `test_walk_forward_no_lookahead()`
6. Implement `test_equity_curve_includes_unrealized()`
7. Add test fixtures and mock data

**Test Cases:**
```python
def test_subprocess_synchronization():
    """Verify thinker and trader stay synchronized during replay."""

def test_missing_candle_data():
    """Ensure graceful handling when cache has gaps."""

def test_realistic_execution_slippage():
    """Verify slippage calculation is realistic."""

def test_walk_forward_no_lookahead():
    """Verify neural model only trains on past data."""

def test_equity_curve_includes_unrealized():
    """Verify max drawdown includes unrealized losses."""
```

**Validation:**
- All tests pass
- Code coverage > 75%
- Tests run in < 2 minutes

---

### Task 6.3: Create End-to-End Smoke Test
**Estimated Time:** 2 hours
**Priority:** HIGH
**Dependencies:** Task 6.2
**File:** `tests/test_e2e_smoke.py` (NEW)

**Actions:**
1. Create minimal backtest (1 week of data)
2. Run complete backtest workflow
3. Verify all components execute
4. Verify report generated
5. Verify metrics calculated

**Smoke Test Flow:**
1. Warm cache for 1 week BTC data
2. Run backtest
3. Generate analytics
4. Verify report exists

**Validation:**
- Test completes in < 5 minutes
- All files generated
- No errors in logs

---

### Task 6.4: Create Performance Benchmark Test
**Estimated Time:** 1.5 hours
**Priority:** MEDIUM
**Dependencies:** Task 6.3
**File:** `tests/test_performance.py` (NEW)

**Actions:**
1. Create benchmark test for 6-month backtest
2. Measure execution time
3. Verify meets target (< 10 minutes)
4. Measure memory usage
5. Add profiling

**Performance Targets:**
- 6-month backtest: < 10 minutes
- Memory usage: < 2GB
- Cache size: < 500MB per coin

**Validation:**
- Run benchmark test
- Verify meets all targets
- Generate performance report

---

### Task 6.5: Create Documentation
**Estimated Time:** 3 hours
**Priority:** MEDIUM
**Dependencies:** All previous tasks
**File:** `docs/BACKTESTING_GUIDE.md` (NEW)

**Actions:**
1. Create user guide for running backtests
2. Document CLI commands with examples
3. Document configuration options
4. Add troubleshooting section
5. Add FAQ section
6. Add architecture diagram

**Sections:**
- Quick Start (5-minute tutorial)
- CLI Reference
- Configuration Options
- Understanding Reports
- Troubleshooting
- Advanced Usage
- FAQ

**Validation:**
- Follow quick start guide (fresh perspective)
- Verify all commands work
- Test all examples

---

### Task 6.6: Final Integration Testing
**Estimated Time:** 2 hours
**Priority:** CRITICAL
**Dependencies:** All tasks complete
**File:** N/A (manual testing)

**Actions:**
1. Run full 6-month backtest on BTC
2. Verify all success criteria met
3. Run 3-month multi-coin backtest (BTC, ETH, DOGE)
4. Compare results against expectations
5. Verify realistic execution (slippage, fees)
6. Verify no look-ahead bias
7. Generate final analytics report

**Success Criteria Validation:**
- ✅ 6-month BTC backtest completes in < 10 minutes
- ✅ Report includes Sharpe, Sortino, Calmar, max DD
- ✅ Max DD includes unrealized losses
- ✅ Walk-forward validation passes (no look-ahead)
- ✅ Execution model realistic (within ±2% of live)

**Validation:**
- All criteria met
- No critical bugs
- Performance acceptable
- Documentation complete

---

## Task Dependency Graph

```
Phase 1: Data Caching Infrastructure
├── 1.1 Directory Structure
├── 1.2 Atomic Write Helper → depends on 1.1
├── 1.3 KuCoin Fetcher → depends on 1.2
├── 1.4 Cache Index → depends on 1.3
└── 1.5 Cache Warming CLI → depends on 1.4

Phase 2: Realistic Execution Engine
├── 2.1 RealisticExecutionEngine → depends on Phase 1
├── 2.2 Execution Config → depends on 2.1
└── 2.3 Mock Order Execution → depends on 2.1

Phase 3: Replay Mode Support
├── 3.1 pt_thinker.py replay → depends on Phase 2
├── 3.2 pt_trader.py replay → depends on 3.1
├── 3.3 Atomic State File → depends on 3.2
├── 3.4 Time Progression → depends on 3.3
├── 3.5 Subprocess Management → depends on 3.4
└── 3.6 Main CLI → depends on 3.5

Phase 4: Walk-Forward Validation
├── 4.1 pt_incremental_trainer → depends on Phase 3
├── 4.2 Integration into Replay → depends on 4.1
└── 4.3 Walk-Forward Tests → depends on 4.2

Phase 5: Analytics & Metrics
├── 5.1 Corrected Sharpe → depends on Phase 4
├── 5.2 True Max Drawdown → depends on 5.1
├── 5.3 Sortino & Calmar → depends on 5.2
├── 5.4 Buy-and-Hold Benchmark → depends on 5.3
├── 5.5 Regime Detection → depends on 5.4
├── 5.6 Data Source Validation → depends on 5.5
├── 5.7 Report Generator → depends on 5.1-5.6
└── 5.8 Analytics CLI → depends on 5.7

Phase 6: Observability & Testing
├── 6.1 Structured Logging → depends on Phase 5
├── 6.2 Integration Tests → depends on 6.1
├── 6.3 E2E Smoke Test → depends on 6.2
├── 6.4 Performance Benchmark → depends on 6.3
├── 6.5 Documentation → depends on all
└── 6.6 Final Integration → depends on all
```

---

## Critical Path (Minimum Viable Backtest)

For fastest path to working backtest (skip non-critical tasks):

1. **Phase 1 Tasks:** 1.1, 1.2, 1.3, 1.4 (skip 1.5, use manual caching)
2. **Phase 2 Tasks:** 2.1 only (use default config)
3. **Phase 3 Tasks:** 3.1, 3.2, 3.4, 3.5, 3.6 (skip 3.3 for MVP)
4. **Phase 4 Tasks:** Skip entirely for MVP (add later)
5. **Phase 5 Tasks:** 5.1, 5.2 only (basic metrics)
6. **Phase 6 Tasks:** 6.3 only (smoke test)

**MVP Timeline:** ~10-12 days (vs. full 21-28 days)

---

## Risk Mitigation

### High-Risk Tasks
- **Task 3.3 (Atomic State File):** Complex synchronization logic
  - Mitigation: Extensive testing with race condition scenarios
  - Fallback: Use simpler multi-file IPC with sleep delays

- **Task 4.1-4.2 (Walk-Forward):** Critical for accuracy
  - Mitigation: Comprehensive tests, manual data inspection
  - Validation: Compare backtest results to paper trading

- **Task 3.5 (Subprocess Management):** Process coordination issues
  - Mitigation: Robust error handling, timeouts, cleanup
  - Testing: Stress tests with subprocess failures

### Medium-Risk Tasks
- **Task 1.3 (KuCoin Fetcher):** API rate limits
  - Mitigation: Conservative rate limiting, retry logic
  - Fallback: Manual data download if API fails

- **Task 2.1 (Execution Engine):** Model accuracy
  - Mitigation: Validate against live trading data
  - Calibration: Tune parameters based on actual slippage

---

## Success Metrics by Phase

### Phase 1
- ✅ Can fetch and cache 6 months of BTC data
- ✅ Cache size reasonable (< 100MB per coin)
- ✅ Second fetch uses cache (no API calls)

### Phase 2
- ✅ Slippage realistic (0.05-0.2% on normal volatility)
- ✅ Partial fills work correctly
- ✅ Fees calculated accurately

### Phase 3
- ✅ pt_thinker and pt_trader run in replay mode
- ✅ No live API calls during replay
- ✅ Output written to backtest directory

### Phase 4
- ✅ Walk-forward tests pass (no look-ahead bias)
- ✅ Model retrains incrementally
- ✅ Training uses only past data

### Phase 5
- ✅ Sharpe ratio uses 365-day annualization
- ✅ Max DD includes unrealized losses
- ✅ Report generates correctly

### Phase 6
- ✅ All integration tests pass
- ✅ 6-month backtest < 10 minutes
- ✅ Documentation complete

---

## Next Steps After Blueprint Approval

1. **Create feature branch:** `feature/backtesting-system`
2. **Begin Phase 1, Task 1.1:** Directory structure
3. **Set up daily standup:** Track progress, blockers
4. **Weekly review:** Demo progress, adjust timeline
5. **Continuous testing:** Run tests after each task
6. **Documentation as you go:** Update docs with learnings

---

## Appendix: File Reference Map

| Task | File Modified/Created | Lines | Reference |
|------|----------------------|-------|-----------|
| 1.2 | pt_replay.py (NEW) | 1-50 | CLAUDE.md:42-47 |
| 1.3 | pt_replay.py | 51-200 | pt_trainer.py:403-441 |
| 1.4 | pt_replay.py | 201-300 | - |
| 1.5 | pt_replay.py | 701-900 | - |
| 2.1 | pt_replay.py | 301-500 | - |
| 3.1 | pt_thinker.py | 430, 556, 1095 | pt_thinker.py:477-1091 |
| 3.2 | pt_trader.py | 692, 737, 1420 | pt_trader.py:692-852 |
| 3.4 | pt_replay.py | 501-700 | - |
| 3.5 | pt_replay.py | 701-900 | pt_hub.py:1515-1675 |
| 4.1 | pt_incremental_trainer.py (NEW) | All | pt_trainer.py |
| 5.1-5.8 | pt_analyze.py (NEW) | All | - |
| 6.1 | All files | Various | - |
| 6.2-6.4 | tests/ (NEW) | All | - |

---

**Blueprint Status:** ✅ Complete and ready for implementation
**Last Updated:** 2025-12-30
**Version:** 1.0
