# Backtesting & Simulation Feature Proposal

**Status:** Proposed
**Created:** 2025-12-30
**Author:** Claude (Brainstorm Workflow)
**Estimated Timeline:** 14 days
**Priority:** High

---

## Executive Summary

This proposal defines a comprehensive backtesting and simulation capability for PowerTrader AI, enabling traders to validate strategies on historical data before risking capital. The implementation uses a **Hybrid Replay Mode** that preserves the system's file-based IPC architecture while delivering professional-grade performance analytics.

**Key Benefits:**
- ✅ Validate strategies on 1+ years of historical data
- ✅ Optimize DCA parameters with quantitative metrics
- ✅ Identify max drawdown and risk exposure before live trading
- ✅ A/B test strategy variations in hours instead of weeks

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [User Value & Business Logic](#user-value--business-logic)
3. [Implementation Approaches Evaluated](#implementation-approaches-evaluated)
4. [Recommended Solution: Hybrid Replay Mode](#recommended-solution-hybrid-replay-mode)
5. [Technical Specification](#technical-specification)
6. [Implementation Plan](#implementation-plan)
7. [Testing & Validation](#testing--validation)
8. [Risk Mitigation](#risk-mitigation)
9. [Success Metrics](#success-metrics)
10. [Appendix: Code References](#appendix-code-references)

---

## Problem Statement

### Current Limitations

PowerTrader AI currently lacks the ability to:

1. **Historical Validation** - No way to test if the DCA strategy would have been profitable during past bear/bull markets
2. **Safe Experimentation** - Strategy changes must be tested with real money, risking capital on unproven modifications
3. **Parameter Optimization** - DCA levels (`[-2.5%, -5%, -10%, -20%, -30%, -40%, -50%]`) are hardcoded with no data-driven method to find optimal thresholds
4. **Performance Metrics** - No Sharpe ratio, maximum drawdown, win rate, or profit factor beyond basic PnL tracking
5. **Strategy Comparison** - Cannot A/B test "neural-only" vs "DCA-hybrid" vs "trailing PM adjustments"

### Impact on Trading

**Without Backtesting:**
- ❌ Test new DCA threshold live → -$500 loss in 2 weeks → revert changes
- ❌ Unknown max drawdown until experienced in live trading
- ❌ No confidence in strategy during volatile markets

**With Backtesting:**
- ✅ Test on 6 months of historical data → see +15% improvement → deploy with confidence
- ✅ Know max drawdown was -18% in worst-case historical scenario
- ✅ Validate strategy works in both bull and bear markets

---

## User Value & Business Logic

### For Individual Traders

**Confidence & Risk Management:**
- Validate strategies on 1+ years of historical data before risking capital
- Understand maximum drawdown exposure (e.g., -25% in March 2024)
- Test strategy behavior during specific market events (crashes, pumps, sideways)

**Optimization & Performance:**
- Discover optimal DCA levels (e.g., -7.5% performs better than current -10%)
- Compare trailing profit margin starting points (+3% vs +5%)
- Identify which neural signal levels provide best entry timing

**Education & Debugging:**
- Replay specific dates to understand why trades succeeded or failed
- Learn when neural pattern matching works vs. when it produces false signals
- Build intuition about market conditions through accelerated replay

### For Strategy Development

**Faster Iteration:**
- Test 10 parameter variations in 1 hour vs. 10 weeks of live trading
- Rapidly prototype new DCA level configurations
- Validate neural threshold adjustments without risking capital

**Data-Driven Decisions:**
- Replace gut-feel tuning with quantitative Sharpe ratio and win rate metrics
- Compare strategy performance vs. buy-and-hold baseline
- Use walk-forward analysis to validate out-of-sample performance

**What-If Analysis:**
- "What if I had started trailing PM at +3% instead of +5% during Jan 2024?"
- "How would the strategy have performed without DCA assistance?"
- "What if I only traded when neural level 5+ instead of level 3+?"

### ROI Example

| Scenario | Time Investment | Capital Risk | Learning |
|----------|----------------|--------------|----------|
| **Live Testing** | 2-4 weeks per variation | -$500 potential loss | Slow, expensive |
| **Backtesting** | 1 hour per variation | $0 risk | Fast, comprehensive |

---

## Implementation Approaches Evaluated

### Approach 1: Minimal Simulation Mode (Quick Win)

**Concept:** Add `--simulation` flag to existing components with minimal refactoring.

**Architecture:**
- Add flag to `pt_trader.py` and `pt_thinker.py`
- Mock Robinhood API calls with simulated fills
- In-memory balance tracking
- Write to `sim_trade_history.jsonl`

**Pros:**
- ✅ Fastest implementation (1-2 days)
- ✅ Reuses 100% of existing trading logic
- ✅ No database required

**Cons:**
- ❌ No historical replay (still uses live market data)
- ❌ No parameter optimization
- ❌ Cannot test past market conditions
- ❌ Real-time only (1 hour = 1 hour)

**Verdict:** Good for paper trading, insufficient for backtesting.

---

### Approach 2: Full-Featured Backtester (Comprehensive)

**Concept:** Build standalone backtesting engine with database, parameter optimization, and professional analytics.

**Architecture:**
- New `pt_backtester.py` orchestrator
- SQLite cache for historical OHLCV data
- `backtest/` module with:
  - `data_manager.py` - Database layer
  - `mock_exchange.py` - Simulated order matching
  - `analytics.py` - Metrics calculator
  - `optimizer.py` - Grid search / Bayesian optimization
- Refactor trading logic into reusable `strategy/` module

**Pros:**
- ✅ Professional-grade backtesting
- ✅ Automated parameter optimization
- ✅ Walk-forward analysis support
- ✅ Very fast (years of data in minutes)

**Cons:**
- ❌ 3-4 weeks development time
- ❌ High refactoring risk (could break live trading)
- ❌ Maintenance burden (dual codepaths may diverge)
- ❌ Requires SQL and new dependencies

**Verdict:** Most powerful, but high risk and long timeline.

---

### Approach 3: Hybrid Replay Mode ⭐ RECOMMENDED

**Concept:** Add "replay mode" to existing components + post-processing analytics script.

**Architecture:**
- **pt_replay.py** - New orchestrator script
  - Fetches/caches historical data from KuCoin
  - Controls time progression via file-based IPC
  - Manages pt_thinker.py + pt_trader.py subprocesses
- **Minimal Component Changes:**
  - Add `--replay` flag to pt_thinker.py / pt_trader.py
  - Read prices from files instead of APIs
  - Mock order execution via IPC files
- **pt_analyze.py** - New analytics script
  - Computes Sharpe, drawdown, win rate
  - Generates HTML report with equity curve

**Pros:**
- ✅ Moderate dev time (1-2 weeks)
- ✅ Preserves file-based IPC architecture
- ✅ Reuses existing components with minimal changes
- ✅ Can test historical scenarios
- ✅ Simple JSON caching (no SQL complexity)
- ✅ Low risk to live trading

**Cons:**
- ❌ No automated parameter optimization (manual A/B testing)
- ❌ Slower than purpose-built backtester
- ❌ Trainer must re-run for each replay

**Verdict:** Best balance of value, risk, and timeline. Can upgrade to Approach 2 later if needed.

---

## Recommended Solution: Hybrid Replay Mode

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     pt_replay.py                            │
│                  (Replay Orchestrator)                      │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │ Cache Mgr    │  │ Time Engine   │  │ Mock Exchange   │ │
│  │ (KuCoin)     │  │ (IPC Writer)  │  │ (Order Fills)   │ │
│  └──────────────┘  └───────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ File-based IPC
                             ▼
        ┌────────────────────────────────────────┐
        │       replay_data/ (IPC Files)         │
        │  • current_timestamp.txt               │
        │  • BTC_current_price.txt               │
        │  • sim_orders.jsonl                    │
        │  • sim_fills.jsonl                     │
        └────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ pt_thinker.py│          │ pt_trader.py │
        │ --replay     │          │ --replay     │
        │              │          │              │
        │ Reads cached │          │ Mock orders  │
        │ candles      │          │ Mock balance │
        └──────────────┘          └──────────────┘
                │                         │
                │                         ▼
                │              backtest_results/<run_id>/
                │              • trade_history.jsonl
                │              • trader_status.jsonl
                │              • config.json
                │
                └───────────────┐
                                ▼
                        ┌──────────────┐
                        │ pt_analyze.py│
                        │              │
                        │ • Sharpe     │
                        │ • Drawdown   │
                        │ • Win Rate   │
                        │ • HTML Report│
                        └──────────────┘
```

### Key Design Principles

1. **Preserve File-Based IPC** - All communication uses files (no shared memory, no SQL)
2. **Minimal Refactoring** - Existing trading logic unchanged, only data source switches
3. **Atomic Writes** - All JSON files use `.tmp` → `os.replace()` pattern (per CLAUDE.md:42-47)
4. **Separate Execution Contexts** - Live vs. replay controlled by command-line flags
5. **Incremental Enhancement** - Can add parameter optimization later without redesign

---

## Technical Specification

### 1. File Structure

```
/home/user/power-trader-ai/
├── pt_replay.py                    # Replay orchestrator (NEW)
├── pt_analyze.py                   # Analytics generator (NEW)
├── backtest_cache/                 # Historical data cache (NEW)
│   ├── BTC-USDT_1hour_1704067200_1706745600.json
│   ├── ETH-USDT_1hour_1704067200_1706745600.json
│   └── cache_index.json           # Cache metadata
├── replay_data/                    # Replay IPC directory (NEW)
│   ├── current_price.txt          # Current simulated price
│   ├── current_timestamp.txt      # Current simulated time
│   ├── replay_control.json        # Orchestration control
│   ├── sim_orders.jsonl           # Mock order requests
│   └── sim_fills.jsonl            # Mock order fills
└── backtest_results/              # Output directory (NEW)
    ├── backtest_2025-01-15_143022/
    │   ├── config.json            # Backtest parameters
    │   ├── trade_history.jsonl    # Simulated trades
    │   ├── trader_status.jsonl    # Timestamped snapshots
    │   ├── analytics_report.html  # Performance metrics
    │   └── equity_curve.png       # Account value chart
    └── latest -> backtest_2025-01-15_143022/
```

### 2. Modified Files

**pt_thinker.py**
- Add `--replay` command-line flag
- Replace `market.get_kline()` calls with cache reads when in replay mode
- Read replay timestamp from `replay_data/current_timestamp.txt` instead of `time.time()`

**pt_trader.py**
- Add `--replay` command-line flag
- Replace Robinhood price API calls with reads from `replay_data/current_price.txt`
- Replace `place_buy_order()` / `place_sell_order()` with mock execution
- Redirect `hub_data/` writes to `backtest_results/<run_id>/`

### 3. File-based IPC Protocol

#### New IPC Files

**replay_data/current_timestamp.txt**
- **Format:** Unix timestamp (integer)
- **Written by:** pt_replay.py
- **Read by:** pt_thinker.py (in replay mode)
- **Example:**
  ```
  1704067200
  ```

**replay_data/BTC_current_price.txt** (per-coin)
- **Format:** Single float (close price)
- **Written by:** pt_replay.py
- **Read by:** pt_trader.py (in replay mode)
- **Example:**
  ```
  98765.43
  ```

**replay_data/replay_control.json**
- **Format:** JSON object (atomic writes)
- **Schema:**
  ```json
  {
    "status": "running|paused|completed|error",
    "current_ts": 1704067200,
    "start_ts": 1704000000,
    "end_ts": 1706745600,
    "progress_pct": 12.5,
    "candles_processed": 120,
    "trades_executed": 8,
    "error_message": null
  }
  ```

**replay_data/sim_orders.jsonl**
- **Format:** JSONL (append-only)
- **Written by:** pt_trader.py
- **Read by:** pt_replay.py
- **Schema:**
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

**replay_data/sim_fills.jsonl**
- **Format:** JSONL (append-only)
- **Written by:** pt_replay.py
- **Read by:** pt_trader.py
- **Schema:**
  ```json
  {
    "ts": 1704067201.0,
    "client_order_id": "uuid-...",
    "side": "buy",
    "symbol": "BTC-USD",
    "qty": 0.001,
    "fill_price": 98800.12,
    "state": "filled"
  }
  ```

#### Time Synchronization Protocol

1. **pt_replay.py** advances time:
   - Write new timestamp to `current_timestamp.txt`
   - Update `{COIN}_current_price.txt` files
   - Sleep 0.1-0.5s to allow components to process

2. **pt_thinker.py** (replay mode):
   - Read `current_timestamp.txt` instead of `time.time()`
   - Use cached candles filtered by timestamp
   - Write signals to normal IPC files (unchanged)

3. **pt_trader.py** (replay mode):
   - Read prices from `{COIN}_current_price.txt`
   - Write orders to `sim_orders.jsonl`
   - Wait for fills in `sim_fills.jsonl` (blocking read with timeout)

### 4. Data Caching Strategy

#### Cache File Format

**backtest_cache/{SYMBOL}_{TIMEFRAME}_{START_TS}_{END_TS}.json**

```json
{
  "symbol": "BTC-USDT",
  "timeframe": "1hour",
  "start_ts": 1704067200,
  "end_ts": 1706745600,
  "candles": [
    {
      "time": 1704067200,
      "open": 98000.0,
      "high": 98500.0,
      "low": 97800.0,
      "close": 98200.0,
      "volume": 1234.56
    }
  ]
}
```

#### Cache Index

**backtest_cache/cache_index.json**

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

#### Cache Warming

Pre-fetch data before running backtest:

```bash
python pt_replay.py --warm-cache \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC,ETH,DOGE
```

### 5. Configuration

#### Command-Line Interface

**pt_replay.py**

```bash
python pt_replay.py [OPTIONS]

Required:
  --start-date YYYY-MM-DD       Start date for backtest
  --end-date YYYY-MM-DD         End date for backtest

Optional:
  --coins BTC,ETH,DOGE          Comma-separated coin list
  --output-dir DIR              Output directory
  --speed FLOAT                 Replay speed multiplier (default: 1.0)
  --warm-cache                  Only fetch/cache data, don't run backtest
  --cache-dir DIR               Cache directory (default: backtest_cache)
  --initial-capital USD         Starting account value
  --no-thinker                  Skip pt_thinker.py (use pre-generated signals)
  --no-trader                   Skip pt_trader.py (dry-run only)
```

**pt_analyze.py**

```bash
python pt_analyze.py <backtest_results_directory>

Example:
python pt_analyze.py backtest_results/backtest_2025-01-15_143022
```

#### Backtest Configuration File

**backtest_results/<run_id>/config.json**

```json
{
  "run_id": "backtest_2025-01-15_143022",
  "start_date": "2024-01-01",
  "end_date": "2024-02-01",
  "start_ts": 1704067200,
  "end_ts": 1706745600,
  "coins": ["BTC", "ETH", "DOGE"],
  "initial_capital_usd": 10000.0,
  "replay_speed": 1.0,
  "components": {
    "thinker_enabled": true,
    "trader_enabled": true
  },
  "git_commit": "20a2149",
  "created_at": 1736950822.5
}
```

---

## Implementation Plan

### Phase 1: Data Caching Infrastructure (Days 1-3)

#### Task 1.1: Create Cache Directory Structure

```bash
mkdir -p backtest_cache replay_data backtest_results
```

#### Task 1.2: Implement pt_replay.py Foundation

**File:** Create `/home/user/power-trader-ai/pt_replay.py`

**Lines 1-250: Cache Management Module**

```python
import os
import json
import time
from kucoin.client import Market

def _atomic_write_json(path, data):
    """Atomic write pattern from CLAUDE.md:42-47"""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def fetch_historical_klines(symbol, timeframe, start_ts, end_ts):
    """
    Fetch OHLCV candles from KuCoin with rate limiting.
    Based on pt_trainer.py:403-441

    Returns:
        List[dict]: [{"time": ts, "open": o, "close": c, "high": h, "low": l, "volume": v}, ...]
    """
    market = Market()
    cache_file = f"backtest_cache/{symbol}_{timeframe}_{start_ts}_{end_ts}.json"

    # Check cache first
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)["candles"]

    # Fetch from KuCoin API
    print(f"Fetching {symbol} {timeframe} from {start_ts} to {end_ts}...")
    candles = []

    # KuCoin returns max 1500 candles per request
    # Similar logic to pt_trainer.py:403-441
    current_start = start_ts
    while current_start < end_ts:
        try:
            response = market.get_kline(
                symbol=symbol,
                kline_type=timeframe,
                startAt=current_start,
                endAt=end_ts
            )

            # Parse response (format: [timestamp, open, close, high, low, volume])
            batch = json.loads(str(response))
            if not batch:
                break

            for candle in batch:
                candles.append({
                    "time": int(candle[0]),
                    "open": float(candle[1]),
                    "close": float(candle[2]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "volume": float(candle[5])
                })

            # Update for next batch
            current_start = candles[-1]["time"] + 1

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching candles: {e}")
            time.sleep(1)  # Backoff on error
            continue

    # Cache result
    data = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "candles": candles
    }
    _atomic_write_json(cache_file, data)

    print(f"Cached {len(candles)} candles to {cache_file}")
    return candles
```

**Reference:** Based on `/home/user/power-trader-ai/pt_trainer.py:403-441`

#### Task 1.3: Implement Cache Index Management

```python
def update_cache_index(symbol, timeframe, start_ts, end_ts, candle_count):
    """Update cache index for fast lookups."""
    index_file = "backtest_cache/cache_index.json"

    # Load existing index
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
    else:
        index = {}

    # Update index
    key = f"{symbol}_{timeframe}"
    if key not in index:
        index[key] = []

    index[key].append({
        "start_ts": start_ts,
        "end_ts": end_ts,
        "file": f"{symbol}_{timeframe}_{start_ts}_{end_ts}.json",
        "candle_count": candle_count,
        "created_at": int(time.time())
    })

    # Write atomically
    _atomic_write_json(index_file, index)
```

---

### Phase 2: Replay Mode Support (Days 4-6)

#### Task 2.1: Modify pt_thinker.py for Replay

**File:** `/home/user/power-trader-ai/pt_thinker.py`

**Addition at lines ~1095-1115 (before main loop):**

```python
import argparse

# Parse command-line arguments
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

**Modification at line ~556 (in `step_coin` function):**

**Current code:**
```python
history = str(market.get_kline(coin, tf_choices[index]))
```

**New code:**
```python
if REPLAY_MODE:
    # Read from cached data based on current replay timestamp
    with open('replay_data/current_timestamp.txt', 'r') as f:
        current_ts = int(f.read().strip())
    history = _read_cached_kline_for_replay(coin, tf_choices[index], current_ts)
else:
    history = str(market.get_kline(coin, tf_choices[index]))
```

**Add helper function (around line ~430):**

```python
def _read_cached_kline_for_replay(coin, timeframe, current_ts):
    """
    Read cached historical candles for replay mode.
    Returns same format as KuCoin API for compatibility.
    """
    # Find appropriate cache file from index
    index_file = os.path.join(REPLAY_CACHE_DIR, "cache_index.json")
    with open(index_file, "r") as f:
        index = json.load(f)

    symbol = f"{coin}-USDT"
    key = f"{symbol}_{timeframe}"

    if key not in index:
        raise FileNotFoundError(f"No cached data for {key}")

    # Load candles from most recent cache file
    cache_entry = index[key][-1]
    cache_file = os.path.join(REPLAY_CACHE_DIR, cache_entry["file"])

    with open(cache_file, "r") as f:
        data = json.load(f)

    # Filter candles up to current_ts and format for compatibility
    candles = [c for c in data["candles"] if c["time"] <= current_ts]

    # Convert back to KuCoin API format: [[time, open, close, high, low, volume], ...]
    formatted = [
        [c["time"], c["open"], c["close"], c["high"], c["low"], c["volume"]]
        for c in candles
    ]

    return str(formatted)
```

**Reference:** Signal generation logic at `/home/user/power-trader-ai/pt_thinker.py:477-1091`

#### Task 2.2: Modify pt_trader.py for Replay

**File:** `/home/user/power-trader-ai/pt_trader.py`

**Addition at lines ~1420-1435 (in `if __name__ == "__main__"` block):**

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--replay', action='store_true', help='Run in replay mode')
parser.add_argument('--replay-output-dir', default='backtest_results/latest', help='Output directory')
args = parser.parse_args()

REPLAY_MODE = args.replay
REPLAY_OUTPUT_DIR = args.replay_output_dir

# Redirect hub data directory in replay mode
if REPLAY_MODE:
    HUB_DATA_DIR = os.path.join(REPLAY_OUTPUT_DIR, "hub_data")
    os.makedirs(HUB_DATA_DIR, exist_ok=True)
    print("=" * 60)
    print(f"REPLAY MODE ACTIVE - Output to {HUB_DATA_DIR}")
    print("=" * 60)
```

**Modification at lines ~20-27 (HUB_DATA_DIR setup):**

```python
# Use replay output dir if in replay mode
if 'REPLAY_MODE' in globals() and REPLAY_MODE:
    HUB_DATA_DIR = os.path.join(REPLAY_OUTPUT_DIR, "hub_data")
else:
    HUB_DATA_DIR = os.environ.get("POWERTRADER_HUB_DIR", "hub_data")

os.makedirs(HUB_DATA_DIR, exist_ok=True)
```

**Modification at lines ~692-734 (`get_price` method):**

**Current code:**
```python
def get_price(self, symbol, side='buy'):
    """Fetch current buy or sell price from Robinhood."""
    response = self.make_api_request("GET", f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}")
    # ... parsing logic ...
```

**New code:**
```python
def get_price(self, symbol, side='buy'):
    """Fetch current buy or sell price (live or simulated)."""
    if REPLAY_MODE:
        # Read from replay IPC file
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
        # Real Robinhood API call
        response = self.make_api_request("GET", f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}")
        # ... existing parsing logic ...
```

**Modification at lines ~737-810 (`place_buy_order`):**

```python
def place_buy_order(self, symbol, qty):
    """Place buy order (live or simulated)."""
    if REPLAY_MODE:
        # Mock order execution
        import uuid
        order = {
            "ts": time.time(),
            "client_order_id": str(uuid.uuid4()),
            "side": "buy",
            "symbol": symbol,
            "qty": qty,
            "order_type": "market"
        }

        # Write to sim_orders.jsonl (append-only)
        with open("replay_data/sim_orders.jsonl", "a") as f:
            f.write(json.dumps(order) + "\n")

        # In replay mode, fills are instant (written by pt_replay.py)
        # For simplicity, return immediate fill
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
        # ... current implementation ...
```

**Similar modification for `place_sell_order` at lines ~813-852**

**Reference:** Order execution logic at `/home/user/power-trader-ai/pt_trader.py:737-852`

---

### Phase 3: Replay Orchestration (Days 7-9)

#### Task 3.1: Implement Time Progression Engine

**File:** `/home/user/power-trader-ai/pt_replay.py` (lines 300-500)

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
    print(f"\n{'='*60}")
    print(f"Starting replay from {start_ts} to {end_ts}")
    print(f"Coins: {', '.join(coins)}")
    print(f"Speed: {speed}x")
    print(f"{'='*60}\n")

    # Load all cached candles
    candles_by_coin = {}
    for coin in coins:
        for tf in ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']:
            symbol = f"{coin}-USDT"
            candles = fetch_historical_klines(symbol, tf, start_ts, end_ts)
            candles_by_coin[f"{coin}_{tf}"] = candles

    # Merge into unified timeline (sorted by timestamp)
    all_timestamps = set()
    for candles in candles_by_coin.values():
        all_timestamps.update([c["time"] for c in candles])
    timeline = sorted(all_timestamps)

    print(f"Timeline: {len(timeline)} unique timestamps")

    # Iterate through timeline
    for idx, ts in enumerate(timeline):
        # Write current timestamp
        with open("replay_data/current_timestamp.txt", "w") as f:
            f.write(str(ts))

        # Update prices for each coin
        for coin in coins:
            # Use 1hour timeframe as primary price source
            candles = candles_by_coin.get(f"{coin}_1hour", [])
            matching = [c for c in candles if c["time"] == ts]

            if matching:
                price_file = f"replay_data/{coin}_current_price.txt"
                with open(price_file, "w") as f:
                    f.write(str(matching[0]["close"]))

        # Update control status
        progress = ((idx + 1) / len(timeline)) * 100
        control = {
            "status": "running",
            "current_ts": ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "progress_pct": round(progress, 2),
            "candles_processed": idx + 1,
            "total_candles": len(timeline)
        }
        _atomic_write_json("replay_data/replay_control.json", control)

        # Progress indicator
        if idx % 100 == 0:
            print(f"Progress: {progress:.1f}% ({idx+1}/{len(timeline)})")

        # Sleep to control replay speed (scaled by speed multiplier)
        # Assuming 1hour candles, real-time = 3600s between candles
        time.sleep(3600 / speed / 100)  # Scaled down for faster replay

    # Mark as completed
    control["status"] = "completed"
    _atomic_write_json("replay_data/replay_control.json", control)
    print(f"\n{'='*60}")
    print("Replay completed!")
    print(f"{'='*60}\n")
```

#### Task 3.2: Implement Subprocess Management

**File:** `/home/user/power-trader-ai/pt_replay.py` (lines 500-700)

```python
import subprocess
import signal
import sys

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
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("replay_data", exist_ok=True)

    # Convert dates to timestamps
    start_ts = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
    end_ts = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))

    # Save config
    config = {
        "run_id": os.path.basename(output_dir),
        "start_date": start_date,
        "end_date": end_date,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "coins": coins,
        "replay_speed": speed,
        "created_at": time.time()
    }
    _atomic_write_json(os.path.join(output_dir, "config.json"), config)

    # Launch thinker subprocess
    print("Launching pt_thinker.py in replay mode...")
    thinker_proc = subprocess.Popen(
        ["python", "pt_thinker.py", "--replay", "--replay-cache-dir", "backtest_cache"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Launch trader subprocess
    print("Launching pt_trader.py in replay mode...")
    trader_proc = subprocess.Popen(
        ["python", "pt_trader.py", "--replay", "--replay-output-dir", output_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Give subprocesses time to initialize
    time.sleep(2)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutdown signal received. Stopping replay...")
        thinker_proc.terminate()
        trader_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run time progression (blocking)
    try:
        replay_time_progression(start_ts, end_ts, coins, speed)
    except Exception as e:
        print(f"ERROR during replay: {e}")
    finally:
        # Graceful shutdown
        print("Stopping subprocesses...")
        thinker_proc.terminate()
        trader_proc.terminate()

        # Wait for termination
        thinker_proc.wait(timeout=5)
        trader_proc.wait(timeout=5)

        print("Subprocesses stopped.")
```

**Reference:** Subprocess management based on `/home/user/power-trader-ai/pt_hub.py:1515-1675`

#### Task 3.3: Add CLI and Main Entry Point

**File:** `/home/user/power-trader-ai/pt_replay.py` (lines 700-900)

```python
def main():
    """Main entry point for backtesting."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description='PowerTrader AI Backtesting')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--coins', default='BTC,ETH', help='Comma-separated coin list')
    parser.add_argument('--output-dir', help='Output directory (default: auto-generated)')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed (default: 1.0)')
    parser.add_argument('--warm-cache', action='store_true', help='Only fetch/cache data')
    parser.add_argument('--cache-dir', default='backtest_cache', help='Cache directory')

    args = parser.parse_args()

    # Parse coins
    coins = [c.strip() for c in args.coins.split(',')]

    # Generate output directory if not specified
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f"backtest_results/backtest_{timestamp}"

    print(f"\n{'='*60}")
    print("PowerTrader AI - Backtesting Mode")
    print(f"{'='*60}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Coins: {', '.join(coins)}")
    print(f"Output: {output_dir}")
    print(f"Speed: {args.speed}x")
    print(f"{'='*60}\n")

    # Warm cache (fetch historical data)
    start_ts = int(time.mktime(time.strptime(args.start_date, "%Y-%m-%d")))
    end_ts = int(time.mktime(time.strptime(args.end_date, "%Y-%m-%d")))

    print("Warming cache (fetching historical data)...")
    for coin in coins:
        for tf in ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']:
            symbol = f"{coin}-USDT"
            fetch_historical_klines(symbol, tf, start_ts, end_ts)

    if args.warm_cache:
        print("\nCache warming complete. Exiting (--warm-cache mode).")
        return

    # Run backtest
    run_backtest(args.start_date, args.end_date, coins, output_dir, args.speed)

    print(f"\nBacktest complete! Results saved to: {output_dir}")
    print(f"Run analytics: python pt_analyze.py {output_dir}")

if __name__ == "__main__":
    main()
```

---

### Phase 4: Analytics Engine (Days 10-12)

#### Task 4.1: Create Performance Metrics Calculator

**File:** Create `/home/user/power-trader-ai/pt_analyze.py`

```python
import json
import os
import sys
from datetime import datetime
import math

def load_trade_history(backtest_dir):
    """Load trade_history.jsonl from backtest results."""
    trade_file = os.path.join(backtest_dir, "trade_history.jsonl")

    if not os.path.exists(trade_file):
        print(f"ERROR: No trade history found at {trade_file}")
        return []

    trades = []
    with open(trade_file, "r") as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    return trades

def calculate_total_return(trades):
    """Calculate total return % from trades."""
    total_profit = sum(trade.get("realized_profit_usd", 0) for trade in trades if trade["side"] == "sell")

    # Estimate initial capital from first buy
    first_buy = next((t for t in trades if t["side"] == "buy"), None)
    if not first_buy:
        return 0.0

    initial_capital = first_buy["price"] * first_buy["qty"]

    if initial_capital == 0:
        return 0.0

    return (total_profit / initial_capital) * 100

def calculate_sharpe_ratio(trades):
    """
    Calculate Sharpe ratio from trade returns.
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    Assuming risk-free rate = 0 for simplicity.
    """
    returns = [trade["pnl_pct"] for trade in trades if trade["side"] == "sell" and "pnl_pct" in trade]

    if not returns or len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    # Annualize (assuming daily trades)
    sharpe = (mean_return / std_dev) * math.sqrt(252)
    return sharpe

def calculate_max_drawdown(trades):
    """
    Calculate maximum drawdown % from cumulative PnL.
    """
    cumulative_pnl = 0
    peak = 0
    max_dd = 0

    for trade in trades:
        if trade["side"] == "sell":
            cumulative_pnl += trade.get("realized_profit_usd", 0)

            if cumulative_pnl > peak:
                peak = cumulative_pnl

            drawdown = peak - cumulative_pnl
            if drawdown > max_dd:
                max_dd = drawdown

    if peak == 0:
        return 0.0

    return (max_dd / peak) * 100

def calculate_win_rate(trades):
    """Calculate win rate % (profitable trades / total trades)."""
    sell_trades = [t for t in trades if t["side"] == "sell"]

    if not sell_trades:
        return 0.0

    winning_trades = sum(1 for t in sell_trades if t.get("pnl_pct", 0) > 0)
    return (winning_trades / len(sell_trades)) * 100

def calculate_profit_factor(trades):
    """
    Calculate profit factor (gross profit / gross loss).
    Values > 1.0 indicate profitable strategy.
    """
    gross_profit = sum(t.get("realized_profit_usd", 0) for t in trades if t["side"] == "sell" and t.get("realized_profit_usd", 0) > 0)
    gross_loss = abs(sum(t.get("realized_profit_usd", 0) for t in trades if t["side"] == "sell" and t.get("realized_profit_usd", 0) < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss

def calculate_avg_trade_duration(trades):
    """Calculate average time between buy and sell."""
    durations = []
    open_positions = {}  # symbol -> buy timestamp

    for trade in trades:
        symbol = trade["symbol"]

        if trade["side"] == "buy":
            open_positions[symbol] = trade["ts"]
        elif trade["side"] == "sell" and symbol in open_positions:
            duration = trade["ts"] - open_positions[symbol]
            durations.append(duration)
            del open_positions[symbol]

    if not durations:
        return 0.0

    avg_seconds = sum(durations) / len(durations)
    return avg_seconds / 3600  # Convert to hours

def generate_analytics_report(backtest_dir):
    """Generate comprehensive analytics report."""
    print(f"\nAnalyzing backtest: {backtest_dir}")

    # Load data
    trades = load_trade_history(backtest_dir)

    if not trades:
        print("No trades found. Cannot generate report.")
        return

    # Load config
    config_file = os.path.join(backtest_dir, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Calculate metrics
    total_return = calculate_total_return(trades)
    sharpe_ratio = calculate_sharpe_ratio(trades)
    max_drawdown = calculate_max_drawdown(trades)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    avg_duration = calculate_avg_trade_duration(trades)

    total_trades = len([t for t in trades if t["side"] == "sell"])

    # Print summary
    print(f"\n{'='*60}")
    print("BACKTEST ANALYTICS SUMMARY")
    print(f"{'='*60}")
    print(f"Period: {config['start_date']} to {config['end_date']}")
    print(f"Coins: {', '.join(config['coins'])}")
    print(f"Total Trades: {total_trades}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Trade Duration: {avg_duration:.1f} hours")
    print(f"{'='*60}\n")

    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PowerTrader AI - Backtest Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }}
        h1 {{
            color: #00ff00;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #00aaff;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: #2a2a2a;
        }}
        th, td {{
            border: 1px solid #444;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #333;
            color: #00ff00;
        }}
        .positive {{ color: #00ff00; }}
        .negative {{ color: #ff4444; }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>PowerTrader AI - Backtest Report</h1>

    <h2>Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Start Date</td><td>{config['start_date']}</td></tr>
        <tr><td>End Date</td><td>{config['end_date']}</td></tr>
        <tr><td>Coins Traded</td><td>{', '.join(config['coins'])}</td></tr>
        <tr><td>Replay Speed</td><td>{config.get('replay_speed', 1.0)}x</td></tr>
        <tr><td>Total Trades</td><td>{total_trades}</td></tr>
    </table>

    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr>
            <td>Total Return</td>
            <td class="{'positive' if total_return > 0 else 'negative'} metric-value">
                {total_return:+.2f}%
            </td>
        </tr>
        <tr><td>Sharpe Ratio</td><td class="metric-value">{sharpe_ratio:.2f}</td></tr>
        <tr><td>Maximum Drawdown</td><td class="negative metric-value">{max_drawdown:.2f}%</td></tr>
        <tr><td>Win Rate</td><td class="metric-value">{win_rate:.1f}%</td></tr>
        <tr><td>Profit Factor</td><td class="metric-value">{profit_factor:.2f}</td></tr>
        <tr><td>Avg Trade Duration</td><td>{avg_duration:.1f} hours</td></tr>
    </table>

    <h2>Trade Log (Last 20 Trades)</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Price</th>
            <th>PnL %</th>
            <th>Realized Profit</th>
        </tr>
"""

    # Add last 20 trades
    for trade in trades[-20:]:
        ts_str = datetime.fromtimestamp(trade["ts"]).strftime("%Y-%m-%d %H:%M")
        pnl_pct = trade.get("pnl_pct", 0)
        realized_profit = trade.get("realized_profit_usd", 0)

        pnl_class = "positive" if pnl_pct > 0 else "negative"

        html += f"""
        <tr>
            <td>{ts_str}</td>
            <td>{trade['symbol']}</td>
            <td>{trade['side'].upper()}</td>
            <td>{trade['qty']:.6f}</td>
            <td>${trade['price']:.2f}</td>
            <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            <td class="{pnl_class}">${realized_profit:+.2f}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Interpretation Guide</h2>
    <ul>
        <li><strong>Sharpe Ratio:</strong> Measures risk-adjusted returns. Values > 1.0 are good, > 2.0 are excellent.</li>
        <li><strong>Max Drawdown:</strong> Largest peak-to-trough decline. Lower is better.</li>
        <li><strong>Win Rate:</strong> Percentage of profitable trades. 50%+ indicates edge.</li>
        <li><strong>Profit Factor:</strong> Gross profit / gross loss. Values > 1.5 are strong.</li>
    </ul>

    <p><em>Report generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
</body>
</html>
"""

    # Write report
    report_path = os.path.join(backtest_dir, "analytics_report.html")
    with open(report_path, "w") as f:
        f.write(html)

    print(f"HTML report generated: {report_path}")
    print(f"Open in browser: file://{os.path.abspath(report_path)}")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python pt_analyze.py <backtest_results_directory>")
        print("Example: python pt_analyze.py backtest_results/backtest_2025-01-15_143022")
        sys.exit(1)

    backtest_dir = sys.argv[1]

    if not os.path.exists(backtest_dir):
        print(f"ERROR: Directory not found: {backtest_dir}")
        sys.exit(1)

    generate_analytics_report(backtest_dir)

if __name__ == "__main__":
    main()
```

**Reference:** Trade history format from `/home/user/power-trader-ai/pt_trader.py:260-300`

---

### Phase 5: Testing & Documentation (Days 13-14)

#### Task 5.1: End-to-End Integration Test

```bash
# Step 1: Warm cache
python pt_replay.py --warm-cache \
  --start-date 2024-01-01 \
  --end-date 2024-01-05 \
  --coins BTC

# Step 2: Run backtest
python pt_replay.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-05 \
  --coins BTC \
  --output-dir backtest_results/integration_test \
  --speed 10.0

# Step 3: Generate analytics
python pt_analyze.py backtest_results/integration_test

# Step 4: Validate output
ls -lh backtest_results/integration_test/
# Expected files:
# - config.json
# - trade_history.jsonl
# - analytics_report.html
```

#### Task 5.2: Update Documentation

**File:** `/home/user/power-trader-ai/CLAUDE.md`

Add new section after "## 2. Core Commands":

```markdown
### Backtesting
- **Warm Cache:** `python pt_replay.py --warm-cache --start-date 2024-01-01 --end-date 2024-02-01 --coins BTC,ETH`
- **Run Backtest:** `python pt_replay.py --start-date 2024-01-01 --end-date 2024-02-01 --coins BTC,ETH`
- **Analyze Results:** `python pt_analyze.py backtest_results/latest`
- **View Report:** Open `backtest_results/latest/analytics_report.html` in browser

**Tips:**
- Use `--speed 10.0` for faster replay (10x real-time)
- Run `--warm-cache` first to download historical data
- Compare different DCA configurations by running multiple backtests
```

---

## Testing & Validation

### Unit Tests

Create `/home/user/power-trader-ai/tests/test_backtest.py`:

```python
import pytest
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pt_replay import fetch_historical_klines, _atomic_write_json
from pt_analyze import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate

def test_atomic_write():
    """Test atomic write pattern."""
    test_file = "test_atomic.json"
    data = {"test": "value"}

    _atomic_write_json(test_file, data)

    # Verify file exists and contains correct data
    assert os.path.exists(test_file)
    with open(test_file, "r") as f:
        loaded = json.load(f)
    assert loaded == data

    # Cleanup
    os.remove(test_file)

def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation."""
    trades = [
        {"side": "sell", "pnl_pct": 5.0},
        {"side": "sell", "pnl_pct": -2.0},
        {"side": "sell", "pnl_pct": 3.0},
        {"side": "sell", "pnl_pct": 1.0}
    ]

    sharpe = calculate_sharpe_ratio(trades)
    assert sharpe > 0  # Positive returns should yield positive Sharpe

def test_win_rate_calculation():
    """Test win rate calculation."""
    trades = [
        {"side": "sell", "pnl_pct": 5.0},
        {"side": "sell", "pnl_pct": -2.0},
        {"side": "sell", "pnl_pct": 3.0},
        {"side": "sell", "pnl_pct": -1.0}
    ]

    win_rate = calculate_win_rate(trades)
    assert win_rate == 50.0  # 2 wins out of 4 trades

def test_max_drawdown_calculation():
    """Test max drawdown calculation."""
    trades = [
        {"side": "sell", "realized_profit_usd": 100},
        {"side": "sell", "realized_profit_usd": -50},
        {"side": "sell", "realized_profit_usd": -30},
        {"side": "sell", "realized_profit_usd": 200}
    ]

    max_dd = calculate_max_drawdown(trades)
    assert max_dd > 0  # Should have some drawdown

if __name__ == "__main__":
    pytest.main([__file__])
```

Run tests:
```bash
pip install pytest
pytest tests/test_backtest.py -v
```

### Edge Cases to Handle

1. **Weekend/Holiday Gaps:**
   - KuCoin may have no candles during low-liquidity periods
   - Solution: Skip timestamps with no data, log gaps in replay_control.json

2. **API Rate Limits:**
   - KuCoin limits: 10 requests/second for public endpoints
   - Solution: Add 0.1s delay between fetch requests, retry with exponential backoff

3. **Missing Neural Training Files:**
   - Backtest may fail if `memories_{timeframe}.txt` is missing
   - Solution: Check for required files before starting, provide clear error message

4. **Coin Added Mid-Backtest:**
   - User config changes during replay
   - Solution: Freeze coin list at backtest start (from config.json)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|---------|-----------|
| **Breaking live trading logic** | Low | High | Use feature flags (`--replay`), extensive testing, no refactoring of core logic |
| **KuCoin API rate limits** | Medium | Medium | Add delays, exponential backoff, cache aggressively |
| **Replay diverges from live** | Medium | High | Unit test mock execution, compare replay vs. live on same period |
| **Large data cache size** | Low | Low | Compress JSON, add TTL expiry, document cleanup process |
| **Incomplete neural training** | Medium | Medium | Check for training files before start, document prerequisites |

---

## Success Metrics

After implementation, you should be able to:

1. ✅ Run a 6-month backtest on BTC in under 10 minutes
2. ✅ Generate a report showing Sharpe ratio, max drawdown, and win rate
3. ✅ Compare two different DCA level configurations side-by-side
4. ✅ Identify specific dates when strategy failed and debug why
5. ✅ Validate that neural signals would have predicted Jan 2024 market movements

### Acceptance Criteria

- [ ] Successfully fetch and cache 1 month of historical data for BTC
- [ ] Run end-to-end backtest without errors
- [ ] Generate analytics report with all metrics calculated
- [ ] Replay mode produces trade history in same format as live trading
- [ ] Live trading continues to work unchanged (no regressions)

---

## Appendix: Code References

### Key Files Modified

| Component | File | Lines | Changes |
|-----------|------|-------|---------|
| Signal Generation | `pt_thinker.py` | 556, 1095-1115 | Add `--replay` flag, read cached data |
| Order Execution | `pt_trader.py` | 692-734, 737-852 | Add `--replay` flag, mock orders |
| Trading Logic | `pt_trader.py` | 20-27, 1420-1435 | Redirect output directory |

### Key Files Created

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `pt_replay.py` | Replay orchestrator | ~900 |
| `pt_analyze.py` | Analytics generator | ~400 |

### Existing Code References

- **DCA Logic:** `/home/user/power-trader-ai/pt_trader.py:187, 1197-1273`
- **Signal Generation:** `/home/user/power-trader-ai/pt_thinker.py:477-1091`
- **Data Fetching:** `/home/user/power-trader-ai/pt_trainer.py:403-441`
- **Trade Logging:** `/home/user/power-trader-ai/pt_trader.py:260-300`
- **Subprocess Management:** `/home/user/power-trader-ai/pt_hub.py:1515-1675`
- **Atomic Writes:** `/home/user/power-trader-ai/CLAUDE.md:42-47`

---

## Implementation Checklist

### Phase 1: Data Caching ✅
- [ ] Create `backtest_cache/`, `replay_data/`, `backtest_results/` directories
- [ ] Implement `pt_replay.py` with `fetch_historical_klines()`
- [ ] Implement `_atomic_write_json()` helper
- [ ] Implement cache index management
- [ ] Test cache fetch with 1 week of BTC data

### Phase 2: Replay Mode ✅
- [ ] Add `--replay` flag to `pt_thinker.py`
- [ ] Modify KuCoin API calls in `pt_thinker.py:556`
- [ ] Add helper function `_read_cached_kline_for_replay()`
- [ ] Add `--replay` flag to `pt_trader.py`
- [ ] Modify price fetching in `pt_trader.py:692-734`
- [ ] Implement mock order execution in `pt_trader.py:737-852`
- [ ] Redirect output directory in `pt_trader.py:20-27`

### Phase 3: Orchestration ✅
- [ ] Implement `replay_time_progression()` in `pt_replay.py`
- [ ] Implement `run_backtest()` subprocess management
- [ ] Add CLI argument parsing
- [ ] Add signal handler for graceful shutdown
- [ ] Test subprocess communication

### Phase 4: Analytics ✅
- [ ] Create `pt_analyze.py`
- [ ] Implement `calculate_sharpe_ratio()`
- [ ] Implement `calculate_max_drawdown()`
- [ ] Implement `calculate_win_rate()`
- [ ] Implement `calculate_profit_factor()`
- [ ] Generate HTML report template
- [ ] Add trade log table

### Phase 5: Testing ✅
- [ ] Create `tests/test_backtest.py`
- [ ] Write unit tests for metrics calculations
- [ ] Run end-to-end test with 5-day backtest
- [ ] Validate trade count matches expected
- [ ] Verify metrics calculation accuracy
- [ ] Update `CLAUDE.md` documentation
- [ ] Create usage examples

---

## Next Steps After Implementation

### Short-Term Enhancements (Optional)

1. **Parameter Optimization:**
   - Add `--grid-search` mode to test DCA level variations
   - Output comparison table of all configurations

2. **Equity Curve Visualization:**
   - Use matplotlib to generate account value chart
   - Save as `equity_curve.png` in backtest results

3. **Multi-Strategy Testing:**
   - Support A/B testing (e.g., with/without neural signals)
   - Compare performance metrics side-by-side

### Long-Term Upgrades (Future Proposals)

1. **Walk-Forward Analysis:**
   - Train on period N, test on period N+1
   - Validate out-of-sample performance

2. **Monte Carlo Simulation:**
   - Randomize trade order to test robustness
   - Generate confidence intervals for metrics

3. **Automated Parameter Optimization:**
   - Implement Bayesian optimization for DCA levels
   - Find optimal threshold combinations

---

**Document Version:** 1.0
**Last Updated:** 2025-12-30
**Status:** Ready for Implementation
