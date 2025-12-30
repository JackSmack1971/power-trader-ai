# PowerTrader AI Backtesting Guide

**Version:** 1.0
**Last Updated:** 2025-12-30
**Status:** Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [CLI Reference](#cli-reference)
4. [Configuration Options](#configuration-options)
5. [Understanding Reports](#understanding-reports)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Quick Start

Get your first backtest running in 5 minutes:

### Step 1: Warm the Cache

Pre-fetch historical data from KuCoin:

```bash
python pt_replay.py --warm-cache \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC
```

This downloads and caches 1 month of BTC price data.

### Step 2: Run the Backtest

Execute the backtest at 10x speed:

```bash
python pt_replay.py --backtest \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC \
  --speed 10.0
```

The backtest will:
- Launch `pt_thinker.py` and `pt_trader.py` in replay mode
- Simulate trading with realistic slippage and fees
- Generate results in `backtest_results/backtest_YYYY-MM-DD_HHMMSS/`

### Step 3: Analyze Results

Generate analytics report:

```bash
python pt_analyze.py backtest_results/backtest_2024-12-30_120000
```

Open `analytics_report.html` in your browser to view:
- Total return and Sharpe ratio
- Max drawdown (including unrealized losses)
- Sortino and Calmar ratios
- Buy-and-hold benchmark comparison

---

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Internet connection for data fetching

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `kucoin-python` - KuCoin API client
- `numpy` - Numerical computing
- `psutil` - System monitoring (for performance tests)

---

## CLI Reference

### pt_replay.py

Main backtesting orchestrator.

#### Cache Warming

```bash
python pt_replay.py --warm-cache \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --coins COIN1,COIN2,... \
  [--timeframes TF1,TF2,...] \
  [--cache-dir DIR]
```

**Arguments:**
- `--warm-cache`: Enable cache warming mode
- `--start-date`: Start date (format: YYYY-MM-DD)
- `--end-date`: End date (format: YYYY-MM-DD)
- `--coins`: Comma-separated coin list (e.g., BTC,ETH,DOGE)
- `--timeframes`: Timeframes to cache (default: 1hour,2hour,4hour,8hour,12hour,1day,1week)
- `--cache-dir`: Cache directory (default: backtest_cache)

**Example:**
```bash
# Cache 6 months of data for multiple coins
python pt_replay.py --warm-cache \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --coins BTC,ETH,DOGE
```

#### Running Backtests

```bash
python pt_replay.py --backtest \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --coins COIN1,COIN2,... \
  [--speed MULTIPLIER] \
  [--output-dir DIR] \
  [--cache-dir DIR] \
  [--no-walk-forward] \
  [--retrain-interval DAYS]
```

**Arguments:**
- `--backtest`: Enable backtest mode
- `--speed`: Replay speed multiplier (default: 10.0)
  - `1.0` = real-time
  - `10.0` = 10x faster (recommended)
  - `100.0` = maximum speed (may skip synchronization)
- `--output-dir`: Output directory (default: auto-generated)
- `--no-walk-forward`: Disable walk-forward validation (faster, but less realistic)
- `--retrain-interval`: Days between model retraining (default: 7)

**Example:**
```bash
# Run 3-month backtest at 10x speed
python pt_replay.py --backtest \
  --start-date 2024-01-01 \
  --end-date 2024-04-01 \
  --coins BTC,ETH \
  --speed 10.0
```

#### Running Tests

```bash
# Run internal unit tests
python pt_replay.py --test
```

### pt_analyze.py

Analytics and report generation.

```bash
python pt_analyze.py BACKTEST_DIR [--output FILE] [--show-regime]
```

**Arguments:**
- `BACKTEST_DIR`: Path to backtest results directory
- `--output, -o`: Output HTML file path (optional)
- `--show-regime`: Include market regime analysis (experimental)

**Example:**
```bash
python pt_analyze.py backtest_results/backtest_2024-12-30_120000
```

---

## Configuration Options

### Execution Model

Edit `backtest_config.json` in the output directory:

```json
{
  "execution_model": {
    "slippage_bps": 5,           // Base slippage (0.05%)
    "fee_bps": 20,                // Transaction fees (0.20%)
    "max_volume_pct": 1.0,        // Max order as % of volume
    "latency_ms": [50, 500],      // Latency range
    "partial_fill_threshold": 0.01
  }
}
```

**Parameters:**
- **slippage_bps**: Base slippage in basis points
  - Typical: 5-10 bps for liquid markets
  - Adjusted automatically based on volatility
- **fee_bps**: Transaction fees (maker/taker combined)
  - Robinhood Crypto: ~20 bps (0.20%)
  - Adjust based on your exchange
- **max_volume_pct**: Maximum order size as % of candle volume
  - Prevents unrealistic fills on low liquidity
  - Typical: 0.5-2.0%
- **latency_ms**: Network latency range [min, max]
  - Simulates order submission delays
  - Adds 0-2 bps additional slippage

### Walk-Forward Validation

Enabled by default to prevent look-ahead bias.

```json
{
  "walk_forward_enabled": true,
  "retrain_interval_days": 7
}
```

**How it works:**
1. Model trains on data available up to current backtest time
2. Retraining happens every N days (default: 7)
3. Model never sees future data
4. Ensures realistic performance estimates

**Disable for faster testing:**
```bash
python pt_replay.py --backtest --no-walk-forward ...
```

⚠️ **Warning:** Disabling walk-forward may inflate performance metrics due to look-ahead bias.

---

## Understanding Reports

### Executive Summary

The top section shows key metrics at a glance:

- **Total Return**: Overall % gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
  - `< 1.0`: Poor
  - `1.0 - 2.0`: Good
  - `> 2.0`: Excellent
- **Max Drawdown**: Largest peak-to-trough decline
  - Includes unrealized losses (critical!)
- **Sortino Ratio**: Like Sharpe, but only penalizes downside volatility
- **Calmar Ratio**: Return / Max Drawdown

### Performance Metrics

**Annualized Return:**
- Normalizes return to 1 year for comparison
- Uses 365-day year (crypto trades 24/7, not just business days)

**Time Period:**
- Total duration of backtest
- Longer backtests = more reliable metrics

### Risk Metrics

**Sharpe Ratio (365-day):**
- Standard measure of risk-adjusted return
- PowerTrader uses 365-day annualization (not 252-day stock market convention)
- Formula: `(Mean Return - Risk Free Rate) / Std Dev of Returns * sqrt(365)`

**Max Drawdown:**
- Critical metric showing worst loss scenario
- Calculated from total equity (cash + unrealized positions)
- Traditional methods miss unrealized losses - ours doesn't!

**Drawdown Duration:**
- How long it took to reach trough from peak
- Longer durations = more psychological pain

### Buy-and-Hold Comparison

Shows how your strategy performed vs. simply buying and holding.

**Outperformance:**
- Positive = strategy beat buy-and-hold
- Negative = buy-and-hold was better

⚠️ **Note:** Current implementation uses placeholder buy-and-hold calculation. Full implementation coming soon.

---

## Advanced Usage

### Custom Output Directories

Organize multiple backtests:

```bash
python pt_replay.py --backtest \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --coins BTC \
  --output-dir my_experiments/test_001
```

### Multi-Coin Backtests

Test portfolio strategies:

```bash
python pt_replay.py --backtest \
  --start-date 2024-01-01 \
  --end-date 2024-03-01 \
  --coins BTC,ETH,DOGE,SOL \
  --speed 10.0
```

### Speed Optimization

**Fast Testing (100x speed):**
```bash
--speed 100.0 --no-walk-forward
```
Use for quick iterations, not final results.

**Accurate Simulation (1x speed):**
```bash
--speed 1.0
```
Real-time replay with full subprocess synchronization.

### Log Analysis

Backtest logs are stored in `backtest_events.jsonl` (JSON Lines format).

**Query with jq:**
```bash
# Count trade executions
cat backtest_events.jsonl | jq 'select(.event_type == "trade_execution")' | wc -l

# Average slippage
cat backtest_events.jsonl | jq -r 'select(.event_type == "trade_execution") | .data.slippage_bps' | awk '{sum+=$1; count++} END {print sum/count}'

# Find errors
cat backtest_events.jsonl | jq 'select(.event_type == "error")'
```

**Query with Python:**
```python
import json

events = []
with open('backtest_events.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Filter trade executions
trades = [e for e in events if e['event_type'] == 'trade_execution']

# Calculate average slippage
avg_slippage = sum(t['data']['slippage_bps'] for t in trades) / len(trades)
print(f"Average slippage: {avg_slippage:.2f} bps")
```

---

## Troubleshooting

### Issue: Cache Warming Fails

**Symptom:**
```
ERROR: Could not fetch data from KuCoin
```

**Solutions:**
1. Check internet connection
2. Verify KuCoin API is accessible: `curl https://api.kucoin.com/api/v1/status`
3. Rate limiting: Add delays between requests (automatic, but may need tuning)
4. Try smaller date ranges first

### Issue: Backtest Exits Prematurely

**Symptom:**
```
ERROR: pt_thinker.py exited prematurely
```

**Solutions:**
1. Check `pt_thinker.py` output in console
2. Verify neural model files exist (run `pt_trainer.py BTC` first)
3. Ensure cache data covers full date range
4. Check for missing dependencies

### Issue: Components Not Synchronized

**Symptom:**
```
TimeoutError: Components did not respond for sequence N
```

**Solutions:**
1. Reduce replay speed (`--speed 1.0`)
2. Increase timeout in `wait_for_components()` (edit `pt_replay.py`)
3. Check if subprocesses crashed (look for error messages)
4. Verify `replay_data/` directory is writable

### Issue: Report Shows Zero Returns

**Symptom:**
Analytics report shows 0% return despite trades executing.

**Solutions:**
1. Verify `trader_status.json` exists in `hub_data/`
2. Check that file contains equity curve data (JSONL format)
3. Ensure backtest ran to completion
4. Check for errors in `backtest_events.jsonl`

### Issue: Memory Usage Too High

**Symptom:**
Process uses >2GB RAM or system slows down.

**Solutions:**
1. Reduce date range (split into multiple backtests)
2. Reduce number of coins
3. Clear cache between runs: `rm -rf backtest_cache/*`
4. Increase system swap space

---

## FAQ

### Q: How long does a 6-month backtest take?

**A:** Approximately 5-10 minutes at 10x speed, depending on:
- Number of coins (1 coin = faster)
- Walk-forward enabled (adds ~2-5 minutes per retrain)
- System performance

Benchmark: 6-month BTC-only backtest should complete in under 10 minutes.

### Q: Can I backtest with live API credentials?

**A:** **No!** Backtesting runs in replay mode, which:
- Uses cached historical data (no live API calls)
- Simulates order execution (no real orders)
- Writes to isolated output directories

Your Robinhood credentials are never used during backtesting.

### Q: What's the difference between backtest and paper trading?

**A:**
- **Backtesting** (pt_replay.py): Historical simulation, fast, uses cached data
- **Paper Trading**: Real-time simulation, slow, uses live market data

Backtesting is for strategy validation. Paper trading is for pre-production testing.

### Q: How accurate is the slippage model?

**A:** Our model simulates:
- Base slippage: 5 bps (0.05%)
- Volatility adjustment: up to 2.5x multiplier
- Network latency: 0-2 bps additional
- Liquidity constraints: partial fills if order too large

This is conservative. Real slippage may be lower on highly liquid assets (BTC) or higher on low-liquidity altcoins.

**Calibration:** Compare backtest results to live trading results over 1-2 weeks. Adjust `slippage_bps` in config if needed.

### Q: Why is walk-forward validation important?

**A:** Without walk-forward validation, the neural model can "see into the future":
- Traditional backtest: Train on ALL data (2024-01 to 2024-12), then test
- Walk-forward: Train only on data available at each point in time

**Example:**
- At backtest time 2024-01-15, model only trained on data before 2024-01-15
- At backtest time 2024-02-15, model retrained with data up to 2024-02-15

This prevents **look-ahead bias**, which inflates performance metrics.

### Q: Can I modify the trading strategy?

**A:** The backtesting system replays existing signals from `pt_thinker.py` and `pt_trader.py`. To test different strategies:

1. Modify `pt_thinker.py` signal logic
2. Run backtest with new signals
3. Compare results

The execution engine (`pt_trader.py`) remains realistic regardless of strategy.

### Q: What timeframes are cached?

**A:** By default:
- 1hour, 2hour, 4hour, 8hour, 12hour, 1day, 1week

These match the timeframes used by `pt_thinker.py` for multi-timeframe analysis.

You can customize with `--timeframes`:
```bash
python pt_replay.py --warm-cache --timeframes 1hour,1day ...
```

### Q: How do I run tests?

**A:**
```bash
# Unit tests
python pt_replay.py --test

# Walk-forward validation tests
python tests/test_walk_forward.py

# Integration tests
python tests/test_backtest_integration.py

# End-to-end smoke test
python tests/test_e2e_smoke.py

# Performance benchmarks
python tests/test_performance.py
```

### Q: Can I backtest multiple coins simultaneously?

**A:** Yes! Specify multiple coins:
```bash
python pt_replay.py --backtest --coins BTC,ETH,DOGE ...
```

The system:
- Loads candles for all coins
- Updates prices atomically for all coins each tick
- Allows `pt_trader.py` to manage portfolio allocation

### Q: Where are results stored?

**A:** Default location: `backtest_results/backtest_YYYY-MM-DD_HHMMSS/`

Contents:
- `config.json` - Backtest configuration
- `backtest_events.jsonl` - Structured event log
- `analytics_report.html` - Analytics report
- `hub_data/trader_status.json` - Equity curve snapshots
- `hub_data/trade_history.jsonl` - Individual trades (if available)

### Q: How do I compare two strategies?

**A:**
1. Run backtest for Strategy A: `--output-dir results/strategy_a`
2. Run backtest for Strategy B: `--output-dir results/strategy_b`
3. Generate reports for both:
   ```bash
   python pt_analyze.py results/strategy_a
   python pt_analyze.py results/strategy_b
   ```
4. Compare Sharpe ratios, max drawdowns, and returns

### Q: What if I find a bug?

**A:** Please report bugs at:
- GitHub: https://github.com/JackSmack1971/power-trader-ai/issues
- Include:
  - Steps to reproduce
  - Expected vs actual behavior
  - Backtest config (sanitize any credentials!)
  - Log files (`backtest_events.jsonl`)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     pt_replay.py (Orchestrator)              │
│  - Time progression                                          │
│  - Subprocess management                                     │
│  - Atomic state files                                        │
└────────────┬───────────────────────────┬─────────────────────┘
             │                           │
             ▼                           ▼
   ┌──────────────────┐        ┌──────────────────┐
   │  pt_thinker.py   │        │   pt_trader.py   │
   │  (--replay mode) │        │  (--replay mode) │
   │                  │        │                  │
   │  - Read cached   │        │  - Read prices   │
   │    candles       │        │    from state    │
   │  - Generate      │        │  - Execute       │
   │    signals       │        │    orders        │
   │  - Write to      │        │  - Write fills   │
   │    signal files  │        │  - Update equity │
   └──────────────────┘        └──────────────────┘
             │                           │
             └───────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │   replay_data/      │
              │  - backtest_state   │
              │  - sim_orders.jsonl │
              │  - sim_fills.jsonl  │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   pt_analyze.py     │
              │  - Build equity     │
              │  - Calculate metrics│
              │  - Generate report  │
              └─────────────────────┘
```

---

## Best Practices

1. **Always warm cache first** - Separates data fetching from backtest execution
2. **Start small** - Test with 1 week before running 6 months
3. **Enable walk-forward** - Unless you're just testing infrastructure
4. **Compare to buy-and-hold** - Ensures strategy adds value
5. **Verify max drawdown** - Make sure you can tolerate worst-case losses
6. **Run multiple backtests** - Test different time periods to avoid overfitting
7. **Check logs for errors** - `backtest_events.jsonl` contains all events
8. **Calibrate slippage** - Compare to live trading to tune realism

---

## Next Steps

After completing your first backtest:

1. **Analyze Results** - Review analytics report thoroughly
2. **Optimize Strategy** - Adjust parameters in `pt_thinker.py`
3. **Test Different Periods** - Bear markets, bull markets, sideways
4. **Multi-Coin Testing** - Diversification improves risk-adjusted returns
5. **Paper Trading** - Test in real-time before going live
6. **Live Trading** - Start small, monitor closely

---

## Version History

**v1.0 (2025-12-30)**
- Initial release
- Complete Phase 1-6 implementation
- Walk-forward validation
- Realistic execution engine
- Analytics & metrics
- Structured logging
- Comprehensive test suite

---

## Support

For questions and support:
- Documentation: `/docs/`
- Examples: Run `python pt_replay.py --help`
- Tests: `/tests/`
- Issues: GitHub Issues

Happy backtesting! 🚀
