---
globs: ["pt_replay.py", "pt_analyze.py", "pt_incremental_trainer.py", "pt_optimizer.py", "tests/**", "backtest_results/**", "backtest_cache/**"]
---
# Backtesting & Optimizer Rules

## Workflow
```bash
# 1. Warm cache (downloads KuCoin OHLCV, run once)
python pt_replay.py --warm-cache --start-date 2024-01-01 --end-date 2024-06-01 --coins BTC,ETH

# 2. Run replay backtest
python pt_replay.py --backtest --start-date 2024-01-01 --end-date 2024-06-01 \
    --coins BTC --speed 10.0 --output-dir backtest_results/my_run

# 3. Generate analytics report
python pt_analyze.py backtest_results/my_run   # → analytics_report.html

# 4. Strategy optimizer (overnight)
python pt_optimizer.py --start-date 2024-01-01 --end-date 2024-06-01 \
    --coins BTC --trials 50 --method diffevo --metric sharpe
# Apply best params: cp optimizer_results/<run>/best_config.json optimizer_config.json
```

## KuCoin Candle Index
`[ts, open, close, high, low, volume, turnover]` — index 2 = close, 3 = high, 4 = low (non-standard).
Replay normalises to named dicts; raw thinker/trainer code uses these integer indices directly.

## Walk-Forward Training
`python pt_incremental_trainer.py BTC --train-until <unix_ts>` — prevents look-ahead bias.
`trainer_last_training_time.txt` per coin; hub and thinker both reject models older than 14 days.

## State File Schema
`replay_data/backtest_state.json`: `{"sequence": int, "timestamp": int, "prices": {"BTC": {"close": f, "high": f, "low": f, "volume": f}}, "status": "ready"}`.
Written atomically by replay; read by thinker + trader subprocesses each tick.

## Performance Targets
6-month backtest: < 10 min at 10× speed, < 2 GB RAM, < 500 MB cache per coin.

## Test Commands
```bash
python -m unittest discover -s tests              # all tests
python -m unittest tests.test_backtest_integration
python -m unittest tests.test_e2e_smoke           # KuCoin is mocked
python -m unittest tests.test_walk_forward
python -m unittest tests.test_performance         # slow — 6-month benchmark
```
Tests that import `pt_trader.py` need `r_key.txt` / `r_secret.txt` present OR must mock them.
