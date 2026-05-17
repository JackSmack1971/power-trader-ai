---
globs: ["pt_thinker.py", "pt_trader.py", "pt_trainer.py"]
---
# Trading Core Rules

## Indentation (Critical)
These three files use **hard tabs**. Never mix with spaces — `TabError` in Python 3.

## Memory Model
- Patterns: `~`-delimited strings in `memories_{tf}.txt`; split with `.split('~')`.
- Weights: space-delimited floats in `memory_weights_{tf}.txt`; split with `.split(' ')`.
- `_memory_cache` is keyed by **timeframe only** (not coin) — multi-coin runs share it; be careful with path assumptions when refactoring.
- Training freshness gate: `trainer_last_training_time.txt` per coin folder; > 14 days = stale, hub refuses to trade.

## Thinker Patterns
- `init_coin(sym)` calls `os.chdir(coin_folder)` — always restore `os.chdir(BASE_DIR)` immediately after. Never thread thinker while this is in use.
- Bare `except:` used throughout for resilience — preserve it in thinker/trainer; use specific types in new code.
- `PrintException()` = custom traceback printer (original code). New code uses `BacktestLogger` for structured JSON.
- `vprint()` = debug output gated by `VERBOSE` flag — do not add unconditional prints.

## Trader Patterns
- Entry gate: `long_signal >= _buy_threshold` (default 3). Overridable via `optimizer_config.json` key `"buy_threshold"`.
- DCA stages 0–3 map to neural levels 4–7; stages 4+ trigger on hardcoded % drops (−30, −40, −50, repeat).
- `_stop_levels` dict: `{symbol: stop_price}` — set on high-signal entry, cleared on any sell event.
- `_apply_risk_sizing(symbol, base_usd, signal_level)` applies ATR + Kelly multipliers — **both off by default**, enable via `optimizer_config.json`.
- Paper mode: `--paper` bypasses Robinhood auth entirely. Account state: `hub_data/paper_account.json`.
- `_apply_optimizer_config()` is called at `__init__` time — it reads `optimizer_config.json` and sets 14 instance params.
