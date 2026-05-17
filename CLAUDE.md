# PowerTrader AI

Multi-process algorithmic crypto trading. Memory-based pattern matcher (not a neural net) generates DCA signals (0–7) from OHLCV history. KuCoin for data, Robinhood Crypto for execution, file-based IPC (optional Redis via `REDIS_URL`).

## Critical Gotchas

1. **`r_key.txt` / `r_secret.txt` must exist before importing `pt_trader.py`** — module-level `SystemExit(1)`. Tests that import trader require these files or use `--paper` flag.
2. **Indentation split** — `pt_thinker.py`, `pt_trader.py`, `pt_trainer.py` use **hard tabs**; every other `.py` file uses **4 spaces**. Mixing causes `TabError`. Match the file you're editing.
3. **Atomic writes mandatory** — never write IPC or state files directly. Always `.tmp` → `os.replace()`. Details: `.claude/rules/ipc.md`.
4. **KuCoin candle index is non-standard**: `[ts, open, close, high, low, vol, turnover]` — index 2 = close, 3 = high, 4 = low.
5. **`os.chdir()` in `pt_thinker.py:init_coin()`** — always restore `os.chdir(BASE_DIR)` after. Never add threading to thinker.
6. **`optimizer_config.json` at project root** — overrides trader params when present; delete to restore defaults.

## Architecture

```
Hub (Tkinter) spawns → thinker ─[signals]→ trader ─[hub_data/]→ Hub
                      → trainer ─[memories/]→ thinker
Replay: pt_replay.py → backtest_cache/ → subprocesses → backtest_results/ → pt_analyze.py
IPC: file-based default | Redis pub/sub opt-in (REDIS_URL env var)
```

## Key Scripts

| File | Role | Indent |
|---|---|---|
| `pt_hub.py` | GUI orchestrator | 4 sp |
| `pt_thinker.py` | Pattern matcher, writes DCA signals | tabs |
| `pt_trader.py` | Order executor, reads signals | tabs |
| `pt_trainer.py` | Memory model builder | tabs |
| `pt_replay.py` | Backtest orchestrator | 4 sp |
| `pt_analyze.py` | Metrics + HTML reports | 4 sp |
| `pt_ipc.py` | Redis/file IPC bridge | 4 sp |
| `pt_optimizer.py` | Strategy parameter search | 4 sp |
| `pt_dashboard.py` | Streamlit monitoring UI | 4 sp |

## Commands

```bash
python -m unittest discover -s tests   # run all tests
python pt_hub.py                       # start full system
python pt_trader.py --paper            # paper trade (no keys needed)
streamlit run pt_dashboard.py          # analytics dashboard
docker compose up                      # full containerised stack
```

## Persistent Rules

- **After every PR merges to `main`**: update `CHANGELOG.md` — new entry under `## Detailed Change History`, update `**Generated:**` date. See `.claude/rules/changelog.md` for the exact format.
- **Modular rules** live in `.claude/rules/` and are scoped by file glob — they inject automatically when you work on matching files.
