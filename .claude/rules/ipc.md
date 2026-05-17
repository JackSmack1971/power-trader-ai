---
globs: ["pt_ipc.py", "pt_trader.py", "pt_thinker.py", "pt_trainer.py", "hub_data/**"]
---
# IPC & File Write Rules

## Atomic Write Pattern (Mandatory)

Never write state files directly — partial writes on crash corrupt the model.

```python
# JSON
def _atomic_write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)

# Plain text (memory/weight files)
def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w+", encoding="utf-8") as f: f.write(text)
    os.replace(tmp, path)
```

## Signal Files

- Path: `{coin_dir}/long_dca_signal.txt` and `short_dca_signal.txt` — integer 0–7.
- BTC uses the project root as its coin dir; all other coins use `{root}/{COIN}/`.
- Memory patterns: `~`-delimited strings in `memories_{tf}.txt`.
- Memory weights: space-delimited floats in `memory_weights_{tf}.txt` (+ `_high_`, `_low_` variants).
- Seven timeframes: `1hour 2hour 4hour 8hour 12hour 1day 1week`.

## Redis IPC (pt_ipc.py)

Use the module-level singleton — never construct a new `IPCBridge` instance:

```python
from pt_ipc import ipc
level = ipc.read_signal("BTC", "long", base_dir=coin_folder)
ipc.write_signal("BTC", "long", value, base_dir=".")
```

Keys: `pt:signal:{COIN}:{long|short}` (last-value string + pub/sub channel).
Thinker dual-writes (file + Redis) so file-IPC readers are never broken.
Redis activates only when `REDIS_URL` env var is set; missing = silent file fallback.
