---
globs: ["Dockerfile*", "docker-compose*.yml", ".env*", ".dockerignore"]
---
# Docker & Infrastructure Rules

## Quick Start
```bash
cp .env.example .env                   # set POWERTRADER_COINS; leave Robinhood keys empty for paper mode
docker compose up dashboard            # Streamlit at http://localhost:8501 (no keys needed)
docker compose up                      # full paper-trading stack
docker compose --profile live up       # live trading (needs keys in .env)
```

## Secrets Policy
- NEVER bake `r_key.txt`, `r_secret.txt`, or `.env` into the image — all three are in `.dockerignore`.
- Pass via `env_file: .env` in `docker-compose.yml` or individual `environment:` keys.
- `ROBINHOOD_API_KEY` and `ROBINHOOD_PRIVATE_KEY_BASE64` must be in `.env` for live trading.

## Volume Mounts (Required for Persistence)
`hub_data/`, `backtest_cache/`, `backtest_results/`, `optimizer_results/`, and per-coin model dirs (`BTC/`, `ETH/`, etc.) must be volume-mounted. The `x-base` anchor in `docker-compose.yml` handles this for all services.

## Multi-Coin Selection
Set `POWERTRADER_COINS=BTC,ETH,DOGE` in `.env`. This overrides `gui_settings.json` in both thinker and trader. Each coin needs trained model files in its subdirectory before trading.

## Redis IPC
`REDIS_URL=redis://redis:6379` enables sub-100ms signal latency via `pt_ipc.py`.
Omit `REDIS_URL` entirely for pure file-based IPC (default, zero extra runtime deps).
The `redis` service in `docker-compose.yml` uses `appendonly yes` for durability.
