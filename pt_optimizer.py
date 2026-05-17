#!/usr/bin/env python3
"""
PowerTrader AI — Strategy Optimizer

Searches over pt_trader.py trading parameters to maximise a back-test
performance metric (Sharpe by default).  Each trial writes an
optimizer_config.json that pt_trader.py hot-loads, runs a full replay
subprocess, then scores the result via pt_analyze metrics.

Usage examples
--------------
# Quick random search (20 trials, overnight)
python pt_optimizer.py \\
    --start-date 2024-01-01 --end-date 2024-06-01 \\
    --coins BTC --trials 20 --method random

# Differential-evolution search (~50 evaluations)
python pt_optimizer.py \\
    --start-date 2024-01-01 --end-date 2024-06-01 \\
    --coins BTC,ETH --trials 50 --method diffevo --metric sortino

# Full grid (enumerate all combinations — can be slow)
python pt_optimizer.py \\
    --start-date 2024-01-01 --end-date 2024-03-01 \\
    --coins BTC --method grid

Output
------
{output_dir}/
    optimizer_results.jsonl   — one JSON line per trial
    best_config.json          — best parameters for live use
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_CONFIG_PATH = os.path.join(BASE_DIR, "optimizer_config.json")

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
PARAM_SPACE: dict[str, list[Any]] = {
    "buy_threshold":           [2, 3, 4, 5],
    "pm_start_pct_no_dca":     [3.0, 5.0, 7.5, 10.0],
    "pm_start_pct_with_dca":   [1.5, 2.5, 4.0],
    "trailing_gap_pct":        [0.25, 0.5, 0.75, 1.0],
    "max_dca_buys_per_24h":    [1, 2, 3],
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _write_trial_config(params: dict) -> None:
    tmp = OPTIMIZER_CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    os.replace(tmp, OPTIMIZER_CONFIG_PATH)


def _remove_trial_config() -> None:
    try:
        os.remove(OPTIMIZER_CONFIG_PATH)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Metric scoring
# ---------------------------------------------------------------------------

def _score_trial(output_dir: str, metric: str) -> float:
    """Import pt_analyze helpers and compute the requested metric.
    Returns -999.0 on any failure."""
    try:
        sys.path.insert(0, BASE_DIR)
        from pt_analyze import (  # type: ignore[import]
            build_equity_curve,
            calculate_sharpe_ratio_corrected,
            calculate_sortino_ratio,
            calculate_max_drawdown_corrected,
            calculate_calmar_ratio,
        )

        hub_dir = os.path.join(output_dir, "hub_data")
        acct_history = os.path.join(hub_dir, "account_value_history.jsonl")
        trader_status = os.path.join(hub_dir, "trader_status.json")
        target = acct_history if os.path.isfile(acct_history) else trader_status
        if not os.path.isfile(target):
            return -999.0

        equity, timestamps = build_equity_curve(target)
        if len(equity) < 10:
            return -999.0

        if metric == "sharpe":
            return float(calculate_sharpe_ratio_corrected(equity, timestamps))

        returns = [
            (equity[i] - equity[i - 1]) / equity[i - 1]
            for i in range(1, len(equity))
            if equity[i - 1] != 0
        ]

        if metric == "sortino":
            return float(calculate_sortino_ratio(returns))

        if metric == "calmar":
            dd = calculate_max_drawdown_corrected(equity)
            years = (
                (timestamps[-1] - timestamps[0]) / (365.25 * 86400)
                if len(timestamps) > 1
                else 1.0
            )
            total_ret = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0
            return float(
                calculate_calmar_ratio(
                    total_ret, dd["max_drawdown_pct"] / 100.0, years
                )
            )
    except Exception as exc:
        print(f"    [score error] {exc}")

    return -999.0


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def _run_trial(
    params: dict,
    trial_idx: int,
    start_date: str,
    end_date: str,
    coins: list[str],
    speed: float,
    base_output_dir: str,
    cache_dir: str,
    metric: str,
    timeout: int,
) -> dict:
    trial_dir = os.path.join(base_output_dir, f"trial_{trial_idx:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    _write_trial_config(params)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "pt_replay.py"),
        "--backtest",
        "--start-date", start_date,
        "--end-date",   end_date,
        "--coins",      ",".join(coins),
        "--speed",      str(speed),
        "--output-dir", trial_dir,
        "--cache-dir",  cache_dir,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            print(f"    [warn] replay exited {proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"    [warn] trial {trial_idx} timed out after {timeout}s")
    elapsed = time.time() - t0

    score = _score_trial(trial_dir, metric)
    result = {
        "trial":   trial_idx,
        "params":  params,
        "score":   score,
        "metric":  metric,
        "elapsed": round(elapsed, 1),
        "dir":     trial_dir,
    }
    return result


# ---------------------------------------------------------------------------
# Parameter sampling strategies
# ---------------------------------------------------------------------------

def _grid_combos(space: dict) -> list[dict]:
    keys = list(space.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*[space[k] for k in keys])
    ]


def _random_samples(space: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [{k: rng.choice(v) for k, v in space.items()} for _ in range(n)]


def _diffevo_samples(
    space: dict,
    max_evals: int,
    seed: int,
    runner_fn,  # callable(params) -> float score
) -> tuple[dict, float]:
    """Run scipy differential_evolution over the integer-indexed param space."""
    try:
        from scipy.optimize import differential_evolution  # type: ignore[import]
    except ImportError:
        print("[optimizer] scipy not installed — falling back to random search.")
        samples = _random_samples(space, max_evals, seed)
        best_p, best_s = None, -999.0
        for p in samples:
            s = runner_fn(p)
            if s > best_s:
                best_s, best_p = s, p
        return best_p, best_s

    keys = list(space.keys())
    bounds = [(0, len(v) - 0.01) for v in space.values()]
    call_log: list[dict] = []

    def _obj(x: list) -> float:
        params = {k: space[k][int(xi)] for k, xi in zip(keys, x)}
        score = runner_fn(params)
        call_log.append({"params": params, "score": score})
        return -score  # minimise negative = maximise score

    pop_size = max(5, min(10, max_evals // 10))
    result = differential_evolution(
        _obj,
        bounds,
        maxiter=max_evals // pop_size,
        popsize=pop_size,
        seed=seed,
        tol=0.001,
        mutation=(0.5, 1.2),
        recombination=0.7,
        updating="immediate",
        workers=1,
    )
    best_params = {k: space[k][int(xi)] for k, xi in zip(keys, result.x)}
    return best_params, -result.fun


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _append_result(results_path: str, result: dict) -> None:
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def _write_best_config(output_dir: str, result: dict) -> str:
    path = os.path.join(output_dir, "best_config.json")
    data = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "metric":    result["metric"],
        "score":     result["score"],
        "params":    result["params"],
        "trial_dir": result.get("dir", ""),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PowerTrader AI — Strategy Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start-date",  required=True, help="Back-test start (YYYY-MM-DD)")
    p.add_argument("--end-date",    required=True, help="Back-test end   (YYYY-MM-DD)")
    p.add_argument("--coins",       default="BTC",  help="Comma-separated list, e.g. BTC,ETH")
    p.add_argument("--speed",       type=float, default=50.0,
                   help="Replay speed multiplier (higher = faster; default 50)")
    p.add_argument("--method",      choices=["grid", "random", "diffevo"], default="random",
                   help="Search strategy (default: random)")
    p.add_argument("--trials",      type=int, default=30,
                   help="Max number of trials (ignored for grid)")
    p.add_argument("--metric",      choices=["sharpe", "sortino", "calmar"], default="sharpe",
                   help="Optimisation objective (default: sharpe)")
    p.add_argument("--seed",        type=int, default=42, help="RNG seed")
    p.add_argument("--output-dir",  default=None,
                   help="Root directory for trial outputs "
                        "(default: optimizer_results/YYYYMMDD_HHMMSS)")
    p.add_argument("--cache-dir",   default=os.path.join(BASE_DIR, "backtest_cache"),
                   help="Path to the candle cache produced by --warm-cache")
    p.add_argument("--trial-timeout", type=int, default=3600,
                   help="Seconds before a single trial is killed (default 3600)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    output_dir = args.output_dir or os.path.join(
        BASE_DIR,
        "optimizer_results",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "optimizer_results.jsonl")

    print(
        f"\n{'='*60}\n"
        f"  PowerTrader AI — Strategy Optimizer\n"
        f"  method={args.method}  metric={args.metric}  "
        f"coins={','.join(coins)}\n"
        f"  {args.start_date} → {args.end_date}  speed={args.speed}×\n"
        f"  output → {output_dir}\n"
        f"{'='*60}\n"
    )

    best_result: dict = {}
    trial_counter = itertools.count(1)

    def _run(params: dict) -> float:
        idx = next(trial_counter)
        print(f"  trial {idx:3d}: {params}")
        result = _run_trial(
            params=params,
            trial_idx=idx,
            start_date=args.start_date,
            end_date=args.end_date,
            coins=coins,
            speed=args.speed,
            base_output_dir=output_dir,
            cache_dir=args.cache_dir,
            metric=args.metric,
            timeout=args.trial_timeout,
        )
        _append_result(results_path, result)
        score = result["score"]
        print(f"           → {args.metric}={score:.4f}  ({result['elapsed']:.0f}s)")
        if not best_result or score > best_result.get("score", -999.0):
            best_result.update(result)
            _write_best_config(output_dir, best_result)
            print(f"           ★ new best!")
        return score

    try:
        if args.method == "grid":
            combos = _grid_combos(PARAM_SPACE)
            print(f"  Grid search: {len(combos)} combinations\n")
            for params in combos:
                _run(params)

        elif args.method == "random":
            samples = _random_samples(PARAM_SPACE, args.trials, args.seed)
            print(f"  Random search: {args.trials} trials\n")
            for params in samples:
                _run(params)

        else:  # diffevo
            print(f"  Differential evolution: up to {args.trials} evaluations\n")
            best_params, best_score = _diffevo_samples(
                PARAM_SPACE, args.trials, args.seed, _run
            )

    except KeyboardInterrupt:
        print("\n[optimizer] interrupted — saving best found so far.")
    finally:
        _remove_trial_config()

    if best_result:
        best_path = _write_best_config(output_dir, best_result)
        print(
            f"\n{'='*60}\n"
            f"  Best {args.metric}: {best_result['score']:.4f}\n"
            f"  Params: {json.dumps(best_result['params'], indent=4)}\n"
            f"  Saved → {best_path}\n"
            f"{'='*60}\n"
        )
        print(
            "  To apply: copy best_config.json to optimizer_config.json\n"
            "  at the project root — pt_trader.py will pick it up automatically.\n"
        )
    else:
        print("\n[optimizer] no trials completed.")


if __name__ == "__main__":
    main()
