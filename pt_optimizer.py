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
# Bayesian optimization (Gaussian-process surrogate via scipy)
# ---------------------------------------------------------------------------

def _bayesian_samples(
    space: dict,
    max_evals: int,
    seed: int,
    runner_fn,  # callable(params) -> float score
    n_initial: int = 5,
) -> tuple[dict, float]:
    """
    Simple Bayesian optimization using a Gaussian-process surrogate model.

    Falls back to random search if scipy/sklearn are not available.

    Strategy:
    1. Evaluate n_initial random points to seed the GP.
    2. For each remaining evaluation, fit a GP to observed (X, y) pairs,
       predict mean + std at candidate points, select the point with the
       highest Upper Confidence Bound (UCB = mean + kappa * std).
    3. Evaluate the selected point, update observations, repeat.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
        from sklearn.gaussian_process.kernels import Matern              # type: ignore
        _sklearn_available = True
    except ImportError:
        _sklearn_available = False

    if not _sklearn_available:
        print("[optimizer] scikit-learn not installed — falling back to random search for Bayesian method.")
        samples = _random_samples(space, max_evals, seed)
        best_p, best_s = None, -999.0
        for p in samples:
            s = runner_fn(p)
            if s > best_s:
                best_s, best_p = s, p
        return best_p or samples[0], best_s

    keys = list(space.keys())
    # Build a dense candidate grid (integer indices per dimension)
    import itertools as _it
    all_index_combos = list(_it.product(*[range(len(v)) for v in space.values()]))

    rng = random.Random(seed)
    observed_X: list[list[int]] = []
    observed_y: list[float] = []

    def _indices_to_params(idx_vec: list) -> dict:
        return {k: space[k][i] for k, i in zip(keys, idx_vec)}

    def _evaluate(idx_vec: list) -> float:
        params = _indices_to_params(idx_vec)
        return runner_fn(params)

    # Phase 1: random seed evaluations
    seed_indices = rng.sample(all_index_combos, min(n_initial, len(all_index_combos)))
    for idx_vec in seed_indices:
        s = _evaluate(list(idx_vec))
        observed_X.append(list(idx_vec))
        observed_y.append(s)

    remaining = max_evals - len(observed_X)
    kappa = 2.576  # UCB exploration parameter

    for _ in range(max(0, remaining)):
        X_obs = [[float(v) for v in row] for row in observed_X]
        y_obs = list(observed_y)

        # Scale y to [-1, 1] for GP stability
        y_arr = [v for v in y_obs if v > -999.0]
        y_min, y_max = (min(y_arr), max(y_arr)) if y_arr else (-1.0, 1.0)
        y_range = y_max - y_min if y_max != y_min else 1.0
        y_scaled = [(y - y_min) / y_range if y > -999.0 else -1.0 for y in y_obs]

        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, n_restarts_optimizer=2)
        try:
            gp.fit(X_obs, y_scaled)
        except Exception:
            # GP failed: pick random unexplored point
            unexplored = [c for c in all_index_combos if list(c) not in observed_X]
            if not unexplored:
                break
            next_idx = list(rng.choice(unexplored))
        else:
            # Evaluate UCB over all candidates
            candidates = [list(c) for c in all_index_combos if list(c) not in observed_X]
            if not candidates:
                break
            cand_X = [[float(v) for v in row] for row in candidates]
            mu, sigma = gp.predict(cand_X, return_std=True)
            ucb = mu + kappa * sigma
            next_idx = candidates[int(ucb.argmax())]

        s = _evaluate(next_idx)
        observed_X.append(next_idx)
        observed_y.append(s)

    if not observed_y:
        return _indices_to_params(list(all_index_combos[0])), -999.0

    best_i = int(max(range(len(observed_y)), key=lambda i: observed_y[i]))
    return _indices_to_params(observed_X[best_i]), observed_y[best_i]


# ---------------------------------------------------------------------------
# Optimizer results visualization (HTML report)
# ---------------------------------------------------------------------------

def _write_optimizer_report(output_dir: str, results: list[dict], metric: str) -> str:
    """Generate an HTML report visualizing all optimizer trial scores."""
    report_path = os.path.join(output_dir, "optimizer_report.html")

    valid = [r for r in results if r.get("score", -999.0) > -999.0]
    scores = [r["score"] for r in valid]
    if not scores:
        return report_path

    best = max(valid, key=lambda r: r["score"])

    # Inline SVG bar chart (no JS dependencies)
    bar_width = 8
    chart_height = 200
    max_score = max(scores) if max(scores) > 0 else 1.0
    min_score = min(scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    bars = []
    for i, r in enumerate(valid):
        s = r["score"]
        pct = max(0.0, (s - min_score) / score_range)
        bar_h = int(pct * chart_height)
        x = i * (bar_width + 1)
        color = "#00FF66" if s == max(scores) else "#3399FF"
        bars.append(f'<rect x="{x}" y="{chart_height - bar_h}" width="{bar_width}" height="{bar_h}" fill="{color}" />')

    svg_w = len(valid) * (bar_width + 1)
    svg_chart = f'<svg width="{svg_w}" height="{chart_height}" style="background:#0B1220">' + "".join(bars) + "</svg>"

    param_rows = ""
    for k, v in best.get("params", {}).items():
        param_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    trial_rows = ""
    for r in sorted(valid, key=lambda x: x.get("score", -999.0), reverse=True)[:20]:
        trial_rows += (
            f"<tr><td>{r['trial']}</td>"
            f"<td>{r['score']:.4f}</td>"
            f"<td>{r['elapsed']:.0f}s</td>"
            f"<td><small>{json.dumps(r['params'])}</small></td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>PowerTrader Optimizer Report</title>
<style>
body {{ background:#070B10; color:#C7D1DB; font-family:monospace; padding:20px; max-width:1200px; margin:0 auto; }}
h1,h2 {{ color:#00FF66; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
th {{ background:#0B1220; color:#00E5FF; padding:8px; text-align:left; }}
td {{ padding:6px 8px; border-bottom:1px solid #243044; }}
.best {{ color:#00FF66; font-weight:bold; }}
</style>
</head>
<body>
<h1>PowerTrader AI — Optimizer Report</h1>
<p>Metric: <b>{metric}</b> &nbsp;|&nbsp; Trials: <b>{len(valid)}</b> &nbsp;|&nbsp; Best score: <b class="best">{best['score']:.4f}</b></p>

<h2>Trial Scores</h2>
{svg_chart}

<h2>Best Configuration (trial {best['trial']})</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{param_rows}</table>

<h2>Top 20 Trials</h2>
<table>
<tr><th>#</th><th>Score</th><th>Time</th><th>Params</th></tr>
{trial_rows}
</table>
<p style="color:#8B949E;font-size:0.8em">Generated {datetime.utcnow().isoformat()}Z</p>
</body>
</html>"""

    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, report_path)
    return report_path


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
    p.add_argument("--method",      choices=["grid", "random", "diffevo", "bayesian"], default="random",
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

    all_results: list[dict] = []

    def _run_and_record(params: dict) -> float:
        idx_before = len(all_results)
        score = _run(params)
        # _run updates best_result in place; capture a snapshot of the latest trial
        # by re-reading the jsonl tail
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                all_results.append(last)
        except Exception:
            pass
        return score

    try:
        if args.method == "grid":
            combos = _grid_combos(PARAM_SPACE)
            print(f"  Grid search: {len(combos)} combinations\n")
            for params in combos:
                _run_and_record(params)

        elif args.method == "random":
            samples = _random_samples(PARAM_SPACE, args.trials, args.seed)
            print(f"  Random search: {args.trials} trials\n")
            for params in samples:
                _run_and_record(params)

        elif args.method == "diffevo":
            print(f"  Differential evolution: up to {args.trials} evaluations\n")
            _diffevo_samples(PARAM_SPACE, args.trials, args.seed, _run_and_record)

        else:  # bayesian
            print(f"  Bayesian optimization: up to {args.trials} evaluations\n")
            _bayesian_samples(PARAM_SPACE, args.trials, args.seed, _run_and_record)

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
        # Generate HTML visualization report
        if all_results:
            rpt = _write_optimizer_report(output_dir, all_results, args.metric)
            print(f"  Visualization report → {rpt}\n")
    else:
        print("\n[optimizer] no trials completed.")


if __name__ == "__main__":
    main()
