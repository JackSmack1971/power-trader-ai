#!/usr/bin/env python3
"""
PowerTrader AI - Backtesting Analytics Module
Implements Phase 5: Analytics & Metrics

This module provides comprehensive analytics for backtest results including:
- Corrected Sharpe Ratio (365-day annualization for crypto)
- True Max Drawdown (including unrealized losses)
- Sortino & Calmar Ratios
- Buy-and-Hold Benchmark comparison
- Market Regime Detection
- Data Source Validation
- HTML Report Generation
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


def build_equity_curve(trader_status_file: str) -> Tuple[List[float], List[int]]:
    """
    Build equity curve from trader_status snapshots.

    Args:
        trader_status_file: Path to trader_status.json or JSONL file with snapshots

    Returns:
        Tuple of (equity_curve, timestamps)
        equity_curve: List of total account values over time
        timestamps: Corresponding unix timestamps
    """
    equity_curve = []
    timestamps = []

    # Check if it's a JSONL file (multiple snapshots) or single JSON
    if not os.path.exists(trader_status_file):
        print(f"Warning: {trader_status_file} not found")
        return [10000.0], [int(datetime.now().timestamp())]

    with open(trader_status_file, 'r') as f:
        content = f.read().strip()

        # Try parsing as JSONL (multiple lines)
        if '\n' in content:
            for line in content.split('\n'):
                if line.strip():
                    try:
                        snapshot = json.loads(line)
                        total_value = snapshot.get('total_value', snapshot.get('cash', 10000.0))
                        timestamp = snapshot.get('timestamp', int(datetime.now().timestamp()))
                        equity_curve.append(total_value)
                        timestamps.append(timestamp)
                    except json.JSONDecodeError:
                        continue
        else:
            # Single JSON snapshot
            try:
                snapshot = json.loads(content)
                total_value = snapshot.get('total_value', snapshot.get('cash', 10000.0))
                timestamp = snapshot.get('timestamp', int(datetime.now().timestamp()))
                equity_curve.append(total_value)
                timestamps.append(timestamp)
            except json.JSONDecodeError:
                pass

    # Default if no data found
    if not equity_curve:
        equity_curve = [10000.0]
        timestamps = [int(datetime.now().timestamp())]

    return equity_curve, timestamps


def calculate_sharpe_ratio_corrected(equity_curve: List[float],
                                    timestamps: List[int],
                                    risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio using equity curve time series.
    Corrected for crypto 24/7 trading (365-day annualization).

    Args:
        equity_curve: List of account values over time
        timestamps: Corresponding unix timestamps
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        float: Annualized Sharpe ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate returns from equity curve
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    # Calculate time period in days
    time_span_seconds = timestamps[-1] - timestamps[0]
    time_span_days = time_span_seconds / 86400.0

    if time_span_days == 0:
        return 0.0

    # Annualization factor for crypto (365 days, not 252)
    periods_per_year = 365.0 / (time_span_days / len(returns))

    # Calculate Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    # Adjust for risk-free rate (convert annual to per-period)
    risk_free_per_period = risk_free_rate / periods_per_year

    sharpe = (mean_return - risk_free_per_period) / std_return * np.sqrt(periods_per_year)

    return float(sharpe)


def calculate_max_drawdown_corrected(equity_curve: List[float]) -> Dict[str, float]:
    """
    Calculate max drawdown from total account equity.
    Includes unrealized losses (cash + positions).

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
    if len(equity_curve) < 2:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_value": 0.0,
            "peak_idx": 0,
            "trough_idx": 0,
            "duration_periods": 0
        }

    equity_array = np.array(equity_curve)

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)

    # Calculate drawdown at each point
    drawdown = (equity_array - running_max) / running_max

    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd_pct = float(drawdown[max_dd_idx])

    # Find the peak before this drawdown
    peak_idx = np.argmax(equity_array[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    return {
        "max_drawdown_pct": abs(max_dd_pct) * 100,  # Convert to percentage
        "max_drawdown_value": float(running_max[max_dd_idx] - equity_array[max_dd_idx]),
        "peak_idx": int(peak_idx),
        "trough_idx": int(max_dd_idx),
        "duration_periods": int(max_dd_idx - peak_idx)
    }


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        float: Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    # Only consider negative returns for downside deviation
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0

    # Calculate downside deviation
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    # Annualization factor (365 days for crypto)
    annualization = np.sqrt(365)

    mean_return = np.mean(returns)
    sortino = (mean_return - risk_free_rate / 365) / downside_std * annualization

    return float(sortino)


def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return: Total return as percentage (e.g., 25.0 for 25%)
        max_drawdown: Max drawdown as percentage (e.g., 10.0 for 10%)
        years: Time period in years

    Returns:
        float: Calmar ratio
    """
    if max_drawdown == 0 or years == 0:
        return 0.0

    annualized_return = (total_return / 100) / years
    max_dd_decimal = max_drawdown / 100

    calmar = annualized_return / max_dd_decimal

    return float(calmar)


def calculate_buy_and_hold_benchmark(backtest_dir: str,
                                     initial_capital: float = 10000.0) -> Dict[str, float]:
    """
    Calculate buy-and-hold return for comparison.

    Strategy: At start, allocate capital equally across all coins.
              Hold until end without rebalancing.

    Args:
        backtest_dir: Directory containing backtest results
        initial_capital: Starting capital

    Returns:
        {
            "total_return_pct": float,
            "final_value": float,
            "coins": dict of individual coin returns
        }
    """
    # Try to load config to get coin list and date range
    config_file = os.path.join(backtest_dir, 'config.json')

    if not os.path.exists(config_file):
        return {
            "total_return_pct": 0.0,
            "final_value": initial_capital,
            "coins": {}
        }

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        coins = config.get('coins', ['BTC'])

        # For now, return placeholder values
        # In a real implementation, this would:
        # 1. Load cached price data for start and end dates
        # 2. Calculate equal allocation at start
        # 3. Calculate final value based on end prices

        return {
            "total_return_pct": 0.0,  # Placeholder
            "final_value": initial_capital,
            "coins": {coin: 0.0 for coin in coins}
        }
    except Exception as e:
        print(f"Error calculating buy-and-hold benchmark: {e}")
        return {
            "total_return_pct": 0.0,
            "final_value": initial_capital,
            "coins": {}
        }


class MarketRegimeDetector:
    """
    Detect market regimes and analyze performance by regime.
    """

    def __init__(self, sma_short: int = 50, sma_long: int = 200):
        """
        Initialize regime detector.

        Args:
            sma_short: Short moving average period
            sma_long: Long moving average period
        """
        self.sma_short = sma_short
        self.sma_long = sma_long

    def detect_regime(self, prices: List[float]) -> str:
        """
        Classify market regime.

        Args:
            prices: List of prices

        Returns:
            str: Regime classification (bull_low_vol, bull_high_vol, bear_low_vol,
                 bear_high_vol, sideways)
        """
        if len(prices) < self.sma_long:
            return "sideways"

        prices_array = np.array(prices)

        # Calculate moving averages
        sma_short = np.mean(prices_array[-self.sma_short:])
        sma_long = np.mean(prices_array[-self.sma_long:])

        # Determine trend
        if sma_short > sma_long * 1.02:
            trend = "bull"
        elif sma_short < sma_long * 0.98:
            trend = "bear"
        else:
            trend = "sideways"

        # Calculate volatility (standard deviation of returns)
        returns = np.diff(prices_array[-30:]) / prices_array[-31:-1]
        volatility = np.std(returns)

        # Classify volatility
        if volatility > 0.03:  # 3% daily volatility
            vol_class = "high_vol"
        else:
            vol_class = "low_vol"

        # Combine trend and volatility
        if trend == "sideways":
            return "sideways"
        else:
            return f"{trend}_{vol_class}"

    def analyze_performance_by_regime(self,
                                     equity_curve: List[float],
                                     prices: List[float],
                                     timestamps: List[int]) -> Dict[str, Dict]:
        """
        Break down performance by regime.

        Args:
            equity_curve: Account equity over time
            prices: Market prices over time
            timestamps: Corresponding timestamps

        Returns:
            Dict mapping regime to performance metrics
        """
        regimes = {}
        current_regime = None
        regime_start_idx = 0

        for i in range(len(prices)):
            regime = self.detect_regime(prices[:i+1])

            if regime != current_regime:
                # Save previous regime performance
                if current_regime and regime_start_idx < i:
                    regime_equity = equity_curve[regime_start_idx:i]
                    if len(regime_equity) > 1:
                        regime_return = (regime_equity[-1] - regime_equity[0]) / regime_equity[0] * 100

                        if current_regime not in regimes:
                            regimes[current_regime] = {
                                "returns": [],
                                "duration_periods": 0
                            }

                        regimes[current_regime]["returns"].append(regime_return)
                        regimes[current_regime]["duration_periods"] += (i - regime_start_idx)

                current_regime = regime
                regime_start_idx = i

        # Calculate aggregate statistics for each regime
        regime_stats = {}
        for regime, data in regimes.items():
            if data["returns"]:
                regime_stats[regime] = {
                    "avg_return": np.mean(data["returns"]),
                    "total_periods": data["duration_periods"],
                    "occurrences": len(data["returns"])
                }

        return regime_stats


def validate_data_sources(kucoin_prices: List[float],
                          robinhood_prices: List[float],
                          tolerance_pct: float = 0.5) -> Dict[str, float]:
    """
    Compare KuCoin and Robinhood prices to measure divergence.

    Args:
        kucoin_prices: List of prices from KuCoin
        robinhood_prices: List of prices from Robinhood
        tolerance_pct: Warning threshold for divergence

    Returns:
        {
            "mean_divergence_pct": float,
            "max_divergence_pct": float,
            "correlation": float,
            "adjustment_factor": float
        }
    """
    if len(kucoin_prices) != len(robinhood_prices) or len(kucoin_prices) == 0:
        return {
            "mean_divergence_pct": 0.0,
            "max_divergence_pct": 0.0,
            "correlation": 0.0,
            "adjustment_factor": 1.0
        }

    kucoin_array = np.array(kucoin_prices)
    robinhood_array = np.array(robinhood_prices)

    # Calculate divergence
    divergence = np.abs(kucoin_array - robinhood_array) / robinhood_array * 100

    mean_divergence = np.mean(divergence)
    max_divergence = np.max(divergence)

    # Calculate correlation
    correlation = np.corrcoef(kucoin_array, robinhood_array)[0, 1]

    # Calculate adjustment factor (average ratio)
    adjustment_factor = np.mean(robinhood_array / kucoin_array)

    return {
        "mean_divergence_pct": float(mean_divergence),
        "max_divergence_pct": float(max_divergence),
        "correlation": float(correlation),
        "adjustment_factor": float(adjustment_factor)
    }


def generate_analytics_report(backtest_dir: str, output_file: Optional[str] = None) -> str:
    """
    Generate comprehensive analytics report from backtest results.

    Args:
        backtest_dir: Directory containing backtest results
        output_file: Optional output HTML file path

    Returns:
        str: Path to generated HTML report
    """
    if output_file is None:
        output_file = os.path.join(backtest_dir, 'analytics_report.html')

    # Load trader status to build equity curve
    trader_status_file = os.path.join(backtest_dir, 'hub_data', 'trader_status.json')
    equity_curve, timestamps = build_equity_curve(trader_status_file)

    # Load config
    config_file = os.path.join(backtest_dir, 'config.json')
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Calculate metrics
    initial_capital = equity_curve[0] if equity_curve else 10000.0
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # Time period
    time_span_days = (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) > 1 else 1.0
    years = time_span_days / 365.0

    # Risk metrics
    sharpe = calculate_sharpe_ratio_corrected(equity_curve, timestamps)
    max_dd = calculate_max_drawdown_corrected(equity_curve)

    # Calculate returns for Sortino
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1] if len(equity_array) > 1 else np.array([0.0])
    sortino = calculate_sortino_ratio(returns)

    calmar = calculate_calmar_ratio(total_return, max_dd["max_drawdown_pct"], years)

    # Buy-and-hold benchmark
    buy_hold = calculate_buy_and_hold_benchmark(backtest_dir, initial_capital)

    # Market regime analysis
    # For regime detection, we would need price data - placeholder for now
    regime_stats = {}

    # Generate HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PowerTrader AI - Backtest Analytics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-value.positive {{
            color: #27ae60;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .config-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>PowerTrader AI - Backtest Analytics Report</h1>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <h2>Executive Summary</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if total_return > 0 else 'negative'}">
                {total_return:+.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{sharpe:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{max_dd['max_drawdown_pct']:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{sortino:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">{calmar:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Final Capital</div>
            <div class="metric-value">${final_capital:,.2f}</div>
        </div>
    </div>

    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Initial Capital</td>
            <td>${initial_capital:,.2f}</td>
            <td>Starting account balance</td>
        </tr>
        <tr>
            <td>Final Capital</td>
            <td>${final_capital:,.2f}</td>
            <td>Ending account balance</td>
        </tr>
        <tr>
            <td>Total Return</td>
            <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:+.2f}%</td>
            <td>Overall return percentage</td>
        </tr>
        <tr>
            <td>Time Period</td>
            <td>{time_span_days:.1f} days ({years:.2f} years)</td>
            <td>Backtest duration</td>
        </tr>
        <tr>
            <td>Annualized Return</td>
            <td>{(total_return / years) if years > 0 else 0:.2f}%</td>
            <td>Return normalized to one year</td>
        </tr>
    </table>

    <h2>Risk Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Sharpe Ratio (365-day)</td>
            <td>{sharpe:.3f}</td>
            <td>Risk-adjusted return (crypto annualization)</td>
        </tr>
        <tr>
            <td>Sortino Ratio</td>
            <td>{sortino:.3f}</td>
            <td>Return vs downside volatility only</td>
        </tr>
        <tr>
            <td>Calmar Ratio</td>
            <td>{calmar:.3f}</td>
            <td>Annualized return / max drawdown</td>
        </tr>
        <tr>
            <td>Max Drawdown</td>
            <td class="negative">{max_dd['max_drawdown_pct']:.2f}%</td>
            <td>Largest peak-to-trough decline</td>
        </tr>
        <tr>
            <td>Max Drawdown Value</td>
            <td>${max_dd['max_drawdown_value']:,.2f}</td>
            <td>Dollar amount of max drawdown</td>
        </tr>
        <tr>
            <td>Drawdown Duration</td>
            <td>{max_dd['duration_periods']} periods</td>
            <td>Time from peak to trough</td>
        </tr>
    </table>

    <h2>Buy-and-Hold Comparison</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Return</th>
            <th>Final Value</th>
        </tr>
        <tr>
            <td>PowerTrader AI</td>
            <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:+.2f}%</td>
            <td>${final_capital:,.2f}</td>
        </tr>
        <tr>
            <td>Buy-and-Hold</td>
            <td>{buy_hold['total_return_pct']:+.2f}%</td>
            <td>${buy_hold['final_value']:,.2f}</td>
        </tr>
        <tr>
            <td><strong>Outperformance</strong></td>
            <td><strong>{(total_return - buy_hold['total_return_pct']):+.2f}%</strong></td>
            <td><strong>${(final_capital - buy_hold['final_value']):+,.2f}</strong></td>
        </tr>
    </table>

    <h2>Configuration</h2>
    <div class="config-section">
        <pre>{json.dumps(config, indent=2)}</pre>
    </div>

    <div class="timestamp">Report End</div>
</body>
</html>
"""

    # Write report
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Analytics report generated: {output_file}")
    return output_file


def main():
    """
    Main CLI entry point for analytics.

    Usage:
        python pt_analyze.py backtest_results/backtest_2025-01-15_143022
    """
    parser = argparse.ArgumentParser(description='PowerTrader AI - Backtest Analytics')
    parser.add_argument('backtest_dir', help='Path to backtest results directory')
    parser.add_argument('--output', '-o', help='Output HTML file path (optional)')
    parser.add_argument('--show-regime', action='store_true',
                       help='Include market regime analysis')

    args = parser.parse_args()

    if not os.path.exists(args.backtest_dir):
        print(f"Error: Backtest directory not found: {args.backtest_dir}")
        return 1

    print("=" * 60)
    print("PowerTrader AI - Backtest Analytics")
    print("=" * 60)
    print(f"Analyzing backtest: {args.backtest_dir}")
    print()

    # Generate report
    report_path = generate_analytics_report(args.backtest_dir, args.output)

    print()
    print("=" * 60)
    print(f"Report generated successfully!")
    print(f"View report: {report_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
