"""
Realistic Execution Engine for PowerTrader AI Backtesting
==========================================================

This module implements a professional-grade execution simulator that models:
1. Exponential slippage based on order size vs volume
2. Robinhood spread penalty (hidden costs of "free" trading)
3. Proper partial fill handling with configurable behavior
4. Volatility-based slippage multipliers
5. Network latency simulation

Author: PowerTrader AI (Expert Quant Review v2.0)
Created: 2025-12-30
"""

import random
import math
from typing import Dict, Optional, Literal


class RealisticExecutionEngine:
    """
    Simulates realistic order execution with slippage, fees, and liquidity constraints.

    Key Features:
    - Exponential slippage curve based on order size vs volume
    - Robinhood spread penalty (0.20% default)
    - Partial fill handling with configurable behavior
    - Volatility-based slippage multipliers
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        fee_bps: float = 20.0,
        robinhood_spread_bps: float = 20.0,
        max_volume_pct: float = 1.0,
        latency_ms_range: tuple = (50, 500),
        partial_fill_behavior: Literal["cancel", "retry_next", "limit"] = "retry_next"
    ):
        """
        Initialize execution engine.

        Args:
            slippage_bps: Base slippage in basis points (0.05% = 5 bps)
            fee_bps: Trading fees in basis points (0.20% = 20 bps)
            robinhood_spread_bps: Robinhood implicit spread penalty (0.20% = 20 bps)
            max_volume_pct: Maximum order size as % of candle volume (1.0 = 1%)
            latency_ms_range: Network latency range in milliseconds
            partial_fill_behavior:
                - "cancel": Cancel unfilled portion immediately
                - "retry_next": Retry unfilled portion on next candle
                - "limit": Leave unfilled portion as limit order (not implemented in backtest)
        """
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.robinhood_spread_bps = robinhood_spread_bps
        self.max_volume_pct = max_volume_pct
        self.latency_ms_range = latency_ms_range
        self.partial_fill_behavior = partial_fill_behavior

        # Track pending orders for retry logic
        self.pending_orders = []

    def calculate_exponential_slippage(
        self,
        order_value: float,
        candle_volume: float,
        base_slippage_bps: float
    ) -> float:
        """
        Calculate slippage using exponential impact curve.

        Formula: Expected Slippage % = (Order Size / Candle Volume) * Volatility_Factor * Constant

        If order size > 1% of volume, slippage increases exponentially, not linearly.

        Args:
            order_value: Total value of order in USD
            candle_volume: Candle's volume in USD
            base_slippage_bps: Base slippage in basis points

        Returns:
            Adjusted slippage in basis points
        """
        if candle_volume <= 0:
            # No volume data - assume high slippage
            return base_slippage_bps * 5.0

        # Calculate order size as % of volume
        volume_pct = (order_value / candle_volume) * 100.0

        # Exponential impact: slippage doubles for every 1% of volume
        if volume_pct > self.max_volume_pct:
            # Orders > 1% of volume get exponentially worse slippage
            impact_multiplier = math.exp(volume_pct - self.max_volume_pct)
            return base_slippage_bps * impact_multiplier
        else:
            # Small orders get base slippage
            return base_slippage_bps

    def simulate_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        current_price: float,
        candle: Dict[str, float],
        is_retry: bool = False
    ) -> Dict[str, any]:
        """
        Simulate realistic order execution with all market frictions.

        Args:
            side: 'buy' or 'sell'
            qty: Order quantity in asset units
            current_price: Close price from candle
            candle: Full OHLCV data
                {
                    "close": 98765.43,
                    "high": 99000.00,
                    "low": 98500.00,
                    "volume": 1234.56,  # Volume in USD or asset units
                    "timestamp": 1704067200
                }
            is_retry: True if this is a retry of a partial fill

        Returns:
            {
                "fill_price": Actual fill price after all costs,
                "fill_qty": Quantity filled (may be partial),
                "fees": Total fees in USD,
                "status": "filled" | "partial" | "rejected",
                "slippage_bps": Slippage in basis points,
                "unfilled_qty": Remaining unfilled quantity,
                "pending_order": True if unfilled portion is pending retry
            }
        """
        # 1. Calculate volatility multiplier (volatile candles = worse slippage)
        volatility = (candle['high'] - candle['low']) / candle['close']
        volatility_multiplier = 1.0 + (volatility * 2.0)  # 2x slippage on volatile candles

        # 2. Calculate order value and exponential slippage
        order_value = qty * current_price
        candle_volume_usd = candle.get('volume', 0) * current_price  # Convert to USD if needed

        base_slippage = self.slippage_bps * volatility_multiplier
        exponential_slippage_bps = self.calculate_exponential_slippage(
            order_value, candle_volume_usd, base_slippage
        )

        # 3. Apply Robinhood spread penalty (hidden cost of "free" trading)
        # Robinhood prices are worse than exchange prices by ~20 bps on average
        total_slippage_bps = exponential_slippage_bps + self.robinhood_spread_bps

        # 4. Calculate fill price with slippage
        slippage_multiplier = total_slippage_bps / 10000.0

        if side == 'buy':
            # Buy slippage makes price worse (higher)
            fill_price = current_price * (1.0 + slippage_multiplier)
        else:
            # Sell slippage makes price worse (lower)
            fill_price = current_price * (1.0 - slippage_multiplier)

        # 5. Check liquidity constraints (partial fills if order too large)
        volume_limit_usd = candle_volume_usd * (self.max_volume_pct / 100.0)

        fill_qty = qty
        fill_status = "filled"
        unfilled_qty = 0.0
        pending_order = False

        if order_value > volume_limit_usd and volume_limit_usd > 0:
            # Partial fill - only fill what liquidity allows
            fill_qty = volume_limit_usd / fill_price
            unfilled_qty = qty - fill_qty
            fill_status = "partial"

            # Handle unfilled portion based on configuration
            if self.partial_fill_behavior == "retry_next" and not is_retry:
                # Queue for retry on next candle
                self.pending_orders.append({
                    "side": side,
                    "qty": unfilled_qty,
                    "original_price": current_price
                })
                pending_order = True
            elif self.partial_fill_behavior == "cancel":
                # Cancel unfilled portion (default conservative behavior)
                pass
            # "limit" behavior would be handled by caller (not in backtest scope)

        # 6. Calculate trading fees (taker fees for market orders)
        fees = fill_qty * fill_price * (self.fee_bps / 10000.0)

        # 7. Add network latency simulation (random additional slippage)
        latency_ms = random.uniform(*self.latency_ms_range)
        # Assume price can move 0-2 bps during latency
        latency_slippage_bps = random.uniform(0, 2.0)
        latency_slippage = current_price * (latency_slippage_bps / 10000.0)

        if side == 'buy':
            fill_price += latency_slippage
        else:
            fill_price -= latency_slippage

        total_slippage_bps += latency_slippage_bps

        # 8. Validate fill is realistic (price within candle range)
        # This prevents unrealistic fills outside the candle's high/low
        if side == 'buy' and fill_price > candle['high'] * 1.005:
            # Fill price more than 0.5% above candle high - reject
            return {
                "fill_price": 0.0,
                "fill_qty": 0.0,
                "fees": 0.0,
                "status": "rejected",
                "slippage_bps": 0.0,
                "unfilled_qty": qty,
                "pending_order": False,
                "reject_reason": "Fill price exceeds realistic range (>0.5% above candle high)"
            }
        elif side == 'sell' and fill_price < candle['low'] * 0.995:
            # Fill price more than 0.5% below candle low - reject
            return {
                "fill_price": 0.0,
                "fill_qty": 0.0,
                "fees": 0.0,
                "status": "rejected",
                "slippage_bps": 0.0,
                "unfilled_qty": qty,
                "pending_order": False,
                "reject_reason": "Fill price exceeds realistic range (<0.5% below candle low)"
            }

        return {
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "fees": fees,
            "status": fill_status,
            "slippage_bps": total_slippage_bps,
            "unfilled_qty": unfilled_qty,
            "pending_order": pending_order,
            "latency_ms": latency_ms,
            "volatility": volatility,
            "volume_pct": (order_value / candle_volume_usd * 100.0) if candle_volume_usd > 0 else 0.0
        }

    def process_pending_orders(self, candle: Dict[str, float]) -> list:
        """
        Process pending orders from previous partial fills.

        Args:
            candle: Current candle data

        Returns:
            List of fill results for pending orders
        """
        if not self.pending_orders:
            return []

        fills = []
        remaining_orders = []

        for order in self.pending_orders:
            result = self.simulate_fill(
                side=order["side"],
                qty=order["qty"],
                current_price=candle["close"],
                candle=candle,
                is_retry=True
            )

            fills.append(result)

            # If still partially filled, keep in queue
            if result["status"] == "partial" and self.partial_fill_behavior == "retry_next":
                remaining_orders.append({
                    "side": order["side"],
                    "qty": result["unfilled_qty"],
                    "original_price": order["original_price"]
                })

        self.pending_orders = remaining_orders
        return fills

    def get_effective_cost(self, side: str, price: float) -> float:
        """
        Calculate the effective cost for an order accounting for all frictions.

        This is useful for estimating total cost before placing an order.

        Args:
            side: 'buy' or 'sell'
            price: Market price

        Returns:
            Effective price after slippage + fees + spread
        """
        # Base costs
        total_cost_bps = self.slippage_bps + self.fee_bps + self.robinhood_spread_bps
        cost_multiplier = total_cost_bps / 10000.0

        if side == 'buy':
            return price * (1.0 + cost_multiplier)
        else:
            return price * (1.0 - cost_multiplier)
