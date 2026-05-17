#!/usr/bin/env python3
"""Tests for monte_carlo_simulation() in pt_analyze.py."""

import os
import sys
import unittest
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pt_analyze import monte_carlo_simulation


class TestMonteCarlo(unittest.TestCase):

    def _flat_curve(self, n: int = 100, value: float = 10000.0):
        return [value] * n

    def _trending_up(self, n: int = 100, start: float = 10000.0, growth: float = 0.001):
        out = []
        v = start
        for _ in range(n):
            out.append(v)
            v *= (1 + growth)
        return out

    def _trending_down(self, n: int = 100, start: float = 10000.0, decay: float = 0.002):
        out = []
        v = start
        for _ in range(n):
            out.append(v)
            v *= (1 - decay)
        return out

    # ── Basic structure ──────────────────────────────────────────────────

    def test_returns_expected_keys(self):
        result = monte_carlo_simulation(self._trending_up(), n_simulations=50, seed=0)
        for key in ["paths", "median_final", "var_95", "cvar_95", "var_99",
                    "prob_ruin", "initial", "percentiles", "n_simulations"]:
            self.assertIn(key, result)

    def test_n_paths_matches_n_simulations(self):
        n = 80
        result = monte_carlo_simulation(self._trending_up(), n_simulations=n, seed=0)
        self.assertEqual(len(result["paths"]), n)

    def test_initial_value_correct(self):
        curve = self._trending_up(start=5000.0)
        result = monte_carlo_simulation(curve, n_simulations=50, seed=0)
        self.assertAlmostEqual(result["initial"], 5000.0, places=2)

    def test_percentiles_keys(self):
        result = monte_carlo_simulation(self._trending_up(), n_simulations=50, seed=0)
        for p in ["5", "25", "50", "75", "95"]:
            self.assertIn(p, result["percentiles"])

    # ── Statistical properties ────────────────────────────────────────────

    def test_flat_curve_median_near_initial(self):
        """Flat returns ⇒ all paths stay near initial value."""
        result = monte_carlo_simulation(self._flat_curve(), n_simulations=200, seed=1)
        self.assertAlmostEqual(result["median_final"], result["initial"], delta=1.0)

    def test_trending_up_median_above_initial(self):
        result = monte_carlo_simulation(self._trending_up(), n_simulations=200, seed=2)
        self.assertGreater(result["median_final"], result["initial"])

    def test_trending_down_prob_ruin_nonzero(self):
        curve = self._trending_down(n=200, decay=0.005)
        result = monte_carlo_simulation(curve, n_simulations=300, seed=3)
        self.assertGreater(result["prob_ruin"], 0.0)

    def test_prob_ruin_zero_for_strong_uptrend(self):
        """Very strong positive returns should give near-zero ruin probability."""
        curve = self._trending_up(n=50, growth=0.02)  # 2% per period
        result = monte_carlo_simulation(curve, n_simulations=200, seed=4)
        self.assertAlmostEqual(result["prob_ruin"], 0.0, delta=0.05)

    def test_var_95_le_var_99(self):
        """99% VaR should be worse (more negative) than 95% VaR."""
        result = monte_carlo_simulation(self._trending_up(), n_simulations=300, seed=5)
        self.assertLessEqual(result["var_99"], result["var_95"])

    def test_cvar_95_le_var_95(self):
        """CVaR (expected loss beyond VaR) is always >= VaR in magnitude."""
        result = monte_carlo_simulation(self._trending_up(), n_simulations=300, seed=6)
        self.assertLessEqual(result["cvar_95"], result["var_95"] + 1e-6)

    def test_percentile_order(self):
        result = monte_carlo_simulation(self._trending_up(), n_simulations=200, seed=7)
        p = result["percentiles"]
        self.assertLessEqual(float(p["5"]), float(p["25"]))
        self.assertLessEqual(float(p["25"]), float(p["50"]))
        self.assertLessEqual(float(p["50"]), float(p["75"]))
        self.assertLessEqual(float(p["75"]), float(p["95"]))

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_too_short_returns_empty(self):
        result = monte_carlo_simulation([10000.0, 10100.0, 10050.0], n_simulations=10)
        # < 4 data points returns empty result
        self.assertEqual(result["paths"], [])

    def test_single_point_returns_empty(self):
        result = monte_carlo_simulation([10000.0], n_simulations=10)
        self.assertEqual(result["paths"], [])

    def test_reproducibility_with_same_seed(self):
        curve = self._trending_up()
        r1 = monte_carlo_simulation(curve, n_simulations=100, seed=99)
        r2 = monte_carlo_simulation(curve, n_simulations=100, seed=99)
        self.assertAlmostEqual(r1["median_final"], r2["median_final"], places=4)

    def test_different_seeds_give_different_results(self):
        # Use a volatile curve so random resampling produces meaningfully different paths
        import math
        curve = [10000.0 * (1 + 0.02 * math.sin(i * 0.3)) for i in range(50)]
        r1 = monte_carlo_simulation(curve, n_simulations=200, seed=1)
        r2 = monte_carlo_simulation(curve, n_simulations=200, seed=999)
        # 5th percentiles should differ since different paths are sampled
        p1 = r1["percentiles"]["5"]
        p2 = r2["percentiles"]["5"]
        self.assertNotEqual(round(p1, 2), round(p2, 2))


if __name__ == "__main__":
    unittest.main()
