"""
Monte Carlo generator and simulator tests: each method reproduces the
statistics it claims to, seeds reproduce, inflation deflates.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import SimulatedReturns
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig

BDAYS = pd.bdate_range('2010-01-01', '2019-12-31')


def asset_from_returns(ticker, r: np.ndarray, dates=BDAYS):
    prices = pd.Series(100 * np.cumprod(1 + r), index=dates[:len(r)])
    return {'ticker': ticker, 'full_data': pd.DataFrame({'Adj Close': prices}),
            'historical_returns': prices.pct_change().dropna()}


def regime_returns(n, seed=0):
    """Volatility clustering: 60-day calm/turbulent regimes."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 60) % 2 == 0, 0.005, 0.03)
    return rng.normal(0.0004, vol)


def abs_autocorr(x):
    a = np.abs(x) - np.abs(x).mean()
    return (a[1:] * a[:-1]).mean() / a.var()


class TestGenerators:
    def test_gbm_matches_log_return_moments(self):
        rng = np.random.default_rng(0)
        hist = rng.normal(0.0005, 0.012, (2500, 1))
        src = SimulatedReturns(hist, 'geometric_brownian', 4000, np.random.default_rng(1))
        sim = np.log1p(src.chunk(0, 252))
        se = np.log1p(hist).std() / np.sqrt(sim.size)
        assert sim.mean() == pytest.approx(np.log1p(hist).mean(), abs=4 * se)
        assert sim.std() == pytest.approx(np.log1p(hist).std(), rel=0.01)

    def test_parametric_matches_simple_return_mean(self):
        """The old engine subtracted 0.5*var from parametric returns (wrong space)."""
        rng = np.random.default_rng(0)
        hist = rng.normal(0.0005, 0.02, (2500, 1))
        src = SimulatedReturns(hist, 'parametric', 4000, np.random.default_rng(1))
        sim = src.chunk(0, 252)
        assert sim.mean() == pytest.approx(hist.mean(), abs=4 * hist.std() / np.sqrt(sim.size))

    def test_gbm_preserves_correlation(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((2500, 2))
        hist = 0.01 * np.column_stack([z[:, 0], 0.7 * z[:, 0] + np.sqrt(1 - 0.49) * z[:, 1]])
        src = SimulatedReturns(hist, 'geometric_brownian', 2000, np.random.default_rng(1))
        sim = src.chunk(0, 252).reshape(-1, 2)
        assert np.corrcoef(sim.T)[0, 1] == pytest.approx(np.corrcoef(hist.T)[0, 1], abs=0.02)

    def test_block_bootstrap_draws_consecutive_days(self):
        hist = np.arange(1000, dtype=float)[:, None]
        src = SimulatedReturns(hist, 'block_bootstrap', 5, np.random.default_rng(0), block_size=21)
        path = src.chunk(0, 63)[0, :, 0]
        steps = np.diff(path)
        # Consecutive within blocks; jumps only at block boundaries (or the circular wrap)
        jumps = np.flatnonzero((steps != 1) & (steps != -999))
        assert set(jumps + 1) <= {21, 42}

    def test_block_bootstrap_keeps_volatility_clustering(self):
        hist = regime_returns(2500)[:, None]
        iid = SimulatedReturns(hist, 'bootstrap', 200, np.random.default_rng(1)).chunk(0, 1000)
        blk = SimulatedReturns(hist, 'block_bootstrap', 200, np.random.default_rng(1),
                               block_size=60).chunk(0, 1000)
        target = abs_autocorr(hist[:, 0])
        ac_iid = np.mean([abs_autocorr(p[:, 0]) for p in iid])
        ac_blk = np.mean([abs_autocorr(p[:, 0]) for p in blk])
        assert abs(ac_iid) < 0.02
        assert ac_blk == pytest.approx(target, abs=0.05)


class TestSimulator:
    def _sim(self, **kw):
        cfg = dict(initial_capital=10000, years=5, simulations=500, method='bootstrap', seed=7)
        cfg.update(kw)
        return PortfolioSimulator(None, SimulationConfig(**cfg))

    def test_seed_reproducible(self):
        a = asset_from_returns('A', np.random.default_rng(0).normal(0.0004, 0.01, 2500))
        r1 = self._sim().simulate_portfolio([a], [1.0])
        r2 = self._sim().simulate_portfolio([a], [1.0])
        r3 = self._sim(seed=8).simulate_portfolio([a], [1.0])
        np.testing.assert_array_equal(r1['final_values'], r2['final_values'])
        assert not np.array_equal(r1['final_values'], r3['final_values'])

    def test_inflation_reports_real_dollars(self):
        r = np.full(2500, 1.08 ** (1 / 252) - 1)
        a = asset_from_returns('A', r)
        nominal = self._sim().simulate_portfolio([a], [1.0])
        real = self._sim(inflation_rate=0.03).simulate_portfolio([a], [1.0])
        dpy = nominal['days_per_year']
        g = (1 + r[0]) ** dpy
        assert nominal['stats']['median_cagr'] == pytest.approx(g - 1, abs=1e-9)
        assert real['stats']['median_cagr'] == pytest.approx(g / 1.03 - 1, abs=1e-9)
        assert real['real_dollars'] and not nominal['real_dollars']

    def test_probabilities_and_percentiles(self):
        a = asset_from_returns('A', np.random.default_rng(3).normal(0.0004, 0.012, 2500))
        res = self._sim(simulations=2000).simulate_portfolio([a], [1.0])
        probs, stats = res['probabilities'], res['stats']
        assert len(probs['years']) == 5
        # Loss probability falls with horizon for a positive-drift asset
        assert probs['prob_loss'][0] > probs['prob_loss'][-1]
        p = stats['percentiles']['final_value']
        assert p[10] < p[25] < p[50] < p[75] < p[90]
        assert p[50] == pytest.approx(stats['median_final_value'])
        assert 0.10 < stats['median_volatility'] < 0.25
        assert 'sharpe_ratio' not in stats  # cross-sectional "Sharpe" was not a Sharpe ratio

    def test_rebalancing_changes_simulated_outcome(self):
        rng = np.random.default_rng(4)
        a = asset_from_returns('A', rng.normal(0.0006, 0.015, 2500))
        b = asset_from_returns('B', rng.normal(0.0001, 0.003, 2500))
        hold = self._sim().simulate_portfolio([a, b], [0.5, 0.5], rebalance='none')
        reb = self._sim().simulate_portfolio([a, b], [0.5, 0.5], rebalance='annual')
        # Same seed, same draws: buy-and-hold drifts toward the stronger asset -> wider spread
        assert np.std(hold['cagr']) > np.std(reb['cagr'])
