"""Efficient frontier tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig

BDAYS = pd.bdate_range('2014-01-01', '2020-12-31')


def make_assets(specs, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t, mu, sd in specs:
        p = pd.Series(100 * np.cumprod(1 + rng.normal(mu, sd, len(BDAYS))), index=BDAYS)
        out.append({'ticker': t, 'full_data': pd.DataFrame({'Adj Close': p})})
    return out


@pytest.fixture(scope='module')
def setup():
    assets = make_assets([('A', 0.0007, 0.015), ('B', 0.0002, 0.004), ('C', 0.0004, 0.010)])
    opt = PortfolioOptimizer(PortfolioSimulator(None, SimulationConfig(simulations=10)), None)
    opt.verbose = False
    return opt, assets, opt.efficient_frontier(assets, n_points=30)


def test_frontier_shape(setup):
    opt, assets, f = setup
    assert len(f['vol']) >= 25
    assert np.all(np.diff(f['ret']) >= -1e-9)          # returns increase along the curve
    assert np.all(np.diff(f['vol']) >= -1e-6)          # so does risk (upper branch)
    assert f['ret'][-1] == pytest.approx(max(f['asset_ret']), rel=1e-4)
    for w in f['weights']:
        assert w.min() >= -1e-9 and w.sum() == pytest.approx(1.0)


def test_min_vol_endpoint_matches_two_asset_formula():
    assets = make_assets([('A', 0.0006, 0.015), ('B', 0.0002, 0.004)], seed=1)
    opt = PortfolioOptimizer(PortfolioSimulator(None, SimulationConfig(simulations=10)), None)
    opt.verbose = False
    f = opt.efficient_frontier(assets, n_points=10)
    c = f['cov']
    w_a = (c[1, 1] - c[0, 1]) / (c[0, 0] + c[1, 1] - 2 * c[0, 1])
    w = np.array([w_a, 1 - w_a])
    assert f['vol'][0] == pytest.approx(np.sqrt(w @ c @ w), rel=1e-4)


def test_optimized_portfolios_sit_on_or_inside_frontier(setup):
    opt, assets, f = setup
    for res in (opt.optimize_sharpe_ratio(assets), opt.optimize_min_volatility(assets)):
        alloc = dict(zip(f['tickers'], res['allocations']))
        vol, ret = opt.frontier_point(f, alloc)
        frontier_vol = np.interp(ret, f['ret'], f['vol'])
        assert vol >= frontier_vol - 1e-3


def test_frontier_point_rejects_outside_assets(setup):
    opt, _, f = setup
    assert opt.frontier_point(f, {'A': 0.5, 'Z': 0.5}) is None
    assert opt.frontier_point(f, {'A': 1.0, 'Z': 0.0}) is not None
