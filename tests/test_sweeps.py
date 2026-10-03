"""Parameter sweep tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_config import SimulationConfig, StrategyConfig, SweepConfig
from sweeps import metric_grid, run_sweep
from synthetic_data import synthetic_ohlcv


def asset_map(*tickers):
    out = {}
    for t in tickers:
        df = synthetic_ohlcv(t, pd.Timestamp('2012-01-01').date(), pd.Timestamp('2020-12-31').date())
        out[t] = {'ticker': t, 'full_data': df, 'historical_returns': df['Adj Close'].pct_change().dropna()}
    return out


SIM = SimulationConfig(initial_capital=10000, years=2, simulations=100, seed=3,
                       contribution_amount=200, contribution_frequency='monthly', rebalance='none')


def test_inert_strategy_cells_equal_baseline():
    sweep = SweepConfig('inert', {'VOO': 0.7, 'QQQ': 0.3},
                        StrategyConfig('buy_the_dip', {'target_ticker': 'QQQ', 'aggressive_weight': 0.9}),
                        grid={'threshold': [0.99, 0.995]})
    res = run_sweep(sweep, asset_map('VOO', 'QQQ'), SIM)
    for cell in res['cells']:
        for k, v in res['baseline'].items():
            assert cell['metrics'][k] == pytest.approx(v), k


def test_grid_orientation_and_common_random_numbers():
    sweep = SweepConfig('dd', {'VOO': 0.8, 'BND': 0.2},
                        StrategyConfig('drawdown_protection', apply_to='rebalance', check_frequency='daily',
                                       params={'risk_off_allocation': {'BND': 1.0}}),
                        grid={'threshold': [0.10, 0.20, 0.30], 'recovery_threshold': [0.02, 0.02]},
                        simulations=100)
    res = run_sweep(sweep, asset_map('VOO', 'BND'), SIM)
    z = metric_grid(res, 'CAGR')
    assert z.shape == (2, 3)                      # rows = second key, columns = first key
    for c in res['cells']:
        col = res['values'][0].index(c['params']['threshold'])
        assert c['metrics']['CAGR'] in z[:, col]
    # Duplicate parameter values -> identical cells, including Monte Carlo (shared seed)
    np.testing.assert_allclose(z[0], z[1])
    mc = metric_grid(res, 'mc_median_cagr')
    np.testing.assert_allclose(mc[0], mc[1])
    # The strategy actually changes something
    assert not np.allclose(z[0], res['baseline']['CAGR'])


@pytest.mark.parametrize('grid', [{}, {'a': [1], 'b': [2], 'c': [3]}, {'a': []}])
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        SweepConfig('bad', {'VOO': 1.0}, StrategyConfig('static'), grid=grid)
