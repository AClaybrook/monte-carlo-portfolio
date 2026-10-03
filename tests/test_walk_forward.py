"""Walk-forward optimizer check tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import Backtester
from portfolio_optimizer import PortfolioOptimizer
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig
from strategies import ScheduledWeightsStrategy
from walk_forward import refit_schedule, run_walk_forward

BDAYS = pd.bdate_range('2012-01-02', '2019-12-31')
SIM = SimulationConfig(initial_capital=10000, simulations=10, years=1, rebalance='annual',
                       contribution_amount=100, contribution_frequency='monthly')


def asset(ticker, mu, sigma, seed, dates=BDAYS):
    rng = np.random.default_rng(seed)
    p = pd.Series(100 * np.cumprod(1 + rng.normal(mu, sigma, len(dates))), index=dates)
    return {'ticker': ticker, 'full_data': pd.DataFrame({'Adj Close': p}),
            'historical_returns': p.pct_change().dropna()}


A, B = asset('A', 0.0006, 0.012, 1), asset('B', 0.0002, 0.004, 2)


def test_scheduled_weights_switch_on_dates():
    schedule = [(pd.Timestamp('2014-01-02'), [1.0, 0.0]), (pd.Timestamp('2016-06-01'), [0.2, 0.8])]
    res = Backtester().run_backtest([A, B], [0.5, 0.5], 10000, rebalance='none',
                                    strategy=ScheduledWeightsStrategy(schedule), apply_to='rebalance',
                                    check_frequency='daily')
    signals = [e for e in res['events'] if e['trigger'] == 'signal']
    assert [e['date'] for e in signals] == [pd.Timestamp('2014-01-02'), pd.Timestamp('2016-06-01')]
    np.testing.assert_allclose(res['weights'].loc['2016-06-01'].values, [0.2, 0.8])


def test_refits_only_see_the_past():
    seen = []

    def fit(train):
        seen.append(max(a['full_data'].index.max() for a in train))
        return {'allocations': [0.5, 0.5]}

    schedule = refit_schedule(fit, [A, B], None, None, train_years=3, test_years=1)
    assert len(schedule) == 5                     # 2015, 2016, 2017, 2018, 2019
    for (t, _), last_seen in zip(schedule, seen):
        assert last_seen < t
        assert t - last_seen < pd.Timedelta(days=5)


def test_constant_fit_matches_in_sample():
    """If every refit returns the in-sample weights, walk-forward equals in-sample exactly."""
    wf = run_walk_forward('const', lambda train: {'allocations': [0.6, 0.4]}, [A, B], [0.6, 0.4], SIM,
                          train_years=2, test_years=1)
    np.testing.assert_allclose(wf['oos']['values'], wf['in_sample']['values'], rtol=1e-12)
    assert wf['mean_refit_turnover'] == pytest.approx(0.0)


def test_too_short_history_returns_none():
    short = [asset('A', 0.0005, 0.01, 3, BDAYS[:300]), asset('B', 0.0002, 0.004, 4, BDAYS[:300])]
    assert run_walk_forward('x', lambda t: {'allocations': [0.5, 0.5]}, short, [0.5, 0.5], SIM,
                            train_years=3) is None


def test_optimizer_cache_distinguishes_slices():
    """Refits pass sliced copies of the same tickers; the cache must not reuse another window."""
    calm, wild = BDAYS[:1000], BDAYS[1000:]
    rng = np.random.default_rng(5)
    a_r = np.concatenate([rng.normal(0, 0.002, 1000), rng.normal(0, 0.03, len(wild))])
    b_r = np.concatenate([rng.normal(0, 0.03, 1000), rng.normal(0, 0.002, len(wild))])
    mk = lambda t, r: {'ticker': t, 'full_data': pd.DataFrame(
        {'Adj Close': pd.Series(100 * np.cumprod(1 + r), index=BDAYS)})}
    full = [mk('A', a_r), mk('B', b_r)]
    opt = PortfolioOptimizer(PortfolioSimulator(None, SimulationConfig(simulations=10)), None)
    opt.verbose = False
    slice_ = lambda a, idx: {'ticker': a['ticker'], 'full_data': a['full_data'].loc[idx]}
    w_calm = opt.optimize_min_volatility([slice_(a, calm) for a in full])['allocations']
    w_wild = opt.optimize_min_volatility([slice_(a, wild) for a in full])['allocations']
    assert w_calm[0] > 0.9 and w_wild[1] > 0.9
