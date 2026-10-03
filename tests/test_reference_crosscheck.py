"""
Cross-check the production engine and metrics against the naive reference
implementation in reference_impl.py on many randomized scenarios.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant_analytics as qa
import reference_impl as ref
from backtester import Backtester
from engine import align_asset_prices
from run_config import RebalanceConfig
from strategies import ScheduledWeightsStrategy


def make_assets(seed, n_assets, crypto=False):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2014-01-01') + pd.Timedelta(days=int(rng.integers(0, 400)))
    end = start + pd.Timedelta(days=int(rng.integers(500, 2500)))
    assets = []
    for k in range(n_assets):
        is_crypto = crypto and k == n_assets - 1
        dates = pd.date_range(start, end, freq='D' if is_crypto else 'B')
        mu, sd = rng.uniform(-0.0003, 0.0012), rng.uniform(0.002, 0.03 if not is_crypto else 0.05)
        p = pd.Series(100 * np.cumprod(1 + rng.normal(mu, sd, len(dates))), index=dates)
        assets.append({'ticker': f'T{k}' + ('-USD' if is_crypto else ''),
                       'full_data': pd.DataFrame({'Adj Close': p})})
    return assets, rng


SCENARIOS = list(range(40))


@pytest.mark.parametrize('seed', SCENARIOS)
def test_engine_matches_reference(seed):
    n_assets = 1 + seed % 4
    assets, rng = make_assets(seed, n_assets, crypto=seed % 5 == 0)
    weights = rng.dirichlet(np.ones(n_assets))
    freq = ['none', 'monthly', 'quarterly', 'annual', 'weekly', 'daily'][seed % 6]
    contrib = [0.0, 250.0, 1000.0][seed % 3]
    contrib_freq = ['monthly', 'quarterly', 'annual', 21][seed % 4]
    threshold = [None, None, 0.05, 0.10][seed % 4] if n_assets > 1 else None
    cost = [0.0, 0.0, 5.0, 25.0][(seed // 2) % 4]

    prices = align_asset_prices(assets)
    dates = prices.index
    if isinstance(contrib_freq, int):
        contribution_dates = {dates[i] for i in range(contrib_freq, len(dates), contrib_freq)}
    else:
        contribution_dates = ref.first_trading_days(dates, contrib_freq)
    bal_ref, twr_ref, trades_ref = ref.reference_backtest(
        prices, weights, 10000, contrib, contribution_dates,
        ref.first_trading_days(dates, freq), threshold, cost)

    res = Backtester().run_backtest(
        assets, weights, 10000, contribution_amount=contrib, contribution_frequency=contrib_freq,
        rebalance=RebalanceConfig(freq, threshold=threshold, transaction_cost_bps=cost))

    np.testing.assert_allclose(res['balance'].values, bal_ref.values, rtol=1e-10)
    np.testing.assert_allclose(res['twr'].values, twr_ref.values, rtol=1e-10)
    assert res['metrics']['Rebalances'] == trades_ref
    assert res['metrics']['Total Contributions'] == pytest.approx(contrib * len(contribution_dates) if contrib else 0)


@pytest.mark.parametrize('seed', range(10))
def test_signal_strategy_matches_reference(seed):
    """A date-driven strategy trading holdings (apply_to='rebalance') vs the reference."""
    assets, rng = make_assets(100 + seed, 3)
    prices = align_asset_prices(assets)
    dates = prices.index
    switch = sorted(rng.choice(dates[5:-5], size=4, replace=False))
    schedule = [(d, rng.dirichlet(np.ones(3))) for d in switch]
    base = np.array([0.4, 0.4, 0.2])

    def target(d):
        live = [w for s, w in schedule if s <= d]
        return live[-1] if live else base

    freq = ['none', 'annual', 'monthly'][seed % 3]
    bal_ref, twr_ref, trades_ref = ref.reference_backtest(
        prices, base, 10000, 300.0, ref.first_trading_days(dates, 'monthly'),
        ref.first_trading_days(dates, freq), target_by_date=target)
    res = Backtester().run_backtest(
        assets, base, 10000, rebalance=freq, contribution_amount=300.0, contribution_frequency='monthly',
        strategy=ScheduledWeightsStrategy(schedule), apply_to='rebalance', check_frequency='daily')
    np.testing.assert_allclose(res['balance'].values, bal_ref.values, rtol=1e-10)
    np.testing.assert_allclose(res['twr'].values, twr_ref.values, rtol=1e-10)
    assert res['metrics']['Rebalances'] == trades_ref


@pytest.mark.parametrize('seed', range(25))
def test_metrics_match_first_principles(seed):
    rng = np.random.default_rng(1000 + seed)
    freq = 'D' if seed % 3 == 0 else 'B'
    start = pd.Timestamp('2010-01-01') + pd.Timedelta(days=int(rng.integers(0, 300)))
    dates = pd.date_range(start, periods=int(rng.integers(400, 3000)), freq=freq)
    idx = pd.Series(np.cumprod(1 + rng.normal(0.0004, rng.uniform(0.003, 0.03), len(dates))), index=dates)
    idx = idx / idx.iloc[0]
    bench = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))), index=dates)
    bench = bench / bench.iloc[0]
    rf = float(rng.uniform(0, 0.05))

    m = qa.compute_performance(idx * 10000, idx, benchmark_index=bench, benchmark_name='B', risk_free_rate=rf)
    monthly = ref.monthly_returns(idx)
    years = ref.calendar_year_returns(idx)

    assert m['CAGR'] == pytest.approx(ref.cagr(idx), rel=1e-10)
    assert m['Max Drawdown'] == pytest.approx(ref.max_drawdown(list(idx.values)), rel=1e-10)
    if len(monthly) >= 12:
        assert m['Stats Frequency'] == 'monthly'
        assert m['Stdev'] == pytest.approx(ref.stdev(monthly) * np.sqrt(12), rel=1e-9)
        assert m['Sharpe'] == pytest.approx(ref.sharpe(monthly, rf), rel=1e-9)
        assert m['Sortino'] == pytest.approx(ref.sortino(monthly, rf), rel=1e-9)
        bm = ref.monthly_returns(bench)
        corr = np.corrcoef(monthly, bm)[0, 1]
        assert m['Correlation'] == pytest.approx(corr, rel=1e-9)
        active = [a - b for a, b in zip(monthly, bm)]
        assert m['Tracking Error'] == pytest.approx(ref.stdev(active) * np.sqrt(12), rel=1e-9)
    # Best/Worst year come from full calendar years when there are any
    first, last = idx.index[0], idx.index[-1]
    full = {y: r for y, r in years.items()
            if not (y == first.year and (first.month, first.day) > (1, 7))
            and not (y == last.year and (last.month, last.day) < (12, 24))}
    pool = full or years
    assert m['Best Year'] == pytest.approx(max(pool.values()), rel=1e-10)
    assert m['Worst Year'] == pytest.approx(min(pool.values()), rel=1e-10)


@pytest.mark.parametrize('seed', range(8))
def test_irr_matches_first_principles(seed):
    assets, rng = make_assets(200 + seed, 2)
    res = Backtester().run_backtest(assets, [0.5, 0.5], 5000, rebalance='quarterly',
                                    contribution_amount=float(rng.integers(100, 2000)),
                                    contribution_frequency='monthly')
    bal, contrib = res['balance'], res['contributions']
    flows = [(bal.index[0], -bal.iloc[0])] + [(d, -a) for d, a in contrib.items()] + [(bal.index[-1], bal.iloc[-1])]
    assert res['metrics']['IRR'] == pytest.approx(ref.irr_bisect(flows), abs=1e-8)
