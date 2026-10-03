"""
Withdrawals (decumulation): engine vs reference, closed-form depletion and
annuity balances, money-weighted return, Monte Carlo success rate.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_impl as ref
from backtester import Backtester
from engine import align_asset_prices, build_cash_flows, simulated_schedule
from portfolio_simulator import PortfolioSimulator
from run_config import RebalanceConfig, SimulationConfig

BDAYS = pd.bdate_range('2012-01-02', '2021-12-31')


def asset(ticker, prices):
    return {'ticker': ticker, 'full_data': pd.DataFrame({'Adj Close': prices})}


def random_assets(seed, n):
    rng = np.random.default_rng(seed)
    return [asset(f'T{k}', pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(BDAYS))), index=BDAYS))
            for k in range(n)], rng


@pytest.mark.parametrize('seed', range(12))
def test_engine_matches_reference_with_withdrawals(seed):
    n = 1 + seed % 3
    assets, rng = random_assets(seed, n)
    weights = rng.dirichlet(np.ones(n))
    prices = align_asset_prices(assets)
    dates = prices.index
    kw = dict(contribution_amount=[0.0, 400.0][seed % 2], contribution_frequency='monthly',
              withdrawal_amount=float(rng.integers(100, 1500)),
              withdrawal_frequency=['monthly', 'quarterly', 'annual'][seed % 3],
              withdrawal_start_years=[0.0, 3.0][(seed // 2) % 2],
              contribution_years=[None, 3.0][(seed // 2) % 2],
              cash_flow_growth=[0.0, 0.03][(seed // 3) % 2])
    freq = ['none', 'annual', 'quarterly'][seed % 3]
    years = np.asarray((dates[1:] - dates[0]).days, dtype=float) / 365.25
    flows = build_cash_flows(len(dates) - 1, years, dates=dates, **kw)
    flows_by_date = {dates[i + 1]: f for i, f in enumerate(flows) if f}
    bal_ref, twr_ref, trades_ref = ref.reference_backtest(
        prices, weights, 20000, rebalance_dates=ref.first_trading_days(dates, freq),
        flows_by_date=flows_by_date, cost_bps=[0, 10][seed % 2])
    res = Backtester().run_backtest(assets, weights, 20000,
                                    rebalance=RebalanceConfig(freq, transaction_cost_bps=[0, 10][seed % 2]), **kw)
    np.testing.assert_allclose(res['balance'].values, bal_ref.values, rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(res['twr'].values, twr_ref.values, rtol=1e-9)
    assert res['metrics']['Rebalances'] == trades_ref


def test_flow_schedule_phases_and_growth():
    n, dpy = 30 * 252, 252
    years = np.arange(1, n + 1) / dpy
    flows = build_cash_flows(n, years, contribution_amount=500, contribution_frequency='monthly',
                             withdrawal_amount=2000, withdrawal_frequency='monthly',
                             withdrawal_start_years=20, contribution_years=20, cash_flow_growth=0.03,
                             days_per_year=dpy)
    contrib, withdraw = flows[flows > 0], flows[flows < 0]
    assert len(contrib) == 20 * 12 and len(withdraw) == 10 * 12
    assert np.all(years[flows > 0] <= 20 + 1e-9) and np.all(years[flows < 0] > 20)
    # Amounts grow 3%/yr: the first withdrawal (year 20 + 1 month) is ~2000 * 1.03^20
    assert -withdraw[0] == pytest.approx(2000 * 1.03 ** years[np.flatnonzero(flows < 0)[0]])


def constant_growth_asset(annual, dates=BDAYS):
    days = np.asarray((dates - dates[0]).days, dtype=float)
    return asset('A', pd.Series(100 * (1 + annual) ** (days / 365.25), index=dates))


def test_depletion_date_and_flat_afterwards():
    """Flat prices, $1,000/month out of $10,000: empty after exactly 10 withdrawals."""
    flat = asset('A', pd.Series(100.0, index=BDAYS[:400]))
    res = Backtester().run_backtest([flat], [1.0], 10000, withdrawal_amount=1000, withdrawal_frequency='monthly')
    month_starts = sorted(ref.first_trading_days(BDAYS[:400], 'monthly'))
    assert res['depleted_on'] == month_starts[9]
    after = res['balance'][res['balance'].index >= res['depleted_on']]
    assert (after == 0).all()
    assert res['metrics']['Total Withdrawals'] == pytest.approx(10000)
    assert res['metrics']['CAGR'] == pytest.approx(0, abs=1e-12)    # flat prices: no investment return


def test_money_weighted_return_with_withdrawals_equals_constant_growth():
    a = constant_growth_asset(0.06)
    res = Backtester().run_backtest([a], [1.0], 100000, withdrawal_amount=500, withdrawal_frequency='monthly',
                                    contribution_amount=0)
    assert res['metrics']['CAGR'] == pytest.approx(0.06, abs=1e-9)
    assert res['metrics']['IRR'] == pytest.approx(0.06, abs=1e-6)
    assert res['depleted_on'] is None


def deterministic_sim(**kw):
    idx = pd.bdate_range('2010-01-01', '2019-12-31')
    p = pd.Series(100 * 1.0002 ** np.arange(len(idx)), index=idx)
    cfg = dict(initial_capital=100000, years=10, simulations=20, seed=1)
    cfg.update(kw)
    res = PortfolioSimulator(None, SimulationConfig(**cfg)).simulate_portfolio([asset('A', p)], [1.0])
    return res, 1.0002


def test_monte_carlo_sustainable_withdrawals_match_annuity_formula():
    res, g = deterministic_sim(withdrawal_amount=600, withdrawal_frequency='monthly')
    dpy = res['days_per_year']
    n = 10 * dpy
    steps = np.flatnonzero(simulated_schedule(n, 'monthly', dpy)) + 1
    expected = 100000 * g ** n - (600 * g ** (n - steps)).sum()
    np.testing.assert_allclose(res['final_values'], expected, rtol=1e-9)
    assert res['stats']['success_rate'] == 1.0
    assert res['stats']['median_withdrawn'] == pytest.approx(600 * len(steps))
    np.testing.assert_allclose(res['irr'], g ** dpy - 1, atol=1e-6)
    assert res['probabilities']['survival'][-1] == 1.0


def test_monte_carlo_unsustainable_withdrawals_deplete_on_schedule():
    res, g = deterministic_sim(withdrawal_amount=2000, withdrawal_frequency='monthly')
    dpy = res['days_per_year']
    n = 10 * dpy
    steps = np.flatnonzero(simulated_schedule(n, 'monthly', dpy)) + 1
    # Walk the balance forward to find the withdrawal that empties it
    bal, last = 100000.0, 0
    for s in steps:
        bal = bal * g ** (s - last) - 2000
        last = s
        if bal <= 0:
            break
    assert res['stats']['success_rate'] == 0.0
    assert res['stats']['median_depletion_year'] == pytest.approx(s / dpy)
    assert np.all(res['final_values'] == 0)
    survival = res['probabilities']['survival']
    assert survival[int(s / dpy) - 1] == 1.0 and survival[-1] == 0.0


def test_monte_carlo_success_rate_is_monotone_in_withdrawal_size():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range('2005-01-01', '2019-12-31')
    p = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.011, len(idx))), index=idx)
    rates = []
    for w in (300, 500, 700, 900):
        res = PortfolioSimulator(None, SimulationConfig(initial_capital=100000, years=25, simulations=1500, seed=3,
                                                        withdrawal_amount=w, withdrawal_frequency='monthly')
                                 ).simulate_portfolio([asset('A', p)], [1.0])
        rates.append(res['stats']['success_rate'])
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]


def test_report_shows_withdrawals_consistently(tmp_path):
    """End to end: Withdrawn column matches an independent recomputation, and the
    survival chart ends at the table's success rate."""
    import re
    import subprocess
    from pathlib import Path
    import test_report_numbers as trn
    from synthetic_data import synthetic_prices

    root = Path(__file__).resolve().parent.parent
    cfg = tmp_path / 'wd_config.py'
    cfg.write_text('''
from run_config import RunConfig, PortfolioConfig, SimulationConfig
config = RunConfig(name="Withdrawals", portfolios=[PortfolioConfig(name='60/40', allocations={'VOO': 0.6, 'BND': 0.4})],
    simulation=SimulationConfig(initial_capital=500000, start_date='2012-01-01', end_date='2023-12-29',
        withdrawal_amount=2500, withdrawal_frequency='monthly', cash_flow_growth=0.02,
        rebalance='annual', years=20, simulations=500, seed=2))
''')
    proc = subprocess.run([sys.executable, 'main.py', str(cfg), '--synthetic'], cwd=root, capture_output=True,
                          text=True, timeout=300, env=dict(os.environ, PYTHONPATH=str(root)))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    path = Path(root, re.search(r'Report saved to: (\S+)', proc.stdout).group(1))
    page = path.read_text()
    path.unlink()
    parser = trn.Tables()
    parser.feed(page)
    figs = trn.json.loads(re.search(r'const FIGS = (\{.*?\});\nconst CMAP', page, re.S).group(1))

    # Independent: reference backtest on raw synthetic prices
    raw = {t: synthetic_prices(t, pd.Timestamp('2012-01-01').date(), pd.Timestamp('2023-12-29').date())
           for t in ('VOO', 'BND')}
    prices = pd.concat([raw['VOO'].rename('VOO'), raw['BND'].rename('BND')], axis=1, join='inner').dropna()
    dates = prices.index
    years = np.asarray((dates[1:] - dates[0]).days, dtype=float) / 365.25
    flows = build_cash_flows(len(dates) - 1, years, withdrawal_amount=2500, withdrawal_frequency='monthly',
                             cash_flow_growth=0.02, dates=dates)
    bal, _, _ = ref.reference_backtest(prices, [0.6, 0.4], 500000,
                                       rebalance_dates=ref.first_trading_days(dates, 'annual'),
                                       flows_by_date={dates[i + 1]: f for i, f in enumerate(flows) if f})
    row = trn.rows_by_name(trn.table(parser.tables, 'Final balance'))['60/40']
    assert trn.num(row['Withdrawn']) == pytest.approx(-flows.sum(), abs=0.51)
    assert trn.num(row['Final balance']) == pytest.approx(bal.iloc[-1], abs=0.51)

    mc = trn.rows_by_name(trn.table(parser.tables, 'Success rate'))['60/40']
    survival = trn.decode(trn.traces(figs, 'mcsurvival', '60/40')[0]['y'])
    assert survival[-1] == pytest.approx(trn.num(mc['Success rate']), abs=0.0005)
    assert 'c-mcloss' not in figs and 'P(loss)' not in page
