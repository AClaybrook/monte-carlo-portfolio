"""
Monte Carlo results vs closed-form answers.

- Geometric Brownian motion: ln(V_T / V_0) ~ Normal(n*mu, n*sigma^2), so terminal
  percentiles, median CAGR and the probability of loss each year are known exactly.
- Bootstrap: days are i.i.d. draws, so E[V_T] = V_0 * (1 + mean daily return)^n,
  and with contributions E[V_T] = V_0 g^n + c * sum(g^(n - s_i)).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import simulated_schedule
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig

BDAYS = pd.bdate_range('2010-01-01', '2019-12-31')


def asset(ticker, r):
    p = pd.Series(100 * np.cumprod(1 + r), index=BDAYS[:len(r)])
    return {'ticker': ticker, 'full_data': pd.DataFrame({'Adj Close': p})}


def sim(**kw):
    cfg = dict(initial_capital=10000, years=5, simulations=20000, seed=11)
    cfg.update(kw)
    return PortfolioSimulator(None, SimulationConfig(**cfg))


@pytest.fixture(scope='module')
def gbm_case():
    rng = np.random.default_rng(0)
    r = np.expm1(rng.normal(0.0004, 0.012, len(BDAYS)))
    a = asset('A', r)
    res = sim(method='geometric_brownian').simulate_portfolio([a], [1.0])
    logs = np.log1p(pd.Series(a['full_data']['Adj Close']).pct_change().dropna().values)
    return res, logs.mean(), logs.std(ddof=1)


def test_gbm_terminal_percentiles(gbm_case):
    res, mu, sd = gbm_case
    n = 5 * res['days_per_year']
    for q in (10, 25, 50, 75, 90):
        analytic = 10000 * np.exp(n * mu + stats.norm.ppf(q / 100) * sd * np.sqrt(n))
        assert res['stats']['percentiles']['final_value'][q] == pytest.approx(analytic, rel=0.015), q


def test_gbm_median_cagr(gbm_case):
    res, mu, sd = gbm_case
    assert res['stats']['median_cagr'] == pytest.approx(np.exp(mu * res['days_per_year']) - 1, abs=0.002)


def test_gbm_probability_of_loss_by_year(gbm_case):
    res, mu, sd = gbm_case
    dpy = res['days_per_year']
    for y, p in zip(res['probabilities']['years'], res['probabilities']['prob_loss']):
        n = y * dpy
        assert p == pytest.approx(stats.norm.cdf(-mu * n / (sd * np.sqrt(n))), abs=0.012), y


def test_gbm_volatility_per_path(gbm_case):
    res, mu, sd = gbm_case
    # Per-path realized vol of simple daily returns ~ sd * sqrt(days per year)
    assert np.median(res['volatility']) == pytest.approx(sd * np.sqrt(res['days_per_year']), rel=0.02)


def _se_check(sample, expected, k=4):
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    assert abs(sample.mean() - expected) < k * se, (sample.mean(), expected, se)


def test_bootstrap_expected_terminal_value():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, len(BDAYS))
    res = sim(method='bootstrap', years=3).simulate_portfolio([asset('A', r)], [1.0])
    hist = pd.Series(np.cumprod(1 + r)).pct_change().dropna().values
    n = 3 * res['days_per_year']
    _se_check(res['final_values'], 10000 * (1 + hist.mean()) ** n)


def test_bootstrap_expected_value_with_contributions():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, len(BDAYS))
    res = sim(method='bootstrap', years=3, contribution_amount=500,
              contribution_frequency='monthly').simulate_portfolio([asset('A', r)], [1.0])
    g = 1 + pd.Series(np.cumprod(1 + r)).pct_change().dropna().values.mean()
    dpy = res['days_per_year']
    n = 3 * dpy
    steps = np.flatnonzero(simulated_schedule(n, 'monthly', dpy)) + 1
    assert len(steps) == 36
    _se_check(res['final_values'], 10000 * g ** n + (500 * g ** (n - steps)).sum())
    assert res['stats']['total_invested'] == pytest.approx(10000 + 36 * 500)


def test_bootstrap_daily_rebalanced_two_assets():
    """With daily rebalancing the portfolio's daily return is w.r, so E[V_T] = V_0 (1 + w.mean_r)^n."""
    rng = np.random.default_rng(3)
    ra, rb = rng.normal(0.0006, 0.015, len(BDAYS)), rng.normal(0.0002, 0.004, len(BDAYS))
    res = sim(method='bootstrap', years=2).simulate_portfolio(
        [asset('A', ra), asset('B', rb)], [0.7, 0.3], rebalance='daily')
    ha = pd.Series(np.cumprod(1 + ra)).pct_change().dropna().values
    hb = pd.Series(np.cumprod(1 + rb)).pct_change().dropna().values
    n = 2 * res['days_per_year']
    _se_check(res['final_values'], 10000 * (1 + 0.7 * ha.mean() + 0.3 * hb.mean()) ** n)


def test_parametric_expected_value():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0005, 0.01, len(BDAYS))
    res = sim(method='parametric', years=2).simulate_portfolio([asset('A', r)], [1.0])
    hist = pd.Series(np.cumprod(1 + r)).pct_change().dropna().values
    _se_check(res['final_values'], 10000 * (1 + hist.mean()) ** (2 * res['days_per_year']))
