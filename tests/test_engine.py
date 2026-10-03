"""
Tests for the shared portfolio engine: alignment, rebalancing policies,
signal-driven strategies, costs, and backtest/Monte Carlo consistency.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import Backtester
from engine import aligned_returns, simulated_schedule, vectorized_irr
from portfolio_simulator import PortfolioSimulator
from run_config import RebalanceConfig, SimulationConfig
from strategies import (BuyTheDipStrategy, DrawdownProtectionStrategy, DualMomentumStrategy,
                        StaticAllocationStrategy, create_strategy)


def asset(ticker, prices: pd.Series):
    df = pd.DataFrame({'Adj Close': prices.astype(float)})
    return {'ticker': ticker, 'name': ticker, 'full_data': df,
            'historical_returns': prices.pct_change().dropna()}


def random_prices(dates, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(mu, sigma, len(dates))), index=dates)


BDAYS = pd.bdate_range('2015-01-01', '2020-12-31')


class TestAlignment:
    def test_weekend_crypto_moves_land_in_monday_return(self):
        cal = pd.date_range('2021-01-01', '2021-03-31', freq='D')
        crypto = pd.Series(np.arange(len(cal), dtype=float) + 100, index=cal)
        equity = pd.Series(100.0, index=pd.bdate_range('2021-01-01', '2021-03-31'))
        r = aligned_returns([asset('EQ', equity), asset('BTC-USD', crypto)])
        monday = pd.Timestamp('2021-01-11')
        friday = pd.Timestamp('2021-01-08')
        assert r.loc[monday, 'BTC-USD'] == pytest.approx(crypto[monday] / crypto[friday] - 1)
        # Every crypto move is accounted for: compounded aligned returns = total move
        total = (1 + r['BTC-USD']).prod()
        assert total == pytest.approx(crypto[r.index[-1]] / crypto[pd.Timestamp('2021-01-01')])


class TestRebalancing:
    def setup_method(self):
        self.a = asset('A', random_prices(BDAYS, 0.0008, 0.012, 1))
        self.b = asset('B', random_prices(BDAYS, 0.0001, 0.004, 2))
        self.bt = Backtester()

    def test_static_strategy_matches_no_strategy(self):
        """The old engine gave different answers depending on code path; now identical."""
        for freq in ('none', 'monthly', 'annual'):
            plain = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance=freq)
            static = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance=freq,
                                          strategy=StaticAllocationStrategy())
            np.testing.assert_allclose(plain['values'], static['values'])

    def test_buy_and_hold_drifts(self):
        res = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance='none')
        pa, pb = self.a['full_data']['Adj Close'], self.b['full_data']['Adj Close']
        expected = 6000 * pa.iloc[-1] / pa.iloc[0] + 4000 * pb.iloc[-1] / pb.iloc[0]
        assert res['values'][-1] == pytest.approx(expected)
        assert res['metrics']['Rebalances'] == 0

    def test_monthly_rebalance_resets_weights_on_first_trading_day(self):
        res = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance='monthly')
        w = res['weights']
        firsts = w.groupby([w.index.year, w.index.month]).head(1).iloc[1:]
        np.testing.assert_allclose(firsts['A'], 0.6, atol=1e-12)
        assert res['metrics']['Rebalances'] == len(firsts)

    def test_threshold_band(self):
        band = RebalanceConfig(frequency='none', threshold=0.05)
        res = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance=band)
        assert res['metrics']['Rebalances'] > 0
        assert (res['weights']['A'] - 0.6).abs().max() <= 0.05 + 0.02  # one day of drift past the band
        assert all(e['trigger'] == 'threshold' for e in res['events'])

    def test_transaction_costs(self):
        free = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance='monthly')
        costly = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000,
                                      rebalance=RebalanceConfig('monthly', transaction_cost_bps=10))
        assert costly['metrics']['Transaction Costs'] > 0
        assert costly['values'][-1] < free['values'][-1]
        assert costly['metrics']['CAGR'] < free['metrics']['CAGR']

    def test_contributions_do_not_change_time_weighted_return(self):
        lump = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance='monthly')
        dca = self.bt.run_backtest([self.a, self.b], [0.6, 0.4], 10000, rebalance='monthly',
                                   contribution_amount=500, contribution_frequency='monthly')
        assert dca['metrics']['CAGR'] == pytest.approx(lump['metrics']['CAGR'], abs=2e-4)
        assert dca['metrics']['Stdev'] == pytest.approx(lump['metrics']['Stdev'], rel=0.02)
        assert dca['metrics']['Total Contributions'] == 500 * (len(dca['contributions']))
        assert len(dca['contributions']) == 71  # one per new month after Jan 2015


def crash_path():
    """Up 20%, crash 40%, recover to a new high."""
    up = np.linspace(100, 120, 250)
    down = np.linspace(120, 72, 120)
    rec = np.linspace(72, 140, 400)
    p = np.concatenate([up, down[1:], rec[1:]])
    return pd.Series(p, index=pd.bdate_range('2016-01-01', periods=len(p)))


class TestSignalStrategies:
    def test_drawdown_protection_switches_with_hysteresis(self):
        risky = crash_path()
        safe = pd.Series(100.0, index=risky.index)
        strat = DrawdownProtectionStrategy(dd_threshold=0.15, recovery_threshold=0.05,
                                           risk_off_allocation={'SAFE': 1.0})
        res = Backtester().run_backtest(
            [asset('RISKY', risky), asset('SAFE', safe)], [0.8, 0.2], 10000,
            rebalance='none', strategy=strat, apply_to='rebalance', check_frequency='daily')
        signals = [e for e in res['events'] if e['trigger'] == 'signal']
        assert len(signals) == 2, signals
        off, on = signals
        assert off['weights'][1] == pytest.approx(1.0)
        assert on['weights'][0] == pytest.approx(0.8)
        # Portfolio drawdown never gets far past the 15% trigger
        assert res['metrics']['Max Drawdown'] > -0.17
        # Signal uses the base-weight (80/20 daily) portfolio's drawdown
        ref = (1 + 0.8 * risky.pct_change().fillna(0)).cumprod()
        ref_dd = ref / ref.cummax() - 1
        assert ref_dd[off['date']] < -0.15 <= ref_dd[off['date'] - pd.tseries.offsets.BDay(1)]
        assert ref_dd[on['date']] > -0.05 >= ref_dd[on['date'] - pd.tseries.offsets.BDay(1)]
        # Sitting out the rebound has a cost: the portfolio itself is still ~15% down at re-entry
        assert res['drawdowns'][on['date']] < -0.10

    def test_buy_the_dip_uses_price_not_holdings(self):
        """Heavy contributions push holdings to new highs while the price is 40% down."""
        risky = crash_path()
        other = pd.Series(100.0, index=risky.index)
        strat = BuyTheDipStrategy('RISKY', threshold=0.20, aggressive_weight=0.9)
        res = Backtester().run_backtest(
            [asset('OTHER', other), asset('RISKY', risky)], [0.5, 0.5], 1000,
            rebalance='none', strategy=strat, contribution_amount=5000, contribution_frequency=5)
        tilted = [e for e in res['events'] if e['type'] == 'contribution']
        assert tilted, "dip contributions should be logged"
        dd = risky / risky.cummax() - 1
        assert all(dd[e['date']] < -0.20 for e in tilted)

    def test_dual_momentum_respects_lookback(self):
        strat = create_strategy('dual_momentum', {'equity_ticker': 'A', 'safe_ticker': 'B',
                                                  'lookback': 126})
        assert isinstance(strat, DualMomentumStrategy)
        assert strat.lookback_days == 126
        a = asset('A', crash_path())
        b = asset('B', pd.Series(100.0, index=crash_path().index) * np.linspace(1, 1.05, len(crash_path())))
        res = Backtester().run_backtest([a, b], [1.0, 0.0], 10000, rebalance='none',
                                        strategy=strat, apply_to='rebalance', check_frequency='daily')
        first = res['events'][0]
        assert first['trigger'] == 'signal'
        assert res['dates'].get_loc(first['date']) >= 126


class TestMonteCarloConsistency:
    def test_constant_history_gives_deterministic_paths(self):
        idx = pd.bdate_range('2015-01-01', '2020-12-31')
        p = pd.Series(100 * 1.0004 ** np.arange(len(idx)), index=idx)
        cfg = SimulationConfig(initial_capital=10000, years=3, simulations=50, seed=1,
                               contribution_amount=100, contribution_frequency='monthly')
        sim = PortfolioSimulator(None, cfg)
        res = sim.simulate_portfolio([asset('A', p)], [1.0])
        dpy = res['days_per_year']
        n = 3 * dpy
        contrib_steps = np.flatnonzero(simulated_schedule(n, 'monthly', dpy)) + 1
        expected = 10000 * 1.0004 ** n + (100 * 1.0004 ** (n - contrib_steps)).sum()
        np.testing.assert_allclose(res['final_values'], expected, rtol=1e-9)
        assert len(contrib_steps) == 36
        assert res['cagr'] == pytest.approx(1.0004 ** dpy - 1)
        np.testing.assert_allclose(res['irr'], 1.0004 ** dpy - 1, atol=1e-6)

    def test_crypto_only_portfolio_simulates_365_day_years(self):
        cal = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        btc = random_prices(cal, 0.001, 0.03, 5)
        sim = PortfolioSimulator(None, SimulationConfig(years=2, simulations=10, seed=0))
        res = sim.simulate_portfolio([asset('BTC-USD', btc)], [1.0])
        assert res['days_per_year'] == 365

    def test_vectorized_irr(self):
        years = np.arange(1, 25) / 12
        r = 0.07
        final = 1000 * (1 + r) ** 2 + (100 * (1 + r) ** (2 - years)).sum()
        assert vectorized_irr(1000, 100, years, np.array([final]), 2.0)[0] == pytest.approx(r, abs=1e-9)
