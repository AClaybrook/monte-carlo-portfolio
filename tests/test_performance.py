"""
Golden tests for quant_analytics: hand-computed values for every headline metric.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant_analytics as qa


def series(values, dates):
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


class TestAnnualization:
    def test_cagr_uses_calendar_span(self):
        s = series([100, 121], ['2020-01-01', '2022-01-01'])
        years = 731 / 365.25
        assert qa.cagr_from_index(s) == pytest.approx(1.21 ** (1 / years) - 1, abs=1e-12)

    def test_crypto_and_equity_calendars_agree(self):
        """Same start/end value over the same dates -> same CAGR regardless of row count."""
        daily = pd.date_range('2018-01-01', '2023-01-01', freq='D')
        bdays = pd.bdate_range('2018-01-01', '2023-01-02')
        crypto = pd.Series(np.linspace(100, 300, len(daily)), index=daily)
        equity = pd.Series(np.linspace(100, 300, len(bdays)), index=bdays)
        equity.index = equity.index.where(equity.index <= daily[-1], daily[-1])
        assert qa.cagr_from_index(crypto) == pytest.approx(qa.cagr_from_index(equity), abs=1e-9)

    def test_infer_periods_per_year(self):
        assert qa.infer_periods_per_year(pd.date_range('2015-01-01', '2024-12-31')) == pytest.approx(365.25, rel=1e-3)
        assert qa.infer_periods_per_year(pd.bdate_range('2015-01-01', '2024-12-31')) == pytest.approx(261, rel=0.01)


class TestPeriodicReturns:
    def test_monthly_returns_from_month_ends(self):
        s = series([100, 105, 110, 99, 120],
                   ['2021-01-04', '2021-01-29', '2021-02-15', '2021-02-26', '2021-03-31'])
        m = qa.monthly_returns(s)
        np.testing.assert_allclose(m.values, [0.05, 99 / 105 - 1, 120 / 99 - 1])

    def test_annual_returns_flags_partial_years(self):
        idx = pd.bdate_range('2019-06-03', '2022-03-31')
        s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        a = qa.annual_returns(s)
        assert list(a.index) == [2019, 2020, 2021, 2022]
        assert list(a['partial']) == [True, False, False, True]
        assert a.loc[2020, 'return'] == pytest.approx(s['2020'].iloc[-1] / s['2019'].iloc[-1] - 1)

    def test_rolling_annualized_return(self):
        idx = pd.date_range('2010-01-01', '2016-01-01', freq='D')
        growth = 1.10 ** ((idx - idx[0]).days / 365.25)
        r = qa.rolling_annualized_return(pd.Series(growth, index=idx), 3)
        assert r.iloc[:1000].isna().all()
        assert r.dropna().values == pytest.approx(0.10, abs=1e-3)


class TestRiskRatios:
    rets = pd.Series([0.01, 0.02, -0.01, 0.03])

    def test_sharpe_hand_computed(self):
        expected = self.rets.mean() / self.rets.std(ddof=1) * np.sqrt(12)
        assert qa.sharpe_ratio(self.rets, 0.0, 12) == pytest.approx(expected)

    def test_sharpe_subtracts_period_risk_free(self):
        rf_m = 1.06 ** (1 / 12) - 1
        ex = self.rets - rf_m
        assert qa.sharpe_ratio(self.rets, 0.06, 12) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(12))

    def test_sortino_hand_computed(self):
        dd = np.sqrt(np.mean(np.minimum(self.rets.values, 0) ** 2)) * np.sqrt(12)
        assert qa.sortino_ratio(self.rets, 0.0, 12) == pytest.approx(self.rets.mean() * 12 / dd)

    def test_capture_ratios(self):
        bench = pd.Series([0.02, -0.01, 0.03, -0.02])
        port = bench * 2
        up, down = qa.capture_ratios(port, bench)
        geo = lambda r: (1 + r).prod() ** (1 / len(r)) - 1
        assert up == pytest.approx(geo(port[bench > 0]) / geo(bench[bench > 0]))
        assert down == pytest.approx(geo(port[bench < 0]) / geo(bench[bench < 0]))
        assert up > 1 and down > 1

    def test_beta_of_levered_series(self):
        rng = np.random.default_rng(0)
        bench = pd.Series(rng.normal(0.01, 0.04, 120))
        reg = qa.regression_stats(bench * 1.5, bench, 0.0, 12)
        assert reg['Beta'] == pytest.approx(1.5)
        assert reg['R2'] == pytest.approx(1.0)


class TestDrawdowns:
    def test_drawdown_periods(self):
        s = series([100, 110, 99, 110, 120, 90, 100],
                   pd.date_range('2020-01-01', periods=7, freq='D'))
        periods = qa.drawdown_periods(s)
        assert len(periods) == 2
        worst, second = periods
        assert worst['depth'] == pytest.approx(-0.25)
        assert worst['end'] is None
        assert second['depth'] == pytest.approx(-0.10)
        assert str(second['start']) == '2020-01-02'
        assert str(second['end']) == '2020-01-04'


class TestMoneyWeighted:
    def test_xirr_single_period(self):
        assert qa.xirr(['2021-01-01', '2022-01-01'], [-1000, 1100]) == pytest.approx(
            1.1 ** (365.25 / 365) - 1, abs=1e-8)

    def test_xirr_with_contribution(self):
        r = 0.08
        dates = pd.to_datetime(['2020-01-01', '2020-07-01', '2022-01-01'])
        t = (dates - dates[0]).days / 365.25
        final = 1000 * (1 + r) ** t[2] + 500 * (1 + r) ** (t[2] - t[1])
        assert qa.xirr(dates, [-1000, -500, final]) == pytest.approx(r, abs=1e-8)


class TestComputePerformance:
    def _paths(self):
        idx = pd.bdate_range('2015-01-01', '2020-12-31')
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
        r.iloc[0] = 0.0
        return qa.growth_index(r)

    def test_contributions_do_not_change_twr_metrics(self):
        twr = self._paths()
        bal_lump = twr * 10000
        contrib = pd.Series(500.0, index=twr.index[21::21])
        # Balance including contributions: each contribution grows with the index afterwards
        units = (contrib / twr.reindex(contrib.index)).reindex(twr.index).fillna(0).cumsum()
        bal_dca = bal_lump + units * twr
        a = qa.compute_performance(bal_lump, twr)
        b = qa.compute_performance(bal_dca, twr, contributions=contrib)
        for k in ('CAGR', 'Stdev', 'Sharpe', 'Max Drawdown', 'Best Year'):
            assert a[k] == pytest.approx(b[k]), k
        assert b['Total Contributions'] == pytest.approx(contrib.sum())

    def test_irr_equals_cagr_when_growth_is_constant(self):
        idx = pd.date_range('2015-01-01', '2020-12-31', freq='D')
        twr = pd.Series(1.07 ** ((idx - idx[0]).days / 365.25), index=idx)
        contrib = pd.Series(500.0, index=idx[30::30])
        units = (contrib / twr.reindex(contrib.index)).reindex(idx).fillna(0).cumsum()
        bal = twr * 10000 + units * twr
        m = qa.compute_performance(bal, twr, contributions=contrib)
        assert m['CAGR'] == pytest.approx(0.07, abs=1e-9)
        assert m['IRR'] == pytest.approx(0.07, abs=1e-6)

    def test_benchmark_against_itself(self):
        twr = self._paths()
        m = qa.compute_performance(twr * 1e4, twr, benchmark_index=twr, benchmark_name='X')
        assert m['Beta'] == pytest.approx(1.0)
        assert m['Tracking Error'] == pytest.approx(0.0, abs=1e-12)
        assert m['Upside Capture'] == pytest.approx(1.0)
        assert m['Active Return'] == pytest.approx(0.0, abs=1e-12)
        assert m['Stats Frequency'] == 'monthly'
