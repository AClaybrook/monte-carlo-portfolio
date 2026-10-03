"""
Performance metrics: the single source of every number in reports.

Conventions (chosen to match Portfolio Visualizer where it matters):
- CAGR is time-weighted and annualized by calendar span: (end/start)^(365.25/days) - 1.
  It never depends on how many rows a series has, so 24/7 crypto and 5-day
  equity calendars annualize correctly.
- Stdev, Sharpe, Sortino, beta, capture ratios and VaR use MONTHLY returns
  (PV's default), falling back to daily returns for histories under a year.
- Sharpe = mean(excess) / std(excess) * sqrt(periods/yr); excess over a
  constant annual risk-free rate converted to the period.
- Sortino = annualized mean excess / downside deviation, where downside
  deviation = sqrt(mean(min(excess, 0)^2)) over ALL periods.
- Max drawdown uses the daily series (deeper than PV's month-end drawdown).
- Best/Worst Year use calendar-year compounded returns (full years only when
  at least one exists).
- With contributions, every return-based metric uses the time-weighted index
  (cash flows removed). Money-weighted return is reported separately as IRR.
"""
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def years_between(start, end) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / DAYS_PER_YEAR


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observed rows per calendar year (~252 for equities, ~365 for crypto)."""
    if len(index) < 2:
        return float(TRADING_DAYS)
    span = years_between(index[0], index[-1])
    return (len(index) - 1) / span if span > 0 else float(TRADING_DAYS)


# ---------------------------------------------------------------------------
# Series transforms
# ---------------------------------------------------------------------------

def growth_index(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1 + returns.fillna(0)).cumprod()


def periodic_returns(index: pd.Series, freq: str = 'ME') -> pd.Series:
    """Returns over calendar periods from a value/growth series.

    The first period runs from the series' first value, so a partial first
    month/year is measured from the actual start.
    """
    ends = index.resample(freq).last().dropna()
    prev = ends.shift(1)
    prev.iloc[0] = index.iloc[0]
    return (ends / prev - 1).dropna()


def monthly_returns(index: pd.Series) -> pd.Series:
    return periodic_returns(index, 'ME')


def annual_returns(index: pd.Series) -> pd.DataFrame:
    """Calendar-year returns with a flag for partial first/last years."""
    r = periodic_returns(index, 'YE')
    first, last = index.index[0], index.index[-1]
    years = r.index.year
    partial = [(y == first.year and (first.month, first.day) > (1, 7)) or
               (y == last.year and (last.month, last.day) < (12, 24)) for y in years]
    return pd.DataFrame({'return': r.values, 'partial': partial}, index=years)


def monthly_returns_table(index: pd.Series) -> pd.DataFrame:
    """Year x month grid of returns plus a calendar-year column (PV style)."""
    m = monthly_returns(index)
    df = pd.DataFrame({'year': m.index.year, 'month': m.index.month, 'r': m.values})
    table = df.pivot(index='year', columns='month', values='r')
    table = table.reindex(columns=range(1, 13))
    table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    table['Year'] = annual_returns(index)['return']
    return table


def drawdown_series(index: pd.Series) -> pd.Series:
    return index / index.cummax() - 1


def max_drawdown(index: pd.Series) -> float:
    if len(index) < 2:
        return 0.0
    return float(drawdown_series(index).min())


def drawdown_periods(index: pd.Series, top: int = 10, min_depth: float = 0.0) -> List[Dict]:
    """Distinct peak-to-recovery episodes, deepest first."""
    dd = drawdown_series(index).values
    dates = index.index
    periods = []
    i, n = 0, len(dd)
    while i < n:
        if dd[i] < 0:
            start = i - 1 if i > 0 else 0
            j = i
            while j < n and dd[j] < 0:
                j += 1
            trough = start + int(np.argmin(dd[start:j]))
            recovered = j < n
            periods.append({
                'start': dates[start].date(),
                'trough': dates[trough].date(),
                'end': dates[j].date() if recovered else None,
                'depth': float(dd[trough]),
                'decline_days': (dates[trough] - dates[start]).days,
                'recovery_days': (dates[j] - dates[trough]).days if recovered else None,
                'underwater_days': ((dates[j] if recovered else dates[-1]) - dates[start]).days,
            })
            i = j
        else:
            i += 1
    periods = [p for p in periods if p['depth'] <= -min_depth]
    return sorted(periods, key=lambda p: p['depth'])[:top]


def rolling_annualized_return(index: pd.Series, years: float) -> pd.Series:
    """Annualized return over the trailing `years` calendar years at each date."""
    offset = pd.DateOffset(days=int(round(years * DAYS_PER_YEAR)))
    lookback = index.index - offset
    valid = lookback >= index.index[0]
    pos = index.index.searchsorted(lookback[valid], side='right') - 1
    past = index.values[pos]
    out = pd.Series(np.nan, index=index.index)
    out[valid] = (index.values[valid] / past) ** (1 / years) - 1
    return out


# ---------------------------------------------------------------------------
# Scalar metrics on a return series (period-based)
# ---------------------------------------------------------------------------

def _rf_per_period(rf_annual: float, ppy: float) -> float:
    return (1 + rf_annual) ** (1 / ppy) - 1


def cagr_from_returns(returns: pd.Series, ppy: float = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    growth = float((1 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (ppy / len(returns)) - 1


def cagr_from_index(index: pd.Series) -> float:
    """Calendar-span CAGR of a value/growth series."""
    yrs = years_between(index.index[0], index.index[-1])
    if yrs <= 0 or index.iloc[0] <= 0:
        return 0.0
    ratio = index.iloc[-1] / index.iloc[0]
    return float(ratio ** (1 / yrs) - 1) if ratio > 0 else -1.0


def volatility(returns: pd.Series, ppy: float = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(ppy))


def downside_deviation(returns: pd.Series, mar_per_period: float = 0.0,
                       ppy: float = TRADING_DAYS) -> float:
    if len(returns) == 0:
        return 0.0
    shortfall = np.minimum(returns.values - mar_per_period, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(ppy))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, ppy: float = TRADING_DAYS) -> float:
    excess = returns - _rf_per_period(rf_annual, ppy)
    sd = excess.std(ddof=1) if len(excess) > 1 else 0.0
    return float(excess.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, ppy: float = TRADING_DAYS) -> float:
    rf = _rf_per_period(rf_annual, ppy)
    dd = downside_deviation(returns, rf, ppy)
    return float((returns.mean() - rf) * ppy / dd) if dd > 0 else 0.0


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    return float(np.percentile(returns, level * 100)) if len(returns) else 0.0


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def capture_ratios(returns: pd.Series, bench: pd.Series):
    """PV-style up/down capture: ratio of geometric mean returns in up/down benchmark periods."""
    def geo(r):
        return (1 + r).prod() ** (1 / len(r)) - 1 if len(r) else np.nan

    up, down = bench > 0, bench < 0
    up_c = geo(returns[up]) / geo(bench[up]) if up.any() else np.nan
    down_c = geo(returns[down]) / geo(bench[down]) if down.any() else np.nan
    return float(up_c), float(down_c)


def regression_stats(returns: pd.Series, bench: pd.Series, rf_annual: float, ppy: float) -> Dict:
    rf = _rf_per_period(rf_annual, ppy)
    ex_p, ex_b = returns - rf, bench - rf
    if len(ex_p) < 3 or ex_b.std() == 0 or ex_p.std() == 0:
        return {'Beta': np.nan, 'Alpha': np.nan, 'R2': np.nan, 'Correlation': np.nan}
    res = stats.linregress(ex_b.values, ex_p.values)
    return {
        'Beta': float(res.slope),
        'Alpha': float((1 + res.intercept) ** ppy - 1),
        'R2': float(res.rvalue ** 2),
        'Correlation': float(np.corrcoef(returns, bench)[0, 1]),
    }


# ---------------------------------------------------------------------------
# Money-weighted return
# ---------------------------------------------------------------------------

def xirr(dates, amounts) -> float:
    """Annualized IRR. amounts: negative = money in, positive = money out/final value."""
    dates = pd.to_datetime(pd.Index(dates))
    t = np.asarray((dates - dates[0]).days, dtype=float) / DAYS_PER_YEAR
    a = np.asarray(amounts, dtype=float)

    def npv(r):
        return np.sum(a / (1 + r) ** t)

    lo, hi = -0.9999, 1.0
    while npv(hi) > 0 and hi < 1e6:
        hi *= 2
    if np.sign(npv(lo)) == np.sign(npv(hi)):
        return np.nan
    return float(optimize.brentq(npv, lo, hi, xtol=1e-10))


# ---------------------------------------------------------------------------
# Full metric set for a backtest
# ---------------------------------------------------------------------------

def compute_performance(balance: pd.Series, twr_index: pd.Series,
                        contributions: Optional[pd.Series] = None,
                        benchmark_index: Optional[pd.Series] = None,
                        benchmark_name: Optional[str] = None,
                        risk_free_rate: float = 0.0) -> Dict:
    """All headline metrics for one backtest.

    Args:
        balance: Portfolio value in dollars (includes contributions).
        twr_index: Time-weighted growth index (cash flows removed), same dates.
        contributions: External cash flows after the start, indexed by date
            (+ money added, - money withdrawn).
        benchmark_index: Benchmark growth index (aligned/overlapping dates).
        risk_free_rate: Annual rate used for Sharpe/Sortino/alpha.
    """
    start_balance = float(balance.iloc[0])
    end_balance = float(balance.iloc[-1])
    contributions = contributions if contributions is not None else pd.Series(dtype=float)
    contributions = contributions[contributions != 0]
    total_contrib = float(contributions[contributions > 0].sum())
    total_withdrawn = float(-contributions[contributions < 0].sum())

    monthly = monthly_returns(twr_index)
    if len(monthly) >= 12:
        rets, ppy, freq = monthly, 12.0, 'monthly'
    else:
        rets = twr_index.pct_change().dropna()
        ppy, freq = infer_periods_per_year(twr_index.index), 'daily'

    annual = annual_returns(twr_index)
    full_years = annual[~annual['partial']]
    year_pool = full_years if len(full_years) else annual

    cagr = cagr_from_index(twr_index)
    mdd = max_drawdown(twr_index)

    flow_dates = [balance.index[0], *contributions.index, balance.index[-1]]
    flow_amts = [-start_balance, *(-contributions.values), end_balance]
    irr = xirr(flow_dates, flow_amts) if len(contributions) else cagr

    m = {
        'Start Balance': start_balance,
        'Total Contributions': total_contrib,
        'Total Withdrawals': total_withdrawn,
        'End Balance': end_balance,
        'CAGR': cagr,
        'IRR': irr,
        'Stdev': volatility(rets, ppy),
        'Best Year': float(year_pool['return'].max()) if len(year_pool) else np.nan,
        'Worst Year': float(year_pool['return'].min()) if len(year_pool) else np.nan,
        'Max Drawdown': mdd,
        'Sharpe': sharpe_ratio(rets, risk_free_rate, ppy),
        'Sortino': sortino_ratio(rets, risk_free_rate, ppy),
        'Calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        'Downside Deviation': downside_deviation(rets, _rf_per_period(risk_free_rate, ppy), ppy),
        'VaR 5%': value_at_risk(rets),
        'CVaR 5%': conditional_var(rets),
        'Skewness': float(rets.skew()) if len(rets) > 2 else np.nan,
        'Excess Kurtosis': float(rets.kurt()) if len(rets) > 3 else np.nan,
        'Positive Periods': float((rets > 0).mean()) if len(rets) else np.nan,
        'Stats Frequency': freq,
        'Years': years_between(twr_index.index[0], twr_index.index[-1]),
        'Benchmark': benchmark_name,
    }

    if benchmark_index is not None and len(benchmark_index) > 1:
        b = benchmark_index.reindex(twr_index.index).ffill().dropna()
        p = twr_index.reindex(b.index)
        if freq == 'monthly':
            pr, br = monthly_returns(p), monthly_returns(b)
        else:
            pr, br = p.pct_change().dropna(), b.pct_change().dropna()
        pr, br = pr.align(br, join='inner')
        active = pr - br
        te = volatility(active, ppy)
        up_c, down_c = capture_ratios(pr, br)
        m.update(regression_stats(pr, br, risk_free_rate, ppy))
        m.update({
            'Active Return': cagr_from_index(p) - cagr_from_index(b),
            'Tracking Error': te,
            'Info Ratio': float(active.mean() * ppy / te) if te > 1e-9 else np.nan,
            'Upside Capture': up_c,
            'Downside Capture': down_c,
        })
    return m


# ---------------------------------------------------------------------------
# Period-count convenience wrapper (daily series, 252/yr unless told otherwise)
# ---------------------------------------------------------------------------

class QuantAnalytics:
    """Thin wrapper over the module functions for a single return series."""

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: float = TRADING_DAYS):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_cagr(self, returns: pd.Series) -> float:
        return cagr_from_returns(returns, self.periods_per_year)

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        return volatility(returns, self.periods_per_year if annualize else 1)

    def calculate_downside_deviation(self, returns: pd.Series, threshold: float = 0,
                                     annualize: bool = True) -> float:
        return downside_deviation(returns, threshold, self.periods_per_year if annualize else 1)

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        return sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year)

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        return sortino_ratio(returns, self.risk_free_rate, self.periods_per_year)

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        idx = growth_index(returns)
        return min(0.0, max_drawdown(pd.concat([pd.Series([1.0]), idx.reset_index(drop=True)])))

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        mdd = self.calculate_max_drawdown(returns)
        return self.calculate_cagr(returns) / abs(mdd) if mdd < 0 else 0.0

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return value_at_risk(returns, 1 - confidence)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        return conditional_var(returns, 1 - confidence)
