"""
Deliberately naive reference implementations used only by tests.

Written from first principles with a different method than the production code
(share counts and date loops instead of vectorized dollar holdings and step
schedules), so agreement between the two is meaningful evidence.
"""
import math
from datetime import date

import numpy as np
import pandas as pd


def first_trading_days(dates: pd.DatetimeIndex, frequency: str) -> set:
    """Dates that start a new calendar period (excluding the very first date)."""
    if frequency in (None, 'none'):
        return set()
    if frequency == 'daily':
        return set(dates[1:])
    out = set()
    for prev, cur in zip(dates[:-1], dates[1:]):
        if frequency == 'monthly':
            new = (cur.year, cur.month) != (prev.year, prev.month)
        elif frequency == 'quarterly':
            new = (cur.year, (cur.month - 1) // 3) != (prev.year, (prev.month - 1) // 3)
        elif frequency == 'annual':
            new = cur.year != prev.year
        elif frequency == 'weekly':
            new = cur.isocalendar()[:2] != prev.isocalendar()[:2]
        else:
            raise ValueError(frequency)
        if new:
            out.add(cur)
    return out


def reference_backtest(prices: pd.DataFrame, weights, initial: float,
                       contribution: float = 0.0, contribution_dates: set = (),
                       rebalance_dates: set = (), threshold: float = None,
                       cost_bps: float = 0.0, target_by_date=None):
    """Share-based simulation. Order each day: mark to market, contribute,
    rebalance (calendar, band breach or target change), record.

    target_by_date: optional callable(date) -> target weights (signal strategy).
    Returns (balance Series, time-weighted index Series, number of trades).
    """
    w = np.asarray(weights, dtype=float) / sum(weights)
    dates = prices.index
    shares = w * initial / prices.iloc[0].values
    target = w.copy()
    balance, twr = [initial], [1.0]
    trades = 0
    prev_value = initial
    for d in dates[1:]:
        px = prices.loc[d].values
        flow = 0.0
        new_target = target if target_by_date is None else np.asarray(target_by_date(d), dtype=float)
        if d in contribution_dates and contribution:
            shares = shares + w * contribution / px
            flow = contribution
        value = float(shares @ px)
        current_w = shares * px / value
        do_trade = (d in rebalance_dates
                    or (threshold is not None and np.max(np.abs(current_w - new_target)) > threshold)
                    or np.max(np.abs(new_target - target)) > 1e-12)
        target = new_target
        if do_trade:
            cost = cost_bps / 1e4 * np.sum(np.abs(target * value - shares * px))
            shares = target * (value - cost) / px
            trades += 1
        end_value = float(shares @ px)
        twr.append(twr[-1] * (end_value - flow) / prev_value)
        balance.append(end_value)
        prev_value = end_value
    return pd.Series(balance, index=dates), pd.Series(twr, index=dates), trades


# ---------------------------------------------------------------------------
# Metrics from first principles (plain Python over grouped month-end values)
# ---------------------------------------------------------------------------

def month_end_values(index: pd.Series):
    """[(year, month, last value)] in order."""
    out = []
    for d, v in index.items():
        key = (d.year, d.month)
        if out and out[-1][0] == key:
            out[-1] = (key, v)
        else:
            out.append((key, v))
    return out


def monthly_returns(index: pd.Series):
    ends = month_end_values(index)
    prev = index.iloc[0]
    rets = []
    for _, v in ends:
        rets.append(v / prev - 1)
        prev = v
    return rets


def calendar_year_returns(index: pd.Series):
    """{year: return} using the last value of each year (first year from the start value)."""
    last = {}
    for d, v in index.items():
        last[d.year] = v
    out, prev = {}, index.iloc[0]
    for y in sorted(last):
        out[y] = last[y] / prev - 1
        prev = last[y]
    return out


def mean(x):
    return sum(x) / len(x)


def stdev(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def sharpe(monthly, rf_annual):
    rf = (1 + rf_annual) ** (1 / 12) - 1
    ex = [r - rf for r in monthly]
    return mean(ex) / stdev(ex) * math.sqrt(12)


def sortino(monthly, rf_annual):
    rf = (1 + rf_annual) ** (1 / 12) - 1
    ex = [r - rf for r in monthly]
    dd = math.sqrt(sum(min(e, 0) ** 2 for e in ex) / len(ex)) * math.sqrt(12)
    return mean(ex) * 12 / dd


def max_drawdown(values):
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


def cagr(index: pd.Series):
    days = (index.index[-1] - index.index[0]).days
    return (index.iloc[-1] / index.iloc[0]) ** (365.25 / days) - 1


def npv(rate, dated_flows):
    t0 = dated_flows[0][0]
    return sum(a / (1 + rate) ** ((d - t0).days / 365.25) for d, a in dated_flows)


def irr_bisect(dated_flows, lo=-0.99, hi=5.0):
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid, dated_flows) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
