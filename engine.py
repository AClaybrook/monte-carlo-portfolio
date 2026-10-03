"""
Portfolio engine shared by the historical backtester and the Monte Carlo simulator.

One day-step loop, vectorized over paths (1 historical path, or N simulated ones),
so the same portfolio definition produces the same math in both:

  1. apply each asset's return to holdings
  2. update price-based indicators (drawdowns, trailing returns)
  3. evaluate the strategy on decision days
  4. add contributions (split by strategy or base weights)
  5. rebalance on calendar days, drift-band breaches, or strategy signal changes
  6. record the time-weighted return for the day (cash flows excluded, costs included)
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies import AllocationStrategy, MarketContext

PERIODS_PER_YEAR = {'weekly': 52, 'monthly': 12, 'quarterly': 4, 'annual': 1}


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------

def _price_series(asset) -> pd.Series:
    df = asset.get('full_data')
    if df is not None and not df.empty:
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        s = df[col]
    else:
        r = asset['historical_returns'].dropna()
        start = pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)])
        s = pd.concat([start, (1 + r).cumprod()])
    return s.astype(float).rename(asset['ticker'])


def align_asset_prices(assets, start=None, end=None) -> pd.DataFrame:
    """Prices on the dates every asset traded.

    Prices are aligned BEFORE returns are computed, so a 24/7 asset's weekend
    move lands in Monday's return instead of being dropped.
    """
    prices = pd.concat([_price_series(a) for a in assets], axis=1, join='inner')
    prices = prices[~prices.index.duplicated(keep='last')].sort_index().dropna()
    if start is not None:
        prices = prices[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices[prices.index <= pd.Timestamp(end)]
    return prices


def aligned_returns(assets, start=None, end=None) -> pd.DataFrame:
    return align_asset_prices(assets, start, end).pct_change().iloc[1:]


# ---------------------------------------------------------------------------
# Schedules: bool arrays of length n_days; True at step t means "after the
# return of day t is applied" (i.e. on dates[t + 1] for historical runs).
# ---------------------------------------------------------------------------

def calendar_schedule(dates: pd.DatetimeIndex, frequency: Optional[str]) -> np.ndarray:
    """First trading day of each new calendar period."""
    n = len(dates) - 1
    if not frequency or frequency == 'none':
        return np.zeros(n, dtype=bool)
    if frequency == 'daily':
        return np.ones(n, dtype=bool)
    if frequency == 'weekly':
        key = np.asarray(dates.to_period('W').astype(str))
    elif frequency == 'monthly':
        key = dates.year * 12 + dates.month
    elif frequency == 'quarterly':
        key = dates.year * 4 + dates.quarter
    elif frequency == 'annual':
        key = dates.year
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    key = np.asarray(key)
    return key[1:] != key[:-1]


def step_schedule(n_days: int, every: int) -> np.ndarray:
    return (np.arange(1, n_days + 1) % max(1, int(every))) == 0


def simulated_schedule(n_days: int, frequency: Optional[str], days_per_year: int) -> np.ndarray:
    if not frequency or frequency == 'none':
        return np.zeros(n_days, dtype=bool)
    if frequency == 'daily':
        return np.ones(n_days, dtype=bool)
    per_year = PERIODS_PER_YEAR[frequency]
    # Evenly spaced so annual events land exactly on year boundaries
    steps = np.round(np.arange(1, n_days * per_year // days_per_year + 1) * days_per_year / per_year)
    out = np.zeros(n_days, dtype=bool)
    steps = steps[(steps >= 1) & (steps <= n_days)].astype(int)
    out[steps - 1] = True
    return out


def contribution_schedule(frequency, n_days: int, dates: Optional[pd.DatetimeIndex] = None,
                          days_per_year: int = 252) -> np.ndarray:
    """int = every N rows; str = calendar ('monthly', ...)."""
    if isinstance(frequency, (int, np.integer)):
        return step_schedule(n_days, frequency)
    if dates is not None:
        return calendar_schedule(dates, frequency)
    return simulated_schedule(n_days, frequency, days_per_year)


# ---------------------------------------------------------------------------
# Return sources
# ---------------------------------------------------------------------------

class HistoricalReturns:
    def __init__(self, returns: np.ndarray):
        self.returns = np.asarray(returns, dtype=float)
        self.n_paths = 1

    def chunk(self, t0: int, t1: int) -> np.ndarray:
        return self.returns[None, t0:t1, :]


def _robust_cholesky(cov: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(cov)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return vecs @ np.diag(np.sqrt(np.clip(vals, 0, None)))


class SimulatedReturns:
    """Draws daily asset returns for many paths, in chunks to bound memory.

    bootstrap        i.i.d. historical days (keeps fat tails and cross-asset correlation)
    block_bootstrap  circular blocks of consecutive days (also keeps trends and
                     volatility clustering, which drawdown/momentum strategies need)
    geometric_brownian  multivariate normal in log space, fitted to log returns
    parametric       multivariate normal in simple-return space
    """

    def __init__(self, historical: np.ndarray, method: str, n_paths: int,
                 rng: np.random.Generator, block_size: int = 21, inflation_per_day: float = 0.0):
        self.hist = np.asarray(historical, dtype=float)
        self.method = method
        self.n_paths = n_paths
        self.rng = rng
        self.block_size = max(1, int(block_size))
        self.deflator = 1.0 + inflation_per_day
        n_obs, k = self.hist.shape
        if method == 'geometric_brownian':
            logs = np.log1p(self.hist)
            self.mu = np.atleast_1d(logs.mean(axis=0))
            self.chol = _robust_cholesky(np.cov(logs, rowvar=False) if k > 1 else np.var(logs, ddof=1))
        elif method == 'parametric':
            self.mu = np.atleast_1d(self.hist.mean(axis=0))
            self.chol = _robust_cholesky(np.cov(self.hist, rowvar=False) if k > 1 else np.var(self.hist, ddof=1))
        elif method == 'block_bootstrap':
            self._pos = np.zeros(n_paths, dtype=np.int64)
            self._left = np.zeros(n_paths, dtype=np.int64)
        elif method != 'bootstrap':
            raise ValueError(f"Unknown simulation method: {method}")

    def chunk(self, t0: int, t1: int) -> np.ndarray:
        n = t1 - t0
        n_obs, k = self.hist.shape
        if self.method == 'bootstrap':
            r = self.hist[self.rng.integers(0, n_obs, (self.n_paths, n))]
        elif self.method == 'block_bootstrap':
            idx = np.empty((self.n_paths, n), dtype=np.int64)
            for j in range(n):
                new = self._left == 0
                if new.any():
                    self._pos[new] = self.rng.integers(0, n_obs, new.sum())
                    self._left[new] = self.block_size
                idx[:, j] = self._pos
                self._pos = (self._pos + 1) % n_obs
                self._left -= 1
            r = self.hist[idx]
        else:
            # float32 draws and a 2-D (BLAS) matmul: ~3x faster than a stacked 3-D matmul
            z = self.rng.standard_normal((self.n_paths * n, k), dtype=np.float32)
            x = (z @ self.chol.T.astype(np.float32) + self.mu.astype(np.float32)).reshape(self.n_paths, n, k)
            r = np.expm1(x) if self.method == 'geometric_brownian' else np.maximum(x, -0.99)
            r = r.astype(np.float64)
        if self.deflator != 1.0:
            r = (1 + r) / self.deflator - 1
        return r


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    record_steps: np.ndarray            # (R,) step index of each record point (0 = start)
    values: np.ndarray                  # (P, R) balance in dollars
    twr: np.ndarray                     # (P, R) time-weighted growth of $1
    weights: np.ndarray                 # (P, R, K) holdings weights at record points
    max_drawdown: np.ndarray            # (P,) of the time-weighted index
    volatility: np.ndarray              # (P,) annualized vol of daily time-weighted returns
    costs: np.ndarray                   # (P,) transaction costs paid
    n_trades: np.ndarray                # (P,) rebalances executed
    contribution_steps: np.ndarray      # step indices (1-based) where cash was added
    events: List[dict] = field(default_factory=list)  # single-path runs only


def _sanitize(weights, base: np.ndarray, n_paths: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.tile(w, (n_paths, 1))
    w = np.where(np.isfinite(w), np.clip(w, 0, None), 0.0)
    sums = w.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= 0
    w = w / np.where(sums > 0, sums, 1)
    if bad.any():
        w[bad] = base
    return w


def run_engine(source, n_days: int, base_weights, tickers: List[str],
               initial_capital: float,
               contribution_amount: float = 0.0,
               contribution_days: Optional[np.ndarray] = None,
               rebalance_days: Optional[np.ndarray] = None,
               rebalance_threshold: Optional[float] = None,
               transaction_cost_bps: float = 0.0,
               strategy: Optional[AllocationStrategy] = None,
               apply_to: str = 'contributions',
               check_days: Optional[np.ndarray] = None,
               periods_per_year: float = 252.0,
               record_steps: Optional[np.ndarray] = None,
               dates: Optional[pd.DatetimeIndex] = None,
               chunk_days: int = 252,
               vectorize: bool = True) -> EngineResult:
    P = source.n_paths
    base = np.asarray(base_weights, dtype=float)
    base = base / base.sum()
    K = len(base)
    ones_k = np.ones(K)  # row sums via matmul: ~10x faster than .sum(axis=1) for small K
    zeros = np.zeros(n_days, dtype=bool)
    contribution_days = zeros if contribution_days is None or not contribution_amount else contribution_days
    rebalance_days = zeros if rebalance_days is None else rebalance_days
    check_days = zeros if check_days is None else check_days
    strat_contrib = strategy is not None and apply_to in ('contributions', 'both')
    strat_rebal = strategy is not None and apply_to in ('rebalance', 'both')
    collect_events = P == 1

    if record_steps is None:
        record_steps = np.arange(n_days + 1)
    record_steps = np.asarray(record_steps, dtype=int)
    rec_pos = {int(s): i for i, s in enumerate(record_steps)}
    R = len(record_steps)

    if vectorize and strategy is None and not rebalance_threshold:
        return _run_segments(source, n_days, base, initial_capital, contribution_amount,
                             contribution_days, rebalance_days, transaction_cost_bps / 1e4,
                             periods_per_year, record_steps, chunk_days)

    holdings = np.tile(base, (P, 1)) * initial_capital
    target = np.tile(base, (P, 1))
    v_prev = holdings @ ones_k
    twr = np.ones(P)
    twr_peak = np.ones(P)
    max_dd = np.zeros(P)
    sum_r = np.zeros(P)
    sum_r2 = np.zeros(P)
    costs = np.zeros(P)
    n_trades = np.zeros(P, dtype=int)
    price_idx = np.ones((P, K))
    price_peak = np.ones((P, K))
    ref_idx = np.ones(P)    # daily-rebalanced base-weight portfolio
    ref_peak = np.ones(P)
    cost_rate = transaction_cost_bps / 1e4

    values_rec = np.empty((P, R))
    twr_rec = np.empty((P, R))
    weights_rec = np.empty((P, R, K))
    if 0 in rec_pos:
        values_rec[:, rec_pos[0]] = v_prev
        twr_rec[:, rec_pos[0]] = 1.0
        weights_rec[:, rec_pos[0]] = base

    W = max(21, strategy.lookback_days if strategy else 0)
    rolling_stats = strategy is not None and getattr(strategy, 'uses_rolling_stats', True)
    all_paths = np.ones(P, dtype=bool)
    if strategy is not None:
        strategy.reset(P, tickers)
        buf = np.zeros((P, W, K))
        pbuf = np.zeros((P, W))
    ptr, filled = 0, 0

    def trailing(n, portfolio=False):
        if n < 2 or n > W or filled < n:
            return None
        idx = (ptr - np.arange(n)) % W
        return pbuf[:, idx] if portfolio else buf[:, idx, :]

    events = []
    contribution_steps = []

    for c0 in range(0, n_days, chunk_days):
        block = source.chunk(c0, min(c0 + chunk_days, n_days))
        for j in range(block.shape[1]):
            t = c0 + j
            r = block[:, j, :]
            holdings *= 1 + r
            v = holdings @ ones_k

            if strategy is not None:
                price_idx *= 1 + r
                np.maximum(price_peak, price_idx, out=price_peak)
                ref_idx *= 1 + r @ base
                np.maximum(ref_peak, ref_idx, out=ref_peak)
                buf[:, ptr, :] = r
                pbuf[:, ptr] = v / np.where(v_prev > 0, v_prev, 1) - 1
                filled = min(filled + 1, W)

            target_s = None
            decide = strategy is not None and (
                contribution_days[t] or (strat_rebal and (check_days[t] or rebalance_days[t])))
            if decide:
                window = trailing(W) if rolling_stats else None
                twr_now = twr * (1 + pbuf[:, ptr])
                ctx = MarketContext(
                    current_holdings=holdings,
                    current_drawdowns=price_idx / price_peak - 1,
                    base_allocations=base,
                    asset_tickers=tickers,
                    current_day=t + 1,
                    total_days=n_days,
                    rolling_returns=None if window is None else window.mean(axis=1) * periods_per_year,
                    rolling_volatility=None if window is None else window.std(axis=1, ddof=1) * np.sqrt(periods_per_year),
                    momentum_score=None if window is None else np.prod(1 + window, axis=1) - 1,
                    portfolio_drawdown=twr_now / np.maximum(twr_peak, twr_now) - 1,
                    reference_drawdown=ref_idx / ref_peak - 1,
                    portfolio_volatility=(None if filled < W or not rolling_stats else
                                          pbuf.std(axis=1, ddof=1) * np.sqrt(periods_per_year)),
                    current_weights=holdings / np.where(v > 0, v, 1)[:, None],
                    current_date=dates[t + 1] if dates is not None else None,
                    periods_per_year=periods_per_year,
                    _trailing_asset_returns=lambda n: trailing(n),
                    _trailing_portfolio_returns=lambda n: trailing(n, portfolio=True),
                )
                if ctx.rolling_returns is not None:
                    ctx.rolling_sharpe = ctx.rolling_returns / (ctx.rolling_volatility + 1e-9)
                target_s = _sanitize(strategy.get_allocation(ctx), base, P)

            flow = 0.0
            if contribution_days[t]:
                w_c = target_s if strat_contrib else base
                holdings += w_c * contribution_amount
                flow = contribution_amount
                contribution_steps.append(t + 1)
                if collect_events and strat_contrib and np.abs(w_c[0] - base).max() > 1e-6:
                    events.append({'step': t + 1, 'type': 'contribution', 'trigger': 'strategy',
                                   'weights': w_c[0].copy()})

            trade = all_paths if rebalance_days[t] else None
            reason = 'calendar' if rebalance_days[t] else None
            if strat_rebal and target_s is not None:
                changed = np.abs(target_s - target).max(axis=1) > 1e-6
                target = target_s
                if trade is None:
                    trade = changed
                    reason = 'signal' if changed.any() else None
            if rebalance_threshold:
                if trade is None:
                    trade = np.zeros(P, dtype=bool)
                v_now = holdings @ ones_k
                drift = np.abs(holdings / np.where(v_now > 0, v_now, 1)[:, None] - target).max(axis=1)
                breach = drift > rebalance_threshold
                if (breach & ~trade).any():
                    reason = reason or 'threshold'
                trade |= breach

            if trade is not None and trade.any():
                v_now = holdings @ ones_k
                before = holdings[trade] / np.where(v_now[trade] > 0, v_now[trade], 1)[:, None]
                desired = target[trade] * v_now[trade, None]
                cost = np.abs(desired - holdings[trade]).sum(axis=1) * cost_rate
                holdings[trade] = target[trade] * (v_now[trade] - cost)[:, None]
                costs[trade] += cost
                n_trades[trade] += 1
                if collect_events:
                    events.append({'step': t + 1, 'type': 'rebalance', 'trigger': reason,
                                   'before': before[0].copy(), 'weights': target[0].copy(),
                                   'turnover': float(np.abs(target[0] - before[0]).sum() / 2)})

            v_end = holdings @ ones_k
            day_r = np.divide(v_end - flow, v_prev, out=np.ones(P), where=v_prev > 0) - 1
            twr *= 1 + day_r
            np.maximum(twr_peak, twr, out=twr_peak)
            np.minimum(max_dd, twr / twr_peak - 1, out=max_dd)
            sum_r += day_r
            sum_r2 += day_r * day_r
            v_prev = v_end
            if strategy is not None:
                ptr = (ptr + 1) % W

            pos = rec_pos.get(t + 1)
            if pos is not None:
                values_rec[:, pos] = v_end
                twr_rec[:, pos] = twr
                weights_rec[:, pos] = holdings / np.where(v_end > 0, v_end, 1)[:, None]

    n = max(n_days, 2)
    var = np.maximum(sum_r2 / n - (sum_r / n) ** 2, 0) * n / (n - 1)
    return EngineResult(
        record_steps=record_steps, values=values_rec, twr=twr_rec, weights=weights_rec,
        max_drawdown=max_dd, volatility=np.sqrt(var * periods_per_year),
        costs=costs, n_trades=n_trades,
        contribution_steps=np.asarray(contribution_steps, dtype=int), events=events,
    )


def _run_segments(source, n_days, base, initial_capital, contribution_amount,
                  contribution_days, rebalance_days, cost_rate, periods_per_year,
                  record_steps, chunk_days) -> EngineResult:
    """run_engine for portfolios without a strategy or drift band.

    Holdings only change on contribution/rebalance days, so each stretch between
    events is a vectorized cumulative product instead of a Python loop per day.
    Produces the same results as the day loop (see tests/test_engine.py).
    """
    P, K = source.n_paths, len(base)
    ones_k = np.ones(K)
    rec_pos = {int(s): i for i, s in enumerate(record_steps)}
    R = len(record_steps)

    holdings = np.tile(base, (P, 1)) * initial_capital
    v_prev = holdings @ ones_k
    twr, twr_peak = np.ones(P), np.ones(P)
    max_dd, sum_r, sum_r2, costs = np.zeros(P), np.zeros(P), np.zeros(P), np.zeros(P)
    n_trades = np.zeros(P, dtype=int)
    values_rec, twr_rec, weights_rec = np.empty((P, R)), np.empty((P, R)), np.empty((P, R, K))
    if 0 in rec_pos:
        values_rec[:, rec_pos[0]] = v_prev
        twr_rec[:, rec_pos[0]] = 1.0
        weights_rec[:, rec_pos[0]] = base
    events, contribution_steps = [], []
    event_days = contribution_days | rebalance_days

    for c0 in range(0, n_days, chunk_days):
        c1 = min(c0 + chunk_days, n_days)
        block = source.chunk(c0, c1)
        cuts = [int(e) for e in np.flatnonzero(event_days[c0:c1])]
        if not cuts or cuts[-1] != c1 - c0 - 1:
            cuts.append(c1 - c0 - 1)
        s = 0
        for e in cuts:
            held = holdings[:, None, :] * np.cumprod(1 + block[:, s:e + 1, :], axis=1)
            vals = held @ ones_k                                   # (P, n) end-of-day values
            t = c0 + e
            holdings = held[:, -1, :].copy()
            flow = 0.0
            if contribution_days[t]:
                holdings += base * contribution_amount
                flow = contribution_amount
                contribution_steps.append(t + 1)
            if rebalance_days[t]:
                v_now = holdings @ ones_k
                before = holdings / np.where(v_now > 0, v_now, 1)[:, None]
                cost = np.abs(base * v_now[:, None] - holdings).sum(axis=1) * cost_rate
                holdings = base * (v_now - cost)[:, None]
                costs += cost
                n_trades += 1
                if P == 1:
                    events.append({'step': t + 1, 'type': 'rebalance', 'trigger': 'calendar',
                                   'before': before[0].copy(), 'weights': base.copy(),
                                   'turnover': float(np.abs(base - before[0]).sum() / 2)})
            v_end = holdings @ ones_k

            ends = vals.copy()
            ends[:, -1] = v_end - flow
            prev = np.concatenate([v_prev[:, None], vals[:, :-1]], axis=1)
            day_r = np.divide(ends, prev, out=np.ones_like(ends), where=prev > 0) - 1
            path = twr[:, None] * np.cumprod(1 + day_r, axis=1)
            peak = np.maximum(np.maximum.accumulate(path, axis=1), twr_peak[:, None])
            np.minimum(max_dd, (path / peak - 1).min(axis=1), out=max_dd)
            twr, twr_peak = path[:, -1].copy(), peak[:, -1].copy()
            sum_r += day_r.sum(axis=1)
            sum_r2 += (day_r * day_r).sum(axis=1)

            last = e - s
            for j in range(last + 1):
                pos = rec_pos.get(c0 + s + j + 1)
                if pos is None:
                    continue
                twr_rec[:, pos] = path[:, j]
                if j == last:
                    values_rec[:, pos] = v_end
                    weights_rec[:, pos] = holdings / np.where(v_end > 0, v_end, 1)[:, None]
                else:
                    values_rec[:, pos] = vals[:, j]
                    weights_rec[:, pos] = held[:, j, :] / np.where(vals[:, j] > 0, vals[:, j], 1)[:, None]
            v_prev = v_end
            s = e + 1

    n = max(n_days, 2)
    var = np.maximum(sum_r2 / n - (sum_r / n) ** 2, 0) * n / (n - 1)
    return EngineResult(
        record_steps=record_steps, values=values_rec, twr=twr_rec, weights=weights_rec,
        max_drawdown=max_dd, volatility=np.sqrt(var * periods_per_year),
        costs=costs, n_trades=n_trades,
        contribution_steps=np.asarray(contribution_steps, dtype=int), events=events,
    )


def vectorized_irr(initial: float, contribution: float, contribution_years: np.ndarray,
                   final_values: np.ndarray, horizon_years: float) -> np.ndarray:
    """Per-path annual IRR for identical cash flows and different end values (bisection)."""
    lo = np.full(len(final_values), -0.9999)
    hi = np.full(len(final_values), 10.0)
    remaining = horizon_years - np.asarray(contribution_years, dtype=float)

    def fv(rate):
        g = 1 + rate[:, None]
        return initial * g[:, 0] ** horizon_years + contribution * (g ** remaining[None, :]).sum(axis=1)

    for _ in range(60):
        mid = (lo + hi) / 2
        too_high = fv(mid) > final_values
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return (lo + hi) / 2
