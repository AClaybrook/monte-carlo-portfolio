"""
Walk-forward check for optimized portfolios.

In-sample optimization picks weights with hindsight. Here the optimizer only sees
the previous `train_years` at each refit, and its weights are traded for the next
`test_years`; the stitched result is what the method would actually have earned.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester import Backtester
from engine import align_asset_prices
from run_config import SimulationConfig
from strategies import ScheduledWeightsStrategy


def _slice_asset(asset: dict, start, end) -> dict:
    df = asset['full_data']
    part = df[(df.index >= start) & (df.index < end)]
    return {'ticker': asset['ticker'], 'name': asset.get('name', asset['ticker']), 'full_data': part,
            'historical_returns': part['Adj Close'].pct_change().dropna()}


def refit_schedule(fit: Callable[[List[dict]], Dict], assets: List[dict], start, end,
                   train_years: int, test_years: int) -> List[tuple]:
    """[(window_start, weights)] fit on the preceding train_years at each window start."""
    dates = align_asset_prices(assets, start, end).index
    if len(dates) < 2:
        return []
    schedule = []
    t = dates[0] + pd.DateOffset(years=train_years)
    while t < dates[-1] - pd.DateOffset(months=1):
        train = [_slice_asset(a, t - pd.DateOffset(years=train_years), t) for a in assets]
        schedule.append((t, np.asarray(fit(train)['allocations'], dtype=float)))
        t += pd.DateOffset(years=test_years)
    return schedule


def run_walk_forward(label: str, fit: Callable[[List[dict]], Dict], assets: List[dict],
                     in_sample_weights, sim_cfg: SimulationConfig, start=None, end=None,
                     train_years: int = 3, test_years: int = 1,
                     benchmark: Optional[dict] = None) -> Optional[Dict]:
    """Backtest refit-as-you-go weights against the full-history (in-sample) weights
    over the same out-of-sample window. Returns None if history is too short."""
    schedule = refit_schedule(fit, assets, start, end, train_years, test_years)
    if not schedule:
        return None
    first = schedule[0][0]
    common = dict(start_date_override=first, end_date=end, contribution_amount=sim_cfg.contribution_amount,
                  contribution_frequency=sim_cfg.contribution_frequency, rebalance=sim_cfg.rebalance,
                  risk_free_rate=sim_cfg.risk_free_rate, benchmark=benchmark)
    bt = Backtester()
    oos = bt.run_backtest(assets, schedule[0][1], sim_cfg.initial_capital,
                          strategy=ScheduledWeightsStrategy(schedule, name=f"{label} (walk-forward)"),
                          apply_to='rebalance', check_frequency='daily', **common)
    ins = bt.run_backtest(assets, in_sample_weights, sim_cfg.initial_capital, **common)
    turnover = [float(np.abs(b - a).sum() / 2) for (_, a), (_, b) in zip(schedule, schedule[1:])]
    return {
        'label': label,
        'tickers': [a['ticker'] for a in assets],
        'schedule': schedule,
        'in_sample_weights': np.asarray(in_sample_weights, dtype=float),
        'train_years': train_years,
        'test_years': test_years,
        'oos': oos,
        'in_sample': ins,
        'mean_refit_turnover': float(np.mean(turnover)) if turnover else 0.0,
    }
