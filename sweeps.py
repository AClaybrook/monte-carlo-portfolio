"""
Strategy parameter sweeps: backtest (and optionally simulate) every cell of a
1-2 parameter grid on the same allocation, cash flows and rebalancing.
"""
import dataclasses
import itertools
from typing import Dict, List, Optional

import numpy as np

from backtester import Backtester
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig, SweepConfig, create_strategy_from_config

# metric key -> (label, higher is better)
SWEEP_METRICS = {
    'CAGR': ('CAGR', True),
    'IRR': ('IRR', True),
    'Sharpe': ('Sharpe', True),
    'Sortino': ('Sortino', True),
    'Max Drawdown': ('Max drawdown', True),   # less negative is better
    'Stdev': ('Stdev', False),
    'mc_median_cagr': ('MC median CAGR', True),
    'mc_p10_cagr': ('MC 10th pct CAGR', True),
    'mc_probability_loss': ('MC P(loss)', False),
    'mc_median_max_drawdown': ('MC median max DD', True),
}


def _cell_metrics(bt: Dict, mc: Optional[Dict]) -> Dict[str, float]:
    m = bt['metrics']
    out = {k: m[k] for k in ('CAGR', 'IRR', 'Sharpe', 'Sortino', 'Max Drawdown', 'Stdev')}
    if mc is not None:
        s = mc['stats']
        out.update({
            'mc_median_cagr': s['median_cagr'],
            'mc_p10_cagr': s['percentiles']['cagr'][10],
            'mc_probability_loss': s['probability_loss'],
            'mc_median_max_drawdown': s['median_max_drawdown'],
        })
    return out


def run_sweep(sweep: SweepConfig, asset_map: Dict[str, dict], sim_cfg: SimulationConfig,
              start=None, end=None, benchmark: Optional[dict] = None) -> Dict:
    """Evaluate every grid cell plus a no-strategy baseline.

    Monte Carlo cells share one seed (common random numbers), so differences
    between cells come from the parameters, not from simulation noise.
    """
    assets = [asset_map[t.upper()] for t in sweep.allocations]
    weights = list(sweep.allocations.values())
    rebalance = sweep.rebalance or sim_cfg.rebalance
    keys = list(sweep.grid)
    backtester = Backtester()
    simulator = None
    if sweep.simulations > 0:
        seed = sim_cfg.seed if sim_cfg.seed is not None else 2024
        simulator = PortfolioSimulator(None, dataclasses.replace(
            sim_cfg, simulations=sweep.simulations, seed=seed))

    def evaluate(strategy_conf):
        strategy = create_strategy_from_config(strategy_conf) if strategy_conf else None
        apply_to = strategy_conf.apply_to if strategy_conf else 'contributions'
        check = strategy_conf.check_frequency if strategy_conf else 'monthly'
        bt = backtester.run_backtest(
            assets, weights, sim_cfg.initial_capital, start_date_override=start, end_date=end,
            strategy=strategy, **sim_cfg.cash_flow_settings(), rebalance=rebalance,
            apply_to=apply_to, check_frequency=check, risk_free_rate=sim_cfg.risk_free_rate,
            benchmark=benchmark)
        mc = None
        if simulator is not None:
            mc = simulator.simulate_portfolio(assets, weights, start_date_override=start,
                                              strategy=strategy, rebalance=rebalance,
                                              apply_to=apply_to, check_frequency=check)
        return _cell_metrics(bt, mc), bt

    baseline, baseline_bt = evaluate(None)
    cells: List[Dict] = []
    for combo in itertools.product(*(sweep.grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        conf = dataclasses.replace(sweep.strategy, params={**sweep.strategy.params, **params}, name=None)
        metrics, _ = evaluate(conf)
        cells.append({'params': params, 'metrics': metrics})

    best = max(cells, key=lambda c: c['metrics']['Sharpe'])
    print(f"  {sweep.name}: {len(cells)} cells | best Sharpe {best['metrics']['Sharpe']:.2f} at "
          + ", ".join(f"{k}={v}" for k, v in best['params'].items())
          + f" | baseline {baseline['Sharpe']:.2f}")
    return {
        'name': sweep.name,
        'strategy_type': sweep.strategy.type,
        'apply_to': sweep.strategy.apply_to,
        'allocations': dict(sweep.allocations),
        'policy': baseline_bt['strategy'],
        'keys': keys,
        'values': [list(sweep.grid[k]) for k in keys],
        'cells': cells,
        'baseline': baseline,
        'metrics': [k for k in SWEEP_METRICS if k in baseline],
        'start': baseline_bt['dates'][0].date(),
        'end': baseline_bt['dates'][-1].date(),
    }


def metric_grid(result: Dict, metric: str) -> np.ndarray:
    """Cells reshaped to (len(values[1]) or 1, len(values[0])) for a heatmap."""
    vals = np.array([c['metrics'][metric] for c in result['cells']], dtype=float)
    nx = len(result['values'][0])
    ny = len(result['values'][1]) if len(result['keys']) == 2 else 1
    # itertools.product varies the LAST key fastest: cells are ordered (x0,y0), (x0,y1), ...
    return vals.reshape(nx, ny).T
