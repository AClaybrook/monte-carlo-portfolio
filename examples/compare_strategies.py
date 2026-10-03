"""
Compare dynamic strategies on the same assets, cash flows and rebalancing,
and write one report per scenario.

Usage:
    python examples/compare_strategies.py                    # all scenarios, real data
    python examples/compare_strategies.py --test crypto      # one scenario
    python examples/compare_strategies.py --synthetic        # offline, generated prices
    python examples/compare_strategies.py --dca 1000 --sims 2000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import date, timedelta
from pathlib import Path

from backtester import Backtester
from data_manager import DataManager
from main import evaluate_portfolio
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig, StrategyConfig
from visualizations import PortfolioVisualizer

SCENARIOS = {
    'crypto': {
        'title': 'Crypto dip buying',
        'allocations': {'VOO': 0.60, 'BND': 0.20, 'BTC-USD': 0.20},
        'strategies': [
            ('Static', None),
            ('BTC opportunistic (30% dip)', StrategyConfig(
                'crypto_opportunistic', {'crypto_ticker': 'BTC-USD', 'dip_threshold': 0.30,
                                         'normal_weight': 0.20, 'dip_weight': 0.50})),
            ('BTC aggressive (20% dip)', StrategyConfig(
                'crypto_opportunistic', {'crypto_ticker': 'BTC-USD', 'dip_threshold': 0.20,
                                         'normal_weight': 0.20, 'dip_weight': 0.60})),
        ],
    },
    'leveraged': {
        'title': 'Leveraged dip buying',
        'allocations': {'SPXL': 0.50, 'TQQQ': 0.50},
        'strategies': [
            ('Static', None),
            ('TQQQ dip buyer (15%)', StrategyConfig(
                'buy_the_dip', {'target_ticker': 'TQQQ', 'threshold': 0.15, 'aggressive_weight': 0.70})),
            ('TQQQ dip buyer (25%)', StrategyConfig(
                'buy_the_dip', {'target_ticker': 'TQQQ', 'threshold': 0.25, 'aggressive_weight': 0.80})),
        ],
    },
    'broad': {
        'title': 'Broad market strategies',
        'allocations': {'VOO': 0.30, 'QQQ': 0.20, 'VGT': 0.10, 'BND': 0.40},
        'strategies': [
            ('Static', None),
            ('Buy the dip (QQQ 12%)', StrategyConfig(
                'buy_the_dip', {'target_ticker': 'QQQ', 'threshold': 0.12, 'aggressive_weight': 0.50})),
            ('Momentum tilt', StrategyConfig('momentum', {'lookback': 63, 'tilt_strength': 0.3})),
            ('Relative value', StrategyConfig('relative_value', {'threshold': 0.08, 'max_tilt': 0.45})),
            ('Drawdown protection (rebalance)', StrategyConfig(
                'drawdown_protection', apply_to='rebalance', check_frequency='daily',
                params={'threshold': 0.10, 'recovery_threshold': 0.03,
                        'risk_off_allocation': {'VOO': 0.20, 'BND': 0.80}})),
            ('Dual momentum (rebalance)', StrategyConfig(
                'dual_momentum', apply_to='rebalance',
                params={'equity_ticker': 'VOO', 'safe_ticker': 'BND', 'lookback': 126})),
        ],
    },
}


def load_assets(dm, tickers, start, end):
    data = dm.bulk_download(tickers, start, end)
    assets = []
    for t in tickers:
        df = data[t.upper()]
        assets.append({'ticker': t, 'name': t, 'full_data': df,
                       'historical_returns': df['Adj Close'].pct_change().dropna()})
    return assets


def run_scenario(key, dm, sim_cfg, start, end):
    sc = SCENARIOS[key]
    print(f"\n{'=' * 70}\n{sc['title'].upper()}\n{'=' * 70}")
    tickers = list(sc['allocations'])
    assets = load_assets(dm, tickers, start, end)
    weights = list(sc['allocations'].values())
    sim, bt = PortfolioSimulator(dm, sim_cfg), Backtester(dm)
    first = max(a['full_data'].index.min() for a in assets)

    results = []
    for label, strategy_conf in sc['strategies']:
        print(f"\n→ {label}")
        results.append(evaluate_portfolio(sim, bt, sim_cfg, label, assets, weights, start=first,
                                          strategy_conf=strategy_conf))

    out = Path('output') / f"strategies_{key}.html"
    out.parent.mkdir(exist_ok=True)
    PortfolioVisualizer(sim).generate_html_report(
        results, str(out), start_date=first.date(), end_date=end, title=sc['title'],
        synthetic=dm.__class__.__name__ == 'SyntheticDataManager')
    print(f"\n✓ Report: {out}")


def main():
    parser = argparse.ArgumentParser(description='Strategy comparison')
    parser.add_argument('--test', choices=[*SCENARIOS, 'all'], default='all')
    parser.add_argument('--dca', type=float, default=500, help='Monthly contribution')
    parser.add_argument('--sims', type=int, default=2000)
    parser.add_argument('--years', type=int, default=10, help='History and simulation horizon')
    parser.add_argument('--rebalance', default='none',
                        help="Calendar rebalancing on top of each strategy (none/monthly/quarterly/annual)")
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    if args.synthetic:
        from synthetic_data import SyntheticDataManager
        dm = SyntheticDataManager()
    else:
        dm = DataManager()

    end = date.today()
    start = end - timedelta(days=int(365.25 * args.years))
    sim_cfg = SimulationConfig(initial_capital=10000, years=args.years, simulations=args.sims,
                               contribution_amount=args.dca, contribution_frequency='monthly',
                               rebalance=args.rebalance)
    for key in (SCENARIOS if args.test == 'all' else [args.test]):
        run_scenario(key, dm, sim_cfg, start, end)
    dm.close()


if __name__ == "__main__":
    main()
