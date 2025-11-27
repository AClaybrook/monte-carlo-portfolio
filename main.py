"""
Main script - Updated for Collapsible Reporting & Modern Optimization.
"""

import sys
import argparse
from pathlib import Path
from run_config import load_config_from_file
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer
from backtester import Backtester

def find_config_file(specified_path: str = None) -> Path:
    if specified_path:
        path = Path(specified_path)
        if not path.exists(): raise FileNotFoundError(f"Config file not found: {specified_path}")
        return path
    if Path('config/my_portfolios.py').exists(): return Path('config/my_portfolios.py')
    if Path('config/example_config.py').exists(): return Path('config/example_config.py')
    raise FileNotFoundError("No config file found.")

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Simulator')
    parser.add_argument('config', nargs='?')
    args = parser.parse_args()

    config_path = find_config_file(args.config)
    print(f"✓ Loading configuration from: {config_path}")
    try:
        config = load_config_from_file(str(config_path))
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        return 1

    # Initialize components
    data_manager = DataManager(db_path=config.database.path)

    # SIMPLIFIED: Pass the whole simulation config object
    sim = PortfolioSimulator(data_manager, config.simulation)

    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)
    backtester = Backtester(data_manager)
    print("\n" + "="*60 + "\nLoading Assets...\n" + "="*60)
    asset_map = {}
    for ticker, asset_config in config.assets.items():
        asset = sim.define_asset_from_ticker(asset_config.ticker, asset_config.name, config.simulation.lookback_years)
        asset_map[ticker.upper()] = asset

    portfolio_results = []

    # 1. Process Defined Portfolios
    print("\n" + "="*60 + "\nProcessing Defined Portfolios...\n" + "="*60)
    for p_conf in config.portfolios:
        print(f"Processing {p_conf.name}...")
        assets = [asset_map[t.upper()] for t in p_conf.allocations.keys()]
        weights = list(p_conf.allocations.values())

        sim_res = sim.simulate_portfolio(assets, weights)
        bt_res = backtester.run_backtest(assets, weights, config.simulation.initial_capital)

        portfolio_results.append({
            'label': p_conf.name,
            'results': sim_res,
            'backtest': bt_res
        })

    # 2. Optimization
    if config.optimization:
        print("\n" + "="*60 + "\nRunning Modern Optimization...\n" + "="*60)
        opt_assets = [asset_map[name.upper()] for name in config.optimization.assets]

        # Run Optimizers
        strategies = [
            optimizer.optimize_sharpe_ratio(opt_assets),
            optimizer.optimize_min_volatility(opt_assets),
            optimizer.optimize_risk_parity(opt_assets),       # NEW
            optimizer.optimize_sortino_ratio(opt_assets)      # NEW
        ]

        for strat in strategies:
            print(f"Optimization Found: {strat['label']}")
            alloc_str = " / ".join([f"{int(w*100)}% {a['ticker']}" for w, a in zip(strat['allocations'], opt_assets)])
            print(f"  Allocation: {alloc_str}")

            # Backtest the optimized result
            bt_res = backtester.run_backtest(opt_assets, strat['allocations'], config.simulation.initial_capital)

            portfolio_results.append({
                'label': strat['label'],
                'results': strat['results'],
                'backtest': bt_res
            })

    # 3. Generate Report
    print("\n" + "="*60 + "\nGenerating Collapsible HTML Report...\n" + "="*60)
    visualizer.generate_html_report(portfolio_results, config.visualization.output_filename)
    print(f"✓ Report saved to: {config.visualization.output_filename}")

    data_manager.close()

if __name__ == "__main__":
    main()