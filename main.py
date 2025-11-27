"""
Main script - Global Time Alignment & Modern Reporting.
"""

import sys
import argparse
import pandas as pd
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

    # Initialize
    data_manager = DataManager(db_path=config.database.path)
    sim = PortfolioSimulator(data_manager, config.simulation)
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)
    backtester = Backtester(data_manager)

    print("\n" + "="*60 + "\nLoading Assets & Aligning Timeframes...\n" + "="*60)
    asset_map = {}

    # 1. Load ALL assets (from portfolios and optimization list)
    all_tickers = set()
    for p in config.portfolios:
        all_tickers.update(p.allocations.keys())
    if config.optimization:
        all_tickers.update(config.optimization.assets)

    start_dates = []

    for ticker in all_tickers:
        # Use config settings if available, otherwise default
        asset_conf = config.assets.get(ticker, None)
        name = asset_conf.name if asset_conf else ticker

        # Load asset
        asset = sim.define_asset_from_ticker(ticker, name)
        asset_map[ticker.upper()] = asset

        # Track start date
        if not asset['historical_returns'].empty:
            start_dates.append(asset['historical_returns'].index.min())

    # 2. Determine Global Common Start Date
    if not start_dates:
        print("Error: No data found for any assets.")
        return 1

    global_start_date = max(start_dates)
    print(f"✓ GLOBAL ALIGNMENT: All analysis will start from {global_start_date.date()}")
    print(f"  (Dictated by the asset with the shortest history)")

    portfolio_results = []

    # 3. Process Defined Portfolios (With Forced Start Date)
    print("\n" + "="*60 + "\nProcessing Defined Portfolios...\n" + "="*60)
    for p_conf in config.portfolios:
        print(f"Processing {p_conf.name}...")
        assets = [asset_map[t.upper()] for t in p_conf.allocations.keys()]
        weights = list(p_conf.allocations.values())

        # Pass global_start_date to force alignment
        sim_res = sim.simulate_portfolio(assets, weights, config.simulation.method, start_date_override=global_start_date)
        bt_res = backtester.run_backtest(assets, weights, config.simulation.initial_capital, start_date_override=global_start_date)

        portfolio_results.append({
            'label': p_conf.name,
            'results': sim_res,
            'backtest': bt_res
        })

    # 4. Optimization
    if config.optimization:
        print("\n" + "="*60 + "\nRunning Modern Optimization...\n" + "="*60)
        opt_assets = [asset_map[name.upper()] for name in config.optimization.assets]

        # Run Optimizers
        strategies = [
            optimizer.optimize_sharpe_ratio(opt_assets, start_date_override=global_start_date),
            optimizer.optimize_min_volatility(opt_assets, start_date_override=global_start_date),
            optimizer.optimize_risk_parity(opt_assets, start_date_override=global_start_date),
            optimizer.optimize_sortino_ratio(opt_assets, start_date_override=global_start_date)
        ]

        for strat in strategies:
            print(f"Optimization Found: {strat['label']}")
            alloc_str = " / ".join([f"{int(w*100)}% {a['ticker']}" for w, a in zip(strat['allocations'], opt_assets)])
            print(f"  Allocation: {alloc_str}")

            # Backtest the optimized result
            bt_res = backtester.run_backtest(opt_assets, strat['allocations'], config.simulation.initial_capital, start_date_override=global_start_date)

            portfolio_results.append({
                'label': strat['label'],
                'results': strat['results'],
                'backtest': bt_res
            })

    # 5. Generate Report
    print("\n" + "="*60 + "\nGenerating Collapsible HTML Report...\n" + "="*60)
    visualizer.generate_html_report(portfolio_results, config.visualization.output_filename)
    print(f"✓ Report saved to: {config.visualization.output_filename}")

    data_manager.close()

if __name__ == "__main__":
    main()