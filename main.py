"""
Main script - With Strategy Support
UPDATED: Uses bulk download to avoid Yahoo Finance rate limiting
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from run_config import load_config_from_file, create_strategy_from_config
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer
from backtester import Backtester
from strategies import StaticAllocationStrategy


def find_config_file(specified_path: str = None) -> Path:
    if specified_path:
        path = Path(specified_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {specified_path}")
        return path
    if Path('config/my_portfolios.py').exists():
        return Path('config/my_portfolios.py')
    if Path('config/example_config.py').exists():
        return Path('config/example_config.py')
    raise FileNotFoundError("No config file found.")


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Simulator')
    parser.add_argument('config', nargs='?')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip optimization step')
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
    sim = PortfolioSimulator(data_manager, config.simulation)
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)
    backtester = Backtester(data_manager)

    print("\n" + "="*60 + "\nLoading Assets & Aligning Timeframes...\n" + "="*60)
    asset_map = {}

    # Collect all unique tickers
    all_tickers = set()
    for p in config.portfolios:
        all_tickers.update([t.upper() for t in p.allocations.keys()])

    if config.optimization:
        all_tickers.update([t.upper() for t in config.optimization.assets])
        all_tickers.add(config.optimization.benchmark_ticker.upper())

    all_tickers = list(all_tickers)
    print(f"Found {len(all_tickers)} unique tickers: {', '.join(sorted(all_tickers))}")

    # ========== BULK DOWNLOAD ALL TICKERS ==========
    # Instead of downloading one by one, fetch all at once
    print(f"\n--- Fetching data for all {len(all_tickers)} tickers ---")
    bulk_data = data_manager.get_data_bulk(all_tickers)

    if not bulk_data:
        print("Error: No data retrieved for any tickers.")
        return 1

    # Build asset_map from bulk data
    start_dates = []
    for ticker in all_tickers:
        ticker = ticker.upper()
        if ticker not in bulk_data:
            print(f"⚠ Skipping {ticker} - no data available")
            continue

        df = bulk_data[ticker]
        asset_conf = config.assets.get(ticker, None)
        name = asset_conf.name if asset_conf else ticker

        # Build asset dict using the pre-fetched data
        asset = sim.define_asset_from_dataframe(ticker, name, df)
        asset_map[ticker] = asset

        if not asset['historical_returns'].empty:
            start_dates.append(asset['historical_returns'].index.min())

    if not start_dates:
        print("Error: No valid data found for any assets.")
        return 1

    global_start_date = max(start_dates)
    global_end_date = sim.end_date
    print(f"\n✓ GLOBAL ALIGNMENT: {global_start_date.date()} to {global_end_date}")
    print(f"✓ Successfully loaded {len(asset_map)} assets")

    portfolio_results = []

    # Add Benchmark Portfolio if optimization is configured
    if config.optimization:
        bench_ticker = config.optimization.benchmark_ticker.upper()
        if bench_ticker in asset_map:
            print(f"\nAdding Benchmark: {bench_ticker}")
            assets = [asset_map[bench_ticker]]
            weights = [1.0]

            sim_res = sim.simulate_portfolio(assets, weights, start_date_override=global_start_date)
            bt_res = backtester.run_backtest(
                assets, weights, config.simulation.initial_capital,
                start_date_override=global_start_date
            )

            portfolio_results.append({
                'label': f"Benchmark ({bench_ticker})",
                'results': sim_res,
                'backtest': bt_res
            })

    # Process Defined Portfolios WITH STRATEGIES
    print("\n" + "="*60 + "\nProcessing Defined Portfolios...\n" + "="*60)

    for p_conf in config.portfolios:
        print(f"\nProcessing: {p_conf.name}")

        # Check all tickers exist
        missing = [t.upper() for t in p_conf.allocations.keys() if t.upper() not in asset_map]
        if missing:
            print(f"  ⚠ Skipping - missing tickers: {missing}")
            continue

        assets = [asset_map[t.upper()] for t in p_conf.allocations.keys()]
        weights = list(p_conf.allocations.values())

        # Create strategy from config if defined
        strategy = None
        if p_conf.strategy:
            strategy = create_strategy_from_config(p_conf.strategy)
            print(f"  > Strategy: {strategy.name}")
            print(f"  > Config: {strategy.get_config_summary()}")

        # Run simulation with strategy
        sim_res = sim.simulate_portfolio(
            assets,
            weights,
            start_date_override=global_start_date,
            strategy=strategy
        )

        # Run backtest with strategy
        bt_res = backtester.run_backtest(
            assets,
            weights,
            config.simulation.initial_capital,
            start_date_override=global_start_date,
            strategy=strategy,
            contribution_amount=config.simulation.contribution_amount,
            contribution_frequency=config.simulation.contribution_frequency
        )

        portfolio_results.append({
            'label': p_conf.name,
            'results': sim_res,
            'backtest': bt_res
        })

        # Print quick summary
        m = bt_res['metrics']
        print(f"  > Hist CAGR: {m['CAGR']*100:.2f}%  |  Max DD: {m['Max Drawdown']*100:.2f}%  |  Sharpe: {m['Sharpe']:.2f}")

    # Run Selected Optimizations
    if config.optimization and not args.no_optimize:
        print("\n" + "="*60 + "\nRunning Selected Optimizations...\n" + "="*60)

        # Check all optimization assets exist
        missing_opt = [t.upper() for t in config.optimization.assets if t.upper() not in asset_map]
        if missing_opt:
            print(f"⚠ Missing optimization assets: {missing_opt}")

        opt_assets = [asset_map[name.upper()] for name in config.optimization.assets if name.upper() in asset_map]

        if not opt_assets:
            print("⚠ No valid assets for optimization")
        else:
            active_strats = config.optimization.active_strategies

            strategy_map = {
                'max_sharpe': optimizer.optimize_sharpe_ratio,
                'min_volatility': optimizer.optimize_min_volatility,
                'risk_parity': optimizer.optimize_risk_parity,
                'max_sortino': optimizer.optimize_sortino_ratio,
                'custom_weighted': optimizer.optimize_custom_weighted
            }

            for strat_name in active_strats:
                if strat_name not in strategy_map:
                    print(f"⚠ Unknown strategy: {strat_name}")
                    continue

                print(f"Running: {strat_name}...")

                if strat_name == 'custom_weighted':
                    strat_result = strategy_map[strat_name](
                        opt_assets,
                        weights_config=config.optimization.objective_weights,
                        start_date_override=global_start_date
                    )
                else:
                    strat_result = strategy_map[strat_name](opt_assets, start_date_override=global_start_date)

                alloc_str = " / ".join([f"{int(w*100)}% {a['ticker']}" for w, a in zip(strat_result['allocations'], opt_assets)])
                print(f"  Result: {alloc_str}")

                bt_res = backtester.run_backtest(
                    opt_assets, strat_result['allocations'],
                    config.simulation.initial_capital,
                    start_date_override=global_start_date
                )

                portfolio_results.append({
                    'label': strat_result['label'],
                    'results': strat_result['results'],
                    'backtest': bt_res
                })

    # Generate Report
    print("\n" + "="*60 + "\nGenerating Report...\n" + "="*60)

    out_dir = 'output'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.visualization.output_filename}"

    visualizer.generate_html_report(
        portfolio_results,
        str(output_path),
        start_date=global_start_date.date(),
        end_date=global_end_date
    )
    print(f"✓ Report saved to: {output_path}")

    data_manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())