"""
Main script - With Strategy Support and Bulk Download
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

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


def collect_all_tickers(config) -> set:
    """Collect all tickers needed from config"""
    all_tickers = set()

    for p in config.portfolios:
        all_tickers.update([t.upper() for t in p.allocations.keys()])

    if config.optimization:
        all_tickers.update([t.upper() for t in config.optimization.assets])
        all_tickers.add(config.optimization.benchmark_ticker.upper())

    return all_tickers


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Simulator')
    parser.add_argument('config', nargs='?')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization step')
    parser.add_argument('--force-download', action='store_true', help='Force re-download all data')
    parser.add_argument('--coverage-report', action='store_true', help='Show data coverage report')
    parser.add_argument('--offline', action='store_true', help='Use cached data only (no yfinance calls)')
    args = parser.parse_args()

    config_path = find_config_file(args.config)
    print(f"✓ Loading configuration from: {config_path}")

    try:
        config = load_config_from_file(str(config_path))
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        return 1

    # Initialize data manager
    data_manager = DataManager(db_path=config.database.path)

    # Coverage report mode
    if args.coverage_report:
        all_tickers = list(collect_all_tickers(config))
        report = data_manager.get_data_coverage_report(all_tickers)
        print("\n" + "="*80)
        print("DATA COVERAGE REPORT")
        print("="*80)
        print(report.to_string(index=False))
        data_manager.close()
        return 0

    # Initialize simulation components
    sim = PortfolioSimulator(data_manager, config.simulation)
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)
    backtester = Backtester(data_manager)

    print("\n" + "="*60)
    print("BULK DOWNLOADING DATA")
    print("="*60)

    # Collect all tickers and bulk download
    all_tickers = list(collect_all_tickers(config))
    print(f"Tickers to load: {', '.join(sorted(all_tickers))}")

    # Calculate date range from config
    start_date, end_date = config.simulation.get_date_range()
    print(f"Date range: {start_date} to {end_date}")

    # Bulk download all data at once (or use cache only in offline mode)
    if args.offline:
        print("OFFLINE MODE - using cached data only (no yfinance calls)")
        cached_end = data_manager.get_latest_cached_date(all_tickers)
        if cached_end is None:
            print("✗ No cached data found for any tickers. Cannot run offline.")
            data_manager.close()
            return 1
        end_date = cached_end
        print(f"  Using cached data through: {end_date}")
        bulk_data = {}
        for ticker in all_tickers:
            df = data_manager._get_from_db(ticker, start_date, end_date)
            if df is not None and not df.empty:
                bulk_data[ticker] = df
    else:
        bulk_data = data_manager.bulk_download(
            all_tickers,
            start_date=start_date,
            end_date=end_date,
            force_update=args.force_download
        )

    print(f"\n✓ Loaded data for {len(bulk_data)} tickers")

    # Build asset map from downloaded data
    print("\n" + "="*60)
    print("BUILDING ASSET MAP")
    print("="*60)

    asset_map = {}
    start_dates = []

    for ticker in all_tickers:
        asset_conf = config.assets.get(ticker, None)
        name = asset_conf.name if asset_conf else ticker

        if ticker in bulk_data:
            df = bulk_data[ticker]
            returns = df['Adj Close'].pct_change().dropna()

            asset = {
                'ticker': ticker,
                'name': name,
                'historical_returns': returns,
                'full_data': df,
                'daily_mean': returns.mean(),
                'daily_std': returns.std()
            }
            asset_map[ticker] = asset

            if not returns.empty:
                start_dates.append(returns.index.min())
                print(f"  ✓ {ticker}: {len(df)} days, {returns.index.min().date()} to {returns.index.max().date()}")
        else:
            print(f"  ⚠ {ticker}: No data available")

    if not start_dates:
        print("Error: No data found for any assets.")
        data_manager.close()
        return 1

    global_start_date = max(start_dates)
    global_end_date = sim.end_date
    print(f"\n✓ GLOBAL ALIGNMENT: {global_start_date.date()} to {global_end_date}")

    portfolio_results = []

    # Add Benchmark Portfolio
    if config.optimization:
        bench_ticker = config.optimization.benchmark_ticker.upper()
        if bench_ticker in asset_map:
            print(f"\n→ Adding Benchmark: {bench_ticker}")
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

    # Process Defined Portfolios
    print("\n" + "="*60)
    print("PROCESSING PORTFOLIOS")
    print("="*60)

    for p_conf in config.portfolios:
        print(f"\n→ {p_conf.name}")

        # Check all assets exist
        missing = [t for t in p_conf.allocations.keys() if t.upper() not in asset_map]
        if missing:
            print(f"  ⚠ Skipping - missing assets: {missing}")
            continue

        assets = [asset_map[t.upper()] for t in p_conf.allocations.keys()]
        weights = list(p_conf.allocations.values())

        # Create strategy if defined
        strategy = None
        if p_conf.strategy:
            strategy = create_strategy_from_config(p_conf.strategy)
            print(f"  Strategy: {strategy.name}")

        # Run simulation
        sim_res = sim.simulate_portfolio(
            assets, weights,
            start_date_override=global_start_date,
            strategy=strategy
        )

        # Run backtest
        bt_res = backtester.run_backtest(
            assets, weights, config.simulation.initial_capital,
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

        # Quick summary
        m = bt_res['metrics']
        print(f"  CAGR: {m['CAGR']*100:.2f}% | Max DD: {m['Max Drawdown']*100:.2f}% | Sharpe: {m['Sharpe']:.2f}")

    # Run Optimizations
    if config.optimization and not args.no_optimize:
        print("\n" + "="*60)
        print("RUNNING OPTIMIZATIONS")
        print("="*60)

        # Filter to available assets
        opt_assets = [
            asset_map[name.upper()]
            for name in config.optimization.assets
            if name.upper() in asset_map
        ]

        if len(opt_assets) < 2:
            print("⚠ Need at least 2 assets for optimization")
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

                print(f"\n→ {strat_name}...")

                if strat_name == 'custom_weighted':
                    strat_result = strategy_map[strat_name](
                        opt_assets,
                        weights_config=config.optimization.objective_weights,
                        start_date_override=global_start_date
                    )
                else:
                    strat_result = strategy_map[strat_name](opt_assets, start_date_override=global_start_date)

                alloc_str = " / ".join([
                    f"{int(w*100)}% {a['ticker']}"
                    for w, a in zip(strat_result['allocations'], opt_assets)
                ])
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
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

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