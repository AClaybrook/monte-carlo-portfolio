"""
Main script using the RunConfig system.

Usage:
    python main.py                           # Uses config/my_portfolios.py
    python main.py config/example_simple.py  # Uses specific config
"""

import sys
import argparse
from pathlib import Path
from run_config import load_config_from_file, RunConfig
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer

def find_config_file(specified_path: str = None) -> Path:
    """Find the config file to use"""
    if specified_path:
        path = Path(specified_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {specified_path}")
        return path

    # Try personal config first
    personal_config = Path('config/my_portfolios.py')
    if personal_config.exists():
        return personal_config

    # Fall back to simple example
    example_config = Path('config/example_config.py')
    if example_config.exists():
        print("⚠ No personal config found, using example_config.py")
        print(f"  Create {personal_config} to customize your portfolios")
        return example_config

    raise FileNotFoundError(
        "No config file found. Create config/my_portfolios.py or config/example_config.py"
    )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Monte Carlo Portfolio Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use config/my_portfolios.py
  python main.py config/example_simple.py     # Use specific config file
  python main.py config/aggressive.py         # Use another config
        """
    )
    parser.add_argument(
        'config',
        nargs='?',
        help='Path to configuration file (default: config/my_portfolios.py)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running simulation'
    )
    args = parser.parse_args()

    # Load configuration
    config_path = find_config_file(args.config)
    print(f"✓ Loading configuration from: {config_path}")

    try:
        config = load_config_from_file(str(config_path))
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        print("\nMake sure your config file:")
        print("  1. Imports from run_config: from run_config import RunConfig, ...")
        print("  2. Defines a variable: config = RunConfig(...)")
        return 1

    # Show configuration summary
    print("\n" + "="*60)
    print(config.summary())
    print("="*60)

    if args.dry_run:
        print("\n✓ Dry run complete (no simulation performed)")
        return 0

    # Initialize components
    data_manager = DataManager(db_path=config.database.path)
    sim = PortfolioSimulator(
        data_manager=data_manager,
        initial_capital=config.simulation.initial_capital,
        years=config.simulation.years,
        simulations=config.simulation.simulations,
        end_date=config.simulation.end_date
    )
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)

    print("\n" + "="*60)
    print("Loading historical data...")
    print("="*60)

    # Load all assets
    asset_map = {}
    for ticker, asset_config in config.assets.items():
        asset = sim.define_asset_from_ticker(
            asset_config.ticker,
            name=asset_config.name,
            lookback_years=config.simulation.lookback_years
        )
        asset_map[ticker.upper()] = asset

    print("\n" + "="*60)
    print("Running Portfolio Simulations")
    print("="*60)

    # Simulate all portfolios
    portfolio_results = []
    for portfolio_config in config.portfolios:
        print(f"\nSimulating: {portfolio_config.name}...")

        # Build asset list and allocations from config
        portfolio_assets = []
        portfolio_allocations = []
        for asset_name, weight in portfolio_config.allocations.items():
            portfolio_assets.append(asset_map[asset_name.upper()])
            portfolio_allocations.append(weight)

        # Run simulation
        results = sim.simulate_portfolio(
            portfolio_assets,
            portfolio_allocations,
            method=config.simulation.method
        )

        portfolio_results.append({
            'label': portfolio_config.name,
            'assets': portfolio_assets,
            'allocations': portfolio_allocations,
            'results': results
        })

        # Save to database
        if config.database.save_results:
            data_manager.save_optimization_result(
                portfolio_name=portfolio_config.name,
                assets=portfolio_assets,
                allocations=portfolio_allocations,
                stats=results['stats'],
                optimization_params={'method': 'manual', 'score': 0}
            )

        # Print statistics
        sim.print_detailed_stats(results, portfolio_config.name)

    # Run optimization if configured
    if config.optimization:
        print("\n" + "="*60)
        print("Running Portfolio Optimization")
        print("="*60)

        # Get assets for optimization
        opt_assets = [asset_map[name.upper()] for name in config.optimization.assets]

        # Run optimization
        if config.optimization.method == 'grid_search':
            top_portfolios = optimizer.grid_search(
                assets=opt_assets,
                objective_weights=config.optimization.objective_weights,
                grid_points=config.optimization.grid_points,
                top_n=config.optimization.top_n
            )
        else:
            top_portfolios = optimizer.random_search(
                assets=opt_assets,
                objective_weights=config.optimization.objective_weights,
                n_iterations=config.optimization.n_iterations,
                top_n=config.optimization.top_n
            )

        # Save optimization results
        optimizer.save_optimized_portfolios(top_portfolios, prefix=f"{config.optimization.method}")

        # Add top 3 to visualization
        for i, opt_result in enumerate(top_portfolios[:3]):
            alloc_str = ' / '.join([
                f"{int(a*100)}% {asset['ticker']}"
                for a, asset in zip(opt_result['allocations'], opt_assets)
            ])

            portfolio_results.append({
                'label': f"Optimized #{i+1}: {alloc_str}",
                'assets': opt_assets,
                'allocations': opt_result['allocations'],
                'results': opt_result['results']
            })

        # Print optimization results
        print("\n" + "="*60)
        print("Top Optimized Portfolios:")
        print("="*60)
        for i, opt_result in enumerate(top_portfolios):
            print(f"\nRank #{i+1} - Score: {opt_result['score']:.4f}")
            print("Allocation:")
            for asset, alloc in zip(opt_assets, opt_result['allocations']):
                print(f"  {asset['ticker']}: {alloc*100:.1f}%")
            stats = opt_result['stats']
            print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
            print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
            print(f"Sortino Ratio: {stats['sortino_ratio']:.3f}")
            print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

    # Create visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    dashboard = visualizer.create_dashboard(portfolio_results)

    if config.visualization.save_html:
        dashboard.write_html(config.visualization.output_filename)
        print(f"✓ Saved dashboard to: {config.visualization.output_filename}")

    if config.visualization.show_browser:
        dashboard.show()

    # Database info
    print("\n" + "="*60)
    print("Database Information")
    print("="*60)
    print(f"Database file: {config.database.path}")

    print(f"\nStored tickers:")
    tickers = data_manager.list_all_tickers()
    for ticker in tickers:
        info = data_manager.get_ticker_info(ticker)
        if info:
            print(f"  {ticker}: {info['record_count']} records "
                  f"({info['start_date']} to {info['end_date']})")

    print(f"\nStored optimization results:")
    opt_results = data_manager.get_optimization_results(limit=10)
    for result in opt_results:
        print(f"  {result['portfolio_name']}")
        print(f"    Score: {result['score']:.4f}, "
              f"CAGR: {result['median_cagr']*100:.2f}%, "
              f"Sharpe: {result['sharpe_ratio']:.3f}")

    # Clean up
    data_manager.close()
    print("\n✓ Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())