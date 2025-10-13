"""
Main script to run Monte Carlo portfolio simulations with configuration files.

Usage:
    python main.py                          # Use default config
    python main.py --config my_config.py    # Use specific config
    python main.py --list-configs           # List available configs
"""

import argparse
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer
from config_loader import ConfigLoader

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Simulator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')
    args = parser.parse_args()

    # Load configuration
    loader = ConfigLoader()

    if args.list_configs:
        print("Available configuration files:")
        for config_file in loader.list_available_configs():
            print(f"  - {config_file}")
        return

    config = loader.load_config(args.config)
    loader.print_config_summary()

    # Extract settings
    sim_settings = config['simulation']
    db_settings = config['database']
    viz_settings = config['visualization']

    # Initialize
    data_manager = DataManager(db_path=db_settings.get('path', 'stock_data.db'))
    sim = PortfolioSimulator(
        data_manager=data_manager,
        initial_capital=sim_settings.get('initial_capital', 100000),
        years=sim_settings.get('years', 10),
        simulations=sim_settings.get('simulations', 10000)
    )
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)

    print("="*60)
    print("Loading and analyzing historical data...")
    print("="*60)

    # Load assets
    assets = {}
    for key, asset_config in config['assets'].items():
        assets[key] = sim.define_asset_from_ticker(
            asset_config['ticker'],
            name=asset_config.get('name', asset_config['ticker']),
            lookback_years=asset_config.get('lookback_years', 10)
        )

    # Run portfolio simulations
    print("\n" + "="*60)
    print("Running Portfolio Simulations")
    print("="*60)

    portfolio_configs = []
    for portfolio_def in config['portfolios']:
        print(f"\nSimulating: {portfolio_def['name']}...")

        # Build asset list and allocations
        portfolio_assets = []
        portfolio_allocations = []
        for asset_key, weight in portfolio_def['allocations'].items():
            portfolio_assets.append(assets[asset_key])
            portfolio_allocations.append(weight)

        # Run simulation
        results = sim.simulate_portfolio(
            portfolio_assets,
            portfolio_allocations,
            method=sim_settings.get('method', 'bootstrap')
        )

        portfolio_configs.append({
            'label': portfolio_def['name'],
            'assets': portfolio_assets,
            'allocations': portfolio_allocations,
            'results': results
        })

        # Save to database
        if db_settings.get('save_results', True):
            data_manager.save_optimization_result(
                portfolio_name=portfolio_def['name'],
                assets=portfolio_assets,
                allocations=portfolio_allocations,
                stats=results['stats'],
                optimization_params={
                    'method': 'manual',
                    'score': 0
                }
            )

        # Print statistics
        sim.print_detailed_stats(results, portfolio_def['name'])

    # Run optimization if enabled
    if config['optimization'].get('enabled', False):
        print("\n" + "="*60)
        print("Running Portfolio Optimization")
        print("="*60)

        opt_config = config['optimization']
        objective_weights = opt_config.get('objective_weights', {
            'return': 0.50,
            'sharpe': 0.20,
            'drawdown': 0.30
        })

        # Main optimization
        optimize_asset_keys = opt_config.get('optimize_assets', [])
        if optimize_asset_keys:
            optimize_assets = [assets[key] for key in optimize_asset_keys]

            if opt_config.get('method') == 'grid_search':
                top_portfolios = optimizer.grid_search(
                    assets=optimize_assets,
                    objective_weights=objective_weights,
                    grid_points=opt_config.get('grid_points', 6),
                    top_n=opt_config.get('top_n', 5)
                )
            else:
                top_portfolios = optimizer.random_search(
                    assets=optimize_assets,
                    objective_weights=objective_weights,
                    n_iterations=opt_config.get('n_iterations', 50),
                    top_n=opt_config.get('top_n', 5)
                )

            # Save optimization results
            optimizer.save_optimized_portfolios(
                top_portfolios,
                prefix=f"{opt_config.get('method', 'optimized')}"
            )

            # Add top results to visualization
            for i, opt_result in enumerate(top_portfolios[:3]):
                alloc_str = ' / '.join([
                    f"{int(a*100)}% {asset['ticker']}"
                    for a, asset in zip(opt_result['allocations'], optimize_assets)
                ])

                portfolio_configs.append({
                    'label': f"Optimized #{i+1}: {alloc_str}",
                    'assets': optimize_assets,
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
                for asset, alloc in zip(optimize_assets, opt_result['allocations']):
                    print(f"  {asset['ticker']}: {alloc*100:.1f}%")
                stats = opt_result['stats']
                print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
                print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
                print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

        # Additional optimizations
        for add_opt in opt_config.get('additional_optimizations', []):
            print(f"\n{add_opt['name']}:")
            add_assets = [assets[key] for key in add_opt['assets']]

            if add_opt.get('method') == 'random_search':
                add_results = optimizer.random_search(
                    assets=add_assets,
                    objective_weights=objective_weights,
                    n_iterations=add_opt.get('n_iterations', 50),
                    top_n=3
                )
            else:
                add_results = optimizer.grid_search(
                    assets=add_assets,
                    objective_weights=objective_weights,
                    grid_points=add_opt.get('grid_points', 5),
                    top_n=3
                )

            optimizer.save_optimized_portfolios(add_results, prefix=add_opt['name'].replace(' ', '_'))

            # Add best result to visualization
            if add_results:
                alloc_str = ' / '.join([
                    f"{int(a*100)}% {asset['ticker']}"
                    for a, asset in zip(add_results[0]['allocations'], add_assets)
                ])
                portfolio_configs.append({
                    'label': f"{add_opt['name']}: {alloc_str}",
                    'assets': add_assets,
                    'allocations': add_results[0]['allocations'],
                    'results': add_results[0]['results']
                })

    # Create visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    dashboard = visualizer.create_dashboard(portfolio_configs)

    if viz_settings.get('save_html', True):
        filename = viz_settings.get('output_filename', 'portfolio_dashboard.html')
        dashboard.write_html(filename)
        print(f"✓ Saved dashboard to: {filename}")

    if viz_settings.get('show_browser', True):
        dashboard.show()

    # Database info
    print("\n" + "="*60)
    print("Database Information")
    print("="*60)
    print(f"Database file: {db_settings.get('path', 'stock_data.db')}")

    print(f"\nStored tickers:")
    tickers = data_manager.list_all_tickers()
    for ticker in tickers:
        info = data_manager.get_ticker_info(ticker)
        if info:
            print(f"  {ticker}: {info['record_count']} records ({info['start_date']} to {info['end_date']})")

    print(f"\nStored optimization results:")
    opt_results = data_manager.get_optimization_results(limit=10)
    for result in opt_results:
        print(f"  {result['portfolio_name']}")
        print(f"    Score: {result['score']:.4f}, CAGR: {result['median_cagr']*100:.2f}%, Sharpe: {result['sharpe_ratio']:.3f}")

    # Clean up
    data_manager.close()
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
