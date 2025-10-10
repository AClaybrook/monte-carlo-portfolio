"""
Main script to run Monte Carlo portfolio simulations with optimization.
"""

from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer

def main():
    # Initialize
    data_manager = DataManager(db_path='stock_data.db')
    sim = PortfolioSimulator(
        data_manager=data_manager,
        initial_capital=100000,
        years=10,
        simulations=10000
    )
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)

    print("="*60)
    print("Monte Carlo Portfolio Simulator with Optimization")
    print("="*60)
    print("Loading and analyzing historical data...")
    print("="*60)

    # Define assets
    voo = sim.define_asset_from_ticker('VOO', name='VOO (S&P 500)')
    tqqq = sim.define_asset_from_ticker('TQQQ', name='TQQQ (3x Nasdaq)')
    bnd = sim.define_asset_from_ticker('BND', name='BND (Bonds)')
    qqq = sim.define_asset_from_ticker('QQQ', name='QQQ (Nasdaq 100)')

    # =================================================================
    # PART 1: Standard Portfolio Analysis
    # =================================================================
    print("\n" + "="*60)
    print("PART 1: Standard Portfolio Analysis")
    print("="*60)

    portfolio_configs = [
        {
            'label': '100% VOO',
            'assets': [voo],
            'allocations': [1.0],
            'results': None
        },
        {
            'label': '80% VOO / 20% TQQQ',
            'assets': [voo, tqqq],
            'allocations': [0.8, 0.2],
            'results': None
        },
        {
            'label': '50% VOO / 50% QQQ',
            'assets': [voo, qqq],
            'allocations': [0.5, 0.5],
            'results': None
        },
        {
            'label': '70% VOO / 20% QQQ / 10% BND',
            'assets': [voo, qqq, bnd],
            'allocations': [0.7, 0.2, 0.1],
            'results': None
        },
        {
            'label': '60% VOO / 30% TQQQ / 10% BND',
            'assets': [voo, tqqq, bnd],
            'allocations': [0.6, 0.3, 0.1],
            'results': None
        }
    ]

    # Run simulations
    for config in portfolio_configs:
        print(f"\nSimulating: {config['label']}...")
        config['results'] = sim.simulate_portfolio(
            config['assets'],
            config['allocations'],
            method='bootstrap'
        )

    # Print statistics
    for config in portfolio_configs:
        sim.print_detailed_stats(config['results'], config['label'])

    # =================================================================
    # PART 2: Portfolio Optimization
    # =================================================================
    print("\n" + "="*60)
    print("PART 2: Portfolio Optimization")
    print("="*60)

    # Define objective function weights
    objective_weights = {
        'return': 0.50,      # 50% weight on expected return
        'sharpe': 0.20,      # 20% weight on Sharpe ratio
        'drawdown': 0.30     # 30% weight on (1 - max drawdown)
    }

    # Run optimization on 3-asset portfolio
    optimization_assets = [voo, qqq, tqqq]

    # Use grid search for 3 assets (manageable)
    top_portfolios = optimizer.grid_search(
        assets=optimization_assets,
        objective_weights=objective_weights,
        grid_points=6,  # 6^3 = 216 combinations
        top_n=5
    )

    # Save optimization results to database
    optimizer.save_optimized_portfolios(top_portfolios, prefix="GridSearch_3Asset")

    # Add top optimized portfolios to visualization
    for i, opt_result in enumerate(top_portfolios[:3]):  # Add top 3
        alloc_str = ' / '.join([
            f"{int(a*100)}% {asset['ticker']}"
            for a, asset in zip(opt_result['allocations'], optimization_assets)
        ])

        portfolio_configs.append({
            'label': f"Optimized #{i+1}: {alloc_str}",
            'assets': optimization_assets,
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
        for asset, alloc in zip(optimization_assets, opt_result['allocations']):
            print(f"  {asset['ticker']}: {alloc*100:.1f}%")
        stats = opt_result['stats']
        print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {stats['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

    # =================================================================
    # PART 3: Advanced Optimization - Random Search with 4 Assets
    # =================================================================
    print("\n" + "="*60)
    print("PART 3: Advanced Optimization (4 Assets - Random Search)")
    print("="*60)

    # For 4+ assets, use random search (grid search becomes too slow)
    optimization_assets_4 = [voo, qqq, tqqq, bnd]

    top_portfolios_4 = optimizer.random_search(
        assets=optimization_assets_4,
        objective_weights=objective_weights,
        n_iterations=10,  # Test 10 random portfolios
        top_n=3
    )

    # Save to database
    optimizer.save_optimized_portfolios(top_portfolios_4, prefix="RandomSearch_4Asset")

    # Add to visualization
    for i, opt_result in enumerate(top_portfolios_4):
        alloc_str = ' / '.join([
            f"{int(a*100)}% {asset['ticker']}"
            for a, asset in zip(opt_result['allocations'], optimization_assets_4)
        ])

        portfolio_configs.append({
            'label': f"4-Asset Opt #{i+1}: {alloc_str}",
            'assets': optimization_assets_4,
            'allocations': opt_result['allocations'],
            'results': opt_result['results']
        })

    print("\n" + "="*60)
    print("Top 4-Asset Optimized Portfolios:")
    print("="*60)
    for i, opt_result in enumerate(top_portfolios_4):
        print(f"\nRank #{i+1} - Score: {opt_result['score']:.4f}")
        print("Allocation:")
        for asset, alloc in zip(optimization_assets_4, opt_result['allocations']):
            print(f"  {asset['ticker']}: {alloc*100:.1f}%")
        stats = opt_result['stats']
        print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {stats['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

    # =================================================================
    # PART 4: Visualizations
    # =================================================================
    print("\n" + "="*60)
    print("Generating interactive visualizations...")
    print("="*60)

    # Create main dashboard with all portfolios (standard + optimized)
    dashboard = visualizer.create_dashboard(portfolio_configs)
    dashboard.write_html('portfolio_dashboard.html')
    print("✓ Saved dashboard to: portfolio_dashboard.html")

    # Show main dashboard in browser
    dashboard.show()

    # =================================================================
    # Database Information
    # =================================================================
    print("\n" + "="*60)
    print("Database Information")
    print("="*60)
    print(f"Database file: stock_data.db")

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
    print("\nInteractive Features:")
    print("  - Click legend items to show/hide portfolios across ALL charts")
    print("  - Hover over any point for detailed information")
    print("  - Zoom and pan on any chart")
    print("  - Double-click legend to isolate a single portfolio")

if __name__ == "__main__":
    main()