"""
Main script to run Monte Carlo portfolio simulations.
"""

from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from visualizations import PortfolioVisualizer

def main():
    # Initialize
    data_manager = DataManager(db_path='stock_data.db')
    sim = PortfolioSimulator(
        data_manager=data_manager,
        initial_capital=1000000,
        years=10,
        simulations=100000
    )
    visualizer = PortfolioVisualizer(sim)

    print("="*60)
    print("Loading and analyzing historical data...")
    print("="*60)

    # Define assets
    voo = sim.define_asset_from_ticker('VOO', name='VOO (S&P 500)')
    tqqq = sim.define_asset_from_ticker('TQQQ', name='TQQQ (3x Nasdaq)')
    bnd = sim.define_asset_from_ticker('BND', name='BND (Bonds)')
    qqq = sim.define_asset_from_ticker('QQQ', name='QQQ (Nasdaq 100)')

    print("\n" + "="*60)
    print("Running Monte Carlo simulations...")
    print("="*60)

    # Define portfolio configurations
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
            'label': '70% VOO / 30% QQQ',
            'assets': [voo, qqq],
            'allocations': [0.7, 0.3],
            'results': None
        },
        {
            'label': '50% VOO / 50% TQQQ',
            'assets': [voo, tqqq],
            'allocations': [0.5, 0.5],
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

    # Create visualizations
    print("\n" + "="*60)
    print("Generating interactive visualizations...")
    print("="*60)

    # Create main dashboard
    dashboard = visualizer.create_dashboard(portfolio_configs)
    dashboard.write_html('portfolio_dashboard.html')
    print("✓ Saved dashboard to: portfolio_dashboard.html")

    # Create individual portfolio analyses
    for config in portfolio_configs:
        filename = f"portfolio_{config['label'].replace('/', '_').replace(' ', '_')}.html"
        fig = visualizer.create_individual_portfolio_analysis(
            config['results'],
            config['label']
        )
        fig.write_html(filename)
        print(f"✓ Saved {config['label']} analysis to: {filename}")

    # Show main dashboard in browser
    dashboard.show()

    # Database info
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

    # Clean up
    data_manager.close()
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
