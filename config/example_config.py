"""
Simple example configuration - good starting point.

Assets are automatically discovered from your portfolio allocations!
Copy this file to config/my_portfolios.py and customize it.
"""

from run_config import RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig, VisualizationConfig, DatabaseConfig

config = RunConfig(
    name="Simple 3-Fund Portfolio Analysis",

    # Just define your portfolios - assets are discovered automatically!
    portfolios=[
        # Conservative: More bonds
        PortfolioConfig(
            name='SP500',
            allocations={
                'VOO': 1.0,  # S&P 500
            },
            description='100% S&P 500 (VOO)'
        ),

        # Balanced: Mix of everything
        PortfolioConfig(
            name='Balanced',
            allocations={
                'VOO': 0.80,
                'BND': 0.20
            },
            description='80% stocks, 20% bonds'
        ),

        # Aggressive: All stocks
        PortfolioConfig(
            name='Aggressive 100% Stocks',
            allocations={
                'VOO': 0.60,
                'QQQ': 0.40
            },
            description='100% stocks, no bonds'
        ),

        PortfolioConfig(
            name='Tech Heavy 100% Stocks',
            allocations={
                'VOO': 0.40,
                'VGT': 0.30,
                'QQQ': 0.30
            },
            description='100% stocks, tech heavy'
        ),

    ],

    # Optional: Customize simulation settings
    simulation=SimulationConfig(
        initial_capital=100000,  # $100k starting
        years=10,                # 10 year horizon
        simulations=1000,       # 10k Monte Carlo runs
        method='geometric_brownian',       # Use historical data resampling
        end_date='2025-10-01',
        lookback_years=5,      # 10 years of historical data (if available)
    ),

    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND', 'TQQQ', 'VCR', 'ARKK', 'VGT'],
        method='grid_search',    # Grid search over allocations
        grid_points=6,          # points per asset (coarse)
        top_n=5,                # Show top 5 results
        objective_weights={      # Weights for custom objective function
            'return': 0.5,       # Maximize return
            'sharpe': 0.25,     # Maximize Sharpe ratio
            'sortino': 0.25,    # Maximize Sortino ratio
            'worst_max_drawdown': 0.0,        # Minimize drawdown
        }
    ),

    visualization=VisualizationConfig(
        save_html=True,  # Save dashboard to HTML file
        show_browser=True,  # Open in web browser automatically
        output_filename='portfolio_dashboard.html'  # Output file name
    ),
)


