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
            name='Conservative 60/40',
            allocations={
                'VOO': 0.40,  # S&P 500
                'QQQ': 0.20,  # Nasdaq 100
                'BND': 0.40   # Bonds
            },
            description='40% stocks, 40% bonds'
        ),

        # Balanced: Mix of everything
        PortfolioConfig(
            name='Balanced',
            allocations={
                'VOO': 0.50,
                'QQQ': 0.30,
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


        # Aggressive: All stocks
        PortfolioConfig(
            name='Custom Aggressive 100% Stocks',
            allocations={
                'VOO': 0.60,
                'NVDA': 0.30,
                'ARKK': 0.10
            },
            description='100% stocks, custom'
        ),
    ],

    # Optional: Customize simulation settings
    simulation=SimulationConfig(
        initial_capital=100000,  # $100k starting
        years=10,                # 10 year horizon
        simulations=1000,       # 10k Monte Carlo runs
        method='bootstrap'       # Use historical data resampling
    ),

    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND', 'TQQQ', 'VCR', 'NVDA', 'ARKK'],
        method='grid_search',    # Grid search over allocations
        grid_points=5,          # 5 points per asset (coarse)
        top_n=5,                # Show top 5 results
        objective_weights={      # Weights for custom objective function
            'return': 0.5,          # Maximize return
            'sharpe': 0.25,          # Maximize Sharpe ratio
            'drawdown': 0.25,        # Minimize drawdown
        }
    ),

    visualization=VisualizationConfig(
        save_html=True,  # Save dashboard to HTML file
        show_browser=True,  # Open in web browser automatically
        output_filename='portfolio_dashboard.html'  # Output file name
    ),
)


