"""
Example configuration for enhanced dashboard testing.

This config demonstrates comparing multiple portfolio strategies:
- Conservative (heavy bonds)
- Balanced (60/40)
- Aggressive (heavy stocks)
- All stocks
"""

from run_config import (
    RunConfig,
    AssetConfig,
    PortfolioConfig,
    SimulationConfig,
    OptimizationConfig
)

config = RunConfig(
    name="Portfolio Strategy Comparison",

    # Define portfolios to compare
    portfolios=[
        PortfolioConfig(
            name='Conservative (40/60)',
            allocations={'VOO': 0.40, 'BND': 0.60},
            description='Low risk, stable returns'
        ),
        PortfolioConfig(
            name='Balanced (60/40)',
            allocations={'VOO': 0.60, 'BND': 0.40},
            description='Classic balanced allocation'
        ),
        PortfolioConfig(
            name='Aggressive (80/20)',
            allocations={'VOO': 0.80, 'BND': 0.20},
            description='Higher risk, higher potential returns'
        ),
        PortfolioConfig(
            name='All Stocks (100/0)',
            allocations={'VOO': 1.00},
            description='Maximum growth potential'
        ),
    ],

    # Optional: Customize asset settings if needed
    assets={
        'VOO': AssetConfig(
            ticker='VOO',
            name='Vanguard S&P 500 ETF',
            lookback_years=10
        ),
        'BND': AssetConfig(
            ticker='BND',
            name='Vanguard Total Bond Market ETF',
            lookback_years=10
        ),
    },

    # Simulation settings
    simulation=SimulationConfig(
        initial_capital=100000,
        years=10,
        simulations=10000,
        method='bootstrap',  # Use historical resampling
        lookback_years=10
    ),

    # Optional: Enable optimization to find best allocation
    optimization=OptimizationConfig(
        assets=['VOO', 'BND'],
        method='grid_search',
        objective_weights={
            'return': 0.40,     # 40% weight on returns
            'sharpe': 0.30,     # 30% weight on risk-adjusted returns
            'drawdown': 0.30    # 30% weight on limiting losses
        },
        grid_points=11,  # Test 0%, 10%, 20%, ..., 100% allocations
        top_n=5
    )
)