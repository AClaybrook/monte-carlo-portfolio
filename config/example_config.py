"""
Simple example configuration - good starting point.

This shows the minimum you need to run a simulation.
Copy this file to config/my_portfolios.py and customize it.
"""

from run_config import (
    RunConfig,
    AssetConfig,
    PortfolioConfig,
    SimulationConfig
)

config = RunConfig(
    name="Simple 3-Fund Portfolio Analysis",

    # Step 1: Define your assets (these will be downloaded from Yahoo Finance)
    assets=[
        AssetConfig(ticker='VOO', name='S&P 500 ETF'),
        AssetConfig(ticker='QQQ', name='Nasdaq 100 ETF'),
        AssetConfig(ticker='BND', name='Total Bond Market'),
    ],

    # Step 2: Define portfolios to test
    portfolios=[
        # Conservative: More bonds
        PortfolioConfig(
            name='Conservative 60/40',
            allocations={
                'voo': 0.40,
                'qqq': 0.20,
                'bnd': 0.40
            },
            description='40% stocks, 40% bonds'
        ),

        # Balanced: Mix of everything
        PortfolioConfig(
            name='Balanced',
            allocations={
                'voo': 0.50,
                'qqq': 0.30,
                'bnd': 0.20
            },
            description='80% stocks, 20% bonds'
        ),

        # Aggressive: All stocks
        PortfolioConfig(
            name='Aggressive 100% Stocks',
            allocations={
                'voo': 0.60,
                'qqq': 0.40
            },
            description='100% stocks, no bonds'
        ),
    ],

    # Step 3: Simulation settings (optional, uses defaults if not specified)
    simulation=SimulationConfig(
        initial_capital=100000,  # $100k starting
        years=10,                # 10 year horizon
        simulations=10000,       # 10k Monte Carlo runs
        method='bootstrap'       # Use historical data resampling
    ),
)