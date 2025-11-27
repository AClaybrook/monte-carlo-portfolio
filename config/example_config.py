"""
Example configuration updated for Efficient Frontier Optimization & Backtesting.

Assets are automatically discovered from your portfolio allocations!
Copy this file to config/my_portfolios.py and customize it.
"""

from run_config import RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig, VisualizationConfig

config = RunConfig(
    name="Long-Term 30 Year Analysis",

    # Define your manual portfolios to test
    portfolios=[
        # Conservative
        PortfolioConfig(
            name='60/40 Classic',
            allocations={
                'VOO': 0.60,
                'BND': 0.40
            },
            description='Standard 60/40 Stocks/Bonds'
        ),

        # Growth
        PortfolioConfig(
            name='Aggressive Growth',
            allocations={
                'VOO': 0.40,
                'QQQ': 0.40,
                'VGT': 0.20
            },
            description='Tech heavy growth'
        ),

        # Hedgefundie / Leverage
        PortfolioConfig(
            name='Hedgefundie Adventure',
            allocations={
                'UPRO': 0.55,
                'TMF': 0.45
            },
            description='Leveraged Risk Parity'
        ),
    ],

    # Simulation Settings
    simulation=SimulationConfig(
        initial_capital=10000,
        years=30,                 # UPDATED: 30 Year Horizon
        simulations=5000,         # 5,000 runs is usually sufficient for Cholesky method
        method='bootstrap',       # 'bootstrap' preserves historical correlations best
        lookback_years=20,        # Use up to 20 years of history if available
    ),

    # Optimization Settings
    # Note: The system now uses SciPy (SLSQP) to find the Efficient Frontier mathematically.
    # It will automatically calculate:
    # 1. Maximum Sharpe Ratio Portfolio
    # 2. Minimum Volatility Portfolio
    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND', 'VGT', 'GLD', 'VTI'],
        # The parameters below are kept for config validation but
        # the new engine uses mathematical optimization (Efficient Frontier)
        # rather than random grid search.
        method='grid_search',
        objective_weights={
            'return': 0.5,
            'sharpe': 0.5,
            'drawdown': 0.0,
        }
    ),

    visualization=VisualizationConfig(
        save_html=True,
        show_browser=True,
        output_filename='portfolio_analysis_30yr.html'
    ),
)