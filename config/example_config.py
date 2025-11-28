"""
Example configuration - Selectable Optimizations & Benchmark.
"""

from run_config import RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig, VisualizationConfig

config = RunConfig(
    name="Strategic Portfolio Analysis",

    portfolios=[
        PortfolioConfig(
            name='60/40 Classic',
            allocations={'VOO': 0.60, 'BND': 0.40},
            description='Classic 60/40 Stock/Bond Portfolio'
        ),
        PortfolioConfig(
            name='Hedgefundie Adventure',
            allocations={
                'SPXL': 0.6,
                'TMF': 0.4
            },
            description='Leveraged Risk Parity'
        ),
        PortfolioConfig(
            name='Aggressive Growth',
            allocations={
                'VOO': 0.40,
                'QQQ': 0.40,
                'VGT': 0.20
            },
            description='Tech heavy growth'
        ),
        PortfolioConfig(
            name='Aggressive + Crypto ',
            allocations={
                'VOO': 0.40,
                'QQQ': 0.25,
                'VGT': 0.25,
                'BTC-USD': 0.10
            },
            description='Tech heavy growth + Crypto'
        ),
        PortfolioConfig(
            name='Leveraged Tech Heavy 100% Stocks',
            allocations={
                'SPXL': 0.5,
                'TQQQ': 0.5
            },
            description='Leveraged 100% stocks'
        ),
        # PortfolioConfig(
        #     name='All Weather',
        #     allocations={'VTI': 0.30, 'TLT': 0.40, 'IEF': 0.15, 'GLD': 0.075, 'DBC': 0.075}
        # )
    ],

    simulation=SimulationConfig(
        initial_capital=10000,
        years=30,
        simulations=5000,
        method='geometric_brownian',
        lookback_years=10,
    ),

    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND', 'VGT', 'GLD', 'VTI', 'TLT', 'TQQQ', 'SPXL', 'TMF', 'BTC-USD'],
        # 1. Active strategies
        active_strategies=[
            'max_sharpe',      # Balanced Growth
            'min_volatility',  # Defensive
            'risk_parity',     # Diversified Risk
            'max_sortino',   # Aggressive Growth (Uncomment to enable)
            'custom_weighted'  # Custom Weighted Strategy
        ],

        # 2. Define your Custom Objective Function
        objective_weights={
            'return':   0.35,  # CAGR
            'sharpe':   0.30,  # risk-adjusted return
            'drawdown': 0.20,  # minimizing deep losses
            'volatility': 0.05,
            'sortino': 0.10
        },

        benchmark_ticker='VFINX'
    ),

    visualization=VisualizationConfig(
        output_filename='portfolio_analysis.html'
    ),
)