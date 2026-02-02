"""
Example configuration - Selectable Optimizations & Benchmark.
"""

from run_config import RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig, VisualizationConfig, StrategyConfig

config = RunConfig(
    name="Strategic Portfolio Analysis",

    portfolios=[
        # PortfolioConfig(
        #     name='60/40 Classic',
        #     allocations={'VOO': 0.60, 'BND': 0.40},
        #     description='Classic 60/40 Stock/Bond Portfolio'
        # ),
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
        # In your config file:
        PortfolioConfig(
            name='Crypto Opportunistic',
            allocations={
                'VOO': 0.70,       # S&P 500 base
                'BND': 0.10,       # Bonds
                'BTC-USD': 0.20,   # Bitcoin base
            },
            strategy=StrategyConfig(
                type='crypto_opportunistic',
                params={
                    'crypto_ticker': 'BTC-USD',
                    'equity_ticker': 'VOO',
                    'dip_threshold': 0.25,      # When BTC down 25%+
                    'normal_weight': 0.20,       # Normal: 20% to BTC
                    'dip_weight': 0.50           # During dip: 50% of new DCA to BTC
                }
            )
        )
        # PortfolioConfig(
        #     name='All Weather',
        #     allocations={'VTI': 0.30, 'TLT': 0.40, 'IEF': 0.15, 'GLD': 0.075, 'DBC': 0.075}
        # )
    ],

    simulation=SimulationConfig(
        initial_capital=10000,
        years=30,
        simulations=10000,
        method='geometric_brownian',
        start_date='2017-01-01',
        end_date='2025-12-01',
        # lookback_years=10,  # Not needed when using explicit dates
        contribution_amount=0.0,
        contribution_frequency=21  # trading days
    ),

    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BTC-USD', 'SPXL', 'SHV', 'VGT'],
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
            'return':   0.60,  # CAGR
            'sharpe':   0.10,  # risk-adjusted return
            'drawdown': 0.10,  # minimizing deep losses
            'volatility': 0.0,
            'sortino': 0.10
        },

        benchmark_ticker='VFINX'
    ),

    visualization=VisualizationConfig(
        output_filename='portfolio_analysis.html'
    ),
)