"""
Fast demo config, meant for `python main.py config/synthetic_demo.py --synthetic`.

Exercises lump sum + DCA, rebalancing, dynamic strategies and optimization in
well under a minute. With --synthetic the prices are generated, not real.
"""
from run_config import (
    RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig,
    VisualizationConfig, StrategyConfig, SweepConfig
)

config = RunConfig(
    name="Synthetic Demo",
    portfolios=[
        PortfolioConfig(name='60/40', allocations={'VOO': 0.60, 'BND': 0.40}),
        PortfolioConfig(name='Growth + BTC', allocations={'VOO': 0.50, 'QQQ': 0.40, 'BTC-USD': 0.10}),
        PortfolioConfig(
            name='Buy the Dip (QQQ)',
            allocations={'VOO': 0.70, 'QQQ': 0.30},
            strategy=StrategyConfig(type='buy_the_dip',
                                    params={'target_ticker': 'QQQ', 'threshold': 0.20,
                                            'aggressive_weight': 0.80}),
        ),
    ],
    simulation=SimulationConfig(
        initial_capital=10000,
        years=10,
        simulations=2000,
        method='bootstrap',
        start_date='2015-01-01',
        end_date='2025-12-31',
        contribution_amount=500.0,
        contribution_frequency=21,
    ),
    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND'],
        active_strategies=['max_sharpe', 'min_volatility'],
        benchmark_ticker='VOO',
    ),
    sweeps=[
        SweepConfig(
            name='Drawdown protection thresholds',
            allocations={'VOO': 0.8, 'BND': 0.2},
            strategy=StrategyConfig('drawdown_protection', apply_to='rebalance', check_frequency='daily',
                                    params={'risk_off_allocation': {'BND': 1.0}}),
            grid={'threshold': [0.08, 0.12, 0.16, 0.20, 0.25],
                  'recovery_threshold': [0.0, 0.03, 0.06]},
            simulations=300,
        ),
    ],
    visualization=VisualizationConfig(output_filename='synthetic_demo.html'),
)
