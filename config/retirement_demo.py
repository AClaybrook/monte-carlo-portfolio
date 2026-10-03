"""
Retirement / withdrawal demo: the "4% rule" on three stock/bond mixes.

$1,000,000 with $40,000 a year withdrawn monthly ($3,333), raised 2.5% a year for
inflation, simulated 30 years with block bootstrap. Run with --synthetic offline.
"""
from run_config import RunConfig, PortfolioConfig, SimulationConfig, VisualizationConfig

config = RunConfig(
    name="Retirement withdrawals (4% rule)",
    benchmark_ticker='VOO',
    portfolios=[
        PortfolioConfig(name='40/60', allocations={'VOO': 0.4, 'BND': 0.6}),
        PortfolioConfig(name='60/40', allocations={'VOO': 0.6, 'BND': 0.4}),
        PortfolioConfig(name='100% stocks', allocations={'VOO': 1.0}),
    ],
    simulation=SimulationConfig(
        initial_capital=1_000_000,
        start_date='2011-01-01', end_date='2025-12-31',
        withdrawal_amount=40_000 / 12, withdrawal_frequency='monthly',
        cash_flow_growth=0.025,
        rebalance='annual',
        years=30, simulations=2000, method='block_bootstrap', block_size=63, seed=7,
    ),
    visualization=VisualizationConfig(output_filename='retirement_demo.html'),
)
