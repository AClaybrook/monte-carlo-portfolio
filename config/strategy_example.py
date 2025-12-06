"""
Example Configuration: Dynamic Allocation Strategies

This config demonstrates various dynamic allocation strategies including:
- Static DCA (baseline)
- Buy the Dip (increase allocation to beaten-down assets)
- Crypto Opportunistic (buy more crypto when it's down)
- Momentum-based allocation
"""

from run_config import (
    RunConfig, PortfolioConfig, SimulationConfig,
    OptimizationConfig, VisualizationConfig, StrategyConfig
)
REPLACEME = "TQQQ"
config = RunConfig(
    name="Dynamic Strategy Analysis",

    portfolios=[
        # =========================================================
        # BASELINE: Static DCA into diversified portfolio
        # =========================================================
        PortfolioConfig(
            name='Baseline: Static 80/20',
            allocations={
                'VOO': 0.80,      # S&P 500
                'BND': 0.20,      # Bonds
            },
            description='Static DCA - always same allocation',
            strategy=StrategyConfig(type='static')  # Default behavior
        ),

        # =========================================================
        # CRYPTO OPPORTUNISTIC: Your specific request!
        # Normally 10% BTC, but ramp up to 40% when BTC crashes
        # =========================================================
        PortfolioConfig(
            name='Crypto Opportunistic (Buy BTC Dips)',
            allocations={
                'VOO': 0.70,      # S&P 500
                'BND': 0.10,      # Bonds
                REPLACEME: 0.20,  # Bitcoin (base allocation)
            },
            description='Increase BTC allocation during major drawdowns',
            strategy=StrategyConfig(
                type='crypto_opportunistic',
                params={
                    'crypto_ticker': REPLACEME,
                    'equity_ticker': 'VOO',
                    'dip_threshold': 0.30,      # When BTC is down 30%+
                    'normal_weight': 0.20,       # Normal: 20% BTC
                    'dip_weight': 0.50           # During dip: 50% of new money to BTC
                }
            )
        ),

        # =========================================================
        # LEVERAGED AGGRESSIVE: Buy TQQQ on dips
        # =========================================================
        PortfolioConfig(
            name='Leveraged Dip Buyer',
            allocations={
                'VOO': 1.00,    # 3x S&P 500
                'TQQQ': 0.00,    # 3x QQQ
            },
            description='Buy more TQQQ when it drops 15%+',
            strategy=StrategyConfig(
                type='buy_the_dip',
                params={
                    'target_ticker': 'TQQQ',
                    'threshold': 0.3,           # 50% drawdown triggers
                    'aggressive_weight': 0.70    # Put 70% into TQQQ during dip
                }
            )
        ),
        PortfolioConfig(
            name='VOO DCA',
            allocations={
                'VOO': 1.00,    # 3x S&P 500
            },
            description='Baseline DCA into VOO',
            strategy=StrategyConfig(
                type='static',
                params={}
            )
        ),
        PortfolioConfig(
            name='Leveraged Dip Buyer Baseline',
            allocations={
                'VOO': 1.00,    # 3x S&P 500
                'TQQQ': 0.00,    # 3x QQQ
            },
            description='Buy more TQQQ when it drops 15%+',
            strategy=StrategyConfig(
                type='buy_the_dip',
                params={
                    'target_ticker': 'TQQQ',
                    'threshold': 1.00,           # 50% drawdown triggers
                    'aggressive_weight': 0.70    # Put 70% into TQQQ during dip
                }
            )
        ),

        # =========================================================
        # DRAWDOWN PROTECTION: De-risk during crashes
        # =========================================================
        # PortfolioConfig(
        #     name='Drawdown Protected Growth',
        #     allocations={
        #         'VOO': 0.70,
        #         'QQQ': 0.20,
        #         'BND': 0.10,
        #     },
        #     description='Shift to bonds when portfolio drops 15%+',
        #     strategy=StrategyConfig(
        #         type='drawdown_protection',
        #         params={
        #             'threshold': 0.15,           # 15% portfolio drawdown
        #             'risk_off_allocation': {     # Defensive allocation
        #                 'VOO': 0.30,
        #                 'BND': 0.70
        #             },
        #             'recovery_threshold': 0.05   # Return to normal at 5% DD
        #         }
        #     )
        # ),

        # =========================================================
        # RELATIVE VALUE: Buy whatever is most beaten down
        # =========================================================
        PortfolioConfig(
            name='Relative Value Rotator',
            allocations={
                'VOO': 0.50,      # S&P 500
                'QQQ': 0.25,      # Nasdaq
                'VGT': 0.25,      # Tech
                REPLACEME: 0.0,      # Emerging Markets
            },
            description='Tilt toward most beaten-down asset',
            strategy=StrategyConfig(
                type='relative_value',
                params={
                    'threshold': 0.25,   # 10% drawdown spread triggers
                    'max_tilt': 0.50     # Max 50% in any single asset
                }
            )
        ),
        PortfolioConfig(
            name='Relative Value Rotator Static',
            allocations={
                'VOO': 0.50,      # S&P 500
                'QQQ': 0.25,      # Nasdaq
                'VGT': 0.25,      # Tech
                REPLACEME: 0.0,      # Emerging Markets
            },
            description='Tilt toward most beaten-down asset',
            strategy=StrategyConfig(
                type='static',
            )
        ),
    ],

    simulation=SimulationConfig(
        initial_capital=10000,
        years=10,
        simulations=10000,
        method='geometric_brownian',
        lookback_years=10,

        # DCA Settings - CRITICAL for strategy testing!
        contribution_amount=500.0,    # $500/month
        contribution_frequency=21     # ~monthly (21 trading days)
    ),

    optimization=OptimizationConfig(
        assets=['VOO', 'QQQ', 'BND'],
        active_strategies=['max_sharpe', 'min_volatility'],
        benchmark_ticker='VOO'
    ),

    visualization=VisualizationConfig(
        output_filename='dynamic_strategy_analysis.html'
    ),
)