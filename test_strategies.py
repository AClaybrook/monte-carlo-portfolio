"""
Strategy Testing Utility

A standalone script for testing and comparing dynamic allocation strategies.
Run directly or import into your analysis workflow.

Usage:
    python test_strategies.py                    # Run default comparison
    python test_strategies.py --ticker BTC-USD   # Test BTC strategies
    python test_strategies.py --dca 1000         # Test with $1000/month DCA
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from backtester import Backtester, StrategyComparison
from run_config import SimulationConfig
from strategies import (
    StaticAllocationStrategy,
    BuyTheDipStrategy,
    MomentumStrategy,
    RelativeValueStrategy,
    DrawdownProtectionStrategy,
    create_crypto_opportunistic_strategy,
)


def load_assets(data_manager, tickers, lookback_years=10):
    """Load asset data for given tickers"""
    from datetime import date

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * lookback_years)

    assets = []
    for ticker in tickers:
        df = data_manager.get_data(ticker, start_date, end_date)
        returns = df['Adj Close'].pct_change().dropna()

        assets.append({
            'ticker': ticker,
            'name': ticker,
            'historical_returns': returns,
            'full_data': df,
            'daily_mean': returns.mean(),
            'daily_std': returns.std()
        })

    return assets


def test_crypto_dip_buying():
    """
    Test the "buy crypto when it's down" strategy.

    Compares:
    1. Static allocation (always same weights)
    2. Opportunistic (buy more BTC when down 25%+)
    3. Aggressive opportunistic (buy more BTC when down 15%+)
    """
    print("\n" + "="*70)
    print("CRYPTO DIP BUYING STRATEGY TEST")
    print("="*70)

    dm = DataManager()

    # Load assets
    tickers = ['VOO', 'BND', 'BTC-USD']
    print(f"\nLoading data for: {tickers}")
    assets = load_assets(dm, tickers, lookback_years=6)  # BTC data limited

    # Base allocation: 60% VOO, 20% BND, 20% BTC
    base_alloc = [0.60, 0.20, 0.20]

    # Define strategies to compare
    strategies = [
        StaticAllocationStrategy(),

        create_crypto_opportunistic_strategy(
            crypto_ticker='BTC-USD',
            equity_ticker='VOO',
            crypto_dip_threshold=0.30,    # 30% dip
            normal_crypto_weight=0.20,
            dip_crypto_weight=0.50
        ),

        create_crypto_opportunistic_strategy(
            crypto_ticker='BTC-USD',
            equity_ticker='VOO',
            crypto_dip_threshold=0.20,    # More aggressive: 20% dip
            normal_crypto_weight=0.20,
            dip_crypto_weight=0.60
        ),
    ]
    # Rename for clarity
    strategies[1].name = "BTC Opportunistic (30% dip)"
    strategies[2].name = "BTC Aggressive (20% dip)"

    # Run comparison
    comparer = StrategyComparison(dm)
    results = comparer.compare_strategies(
        assets=assets,
        base_allocations=base_alloc,
        strategies=strategies,
        initial_capital=10000,
        contribution_amount=500,
        contribution_frequency=21
    )

    # Generate report
    comparer.generate_comparison_report(
        results,
        output_path="output/crypto_strategy_comparison.html"
    )

    dm.close()
    return results


def test_leveraged_dip_buying():
    """
    Test buying leveraged ETFs on dips.
    """
    print("\n" + "="*70)
    print("LEVERAGED DIP BUYING STRATEGY TEST")
    print("="*70)

    dm = DataManager()

    tickers = ['SPXL', 'TQQQ']
    print(f"\nLoading data for: {tickers}")
    assets = load_assets(dm, tickers, lookback_years=10)

    base_alloc = [0.50, 0.50]

    strategies = [
        StaticAllocationStrategy(),

        BuyTheDipStrategy(
            target_ticker='TQQQ',
            threshold=0.15,
            aggressive_weight=0.70
        ),

        BuyTheDipStrategy(
            target_ticker='TQQQ',
            threshold=0.25,
            aggressive_weight=0.80
        ),
    ]
    strategies[1].name = "TQQQ Dip Buyer (15%)"
    strategies[2].name = "TQQQ Dip Buyer (25%)"

    comparer = StrategyComparison(dm)
    results = comparer.compare_strategies(
        assets=assets,
        base_allocations=base_alloc,
        strategies=strategies,
        initial_capital=10000,
        contribution_amount=500,
        contribution_frequency=21
    )

    comparer.generate_comparison_report(
        results,
        output_path="output/leveraged_strategy_comparison.html"
    )

    dm.close()
    return results


def test_broad_market_strategies():
    """
    Compare various strategies on a standard stock/bond portfolio.
    """
    print("\n" + "="*70)
    print("BROAD MARKET STRATEGY COMPARISON")
    print("="*70)

    dm = DataManager()

    tickers = ['VOO', 'QQQ', 'VGT', 'BND']
    print(f"\nLoading data for: {tickers}")
    assets = load_assets(dm, tickers, lookback_years=10)

    # 60% stocks, 40% bonds base
    base_alloc = [0.30, 0.20, 0.10, 0.40]

    strategies = [
        StaticAllocationStrategy(),

        BuyTheDipStrategy(
            target_ticker='QQQ',
            threshold=0.12,
            aggressive_weight=0.50
        ),

        MomentumStrategy(
            momentum_lookback=63,
            tilt_strength=0.3,
            min_weight=0.05
        ),

        RelativeValueStrategy(
            rebalance_threshold=0.08,
            max_tilt=0.45
        ),

        DrawdownProtectionStrategy(
            dd_threshold=0.10,
            risk_off_allocation={'VOO': 0.20, 'BND': 0.80}
        ),
    ]

    comparer = StrategyComparison(dm)
    results = comparer.compare_strategies(
        assets=assets,
        base_allocations=base_alloc,
        strategies=strategies,
        initial_capital=10000,
        contribution_amount=500,
        contribution_frequency=21
    )

    comparer.generate_comparison_report(
        results,
        output_path="output/broad_market_strategy_comparison.html"
    )

    dm.close()
    return results


def run_custom_strategy_test(
    tickers: list,
    allocations: list,
    strategy,
    dca_amount: float = 500,
    initial_capital: float = 10000,
    lookback_years: int = 10
):
    """
    Run a single strategy test with custom parameters.

    Returns detailed backtest results.
    """
    dm = DataManager()
    assets = load_assets(dm, tickers, lookback_years)

    backtester = Backtester(dm)
    results = backtester.run_backtest(
        assets=assets,
        allocations=allocations,
        initial_capital=initial_capital,
        strategy=strategy,
        contribution_amount=dca_amount,
        contribution_frequency=21
    )

    dm.close()
    return results


def main():
    parser = argparse.ArgumentParser(description='Strategy Testing Utility')
    parser.add_argument('--test', choices=['crypto', 'leveraged', 'broad', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--dca', type=float, default=500,
                        help='Monthly DCA amount')
    args = parser.parse_args()

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    if args.test in ['crypto', 'all']:
        test_crypto_dip_buying()

    if args.test in ['leveraged', 'all']:
        test_leveraged_dip_buying()

    if args.test in ['broad', 'all']:
        test_broad_market_strategies()

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nReports saved to output/ directory:")
    print("  - crypto_strategy_comparison.html")
    print("  - leveraged_strategy_comparison.html")
    print("  - broad_market_strategy_comparison.html")


if __name__ == "__main__":
    main()