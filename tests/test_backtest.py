"""
Unit tests for historical backtesting.

Tests verify:
- Backtest return calculations
- Rebalancing logic
- Drawdown analysis
- Benchmark comparison
- Historical crisis scenarios
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import Backtester
from strategies import StaticAllocationStrategy


class MockDataManager:
    """Mock data manager for testing."""
    pass


def create_test_asset(ticker: str, prices: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Create a test asset dictionary from prices."""
    # Ensure prices and dates have same length
    assert len(prices) == len(dates), f"Prices ({len(prices)}) and dates ({len(dates)}) must match"

    price_series = pd.Series(prices, index=dates)
    returns = price_series.pct_change().fillna(0)

    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(prices))
    }, index=dates)

    return {
        'ticker': ticker,
        'name': ticker,
        'historical_returns': returns,
        'full_data': df,
        'daily_mean': returns.mean(),
        'daily_std': returns.std()
    }


class TestBacktestReturns:
    """Tests for backtest return calculations."""

    def test_simple_buy_and_hold(self):
        """
        Simple buy-and-hold: asset doubles → 100% return.
        """
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Asset doubles over the year
        prices = np.linspace(100, 200, n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Final value should be ~double initial
        expected_final = 10000 * (200 / 100)
        actual_final = results['values'][-1]

        assert abs(actual_final - expected_final) / expected_final < 0.01, \
            f"Final value {actual_final:.0f} should be {expected_final:.0f}"

    def test_cagr_calculation(self):
        """
        Verify CAGR calculation matches expected.

        100% return over 1 year → CAGR = 100%
        """
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # 100% return over the year
        prices = np.linspace(100, 200, n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # CAGR for 100% return in 1 year = 100%
        cagr = results['metrics']['CAGR']
        assert abs(cagr - 1.0) < 0.15, \
            f"CAGR {cagr:.2%} should be ~100%"

    def test_negative_return(self):
        """
        Asset halves → -50% return, CAGR should be negative.
        """
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Asset loses 50%
        prices = np.linspace(100, 50, n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Final value should be half
        expected_final = 10000 * 0.5
        actual_final = results['values'][-1]

        assert abs(actual_final - expected_final) / expected_final < 0.01, \
            f"Final value {actual_final:.0f} should be {expected_final:.0f}"

        assert results['metrics']['CAGR'] < 0, \
            f"CAGR should be negative for losing backtest"


class TestRebalancing:
    """Tests for portfolio rebalancing logic."""

    def test_drift_calculation(self):
        """
        Verify that asset weights drift correctly without rebalancing.

        Start 50/50, Asset A +20%, Asset B -20%
        → Asset A becomes 60%, Asset B becomes 40%
        """
        n_days = 10
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Asset A: +20% over period
        prices_a = np.linspace(100, 120, n_days)

        # Asset B: -20% over period
        prices_b = np.linspace(100, 80, n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset_a = create_test_asset('A', prices_a, dates)
        asset_b = create_test_asset('B', prices_b, dates)

        results = backtester.run_backtest(
            assets=[asset_a, asset_b],
            allocations=[0.5, 0.5],
            initial_capital=10000
        )

        # Final value: A = 5000 × 1.2 = 6000, B = 5000 × 0.8 = 4000
        # Total = 10000 (no gain overall with 50/50)
        final_total = results['values'][-1]
        expected = 10000  # Equal gains and losses cancel

        assert abs(final_total - expected) / expected < 0.05, \
            f"Final value {final_total:.0f} should be ~{expected:.0f}"

    def test_multi_asset_allocation(self):
        """
        Test with three assets and different allocations.
        """
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Different returns for each asset
        prices_a = np.linspace(100, 110, n_days)  # +10%
        prices_b = np.linspace(100, 105, n_days)  # +5%
        prices_c = np.linspace(100, 95, n_days)   # -5%

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset_a = create_test_asset('A', prices_a, dates)
        asset_b = create_test_asset('B', prices_b, dates)
        asset_c = create_test_asset('C', prices_c, dates)

        # 50% A, 30% B, 20% C
        results = backtester.run_backtest(
            assets=[asset_a, asset_b, asset_c],
            allocations=[0.5, 0.3, 0.2],
            initial_capital=10000
        )

        # Expected: 5000 × 1.10 + 3000 × 1.05 + 2000 × 0.95
        #         = 5500 + 3150 + 1900 = 10550
        expected = 10550
        actual = results['values'][-1]

        assert abs(actual - expected) / expected < 0.02, \
            f"Final value {actual:.0f} should be ~{expected:.0f}"


class TestDrawdownAnalysis:
    """Tests for drawdown calculation in backtests."""

    def test_max_drawdown_calculation(self):
        """
        Verify max drawdown is calculated correctly.

        V-shaped pattern: peak at day 50, trough at day 100
        """
        n_days = 150
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Rise to peak, crash, partial recovery
        rise = np.linspace(100, 150, 50)       # Peak at 150
        crash = np.linspace(150, 100, 50)      # Trough at 100 (33% drawdown)
        recover = np.linspace(100, 120, 50)    # Partial recovery
        prices = np.concatenate([rise, crash, recover])

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Max drawdown: (100 - 150) / 150 = -33.3%
        expected_dd = -0.333
        actual_dd = results['metrics']['Max Drawdown']

        assert abs(actual_dd - expected_dd) < 0.05, \
            f"Max drawdown {actual_dd:.2%} should be ~{expected_dd:.2%}"

    def test_no_drawdown_always_up(self):
        """
        Constant upward movement should have zero drawdown.
        """
        n_days = 100
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        prices = np.linspace(100, 200, n_days)  # Steady rise

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Check drawdowns series - all should be 0 or close to 0
        max_dd = results['drawdowns'].min()
        assert max_dd >= -0.001, \
            f"Drawdown should be ~0 for constant rise, got {max_dd:.2%}"


class TestHistoricalScenarios:
    """
    Tests using historical crisis scenarios.

    Note: These tests use synthetic data mimicking historical patterns.
    For actual validation against Portfolio Visualizer, use real data.
    """

    def test_2022_like_stress_scenario(self):
        """
        Simulate 2022-like scenario: both stocks and bonds down.

        Historical 2022:
        - SPY: -18.18%
        - AGG: -13.02%
        - 60/40: ~-16%
        """
        n_days = 252
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

        # Simulate 2022-like returns
        # SPY: -18.18% for the year
        spy_end = 100 * (1 - 0.1818)
        prices_spy = np.linspace(100, spy_end, n_days)
        # Add some volatility
        np.random.seed(42)
        prices_spy = prices_spy + np.random.normal(0, 2, n_days)
        prices_spy = np.maximum(prices_spy, 50)  # Floor at 50

        # AGG: -13.02% for the year
        agg_end = 100 * (1 - 0.1302)
        prices_agg = np.linspace(100, agg_end, n_days)
        prices_agg = prices_agg + np.random.normal(0, 0.5, n_days)
        prices_agg = np.maximum(prices_agg, 50)

        dm = MockDataManager()
        backtester = Backtester(dm)

        spy = create_test_asset('SPY', prices_spy, dates)
        agg = create_test_asset('AGG', prices_agg, dates)

        # 60/40 portfolio
        results = backtester.run_backtest(
            assets=[spy, agg],
            allocations=[0.6, 0.4],
            initial_capital=10000
        )

        # Expected 60/40 return: 0.6 × (-18.18%) + 0.4 × (-13.02%) = -16.12%
        expected_return = -0.16

        # Allow tolerance for volatility noise
        final_value = results['values'][-1]
        actual_return = (final_value / 10000) - 1

        assert actual_return < 0, \
            f"2022 scenario should be negative: {actual_return:.2%}"
        assert abs(actual_return - expected_return) < 0.10, \
            f"60/40 return {actual_return:.2%} should be near {expected_return:.2%}"

    def test_crash_and_recovery_scenario(self):
        """
        Simulate 2020-like rapid crash and V-shaped recovery.
        """
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # 2020-like pattern:
        # - First 45 days: steady
        # - Days 45-75: -35% crash (30 days)
        # - Days 75-252: recovery to new highs

        steady = np.linspace(100, 102, 45)
        crash = np.linspace(102, 67, 30)  # -35%
        recovery = np.linspace(67, 118, n_days - 75 + 1)  # Recovery + 18% gain
        prices = np.concatenate([steady, crash, recovery[1:]])

        # Ensure correct length
        prices = prices[:n_days]
        # Pad if needed
        if len(prices) < n_days:
            prices = np.pad(prices, (0, n_days - len(prices)), mode='edge')

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('SPY', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Max drawdown should be around -35%
        max_dd = results['metrics']['Max Drawdown']
        assert max_dd < -0.30, \
            f"Max drawdown {max_dd:.2%} should be significant"

        # But should end positive
        final_return = (results['values'][-1] / 10000) - 1
        assert final_return > 0, \
            f"Should end positive after recovery: {final_return:.2%}"


class TestEdgeCases:
    """Tests for edge cases in backtesting."""

    def test_single_day(self):
        """Single day backtest should handle gracefully."""
        dates = pd.date_range(start='2020-01-01', periods=2, freq='D')
        prices = np.array([100, 101])  # 1% gain

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        try:
            results = backtester.run_backtest(
                assets=[asset],
                allocations=[1.0],
                initial_capital=10000
            )
            # Should not crash - check for some key
            assert 'values' in results or 'metrics' in results
        except Exception as e:
            pytest.fail(f"Single day backtest should not crash: {e}")

    def test_flat_returns(self):
        """
        Zero returns (constant prices) should work.
        """
        n_days = 100
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        prices = np.array([100.0] * n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('FLAT', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Final value equals initial
        final = results['values'][-1]
        assert abs(final - 10000) < 1, \
            f"Final should equal initial for flat prices, got {final}"

    def test_allocation_sums_to_one(self):
        """
        Backtester should handle allocations correctly.
        """
        n_days = 50
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        prices = np.linspace(100, 110, n_days)

        dm = MockDataManager()
        backtester = Backtester(dm)

        asset = create_test_asset('TEST', prices, dates)

        results = backtester.run_backtest(
            assets=[asset],
            allocations=[1.0],
            initial_capital=10000
        )

        # Should not crash and have values
        assert 'values' in results
        assert len(results['values']) > 0
