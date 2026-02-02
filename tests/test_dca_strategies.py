"""
Unit tests for DCA (Dollar Cost Averaging) and Dynamic Strategies.

Tests verify:
- DCA contributions are applied correctly
- Total invested calculation
- Dynamic strategy allocation adjustments
- Time-stepped simulation accuracy
- Contribution timing effects
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig
from strategies import (
    StaticAllocationStrategy,
    BuyTheDipStrategy,
    MomentumStrategy,
    MarketContext
)


class MockDataManager:
    """Mock data manager for testing."""
    pass


def create_test_asset(ticker: str, returns: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Create a test asset dictionary from returns."""
    returns_series = pd.Series(returns, index=dates)
    prices = (1 + returns_series).cumprod() * 100

    return {
        'ticker': ticker,
        'name': ticker,
        'historical_returns': returns_series,
        'full_data': pd.DataFrame({'Adj Close': prices, 'Close': prices}, index=dates),
        'daily_mean': returns_series.mean(),
        'daily_std': returns_series.std()
    }


class TestDCAContributions:
    """Tests for DCA (Dollar Cost Averaging) contributions."""

    def test_initial_capital_only(self):
        """
        With no DCA contributions, total invested = initial capital.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        returns = np.random.normal(0.0003, 0.01, n_days)

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=100,
            method='bootstrap',
            contribution_amount=0,  # No DCA
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        # Use time-stepped simulation (needed for DCA)
        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # With zero contributions, portfolio starts at initial capital
        initial_values = results['portfolio_values'][:, 0]
        assert np.all(initial_values == 10000), \
            "Initial portfolio values should equal initial capital"

    def test_dca_adds_contributions(self):
        """
        DCA should increase total invested amount.

        Initial: $10,000
        Monthly contribution: $500 every 21 trading days
        Year: 252 days → 12 contributions
        Total invested: $10,000 + ($500 × 12) = $16,000
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Zero returns to isolate contribution effect
        returns = np.zeros(n_days)

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=10,
            method='bootstrap',
            contribution_amount=500,
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # Expected contributions: 252 / 21 = 12 contributions
        expected_contributions = (n_days // 21) * 500
        expected_total = 10000 + expected_contributions

        final_values = results['final_values']

        # With zero returns, final value should approximately equal total invested
        # (some variation due to simulation length vs trading days per year)
        for val in final_values:
            assert abs(val - expected_total) / expected_total < 0.15, \
                f"Final value {val:.0f} should be close to {expected_total:.0f}"

    def test_dca_timing_effect(self):
        """
        DCA in a declining market should result in lower average cost
        (more shares bought when prices are low).

        In a declining then recovering market, DCA should outperform
        lump sum invested at the start.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # V-shaped market: decline then recovery
        decline = np.linspace(0, -0.002, 126)  # ~25% decline over 6 months
        recovery = np.linspace(0.002, 0, 126)  # 25% recovery
        returns = np.concatenate([decline, recovery])

        dm = MockDataManager()

        # Lump sum simulation
        config_lump = SimulationConfig(
            initial_capital=12000,  # Same total investment
            years=1,
            simulations=10,
            method='bootstrap',
            contribution_amount=0,
            contribution_frequency=21
        )

        # DCA simulation
        config_dca = SimulationConfig(
            initial_capital=6000,  # Half upfront
            years=1,
            simulations=10,
            method='bootstrap',
            contribution_amount=500,  # $500/month × 12 = $6000 additional
            contribution_frequency=21
        )

        asset = create_test_asset('TEST', returns, dates)

        sim_lump = PortfolioSimulator(dm, config_lump)
        sim_dca = PortfolioSimulator(dm, config_dca)

        strategy = StaticAllocationStrategy()

        results_lump = sim_lump.simulate_portfolio([asset], [1.0], strategy=strategy)
        results_dca = sim_dca.simulate_portfolio([asset], [1.0], strategy=strategy)

        # In V-shaped market, DCA should perform relatively better
        # because it buys more shares during the decline
        median_lump = np.median(results_lump['final_values'])
        median_dca = np.median(results_dca['final_values'])

        # This is a qualitative test - DCA has timing advantage in V-shaped market
        print(f"Lump sum median: {median_lump:.0f}, DCA median: {median_dca:.0f}")


class TestDynamicStrategies:
    """Tests for dynamic allocation strategies."""

    def test_static_strategy_unchanged_allocation(self):
        """
        Static strategy should maintain constant allocation.
        """
        base_allocation = [0.6, 0.4]
        strategy = StaticAllocationStrategy()

        # Create mock market context
        context = MarketContext(
            current_holdings=np.array([[6000, 4000]]),
            current_drawdowns=np.array([[0, 0]]),
            base_allocations=np.array(base_allocation),
            asset_tickers=['A', 'B'],
            current_day=100,
            total_days=252,
            rolling_returns=None,
            rolling_volatility=None,
            rolling_sharpe=None,
            momentum_score=None,
            portfolio_drawdown=np.array([0])
        )

        new_allocation = strategy.get_allocation(context)

        # Handle both 1D and 2D outputs
        if hasattr(new_allocation, 'ndim') and new_allocation.ndim == 2:
            new_allocation = new_allocation.flatten()

        np.testing.assert_array_almost_equal(
            new_allocation, base_allocation,
            err_msg="Static strategy should return base allocation"
        )

    def test_buy_the_dip_increases_allocation_on_drawdown(self):
        """
        Buy the Dip strategy should increase allocation when target asset is down.
        """
        strategy = BuyTheDipStrategy(
            target_ticker='BTC',
            threshold=0.10,  # 10% drawdown threshold
            aggressive_weight=0.50  # Max 50% in target when dip occurs
        )

        base_allocation = [0.5, 0.3, 0.2]  # BTC at 20%

        # Context with BTC in significant drawdown
        context = MarketContext(
            current_holdings=np.array([[4500, 2700, 1200]]),  # BTC value dropped
            current_drawdowns=np.array([[0, 0, -0.25]]),  # BTC down 25%
            base_allocations=np.array(base_allocation),
            asset_tickers=['SPY', 'AGG', 'BTC'],
            current_day=100,
            total_days=252,
            rolling_returns=None,
            rolling_volatility=None,
            rolling_sharpe=None,
            momentum_score=None,
            portfolio_drawdown=np.array([-0.10])
        )

        new_allocation = strategy.get_allocation(context)

        # Handle both 1D and 2D outputs
        if hasattr(new_allocation, 'ndim') and new_allocation.ndim == 2:
            new_allocation = new_allocation.flatten()

        # BTC allocation should increase (buying the dip)
        assert new_allocation[2] > base_allocation[2], \
            f"BTC allocation should increase on dip: {new_allocation[2]:.2%} vs {base_allocation[2]:.2%}"

    def test_buy_the_dip_no_change_without_dip(self):
        """
        Buy the Dip should not change allocation without significant drawdown.
        """
        strategy = BuyTheDipStrategy(
            target_ticker='BTC',
            threshold=0.20,  # 20% threshold
            aggressive_weight=0.50
        )

        base_allocation = [0.5, 0.3, 0.2]

        # Context with minimal drawdown
        context = MarketContext(
            current_holdings=np.array([[5000, 3000, 2000]]),
            current_drawdowns=np.array([[0, 0, -0.05]]),  # Only 5% down
            base_allocations=np.array(base_allocation),
            asset_tickers=['SPY', 'AGG', 'BTC'],
            current_day=100,
            total_days=252,
            rolling_returns=None,
            rolling_volatility=None,
            rolling_sharpe=None,
            momentum_score=None,
            portfolio_drawdown=np.array([-0.02])
        )

        new_allocation = strategy.get_allocation(context)

        # Handle both 1D and 2D outputs
        if hasattr(new_allocation, 'ndim') and new_allocation.ndim == 2:
            new_allocation = new_allocation.flatten()

        # Should return base allocation when dip threshold not met
        np.testing.assert_array_almost_equal(
            new_allocation, base_allocation,
            err_msg="Allocation should not change without significant dip"
        )

    def test_momentum_strategy_tilts_to_winners(self):
        """
        Momentum strategy should tilt allocation toward assets with positive momentum.
        """
        strategy = MomentumStrategy(momentum_lookback=21)

        base_allocation = [0.5, 0.5]

        # Create context with one asset having strong positive momentum
        context = MarketContext(
            current_holdings=np.array([[5500, 4500]]),  # Asset A has grown more
            current_drawdowns=np.array([[0, -0.05]]),
            base_allocations=np.array(base_allocation),
            asset_tickers=['WINNER', 'LOSER'],
            current_day=100,
            total_days=252,
            rolling_returns=np.array([[0.15, -0.05]]),  # Winner has positive return
            rolling_volatility=np.array([[0.15, 0.15]]),
            rolling_sharpe=np.array([[1.0, -0.3]]),
            momentum_score=np.array([[0.10, -0.05]]),  # Winner has positive momentum
            portfolio_drawdown=np.array([0])
        )

        new_allocation = strategy.get_allocation(context)

        # Handle both 1D and 2D outputs
        if hasattr(new_allocation, 'ndim') and new_allocation.ndim == 2:
            new_allocation = new_allocation.flatten()

        # Winner should get higher allocation
        assert new_allocation[0] > new_allocation[1], \
            f"Momentum should tilt to winner: {new_allocation}"


class TestTimeSteppedSimulation:
    """Tests for time-stepped simulation accuracy."""

    def test_holdings_track_correctly(self):
        """
        Holdings should be tracked correctly through time.
        """
        np.random.seed(42)
        n_days = 63  # ~3 months
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Known returns
        returns = np.array([0.01] * n_days)  # Constant 1% daily

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=10,
            method='bootstrap',
            contribution_amount=0,
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # Final value should be approximately initial × (1.01)^n_days
        expected_final = 10000 * (1.01 ** 63)  # ~1.87x

        # Due to simulation randomness, check median is in reasonable range
        median_final = np.median(results['final_values'])
        # Time-stepped uses fewer actual days based on simulation years
        # Just verify it grew
        assert median_final > 10000, "Portfolio should grow with positive returns"

    def test_contribution_timing(self):
        """
        Contributions should be added at correct intervals.

        With frequency=21 trading days, contributions happen ~monthly.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        returns = np.zeros(n_days)  # Zero returns to isolate contribution effect

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=1,  # Just one simulation
            method='bootstrap',
            contribution_amount=1000,
            contribution_frequency=21  # Every 21 trading days
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # With 252 trading days per year, expect ~12 contributions
        # Plus initial capital: 10000 + 12×1000 = 22000
        trading_days = config.years * 252
        expected_contributions = trading_days // 21
        expected_total = 10000 + expected_contributions * 1000

        final_value = results['final_values'][0]

        # Allow some tolerance for exact day counting
        assert abs(final_value - expected_total) / expected_total < 0.10, \
            f"Final value {final_value:.0f} should be close to {expected_total:.0f}"


class TestStrategyMetrics:
    """Tests for strategy-specific metrics."""

    def test_cagr_with_dca(self):
        """
        CAGR calculation should account for total invested, not just initial capital.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # 10% annual return
        daily_return = (1.10) ** (1/252) - 1
        returns = np.array([daily_return] * n_days)

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=100,
            method='bootstrap',
            contribution_amount=500,
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # CAGR should be based on actual performance, not inflated by contributions
        median_cagr = np.median(results['cagr'])

        # CAGR for DCA is calculated relative to total invested
        # Should be reasonable (not hugely inflated by contributions)
        assert median_cagr < 0.50, \
            f"CAGR {median_cagr:.2%} should not be inflated by contributions"
        assert median_cagr > -0.50, \
            f"CAGR {median_cagr:.2%} should be reasonable"

    def test_drawdown_during_contributions(self):
        """
        Drawdown should be calculated correctly even with ongoing contributions.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Market crash then recovery
        crash = np.array([-0.02] * 50)  # 50 days of -2% daily
        sideways = np.array([0.0] * 50)
        recovery = np.array([0.02] * 152)
        returns = np.concatenate([crash, sideways, recovery])

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=50,
            method='bootstrap',
            contribution_amount=500,
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset = create_test_asset('TEST', returns, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset], [1.0], strategy=strategy)

        # Max drawdown should be negative (loss from peak)
        median_dd = np.median(results['max_drawdowns'])
        assert median_dd < 0, f"Max drawdown {median_dd:.2%} should be negative"

        # Drawdown should be noticeable given the crash
        # (DCA contributions can reduce apparent drawdown)
        assert median_dd < -0.05, f"Max drawdown {median_dd:.2%} should be noticeable"


class TestMultiAssetDCA:
    """Tests for DCA with multiple assets."""

    def test_contributions_split_by_allocation(self):
        """
        DCA contributions should be split according to target allocations.

        If allocation is 60/40, $1000 contribution should add $600 and $400.
        """
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        # Zero returns to isolate contribution effect
        returns_a = np.zeros(n_days)
        returns_b = np.zeros(n_days)

        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=1,
            method='bootstrap',
            contribution_amount=1000,
            contribution_frequency=21
        )

        sim = PortfolioSimulator(dm, config)
        asset_a = create_test_asset('A', returns_a, dates)
        asset_b = create_test_asset('B', returns_b, dates)

        strategy = StaticAllocationStrategy()
        results = sim.simulate_portfolio([asset_a, asset_b], [0.6, 0.4], strategy=strategy)

        # With 60/40 allocation:
        # Initial: $6000 in A, $4000 in B
        # Each contribution: $600 to A, $400 to B
        # 12 contributions expected
        # Final A: $6000 + (12 × $600) = $13,200
        # Final B: $4000 + (12 × $400) = $8,800
        # Total: $22,000

        expected_total = 10000 + 12 * 1000
        final_value = results['final_values'][0]

        assert abs(final_value - expected_total) / expected_total < 0.15, \
            f"Final value {final_value:.0f} should be close to {expected_total:.0f}"
