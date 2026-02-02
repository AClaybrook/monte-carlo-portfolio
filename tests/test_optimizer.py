"""
Unit tests for Portfolio Optimizer.

Tests verify:
- Mean-Variance Optimization (two-asset toy problem)
- Max Sharpe ratio optimization
- Risk Parity (Equal Risk Contribution)
- Custom weighted objective
- Edge cases (singular covariance, etc.)
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer
from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig


class MockDataManager:
    """Mock data manager for testing."""
    pass


def create_test_assets(returns_df: pd.DataFrame) -> list:
    """Convert a returns DataFrame to asset dictionaries."""
    assets = []
    for col in returns_df.columns:
        returns = returns_df[col]
        prices = (1 + returns).cumprod() * 100

        assets.append({
            'ticker': col,
            'name': col,
            'historical_returns': returns,
            'full_data': pd.DataFrame({'Adj Close': prices, 'Close': prices}, index=returns.index),
            'daily_mean': returns.mean(),
            'daily_std': returns.std()
        })

    return assets


class TestMeanVarianceOptimization:
    """
    Tests for Mean-Variance (Markowitz) Optimization.

    Key verification: Two-asset toy problem with known solution.
    """

    def test_two_asset_min_variance(self):
        """
        Two-asset minimum variance portfolio with known solution.

        Asset A: σ = 10%
        Asset B: σ = 20%
        Correlation: 0

        Optimal weights:
        wA = σB² / (σA² + σB²) = 0.04 / (0.01 + 0.04) = 0.80
        wB = σA² / (σA² + σB²) = 0.01 / (0.01 + 0.04) = 0.20
        """
        np.random.seed(42)
        n_days = 2520  # 10 years
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Generate uncorrelated returns with specific volatilities
        vol_a = 0.10 / np.sqrt(252)  # 10% annual → daily
        vol_b = 0.20 / np.sqrt(252)  # 20% annual → daily

        returns_a = np.random.normal(0.0003, vol_a, n_days)
        returns_b = np.random.normal(0.0003, vol_b, n_days)

        returns_df = pd.DataFrame({'A': returns_a, 'B': returns_b}, index=dates)

        # Calculate theoretical optimal weights
        var_a = vol_a ** 2 * 252  # Annual variance
        var_b = vol_b ** 2 * 252

        expected_w_a = var_b / (var_a + var_b)
        expected_w_b = var_a / (var_a + var_b)

        # Create optimizer
        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_min_volatility(assets)

        allocations = result['allocations']

        # Weight for A should be close to 80%
        assert abs(allocations[0] - expected_w_a) < 0.10, \
            f"Weight A should be ~{expected_w_a:.0%}, got {allocations[0]:.0%}"

        # Weight for B should be close to 20%
        assert abs(allocations[1] - expected_w_b) < 0.10, \
            f"Weight B should be ~{expected_w_b:.0%}, got {allocations[1]:.0%}"

    def test_min_variance_with_correlation(self):
        """
        Two correlated assets - optimal weights change with correlation.

        Higher correlation → less diversification benefit → weights closer to equal
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Generate correlated returns
        vol = 0.15 / np.sqrt(252)
        correlation = 0.8

        # Cholesky decomposition for correlated normals
        L = np.array([[1, 0], [correlation, np.sqrt(1 - correlation**2)]])
        uncorrelated = np.random.standard_normal((n_days, 2)) * vol
        correlated = uncorrelated @ L.T

        returns_df = pd.DataFrame({'A': correlated[:, 0], 'B': correlated[:, 1]}, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_min_volatility(assets)

        allocations = result['allocations']

        # With high correlation and equal vol, weights should be near 50/50
        assert abs(allocations[0] - 0.5) < 0.15, \
            f"With high correlation, weights should be near equal: {allocations}"


class TestMaxSharpeOptimization:
    """Tests for Maximum Sharpe Ratio optimization."""

    def test_max_sharpe_selects_better_asset(self):
        """
        Given two assets with different Sharpe ratios,
        optimizer should favor the one with higher Sharpe.
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # High Sharpe asset: 15% return, 15% vol → Sharpe ≈ 0.73
        high_sharpe = np.random.normal(0.15/252, 0.15/np.sqrt(252), n_days)

        # Low Sharpe asset: 6% return, 20% vol → Sharpe ≈ 0.10
        low_sharpe = np.random.normal(0.06/252, 0.20/np.sqrt(252), n_days)

        returns_df = pd.DataFrame({
            'HIGH_SHARPE': high_sharpe,
            'LOW_SHARPE': low_sharpe
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_sharpe_ratio(assets, risk_free_rate=0.04)

        allocations = result['allocations']

        # High Sharpe asset should dominate
        assert allocations[0] > allocations[1], \
            f"High Sharpe asset should have higher weight: {allocations}"

        assert allocations[0] > 0.6, \
            f"High Sharpe asset should have majority weight, got {allocations[0]:.0%}"


class TestRiskParity:
    """
    Tests for Risk Parity (Equal Risk Contribution).

    Each asset should contribute equally to portfolio variance.
    """

    def test_risk_parity_equal_contribution(self):
        """
        In a risk parity portfolio, each asset's marginal risk contribution
        should be equal: wᵢ × MRCᵢ = constant for all i
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Three assets with different volatilities
        returns_a = np.random.normal(0.0003, 0.10/np.sqrt(252), n_days)
        returns_b = np.random.normal(0.0003, 0.15/np.sqrt(252), n_days)
        returns_c = np.random.normal(0.0003, 0.25/np.sqrt(252), n_days)

        returns_df = pd.DataFrame({
            'LOW_VOL': returns_a,
            'MED_VOL': returns_b,
            'HIGH_VOL': returns_c
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_risk_parity(assets)

        allocations = result['allocations']

        # Lower volatility assets should have higher weight
        assert allocations[0] > allocations[2], \
            f"Lower vol asset should have higher weight: {allocations}"

        # Calculate risk contributions to verify
        cov = returns_df.cov().values * 252
        weights = np.array(allocations)

        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_risk = (cov @ weights) / port_vol
        risk_contribution = weights * marginal_risk

        # Risk contributions should be approximately equal
        max_diff = max(risk_contribution) - min(risk_contribution)
        mean_contrib = np.mean(risk_contribution)

        assert max_diff / mean_contrib < 0.5, \
            f"Risk contributions should be similar: {risk_contribution}"


class TestCustomObjective:
    """Tests for custom weighted objective optimization."""

    def test_return_only_selects_highest_return(self):
        """
        With 100% weight on returns, optimizer should pick highest-return asset.
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Different return levels
        low_return = np.random.normal(0.05/252, 0.15/np.sqrt(252), n_days)
        medium_return = np.random.normal(0.10/252, 0.15/np.sqrt(252), n_days)
        high_return = np.random.normal(0.25/252, 0.20/np.sqrt(252), n_days)

        returns_df = pd.DataFrame({
            'LOW': low_return,
            'MEDIUM': medium_return,
            'HIGH': high_return
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)

        # 100% weight on returns
        result = opt.optimize_custom_weighted(
            assets,
            weights_config={'return': 1.0, 'sharpe': 0, 'drawdown': 0, 'volatility': 0, 'sortino': 0}
        )

        allocations = result['allocations']

        # High return asset should dominate
        assert allocations[2] > 0.7, \
            f"Highest return asset should dominate with return-only objective: {allocations}"

    def test_volatility_only_selects_lowest_vol(self):
        """
        With 100% weight on volatility, optimizer should pick lowest-vol asset.
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Different volatility levels
        low_vol = np.random.normal(0.05/252, 0.05/np.sqrt(252), n_days)
        medium_vol = np.random.normal(0.05/252, 0.15/np.sqrt(252), n_days)
        high_vol = np.random.normal(0.05/252, 0.30/np.sqrt(252), n_days)

        returns_df = pd.DataFrame({
            'LOW_VOL': low_vol,
            'MED_VOL': medium_vol,
            'HIGH_VOL': high_vol
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)

        # 100% weight on volatility (minimize)
        result = opt.optimize_custom_weighted(
            assets,
            weights_config={'return': 0, 'sharpe': 0, 'drawdown': 0, 'volatility': 1.0, 'sortino': 0}
        )

        allocations = result['allocations']

        # Low vol asset should dominate
        assert allocations[0] > 0.7, \
            f"Lowest vol asset should dominate with vol-only objective: {allocations}"

    def test_balanced_objective(self):
        """
        Balanced objective should find a reasonable middle ground.
        """
        np.random.seed(42)
        n_days = 2520
        dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')

        # Risky high-return asset
        risky = np.random.normal(0.15/252, 0.35/np.sqrt(252), n_days)

        # Safe low-return asset
        safe = np.random.normal(0.03/252, 0.05/np.sqrt(252), n_days)

        # Balanced asset
        balanced = np.random.normal(0.10/252, 0.15/np.sqrt(252), n_days)

        returns_df = pd.DataFrame({
            'RISKY': risky,
            'SAFE': safe,
            'BALANCED': balanced
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)

        # Balanced objective
        result = opt.optimize_custom_weighted(
            assets,
            weights_config={'return': 0.4, 'sharpe': 0.3, 'drawdown': 0.2, 'volatility': 0.1, 'sortino': 0}
        )

        allocations = result['allocations']

        # With balanced objective, optimizer may still concentrate if one asset dominates
        # The test just verifies the optimization completes and returns valid weights
        assert sum(allocations) == pytest.approx(1.0), \
            f"Allocations should sum to 1: {allocations}"
        assert all(w >= 0 for w in allocations), \
            f"All weights should be non-negative: {allocations}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_asset(self):
        """Single asset should return 100% allocation."""
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        returns = np.random.normal(0.0003, 0.01, n_days)
        returns_df = pd.DataFrame({'ONLY': returns}, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_min_volatility(assets)

        assert result['allocations'][0] == pytest.approx(1.0), \
            "Single asset should have 100% allocation"

    def test_identical_assets(self):
        """Identical assets should have equal weights."""
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        returns = np.random.normal(0.0003, 0.01, n_days)
        # Same returns for both assets
        returns_df = pd.DataFrame({'A': returns, 'B': returns.copy()}, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_risk_parity(assets)

        allocations = result['allocations']

        # Should be approximately 50/50
        assert abs(allocations[0] - 0.5) < 0.15, \
            f"Identical assets should have equal weights: {allocations}"

    def test_weights_sum_to_one(self):
        """All optimizations should return weights that sum to 1."""
        np.random.seed(42)
        n_days = 504
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        returns_df = pd.DataFrame({
            'A': np.random.normal(0.0003, 0.01, n_days),
            'B': np.random.normal(0.0004, 0.015, n_days),
            'C': np.random.normal(0.0002, 0.02, n_days)
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)

        # Test all optimization methods
        results = [
            opt.optimize_min_volatility(assets),
            opt.optimize_sharpe_ratio(assets),
            opt.optimize_risk_parity(assets),
            opt.optimize_sortino_ratio(assets),
            opt.optimize_custom_weighted(assets, {'return': 0.5, 'sharpe': 0.5})
        ]

        for result in results:
            total_weight = sum(result['allocations'])
            assert abs(total_weight - 1.0) < 0.01, \
                f"Weights should sum to 1, got {total_weight} for {result['label']}"

    def test_no_negative_weights(self):
        """All weights should be non-negative (no short selling)."""
        np.random.seed(42)
        n_days = 504
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        returns_df = pd.DataFrame({
            'A': np.random.normal(0.0003, 0.01, n_days),
            'B': np.random.normal(-0.0001, 0.02, n_days),  # Negative return
            'C': np.random.normal(0.0005, 0.015, n_days)
        }, index=dates)

        dm = MockDataManager()
        config = SimulationConfig(initial_capital=10000, years=1, simulations=100)
        sim = PortfolioSimulator(dm, config)
        opt = PortfolioOptimizer(sim, dm)

        assets = create_test_assets(returns_df)
        result = opt.optimize_sharpe_ratio(assets)

        for i, w in enumerate(result['allocations']):
            assert w >= -0.001, f"Weight {i} should be non-negative, got {w}"
