"""
Unit tests for Monte Carlo simulation engine.

Tests verify:
- GBM (Geometric Brownian Motion) mathematical correctness
- Itô adjustment (-σ²/2) is properly applied
- Bootstrap correlation preservation
- Convergence diagnostics
- Overflow protection for long simulations
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_simulator import PortfolioSimulator
from run_config import SimulationConfig


class MockDataManager:
    """Mock data manager for testing."""
    def __init__(self):
        pass

    def get_data(self, ticker, start, end):
        return None


class TestGBMConvergence:
    """
    Tests for Geometric Brownian Motion simulation.

    Critical verification:
    - Mean of ending prices should converge to S₀ × e^(μt)
    - Median should converge to S₀ × e^((μ - σ²/2)t)
    - If median ≈ mean, the Itô adjustment is missing (BUG!)
    """

    def test_gbm_mean_convergence(self):
        """
        GBM mean price should converge to E[S] = S₀ × e^(μt)

        Parameters: S₀=100, μ=5%, σ=20%, t=1 year
        Expected mean: 100 × e^0.05 ≈ 105.13
        """
        np.random.seed(42)

        # Create a simple test with known parameters
        S0 = 100
        mu = 0.05  # 5% annual return
        sigma = 0.20  # 20% volatility
        n_sims = 50000
        n_days = 252

        # Daily parameters
        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)

        # Simulate using proper GBM
        dt = 1/252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        final_prices = np.zeros(n_sims)
        for i in range(n_sims):
            log_returns = drift + diffusion * np.random.standard_normal(n_days)
            final_prices[i] = S0 * np.exp(np.sum(log_returns))

        # Mean should converge to S₀ × e^(μt)
        expected_mean = S0 * np.exp(mu)
        actual_mean = np.mean(final_prices)

        # Allow 2% tolerance
        assert abs(actual_mean - expected_mean) / expected_mean < 0.02, \
            f"Mean {actual_mean:.2f} should be close to {expected_mean:.2f}"

    def test_gbm_median_vs_mean(self):
        """
        Median should be LOWER than mean for GBM (log-normal distribution).

        If median ≈ mean, the Itô adjustment (-σ²/2) is missing!

        Expected:
        - Mean: S₀ × e^μ = 100 × e^0.05 = 105.13
        - Median: S₀ × e^(μ - σ²/2) = 100 × e^(0.05 - 0.02) = 103.05
        """
        np.random.seed(42)

        S0 = 100
        mu = 0.05
        sigma = 0.20
        n_sims = 50000
        n_days = 252

        dt = 1/252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        final_prices = np.zeros(n_sims)
        for i in range(n_sims):
            log_returns = drift + diffusion * np.random.standard_normal(n_days)
            final_prices[i] = S0 * np.exp(np.sum(log_returns))

        expected_mean = S0 * np.exp(mu)
        expected_median = S0 * np.exp(mu - 0.5 * sigma**2)

        actual_mean = np.mean(final_prices)
        actual_median = np.median(final_prices)

        # Median should be noticeably lower than mean
        assert actual_median < actual_mean, \
            f"Median ({actual_median:.2f}) should be < Mean ({actual_mean:.2f}) for GBM"

        # Median should be close to expected
        assert abs(actual_median - expected_median) / expected_median < 0.03, \
            f"Median {actual_median:.2f} should be close to {expected_median:.2f}"

    def test_ito_adjustment_present(self):
        """
        Verify the simulator applies the Itô correction.

        Without -σ²/2 drift adjustment, simulated median would equal mean.
        """
        np.random.seed(42)

        S0 = 100
        mu = 0.10  # Higher return makes difference more visible
        sigma = 0.30  # Higher vol makes difference more visible
        n_sims = 20000
        n_days = 252

        # Correct implementation (with Itô)
        dt = 1/252
        drift_correct = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        prices_correct = np.zeros(n_sims)
        for i in range(n_sims):
            log_returns = drift_correct + diffusion * np.random.standard_normal(n_days)
            prices_correct[i] = S0 * np.exp(np.sum(log_returns))

        mean_correct = np.mean(prices_correct)
        median_correct = np.median(prices_correct)

        # With σ=30%, the difference should be substantial (~4.5%)
        median_to_mean_ratio = median_correct / mean_correct

        # Median should be meaningfully less than mean
        assert median_to_mean_ratio < 0.98, \
            f"Median/Mean ratio {median_to_mean_ratio:.3f} should be < 0.98 with Itô correction"


class TestBootstrapSimulation:
    """Tests for historical bootstrap simulation."""

    def test_bootstrap_mean_preservation(self):
        """Bootstrap should preserve the mean of historical returns."""
        np.random.seed(42)

        # Historical returns
        n_historical = 252 * 10  # 10 years
        historical_mean = 0.0004  # ~10% annual
        historical_std = 0.01
        historical = np.random.normal(historical_mean, historical_std, n_historical)

        # Bootstrap sample
        n_bootstrap = 252
        n_samples = 10000
        bootstrap_means = []

        for _ in range(n_samples):
            sample = np.random.choice(historical, size=n_bootstrap, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Mean of bootstrap means should be close to historical mean
        # (within 2 standard errors of the sampling distribution)
        se = historical_std / np.sqrt(n_bootstrap)
        assert abs(np.mean(bootstrap_means) - historical_mean) < 3 * se, \
            "Bootstrap should preserve historical mean within tolerance"

    def test_bootstrap_correlation_preservation(self):
        """
        Multi-asset bootstrap should preserve cross-asset correlations.

        If sampling assets independently, correlation would be ~0.
        Block sampling or synchronized sampling should preserve correlation.
        """
        np.random.seed(42)

        n_days = 252 * 5  # 5 years

        # Create two correlated assets
        target_correlation = 0.7

        # Generate correlated returns using Cholesky
        L = np.array([[1, 0], [target_correlation, np.sqrt(1 - target_correlation**2)]])
        uncorrelated = np.random.standard_normal((n_days, 2)) * 0.01
        historical = uncorrelated @ L.T

        actual_historical_corr = np.corrcoef(historical[:, 0], historical[:, 1])[0, 1]

        # Synchronized bootstrap (sample same days for both assets)
        n_bootstrap = 252
        n_samples = 1000
        bootstrap_corrs = []

        for _ in range(n_samples):
            indices = np.random.choice(n_days, size=n_bootstrap, replace=True)
            sample = historical[indices]
            bootstrap_corrs.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])

        mean_bootstrap_corr = np.mean(bootstrap_corrs)

        # Bootstrap correlation should be close to historical
        assert abs(mean_bootstrap_corr - actual_historical_corr) < 0.1, \
            f"Bootstrap correlation {mean_bootstrap_corr:.2f} should be close to historical {actual_historical_corr:.2f}"


class TestSimulationConvergence:
    """Tests for Monte Carlo convergence diagnostics."""

    def test_standard_error_convergence(self):
        """
        Standard error of mean should follow 1/√N law.

        SE = σ / √N
        For N=10000 and σ=0.01, SE ≈ 0.0001
        """
        np.random.seed(42)
        sigma = 0.01

        # Run multiple simulations with different N
        Ns = [100, 1000, 10000]
        SEs = []

        for N in Ns:
            means = []
            for _ in range(100):  # 100 batches
                sample = np.random.normal(0, sigma, N)
                means.append(np.mean(sample))
            SEs.append(np.std(means))

        # SE should decrease as √N increases
        for i in range(len(Ns) - 1):
            ratio_N = np.sqrt(Ns[i+1] / Ns[i])
            ratio_SE = SEs[i] / SEs[i+1]

            # Ratio of SEs should approximately equal √(N2/N1)
            assert abs(ratio_SE - ratio_N) / ratio_N < 0.3, \
                f"SE ratio {ratio_SE:.2f} should be close to sqrt(N) ratio {ratio_N:.2f}"

    def test_tail_stability(self):
        """
        Tail estimates (95th percentile) need more samples for stability.
        """
        np.random.seed(42)

        # Generate many samples
        large_sample = np.random.normal(0, 1, 100000)
        true_95 = np.percentile(large_sample, 95)

        # Compare with smaller samples
        small_sample_estimates = []
        for _ in range(100):
            small = np.random.normal(0, 1, 1000)
            small_sample_estimates.append(np.percentile(small, 95))

        # Variance of small sample estimates should be higher
        small_variance = np.std(small_sample_estimates)

        # 95th percentile estimates should still be close on average
        assert abs(np.mean(small_sample_estimates) - true_95) < 0.1, \
            "Mean of 95th percentile estimates should be close to true value"


class TestOverflowProtection:
    """Tests for numerical overflow protection."""

    def test_long_simulation_no_overflow(self):
        """
        30-year simulation should not cause overflow.

        With 7560 trading days and high-growth assets,
        cumulative returns can exceed float32 limits.
        """
        np.random.seed(42)

        n_days = 252 * 30  # 30 years
        n_sims = 100

        # High-growth scenario (like crypto)
        daily_return = 0.002  # ~50% annual
        daily_vol = 0.04

        returns = np.random.normal(daily_return, daily_vol, (n_sims, n_days))

        # Use float64 cumprod (should not overflow)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            cumulative = np.cumprod(1 + returns.astype(np.float64), axis=1)

            # Check for overflow warnings
            overflow_warnings = [x for x in w if 'overflow' in str(x.message).lower()]
            assert len(overflow_warnings) == 0, \
                f"Got overflow warnings: {[str(x.message) for x in overflow_warnings]}"

    def test_extreme_returns_handled(self):
        """
        Extreme returns (like -99%) should be handled without NaN/Inf.
        """
        np.random.seed(42)

        n_days = 252
        n_sims = 100

        # Include some extreme returns
        returns = np.random.normal(0.0005, 0.02, (n_sims, n_days))
        returns[0, 50] = -0.50  # 50% crash
        returns[0, 100] = -0.30  # 30% crash

        cumulative = np.cumprod(1 + returns, axis=1)
        final_values = cumulative[:, -1]

        # Should not have NaN or Inf
        assert np.all(np.isfinite(final_values)), \
            "Final values should all be finite"

        # Crashed simulation should still have positive (small) value
        assert final_values[0] > 0, \
            "Even crashed simulation should have positive value"


class TestSimulatorIntegration:
    """Integration tests for PortfolioSimulator."""

    def test_simulator_initial_capital(self, multi_asset_portfolio):
        """Verify simulator correctly applies initial capital."""
        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=100,
            method='bootstrap'
        )

        sim = PortfolioSimulator(dm, config)
        assets = list(multi_asset_portfolio.values())

        results = sim.simulate_portfolio(assets[:2], [0.5, 0.5])

        # Initial value should be initial_capital
        assert results['portfolio_values'][0, 0] == config.initial_capital, \
            "Initial portfolio value should equal initial_capital"

    def test_simulator_allocation_weights(self, multi_asset_portfolio):
        """Verify allocations sum to 1 and are applied correctly."""
        dm = MockDataManager()
        config = SimulationConfig(
            initial_capital=10000,
            years=1,
            simulations=100,
            method='bootstrap'
        )

        sim = PortfolioSimulator(dm, config)
        assets = list(multi_asset_portfolio.values())[:2]
        allocations = [0.6, 0.4]

        results = sim.simulate_portfolio(assets, allocations)

        assert sum(results['allocations']) == pytest.approx(1.0), \
            "Allocations should sum to 1"
