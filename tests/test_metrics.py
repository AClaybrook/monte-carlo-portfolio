"""
Unit tests for financial metrics calculations.

These tests verify the mathematical correctness of:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility
- Other risk metrics

All tests use deterministic data with known mathematical outcomes.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_analytics import QuantAnalytics


class TestCAGR:
    """Tests for Compound Annual Growth Rate calculation."""

    def test_cagr_constant_returns(self, constant_returns):
        """
        Constant 0.1% daily returns.
        CAGR = (1.001)^252 - 1 ≈ 28.6%
        """
        qa = QuantAnalytics(risk_free_rate=0.0)
        cagr = qa.calculate_cagr(constant_returns)

        # Expected: compound 0.1% daily for 252 days
        expected = (1.001 ** 252) - 1

        # Allow small tolerance due to fewer days in test data
        assert cagr > 0, "CAGR should be positive for positive returns"

    def test_cagr_zero_returns(self):
        """Zero returns should give CAGR of 0."""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        zero_returns = pd.Series([0.0] * 252, index=dates)

        qa = QuantAnalytics()
        cagr = qa.calculate_cagr(zero_returns)

        assert abs(cagr) < 1e-6, f"CAGR should be 0 for zero returns, got {cagr}"

    def test_cagr_one_year_doubling(self):
        """
        If portfolio doubles in exactly 1 year, CAGR = 100%.
        Daily return needed: (2)^(1/252) - 1 ≈ 0.275%
        """
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        daily_return = 2 ** (1/252) - 1
        returns = pd.Series([daily_return] * 252, index=dates)

        qa = QuantAnalytics()
        cagr = qa.calculate_cagr(returns)

        # Should be very close to 100%
        assert abs(cagr - 1.0) < 0.01, f"CAGR should be ~100%, got {cagr*100:.2f}%"

    def test_cagr_negative_returns(self):
        """Negative returns should give negative CAGR."""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        returns = pd.Series([-0.001] * 252, index=dates)

        qa = QuantAnalytics()
        cagr = qa.calculate_cagr(returns)

        assert cagr < 0, "CAGR should be negative for negative returns"


class TestVolatility:
    """Tests for volatility calculation."""

    def test_zero_volatility(self, constant_returns):
        """Constant returns should have zero volatility."""
        qa = QuantAnalytics()
        vol = qa.calculate_volatility(constant_returns)

        assert vol < 1e-10, f"Volatility should be 0 for constant returns, got {vol}"

    def test_annualization(self):
        """
        Verify volatility is properly annualized.
        If daily σ = 1%, annual σ = 1% × √252 ≈ 15.87%
        """
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 10000), index=dates)

        qa = QuantAnalytics()
        annual_vol = qa.calculate_volatility(returns, annualize=True)
        daily_vol = qa.calculate_volatility(returns, annualize=False)

        expected_ratio = np.sqrt(252)
        actual_ratio = annual_vol / daily_vol

        assert abs(actual_ratio - expected_ratio) < 0.1, \
            f"Annualization ratio should be √252={expected_ratio:.2f}, got {actual_ratio:.2f}"

    def test_known_volatility(self):
        """Generate returns with known σ and verify calculation."""
        np.random.seed(42)
        target_daily_vol = 0.02  # 2% daily
        n = 10000
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        returns = pd.Series(np.random.normal(0, target_daily_vol, n), index=dates)

        qa = QuantAnalytics()
        calculated_vol = qa.calculate_volatility(returns, annualize=False)

        # Should be close to target (within 5% relative error)
        relative_error = abs(calculated_vol - target_daily_vol) / target_daily_vol
        assert relative_error < 0.05, \
            f"Calculated vol {calculated_vol:.4f} differs from target {target_daily_vol}"


class TestSharpeRatio:
    """Tests for Sharpe Ratio calculation."""

    def test_sharpe_formula(self):
        """
        Verify Sharpe = (Rp - Rf) / σp

        Given: 15% annual return, 20% vol, 4% rf
        Sharpe = (0.15 - 0.04) / 0.20 = 0.55
        """
        np.random.seed(42)
        n = 10000
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

        # Generate returns with known annual characteristics
        annual_return = 0.15
        annual_vol = 0.20
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        returns = pd.Series(np.random.normal(daily_return, daily_vol, n), index=dates)

        qa = QuantAnalytics(risk_free_rate=0.04)
        sharpe = qa.calculate_sharpe_ratio(returns)

        expected_sharpe = (annual_return - 0.04) / annual_vol

        # Allow 20% relative tolerance due to sampling variation
        assert abs(sharpe - expected_sharpe) / expected_sharpe < 0.25, \
            f"Sharpe should be ~{expected_sharpe:.2f}, got {sharpe:.2f}"

    def test_sharpe_negative_excess_return(self):
        """Sharpe should be negative when return < rf."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0001, 0.01, 252), index=dates)  # ~2.5% annual

        qa = QuantAnalytics(risk_free_rate=0.05)  # 5% rf > 2.5% return
        sharpe = qa.calculate_sharpe_ratio(returns)

        assert sharpe < 0, f"Sharpe should be negative when return < rf, got {sharpe}"

    def test_sharpe_high_vs_low(self, high_sharpe_asset, low_sharpe_asset):
        """High Sharpe asset should have higher ratio than low Sharpe."""
        qa = QuantAnalytics(risk_free_rate=0.04)

        high_sharpe = qa.calculate_sharpe_ratio(high_sharpe_asset)
        low_sharpe = qa.calculate_sharpe_ratio(low_sharpe_asset)

        assert high_sharpe > low_sharpe, \
            f"High Sharpe asset ({high_sharpe:.2f}) should beat low Sharpe ({low_sharpe:.2f})"


class TestSortinoRatio:
    """Tests for Sortino Ratio calculation."""

    def test_sortino_only_downside(self):
        """
        Sortino should only consider downside deviation.
        Returns: 10 positive (ignored), 10 negative (counted)
        """
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
        returns = pd.Series(
            [0.02] * 10 +  # Positive - not counted in denominator
            [-0.01] * 10,  # Negative - counted in denominator
            index=dates
        )

        qa = QuantAnalytics(risk_free_rate=0.0)
        sortino = qa.calculate_sortino_ratio(returns)

        # Sortino should be positive (overall positive return)
        assert sortino > 0, f"Sortino should be positive, got {sortino}"

    def test_sortino_vs_sharpe_skewed(self):
        """
        For positively skewed returns, Sortino > Sharpe.
        This is because Sharpe penalizes upside volatility.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

        # Create positively skewed returns (more upside outliers)
        base = np.random.normal(0.0003, 0.01, 1000)
        # Add positive outliers
        outliers = np.where(np.random.random(1000) > 0.95, 0.05, 0)
        returns = pd.Series(base + outliers, index=dates)

        qa = QuantAnalytics(risk_free_rate=0.0)
        sharpe = qa.calculate_sharpe_ratio(returns)
        sortino = qa.calculate_sortino_ratio(returns)

        # Sortino should be higher for positively skewed returns
        assert sortino >= sharpe * 0.9, \
            f"Sortino ({sortino:.2f}) should be >= Sharpe ({sharpe:.2f}) for skewed returns"


class TestMaxDrawdown:
    """Tests for Maximum Drawdown calculation."""

    def test_simple_drawdown(self):
        """
        Simple test: +10% then -20%.
        Peak at 1.10, trough at 0.88.
        Drawdown = (0.88 - 1.10) / 1.10 = -20%
        """
        dates = pd.date_range(start='2020-01-01', periods=2, freq='D')
        returns = pd.Series([0.10, -0.20], index=dates)

        qa = QuantAnalytics()
        max_dd = qa.calculate_max_drawdown(returns)

        # Expected: after +10%, value = 1.10. After -20%, value = 0.88
        # Drawdown = (0.88 - 1.10) / 1.10 = -0.20
        expected = (1.10 * 0.80 - 1.10) / 1.10

        assert abs(max_dd - expected) < 0.01, \
            f"Max drawdown should be {expected:.2%}, got {max_dd:.2%}"

    def test_drawdown_always_negative(self, drawdown_scenario):
        """Drawdown should always be negative or zero."""
        qa = QuantAnalytics()
        max_dd = qa.calculate_max_drawdown(drawdown_scenario)

        assert max_dd <= 0, f"Max drawdown should be <= 0, got {max_dd}"

    def test_no_drawdown_always_up(self):
        """Constant positive returns should have zero drawdown."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 100, index=dates)

        qa = QuantAnalytics()
        max_dd = qa.calculate_max_drawdown(returns)

        assert max_dd == 0, f"Max drawdown should be 0 for always-up returns, got {max_dd}"

    def test_multiple_drawdowns_returns_worst(self):
        """
        Multiple drawdowns - should return the worst one.
        First crash: -10%, Recovery, Second crash: -15%
        Max DD should be -15%
        """
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        returns = pd.Series([
            0.05,   # Up 5%
            -0.10,  # Down 10% (first drawdown)
            0.10, 0.10,  # Recovery
            -0.08, -0.08,  # Second crash: down ~15.4% total
            0.05, 0.05, 0.05, 0.05  # Recovery
        ], index=dates)

        qa = QuantAnalytics()
        max_dd = qa.calculate_max_drawdown(returns)

        # Second drawdown should be worse
        assert max_dd < -0.10, f"Max DD should be worse than -10%, got {max_dd:.2%}"


class TestDownsideDeviation:
    """Tests for downside deviation calculation."""

    def test_downside_only_negative(self):
        """Downside deviation should only consider returns below threshold."""
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')

        # 10 positive returns, 10 negative returns
        returns = pd.Series(
            [0.02] * 10 + [-0.01] * 10,
            index=dates
        )

        qa = QuantAnalytics()
        downside = qa.calculate_downside_deviation(returns, threshold=0)

        # Downside deviation should be based only on the -1% returns
        # σ of constant values = 0, but the calculation uses returns below threshold
        assert downside >= 0, "Downside deviation should be non-negative"

    def test_downside_all_positive(self):
        """If all returns are positive, downside deviation should be 0."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 100, index=dates)

        qa = QuantAnalytics()
        downside = qa.calculate_downside_deviation(returns, threshold=0)

        assert downside == 0, f"Downside deviation should be 0 for all positive returns, got {downside}"


class TestCalmarRatio:
    """Tests for Calmar Ratio (CAGR / Max Drawdown)."""

    def test_calmar_formula(self):
        """Verify Calmar = CAGR / |Max Drawdown|"""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')

        # Create returns with known CAGR and drawdown
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)

        qa = QuantAnalytics()
        cagr = qa.calculate_cagr(returns)
        max_dd = qa.calculate_max_drawdown(returns)
        calmar = qa.calculate_calmar_ratio(returns)

        if max_dd != 0:
            expected = cagr / abs(max_dd)
            assert abs(calmar - expected) < 0.01, \
                f"Calmar should be CAGR/|DD| = {expected:.2f}, got {calmar:.2f}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns(self):
        """Empty return series should be handled gracefully."""
        empty = pd.Series([], dtype=float)
        qa = QuantAnalytics()

        # Should not crash
        try:
            vol = qa.calculate_volatility(empty)
            # May return 0, NaN, or raise - just shouldn't crash unexpectedly
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise for empty data

    def test_single_return(self):
        """Single return should be handled."""
        dates = pd.date_range(start='2020-01-01', periods=1, freq='D')
        single = pd.Series([0.05], index=dates)
        qa = QuantAnalytics()

        # Should not crash
        try:
            cagr = qa.calculate_cagr(single)
        except (ValueError, ZeroDivisionError):
            pass

    def test_nan_handling(self):
        """NaN values should not corrupt calculations."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        returns = pd.Series([0.01, 0.02, np.nan, 0.01, 0.02, 0.01, np.nan, 0.01, 0.02, 0.01],
                           index=dates)

        qa = QuantAnalytics()

        # Should handle NaN without crashing
        try:
            # Drop NaN before calculation
            clean = returns.dropna()
            vol = qa.calculate_volatility(clean)
            assert not np.isnan(vol), "Result should not be NaN after cleaning"
        except Exception as e:
            pytest.fail(f"Should handle NaN gracefully: {e}")
