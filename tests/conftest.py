"""
Pytest fixtures for portfolio testing.
Provides deterministic test data with known mathematical outcomes.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def constant_returns():
    """
    100 days of constant 0.1% daily returns.
    Known CAGR: (1.001)^252 - 1 ≈ 28.6%
    Known volatility: 0
    """
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    returns = pd.Series([0.001] * 100, index=dates)
    return returns


@pytest.fixture
def alternating_returns():
    """
    Returns that alternate: +10%, -10%, +10%, -10%...
    Tests drawdown calculation and volatility.
    """
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    returns = pd.Series([0.10 if i % 2 == 0 else -0.10 for i in range(100)], index=dates)
    return returns


@pytest.fixture
def trending_up_returns():
    """
    Steady upward trend: +1% daily for 252 days.
    CAGR should be very high.
    """
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    returns = pd.Series([0.01] * 252, index=dates)
    return returns


@pytest.fixture
def drawdown_scenario():
    """
    Specific sequence to test drawdown:
    - Days 1-10: +2% daily (peak at ~21.9%)
    - Days 11-20: -3% daily (drawdown to ~-6.3% from peak)
    - Days 21-30: +1% daily (partial recovery)

    Expected max drawdown: approximately -26% from peak
    """
    dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
    returns = pd.Series(
        [0.02] * 10 +   # Growth phase
        [-0.03] * 10 +  # Crash phase
        [0.01] * 10,    # Recovery phase
        index=dates
    )
    return returns


@pytest.fixture
def two_asset_uncorrelated():
    """
    Two uncorrelated assets for mean-variance testing.
    Asset A: lower volatility (σ≈10%)
    Asset B: higher volatility (σ≈20%)
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Generate uncorrelated returns
    returns_a = np.random.normal(0.0004, 0.01, n_days)  # ~10% annual vol
    returns_b = np.random.normal(0.0006, 0.02, n_days)  # ~20% annual vol

    returns_df = pd.DataFrame({
        'A': returns_a,
        'B': returns_b
    }, index=dates)

    return returns_df


@pytest.fixture
def high_sharpe_asset():
    """
    Asset with high Sharpe ratio: 15% return, 10% vol.
    Sharpe = (0.15 - 0.04) / 0.10 = 1.1
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Daily: mean = 15%/252, std = 10%/sqrt(252)
    returns = np.random.normal(0.15/252, 0.10/np.sqrt(252), n_days)
    return pd.Series(returns, index=dates)


@pytest.fixture
def low_sharpe_asset():
    """
    Asset with low Sharpe ratio: 5% return, 25% vol.
    Sharpe = (0.05 - 0.04) / 0.25 = 0.04
    """
    np.random.seed(43)
    n_days = 252
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = np.random.normal(0.05/252, 0.25/np.sqrt(252), n_days)
    return pd.Series(returns, index=dates)


@pytest.fixture
def mock_asset_dict():
    """
    Create a mock asset dictionary matching the structure used by PortfolioSimulator.
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    returns = pd.Series(np.random.normal(0.0004, 0.01, n_days), index=dates)
    prices = (1 + returns).cumprod() * 100

    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)

    return {
        'ticker': 'TEST',
        'name': 'Test Asset',
        'historical_returns': returns,
        'full_data': df,
        'daily_mean': returns.mean(),
        'daily_std': returns.std()
    }


@pytest.fixture
def dca_scenario_returns():
    """
    Returns for DCA testing: 252 days with realistic market behavior.
    Mix of up and down periods to test contribution timing effects.
    """
    np.random.seed(44)
    n_days = 252
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Simulate market with some structure
    returns = np.random.normal(0.0003, 0.012, n_days)  # ~7.5% annual, ~19% vol
    return pd.Series(returns, index=dates)


@pytest.fixture
def multi_asset_portfolio():
    """
    Create 4 assets with different characteristics for portfolio testing.
    """
    np.random.seed(45)
    n_days = 504  # 2 years
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # Asset characteristics (annual)
    assets = {
        'STOCKS': {'return': 0.10, 'vol': 0.20},
        'BONDS': {'return': 0.04, 'vol': 0.05},
        'GOLD': {'return': 0.03, 'vol': 0.15},
        'CASH': {'return': 0.02, 'vol': 0.001}
    }

    result = {}
    for name, params in assets.items():
        daily_return = params['return'] / 252
        daily_vol = params['vol'] / np.sqrt(252)
        returns = np.random.normal(daily_return, daily_vol, n_days)
        prices = (1 + pd.Series(returns)).cumprod() * 100

        df = pd.DataFrame({
            'Adj Close': prices.values,
            'Close': prices.values
        }, index=dates)

        result[name] = {
            'ticker': name,
            'name': name,
            'historical_returns': pd.Series(returns, index=dates),
            'full_data': df,
            'daily_mean': np.mean(returns),
            'daily_std': np.std(returns)
        }

    return result
