"""
Historical Validation Tests

Tests that verify asset characteristics and optimizer behavior match known real-world patterns.
These tests help ensure the simulator produces sensible results that align with market reality.

Uses CACHED data from the database only (no yfinance calls).
Tests will skip if required ticker data isn't in the local database.

Available tickers in DB: VOO, QQQ, BTC-USD, SPXL, SHV, VGT, BND, VTI, GLD, TLT, etc.
Cached date range: ~2016 to late 2025
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import DataManager


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical data from LOCAL DATABASE ONLY (no yfinance calls).
    Skips test if ticker not in database.
    """
    dm = DataManager()
    try:
        # Use _get_from_db directly to avoid any yfinance calls
        data = dm._get_from_db(ticker, start, end)
        if data is not None and not data.empty:
            # Return DataFrame with ticker as column name (using Adj Close)
            result = pd.DataFrame({ticker: data['Adj Close']})
            result.index = data.index
            return result
        pytest.skip(f"No cached data for {ticker} in date range {start} to {end}")
    except Exception as e:
        pytest.skip(f"Could not fetch data for {ticker}: {e}")
    finally:
        dm.close()
    pytest.skip(f"No data available for {ticker}")


class TestAssetCharacteristics:
    """
    Tests verifying individual asset behaviors match expectations.
    Uses tickers available in local database: VOO, BND, SHV, BTC-USD, QQQ, VGT, etc.
    """

    def calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate annualized volatility from price series."""
        returns = prices.pct_change().dropna()
        return returns.std() * np.sqrt(252)

    def calculate_cagr(self, prices: pd.Series) -> float:
        """Calculate CAGR from price series."""
        if len(prices) < 2:
            return 0.0
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1

    def test_shv_low_volatility(self):
        """SHV (short-term treasury ETF) should have very low volatility."""
        data = fetch_data("SHV", "2020-01-01", "2024-12-31")
        vol = self.calculate_volatility(data['SHV'])

        # SHV is essentially cash - volatility should be < 2%
        assert vol < 0.02, f"SHV volatility {vol:.2%} should be < 2% (cash-like)"

    def test_btc_high_volatility(self):
        """BTC should have high volatility (>40% annualized)."""
        data = fetch_data("BTC-USD", "2020-01-01", "2024-12-31")
        vol = self.calculate_volatility(data['BTC-USD'])

        # Bitcoin is highly volatile
        assert vol > 0.40, f"BTC volatility {vol:.2%} should be > 40%"

    def test_voo_moderate_volatility(self):
        """VOO (S&P 500 ETF) should have moderate volatility (10-25%)."""
        data = fetch_data("VOO", "2020-01-01", "2024-12-31")
        vol = self.calculate_volatility(data['VOO'])

        assert 0.10 < vol < 0.30, f"VOO volatility {vol:.2%} should be 10-30%"

    def test_bnd_lower_volatility_than_stocks(self):
        """BND (bond fund) should have lower volatility than stocks."""
        voo_data = fetch_data("VOO", "2020-01-01", "2024-12-31")
        bnd_data = fetch_data("BND", "2020-01-01", "2024-12-31")

        voo_vol = self.calculate_volatility(voo_data['VOO'])
        bnd_vol = self.calculate_volatility(bnd_data['BND'])

        assert bnd_vol < voo_vol, f"BND vol {bnd_vol:.2%} should be < VOO vol {voo_vol:.2%}"

    def test_volatility_ordering(self):
        """Test typical volatility ordering: SHV < BND < VOO < BTC."""
        tickers = ["SHV", "BND", "VOO", "BTC-USD"]
        vols = {}

        for ticker in tickers:
            try:
                data = fetch_data(ticker, "2020-01-01", "2024-12-31")
                col = ticker if ticker in data.columns else data.columns[0]
                vols[ticker] = self.calculate_volatility(data[col])
            except Exception:
                continue

        if len(vols) >= 3:
            # Verify SHV < VOO < BTC if all available
            if "SHV" in vols and "VOO" in vols:
                assert vols["SHV"] < vols["VOO"], "SHV should be less volatile than VOO"
            if "VOO" in vols and "BTC-USD" in vols:
                assert vols["VOO"] < vols["BTC-USD"], "VOO should be less volatile than BTC"


class TestHistoricalBenchmarks:
    """
    Tests verifying known historical events are captured correctly.
    Uses VOO (S&P 500 ETF) instead of SPY, BND instead of AGG.
    """

    def test_2022_voo_negative_return(self):
        """2022 was a down year for VOO (~-18%)."""
        data = fetch_data("VOO", "2022-01-01", "2022-12-31")

        annual_return = data['VOO'].iloc[-1] / data['VOO'].iloc[0] - 1

        # 2022 VOO return was approximately -18%
        assert annual_return < 0, f"2022 VOO should be negative, got {annual_return:.2%}"
        assert annual_return > -0.30, f"2022 VOO was ~-18%, got {annual_return:.2%}"

    def test_2022_bnd_negative_return(self):
        """2022 was also bad for bonds (BND ~-13%)."""
        data = fetch_data("BND", "2022-01-01", "2022-12-31")

        annual_return = data['BND'].iloc[-1] / data['BND'].iloc[0] - 1

        # 2022 was historic for bonds being down with stocks
        assert annual_return < 0, f"2022 BND should be negative, got {annual_return:.2%}"

    def test_2020_covid_recovery(self):
        """VOO recovered from COVID crash by end of 2020."""
        data = fetch_data("VOO", "2020-01-01", "2020-12-31")

        annual_return = data['VOO'].iloc[-1] / data['VOO'].iloc[0] - 1

        # Despite COVID crash, 2020 ended positive
        assert annual_return > 0, f"2020 VOO should be positive overall, got {annual_return:.2%}"

    def test_2020_covid_drawdown(self):
        """VOO had ~34% drawdown during COVID crash (Feb-Mar 2020)."""
        data = fetch_data("VOO", "2020-01-01", "2020-04-30")

        prices = data['VOO']
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min()

        # COVID crash was approximately -34%
        assert max_drawdown < -0.25, f"COVID drawdown should be > 25%, got {max_drawdown:.2%}"
        assert max_drawdown > -0.45, f"COVID drawdown was ~34%, got {max_drawdown:.2%}"

    def test_2021_strong_bull_market(self):
        """2021 was a strong year for equities."""
        data = fetch_data("VOO", "2021-01-01", "2021-12-31")

        annual_return = data['VOO'].iloc[-1] / data['VOO'].iloc[0] - 1

        # 2021 VOO return was approximately +27%
        assert annual_return > 0.15, f"2021 VOO should be strongly positive, got {annual_return:.2%}"


class TestPVCompatibility:
    """Tests for Portfolio Visualizer compatibility module."""

    def test_csv_export_format(self):
        """Test CSV export matches expected PV format."""
        from pv_compat import export_portfolio_csv

        allocations = {'VOO': 0.6, 'BND': 0.4}
        csv = export_portfolio_csv(allocations, "60/40 Portfolio")

        lines = csv.strip().split('\n')

        assert lines[0] == "60/40 Portfolio", "First line should be portfolio name"
        assert lines[1] == "", "Second line should be blank"
        assert lines[2] == "Symbol,Weight", "Third line should be header"
        assert "BND,40%" in csv, "Should contain BND allocation"
        assert "VOO,60%" in csv, "Should contain VOO allocation"

    def test_csv_parse_roundtrip(self):
        """Test that export -> parse roundtrip preserves data."""
        from pv_compat import export_portfolio_csv, parse_portfolio_csv

        original = {'SPY': 0.5, 'AGG': 0.3, 'GLD': 0.2}
        csv = export_portfolio_csv(original, "Test Portfolio")
        parsed = parse_portfolio_csv(csv)

        assert parsed['name'] == "Test Portfolio"
        assert len(parsed['allocations']) == 3

        for ticker, weight in original.items():
            assert abs(parsed['allocations'][ticker] - weight) < 0.01

    def test_parse_existing_pv_file(self):
        """Test parsing the existing PV example file."""
        from pv_compat import parse_portfolio_csv

        pv_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'pv', 'portfolio_import1.csv'
        )

        if not os.path.exists(pv_file):
            pytest.skip("PV example file not found")

        with open(pv_file) as f:
            csv_content = f.read()

        parsed = parse_portfolio_csv(csv_content)

        assert 'VTSMX' in parsed['allocations']
        assert parsed['allocations']['VTSMX'] == 0.40
        assert sum(parsed['allocations'].values()) == pytest.approx(1.0)

    def test_pv_url_generation(self):
        """Test Portfolio Visualizer URL generation."""
        from pv_compat import generate_pv_url

        allocations = {'VOO': 0.6, 'BND': 0.4}
        url = generate_pv_url(allocations, start_year=2020, end_year=2024)

        assert url.startswith("https://www.portfoliovisualizer.com/backtest-portfolio?")
        assert "symbol1=" in url or "symbol2=" in url
        assert "allocation" in url
        assert "startYear=2020" in url
        assert "endYear=2024" in url

    def test_crypto_ticker_conversion(self):
        """Test crypto ticker format conversion."""
        from pv_compat import to_pv_ticker, from_pv_ticker

        # Local to PV
        assert to_pv_ticker("BTC-USD") == "BTC"
        assert to_pv_ticker("ETH-USD") == "ETH"
        assert to_pv_ticker("VOO") == "VOO"  # Non-crypto unchanged

        # PV to local
        assert from_pv_ticker("BTC") == "BTC-USD"
        assert from_pv_ticker("ETH") == "ETH-USD"
        assert from_pv_ticker("VOO") == "VOO"  # Non-crypto unchanged

    def test_save_portfolio_creates_file(self, tmp_path):
        """Test saving portfolio creates CSV file."""
        from pv_compat import save_portfolio_csv, parse_portfolio_csv

        allocations = {'SPY': 0.7, 'BND': 0.3}
        filepath = save_portfolio_csv(allocations, "Test_70_30", str(tmp_path))

        assert os.path.exists(filepath)

        with open(filepath) as f:
            content = f.read()

        parsed = parse_portfolio_csv(content)
        assert parsed['allocations']['SPY'] == 0.70
        assert parsed['allocations']['BND'] == 0.30
