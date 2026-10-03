"""
System Test: Portfolio Visualizer Benchmark Comparison

Validates backtest accuracy by comparing against known Portfolio Visualizer results.
Uses cached data from local database only (no yfinance calls).
Skips if required tickers aren't available.

Reference: https://www.portfoliovisualizer.com/backtest-portfolio?s=y&sl=6rjRrv2DMYbQwP6NE9COc8

PV Configuration:
    Time Period: Month-to-Month
    Date Range: Feb 2016 - Jan 2026 (constrained by GBTC availability across portfolios)
    Initial Amount: $10,000
    Cashflows: None
    Rebalancing: No rebalancing (buy and hold)
    Reinvest Dividends: Yes (via Adj Close prices)
    Benchmark: Vanguard 500 Index Investor (VFINX)

Portfolios tested:
    Portfolio 1: VOO 40%, QQQ 20%, GBTC 10%, VGT 30%
    Portfolio 2: VOO 70%, QQQ 30%  (primary validation - most reliable)
    Portfolio 3: VOO 40%, QQQ 20%, GBTC 10%, VGT 20%, SPXL 10%

Methodology differences from PV:
    - PV uses monthly returns; we use daily (affects std dev, max drawdown depth)
    - PV subtracts risk-free rate for Sharpe/Sortino; we note this in tolerances
    - Daily data captures intra-month drawdowns (our max DD will be deeper)
    - CAGR should match closely since it depends only on start/end values
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import DataManager
from backtester import Backtester

# ============================================================
# Portfolio Visualizer Reference Data
# ============================================================

PV_START_DATE = date(2016, 2, 1)
PV_END_DATE = date(2026, 1, 31)
PV_INITIAL_CAPITAL = 10000

PV_PORTFOLIO_1 = {
    'name': 'Portfolio 1: VOO 40% / QQQ 20% / GBTC 10% / VGT 30%',
    'tickers': ['VOO', 'QQQ', 'GBTC', 'VGT'],
    'weights': {'VOO': 0.40, 'QQQ': 0.20, 'GBTC': 0.10, 'VGT': 0.30},
    'expected': {
        'CAGR': 0.3502,
        'End Balance': 201293,
        'Stdev': 0.5693,
        'Max Drawdown': -0.6999,
        'Sharpe': 0.76,
        'Sortino': 1.50,
        'Best Year': 3.7268,
        'Worst Year': -0.6457,
    },
    'annual_returns': {
        2016: 0.3475, 2017: 3.7268, 2018: -0.6457, 2019: 0.6606,
        2020: 0.0136, 2021: 0.0704, 2022: -0.5093, 2023: 1.6201,
        2024: 0.8683, 2025: -0.0174,
    },
}

PV_PORTFOLIO_2 = {
    'name': 'Portfolio 2: VOO 70% / QQQ 30%',
    'tickers': ['VOO', 'QQQ'],
    'weights': {'VOO': 0.70, 'QQQ': 0.30},
    'expected': {
        'CAGR': 0.1721,
        'End Balance': 48955,
        'Stdev': 0.1578,
        'Max Drawdown': -0.2722,
        'Sharpe': 0.95,
        'Sortino': 1.52,
        'Best Year': 0.3604,
        'Worst Year': -0.2373,
    },
    'annual_returns': {
        2016: 0.1709, 2017: 0.2498, 2018: -0.0313, 2019: 0.3381,
        2020: 0.2845, 2021: 0.2825, 2022: -0.2373, 2023: 0.3604,
        2024: 0.2521, 2025: 0.1896,
    },
}

PV_PORTFOLIO_3 = {
    'name': 'Portfolio 3: VOO 40% / QQQ 20% / GBTC 10% / VGT 20% / SPXL 10%',
    'tickers': ['VOO', 'QQQ', 'GBTC', 'VGT', 'SPXL'],
    'weights': {'VOO': 0.40, 'QQQ': 0.20, 'GBTC': 0.10, 'VGT': 0.20, 'SPXL': 0.10},
    'expected': {
        'CAGR': 0.3543,
        'End Balance': 207564,
        'Stdev': 0.5684,
        'Max Drawdown': -0.7039,
        'Sharpe': 0.76,
        'Sortino': 1.50,
        'Best Year': 3.6838,
        'Worst Year': -0.6468,
    },
    'annual_returns': {
        2016: 0.3808, 2017: 3.6838, 2018: -0.6468, 2019: 0.7053,
        2020: 0.0060, 2021: 0.1604, 2022: -0.6191, 2023: 1.6244,
        2024: 0.8811, 2025: -0.0083,
    },
}

PV_BENCHMARK = {
    'name': 'Vanguard 500 Index Investor (VFINX)',
    'ticker': 'VFINX',
    'expected': {
        'CAGR': 0.1542,
        'End Balance': 41941,
        'Stdev': 0.1499,
        'Max Drawdown': -0.2395,
        'Sharpe': 0.89,
        'Sortino': 1.39,
    },
}


# ============================================================
# Helper Functions
# ============================================================

def get_data_manager():
    """Get DataManager using default database."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'stock_data.db'
    )
    if not os.path.exists(db_path):
        pytest.skip("stock_data.db not found")
    return DataManager(db_path=db_path)


def load_ticker_data(dm, ticker, start_date, end_date):
    """Load ticker data from DB, return None if not available."""
    try:
        df = dm._get_from_db(ticker, start_date, end_date)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def buy_and_hold_portfolio(prices_df, weights, initial_capital=10000):
    """
    Compute true buy-and-hold portfolio value (no rebalancing).

    This matches PV's "No rebalancing" mode: invest weight*capital into each
    asset at start, then hold. Weights drift as assets perform differently.

    Args:
        prices_df: DataFrame with Adj Close prices, columns = tickers
        weights: dict of ticker -> weight (must sum to 1.0)
        initial_capital: starting investment

    Returns:
        Series of daily portfolio values
    """
    initial_prices = prices_df.iloc[0]
    shares = {}
    for col in prices_df.columns:
        shares[col] = (initial_capital * weights[col]) / initial_prices[col]

    portfolio_value = pd.Series(0.0, index=prices_df.index, dtype=float)
    for col in prices_df.columns:
        portfolio_value += shares[col] * prices_df[col]

    return portfolio_value


def calculate_cagr(portfolio_value):
    """CAGR using calendar days (matching PV methodology)."""
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1


def calculate_max_drawdown(portfolio_value):
    """Maximum drawdown from peak."""
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return drawdown.min()


def calculate_annualized_std(portfolio_value):
    """Annualized standard deviation from daily returns."""
    daily_returns = portfolio_value.pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)


def calculate_annual_returns(portfolio_value):
    """
    Calculate calendar-year returns from daily portfolio values.
    Partial years (first/last) use actual start/end dates.
    """
    annual_returns = {}
    years = sorted(set(d.year for d in portfolio_value.index))

    for year in years:
        year_data = portfolio_value[portfolio_value.index.map(
            lambda d: d.year == year
        )]
        if len(year_data) < 2:
            continue
        annual_returns[year] = year_data.iloc[-1] / year_data.iloc[0] - 1

    return annual_returns


def print_comparison(name, actual, expected):
    """Print side-by-side comparison for debugging."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  {'Metric':<20} {'Actual':>12} {'PV Expected':>12} {'Diff':>12}")
    print(f"  {'-'*56}")

    for key in expected:
        if key not in actual:
            continue
        a = actual[key]
        e = expected[key]
        if key == 'End Balance':
            diff = f"{(a/e - 1)*100:+.1f}%"
            print(f"  {key:<20} ${a:>10,.0f} ${e:>10,.0f} {diff:>12}")
        elif isinstance(e, float) and abs(e) > 1:
            diff = f"{a-e:+.2f}"
            print(f"  {key:<20} {a:>12.2f} {e:>12.2f} {diff:>12}")
        else:
            diff = f"{(a-e)*100:+.2f}pp"
            print(f"  {key:<20} {a*100:>11.2f}% {e*100:>11.2f}% {diff:>12}")


def print_annual_comparison(name, actual_annual, expected_annual):
    """Print annual return comparison."""
    print(f"\n  Annual Returns for {name}:")
    print(f"  {'Year':<6} {'Actual':>10} {'PV':>10} {'Diff':>10}")
    print(f"  {'-'*36}")
    for year in sorted(expected_annual.keys()):
        if year in actual_annual:
            a = actual_annual[year]
            e = expected_annual[year]
            diff = (a - e) * 100
            print(f"  {year:<6} {a*100:>9.2f}% {e*100:>9.2f}% {diff:>+9.2f}pp")


# ============================================================
# Test Class
# ============================================================

class TestPortfolioVisualizerBenchmark:
    """
    System test comparing our results to Portfolio Visualizer.

    Tests use cached database data only (no yfinance calls).
    Skips if required tickers aren't available in the database.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up DataManager for tests."""
        self.dm = get_data_manager()
        yield
        self.dm.close()

    def _load_portfolio_prices(self, tickers, weights):
        """Load aligned Adj Close prices for portfolio tickers."""
        price_series = {}
        for ticker in tickers:
            df = load_ticker_data(self.dm, ticker, PV_START_DATE, PV_END_DATE)
            if df is None:
                pytest.skip(f"No cached data for {ticker}")
            price_series[ticker] = df['Adj Close']

        prices = pd.DataFrame(price_series)
        prices = prices.dropna()

        if len(prices) < 252:
            pytest.skip(f"Insufficient data: only {len(prices)} trading days")

        return prices

    def _run_buy_and_hold(self, pv_config):
        """Run buy-and-hold backtest and compute metrics."""
        prices = self._load_portfolio_prices(
            pv_config['tickers'], pv_config['weights']
        )
        portfolio = buy_and_hold_portfolio(
            prices, pv_config['weights'], PV_INITIAL_CAPITAL
        )
        return portfolio, {
            'CAGR': calculate_cagr(portfolio),
            'End Balance': portfolio.iloc[-1],
            'Stdev': calculate_annualized_std(portfolio),
            'Max Drawdown': calculate_max_drawdown(portfolio),
        }

    # ================================================================
    # Portfolio 2: VOO 70% / QQQ 30% — Primary Validation
    # Both tickers are highly liquid with accurate data.
    # ================================================================

    def test_p2_cagr(self):
        """Portfolio 2 CAGR should be close to PV's 17.21%."""
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_2)
        print_comparison(PV_PORTFOLIO_2['name'], metrics, PV_PORTFOLIO_2['expected'])

        expected = PV_PORTFOLIO_2['expected']['CAGR']
        assert abs(metrics['CAGR'] - expected) < 0.02, \
            f"CAGR {metrics['CAGR']:.2%} should be within 2pp of PV {expected:.2%}"

    def test_p2_end_balance(self):
        """Portfolio 2 end balance should be close to PV's $48,955."""
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_2)

        expected = PV_PORTFOLIO_2['expected']['End Balance']
        pct_diff = abs(metrics['End Balance'] / expected - 1.0)
        assert pct_diff < 0.10, \
            f"End balance ${metrics['End Balance']:,.0f} should be within 10% " \
            f"of PV ${expected:,.0f} (diff: {pct_diff:.1%})"

    def test_p2_max_drawdown(self):
        """Portfolio 2 max drawdown should be in the right range.

        PV uses monthly data so reports -27.22%. Daily data captures
        intra-month moves, so our drawdown will be deeper (more negative).
        """
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_2)

        pv_dd = PV_PORTFOLIO_2['expected']['Max Drawdown']
        actual_dd = metrics['Max Drawdown']

        # Our daily DD should be at least as deep as PV's monthly DD (or very close)
        assert actual_dd < 0, "Max drawdown should be negative"
        assert actual_dd > -0.45, f"Max drawdown {actual_dd:.2%} seems unreasonably deep"
        # Daily DD should be similar to or deeper than monthly
        assert actual_dd <= pv_dd + 0.05, \
            f"Daily DD {actual_dd:.2%} should be at least close to monthly DD {pv_dd:.2%}"

    def test_p2_volatility(self):
        """Portfolio 2 annualized std dev should be close to PV's 15.78%.

        PV annualizes monthly std dev (×sqrt(12)). We annualize daily (×sqrt(252)).
        These typically give similar results for liquid assets.
        """
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_2)

        expected = PV_PORTFOLIO_2['expected']['Stdev']
        assert abs(metrics['Stdev'] - expected) < 0.05, \
            f"Std dev {metrics['Stdev']:.2%} should be within 5pp of PV {expected:.2%}"

    def test_p2_annual_returns(self):
        """Portfolio 2 annual returns should match PV within tolerance.

        Full-year returns should be very close since they depend on
        year-boundary prices from the same data source.
        """
        portfolio, _ = self._run_buy_and_hold(PV_PORTFOLIO_2)
        actual_annual = calculate_annual_returns(portfolio)
        expected_annual = PV_PORTFOLIO_2['annual_returns']

        print_annual_comparison(PV_PORTFOLIO_2['name'], actual_annual, expected_annual)

        # Check full calendar years (2017-2024) — skip partial years (2016, 2025/2026)
        for year in range(2017, 2025):
            if year not in actual_annual or year not in expected_annual:
                continue
            diff = abs(actual_annual[year] - expected_annual[year])
            assert diff < 0.03, \
                f"Year {year}: return {actual_annual[year]:.2%} differs from " \
                f"PV {expected_annual[year]:.2%} by {diff:.2%}"

    # ================================================================
    # Benchmark: VFINX (Vanguard 500 Index Investor)
    # ================================================================

    def test_benchmark_cagr(self):
        """VFINX benchmark CAGR should be close to PV's 15.42%."""
        df = load_ticker_data(self.dm, 'VFINX', PV_START_DATE, PV_END_DATE)
        if df is None:
            pytest.skip("No cached data for VFINX")

        prices = df['Adj Close']
        portfolio = prices * (PV_INITIAL_CAPITAL / prices.iloc[0])

        cagr = calculate_cagr(portfolio)
        expected = PV_BENCHMARK['expected']['CAGR']

        print(f"\n  VFINX Benchmark: CAGR = {cagr:.2%} (PV: {expected:.2%})")

        assert abs(cagr - expected) < 0.02, \
            f"VFINX CAGR {cagr:.2%} should be within 2pp of PV {expected:.2%}"

    def test_benchmark_end_balance(self):
        """VFINX end balance should be close to PV's $41,941."""
        df = load_ticker_data(self.dm, 'VFINX', PV_START_DATE, PV_END_DATE)
        if df is None:
            pytest.skip("No cached data for VFINX")

        prices = df['Adj Close']
        end_balance = PV_INITIAL_CAPITAL * (prices.iloc[-1] / prices.iloc[0])
        expected = PV_BENCHMARK['expected']['End Balance']

        pct_diff = abs(end_balance / expected - 1.0)
        assert pct_diff < 0.10, \
            f"VFINX end balance ${end_balance:,.0f} should be within 10% " \
            f"of PV ${expected:,.0f}"

    # ================================================================
    # Portfolio 1: VOO 40% / QQQ 20% / GBTC 10% / VGT 30%
    # Requires GBTC data — will skip if not in database.
    # ================================================================

    def test_p1_cagr(self):
        """Portfolio 1 CAGR should be close to PV's 35.02%.

        Wider tolerance due to GBTC data source differences
        (yfinance GBTC vs PV's GBTC data may differ in dividend/split handling).
        """
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_1)
        print_comparison(PV_PORTFOLIO_1['name'], metrics, PV_PORTFOLIO_1['expected'])

        expected = PV_PORTFOLIO_1['expected']['CAGR']
        assert abs(metrics['CAGR'] - expected) < 0.05, \
            f"CAGR {metrics['CAGR']:.2%} should be within 5pp of PV {expected:.2%}"

    def test_p1_end_balance(self):
        """Portfolio 1 end balance should be close to PV's $201,293."""
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_1)

        expected = PV_PORTFOLIO_1['expected']['End Balance']
        pct_diff = abs(metrics['End Balance'] / expected - 1.0)
        assert pct_diff < 0.25, \
            f"End balance ${metrics['End Balance']:,.0f} should be within 25% " \
            f"of PV ${expected:,.0f}"

    # ================================================================
    # Portfolio 3: VOO 40% / QQQ 20% / GBTC 10% / VGT 20% / SPXL 10%
    # Requires GBTC data — will skip if not in database.
    # ================================================================

    def test_p3_cagr(self):
        """Portfolio 3 CAGR should be close to PV's 35.43%."""
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_3)
        print_comparison(PV_PORTFOLIO_3['name'], metrics, PV_PORTFOLIO_3['expected'])

        expected = PV_PORTFOLIO_3['expected']['CAGR']
        assert abs(metrics['CAGR'] - expected) < 0.05, \
            f"CAGR {metrics['CAGR']:.2%} should be within 5pp of PV {expected:.2%}"

    def test_p3_end_balance(self):
        """Portfolio 3 end balance should be close to PV's $207,564."""
        portfolio, metrics = self._run_buy_and_hold(PV_PORTFOLIO_3)

        expected = PV_PORTFOLIO_3['expected']['End Balance']
        pct_diff = abs(metrics['End Balance'] / expected - 1.0)
        assert pct_diff < 0.25, \
            f"End balance ${metrics['End Balance']:,.0f} should be within 25% " \
            f"of PV ${expected:,.0f}"

    # ================================================================
    # Cross-portfolio validation
    # ================================================================

    def test_p2_outperforms_benchmark(self):
        """Portfolio 2 (VOO/QQQ) should outperform pure VFINX benchmark.

        QQQ's outperformance over this period means 70/30 VOO/QQQ > 100% VFINX.
        """
        _, p2_metrics = self._run_buy_and_hold(PV_PORTFOLIO_2)

        df = load_ticker_data(self.dm, 'VFINX', PV_START_DATE, PV_END_DATE)
        if df is None:
            pytest.skip("No cached data for VFINX")
        prices = df['Adj Close']
        bench_cagr = calculate_cagr(prices * (PV_INITIAL_CAPITAL / prices.iloc[0]))

        assert p2_metrics['CAGR'] > bench_cagr, \
            f"P2 CAGR {p2_metrics['CAGR']:.2%} should exceed VFINX {bench_cagr:.2%}"

    def test_portfolio_ranking(self):
        """Return ranking should match PV: P1 and P3 >> P2.

        Only tests portfolios where data is available.
        """
        cagrs = {}
        for label, pv in [('P1', PV_PORTFOLIO_1), ('P2', PV_PORTFOLIO_2),
                          ('P3', PV_PORTFOLIO_3)]:
            try:
                _, metrics = self._run_buy_and_hold(pv)
                cagrs[label] = metrics['CAGR']
            except Exception:
                continue

        if 'P2' in cagrs and 'P1' in cagrs:
            assert cagrs['P2'] < cagrs['P1'], \
                f"P2 ({cagrs['P2']:.2%}) should underperform P1 ({cagrs['P1']:.2%})"

        if 'P2' in cagrs and 'P3' in cagrs:
            assert cagrs['P2'] < cagrs['P3'], \
                f"P2 ({cagrs['P2']:.2%}) should underperform P3 ({cagrs['P3']:.2%})"

    # ================================================================
    # Backtester Integration
    # Verifies our Backtester class produces reasonable results.
    #
    # Note: The Backtester._run_static_backtest uses constant daily weights
    # (equivalent to daily rebalancing), while PV uses true buy-and-hold.
    # For correlated assets like VOO/QQQ, the difference is small.
    # ================================================================

    def _build_backtester_assets(self, pv_config):
        """Build asset dicts and run backtester for a PV portfolio config."""
        assets = []
        for ticker in pv_config['tickers']:
            df = load_ticker_data(self.dm, ticker, PV_START_DATE, PV_END_DATE)
            if df is None:
                pytest.skip(f"No data for {ticker}")
            returns = df['Adj Close'].pct_change().dropna()
            assets.append({
                'ticker': ticker,
                'name': ticker,
                'historical_returns': returns,
                'full_data': df,
                'daily_mean': returns.mean(),
                'daily_std': returns.std(),
            })

        backtester = Backtester(self.dm)
        weights = [pv_config['weights'][t] for t in pv_config['tickers']]

        # Convert date to pd.Timestamp for index comparison compatibility
        start_ts = pd.Timestamp(PV_START_DATE)

        result = backtester.run_backtest(
            assets, weights, PV_INITIAL_CAPITAL,
            start_date_override=start_ts
        )
        return result

    def test_backtester_p2_cagr(self):
        """Backtester CAGR for Portfolio 2 should be in the same ballpark as PV."""
        pv = PV_PORTFOLIO_2
        result = self._build_backtester_assets(pv)

        bt = result['metrics']
        print(f"\n  Backtester (daily-rebalanced) vs PV (buy-and-hold):")
        print(f"    CAGR:    {bt['CAGR']*100:>7.2f}% vs {pv['expected']['CAGR']*100:.2f}% (PV)")
        print(f"    End Bal: ${bt['End Balance']:>9,.0f} vs ${pv['expected']['End Balance']:,.0f} (PV)")
        print(f"    Max DD:  {bt['Max Drawdown']*100:>7.2f}% vs {pv['expected']['Max Drawdown']*100:.2f}% (PV)")
        print(f"    Sharpe:  {bt['Sharpe']:>7.2f}    vs {pv['expected']['Sharpe']:.2f} (PV, uses risk-free)")

        # Daily-rebalanced CAGR should be close to buy-and-hold for correlated assets
        assert abs(bt['CAGR'] - pv['expected']['CAGR']) < 0.03, \
            f"Backtester CAGR {bt['CAGR']:.2%} too far from PV {pv['expected']['CAGR']:.2%}"

    def test_backtester_p2_end_balance(self):
        """Backtester end balance for Portfolio 2 should be reasonable."""
        pv = PV_PORTFOLIO_2
        result = self._build_backtester_assets(pv)

        end_balance = result['metrics']['End Balance']
        expected = pv['expected']['End Balance']
        pct_diff = abs(end_balance / expected - 1.0)

        assert pct_diff < 0.15, \
            f"Backtester end balance ${end_balance:,.0f} should be within 15% " \
            f"of PV ${expected:,.0f}"

    def test_backtester_p2_drawdown_direction(self):
        """Backtester should capture significant drawdowns."""
        pv = PV_PORTFOLIO_2
        result = self._build_backtester_assets(pv)

        max_dd = result['metrics']['Max Drawdown']
        assert max_dd < -0.20, \
            f"Max drawdown {max_dd:.2%} should capture COVID crash (expected < -20%)"
        assert max_dd > -0.45, \
            f"Max drawdown {max_dd:.2%} seems too deep for VOO/QQQ portfolio"

    # ================================================================
    # Comprehensive summary (runs all portfolios, prints comparison table)
    # ================================================================

    def test_full_comparison_summary(self):
        """Print a comprehensive comparison of all available portfolios.

        This test always passes — it's for informational output only.
        The actual assertions are in the individual tests above.
        """
        print(f"\n{'='*65}")
        print(f"  PORTFOLIO VISUALIZER BENCHMARK COMPARISON")
        print(f"  Date range: {PV_START_DATE} to {PV_END_DATE}")
        print(f"  Initial capital: ${PV_INITIAL_CAPITAL:,}")
        print(f"{'='*65}")

        for label, pv in [('Portfolio 2', PV_PORTFOLIO_2),
                          ('Portfolio 1', PV_PORTFOLIO_1),
                          ('Portfolio 3', PV_PORTFOLIO_3)]:
            try:
                portfolio, metrics = self._run_buy_and_hold(pv)
                print_comparison(pv['name'], metrics, pv['expected'])

                if 'annual_returns' in pv:
                    actual_annual = calculate_annual_returns(portfolio)
                    print_annual_comparison(pv['name'], actual_annual, pv['annual_returns'])
            except Exception as e:
                print(f"\n  {label}: SKIPPED ({e})")

        # Benchmark
        try:
            df = load_ticker_data(self.dm, 'VFINX', PV_START_DATE, PV_END_DATE)
            if df is not None:
                prices = df['Adj Close']
                portfolio = prices * (PV_INITIAL_CAPITAL / prices.iloc[0])
                metrics = {
                    'CAGR': calculate_cagr(portfolio),
                    'End Balance': portfolio.iloc[-1],
                    'Stdev': calculate_annualized_std(portfolio),
                    'Max Drawdown': calculate_max_drawdown(portfolio),
                }
                print_comparison(PV_BENCHMARK['name'], metrics, PV_BENCHMARK['expected'])
        except Exception as e:
            print(f"\n  Benchmark: SKIPPED ({e})")


# ============================================================
# Backtester vs Portfolio Visualizer, metric by metric
# (needs stock_data.db with the tickers below; skipped otherwise)
# ============================================================

# PV computes Sharpe/Sortino against actual 1-month T-bill returns; their
# average over Feb 2016 - Jan 2026 was roughly 1.9%/yr.
PV_APPROX_RISK_FREE = 0.019


class TestBacktesterMatchesPV:
    """The report's numbers come from Backtester + quant_analytics; check them against PV."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dm = get_data_manager()
        yield
        self.dm.close()

    def _run(self, pv_config):
        assets = []
        for ticker in pv_config['tickers']:
            df = load_ticker_data(self.dm, ticker, PV_START_DATE - pd.Timedelta(days=7), PV_END_DATE)
            if df is None:
                pytest.skip(f"No data for {ticker}")
            assets.append({'ticker': ticker, 'full_data': df,
                           'historical_returns': df['Adj Close'].pct_change().dropna()})
        weights = [pv_config['weights'][t] for t in pv_config['tickers']]
        # PV's month-to-month period starts from the prior month-end close (Fri 2016-01-29)
        start = pd.Timestamp('2016-01-29')
        return Backtester(self.dm).run_backtest(
            assets, weights, PV_INITIAL_CAPITAL, start_date_override=start,
            end_date=pd.Timestamp(PV_END_DATE), rebalance='none',
            risk_free_rate=PV_APPROX_RISK_FREE)

    @pytest.mark.parametrize('pv', [PV_PORTFOLIO_2, PV_PORTFOLIO_1], ids=['P2', 'P1'])
    def test_headline_metrics(self, pv):
        res = self._run(pv)
        m, e = res['metrics'], pv['expected']
        print(f"\n  {pv['name']}")
        for k in ('CAGR', 'Stdev', 'Sharpe', 'Sortino', 'Max Drawdown', 'Best Year', 'Worst Year'):
            print(f"    {k:<13} ours {m[k]:>8.4f}   PV {e[k]:>8.4f}")
        assert m['CAGR'] == pytest.approx(e['CAGR'], abs=0.005)
        assert m['End Balance'] == pytest.approx(e['End Balance'], rel=0.03)
        assert m['Stdev'] == pytest.approx(e['Stdev'], abs=0.015)
        assert m['Sharpe'] == pytest.approx(e['Sharpe'], abs=0.10)
        assert m['Best Year'] == pytest.approx(e['Best Year'], abs=0.01)
        assert m['Worst Year'] == pytest.approx(e['Worst Year'], abs=0.01)
        # Daily data sees intra-month lows that PV's month-end series cannot
        assert m['Max Drawdown'] <= e['Max Drawdown'] + 0.005

    @pytest.mark.parametrize('pv', [PV_PORTFOLIO_2, PV_PORTFOLIO_1], ids=['P2', 'P1'])
    def test_annual_returns(self, pv):
        annual = self._run(pv)['annual_returns']['return']
        for year, expected in pv['annual_returns'].items():
            assert annual[year] == pytest.approx(expected, abs=0.01), f"{year}"
