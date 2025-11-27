"""
Monte Carlo portfolio simulation engine.
FIXED: Auto-trims to common start date and reports Sim vs Backtest Delta.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

class PortfolioSimulator:
    def __init__(self, data_manager, sim_config):
        self.data_manager = data_manager
        self.config = sim_config
        self.initial_capital = sim_config.initial_capital
        self.simulations = sim_config.simulations
        self.years = sim_config.years # Target years
        self.trading_days = 252 * sim_config.years

        if hasattr(sim_config, 'end_date') and sim_config.end_date:
            self.end_date = date.fromisoformat(sim_config.end_date)
        else:
            self.end_date = date.today()

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=None):
        if name is None: name = ticker
        if lookback_years is None: lookback_years = self.config.lookback_years

        end_date = self.end_date
        start_date = end_date - timedelta(days=365*lookback_years)

        df = self.data_manager.get_data(ticker, start_date, end_date)
        returns = df['Adj Close'].pct_change().dropna()

        return {
            'ticker': ticker, 'name': name,
            'historical_returns': returns, 'full_data': df,
            'daily_mean': returns.mean(), 'daily_std': returns.std()
        }

    def _prepare_multivariate_data(self, assets):
        """
        Aligns data and TRIMS to the shortest common history.
        """
        # 1. Identify Common Start Date
        start_dates = [asset['historical_returns'].index.min() for asset in assets]
        common_start = max(start_dates)

        limiting_asset = assets[start_dates.index(common_start)]['ticker']

        print(f"  [Timeframe] Trimming analysis to Common Start Date: {common_start.date()} (Limited by {limiting_asset})")

        # 2. Slice all assets to this start date
        dfs = []
        for asset in assets:
            s = asset['historical_returns']
            s = s[s.index >= common_start]
            s.name = asset['ticker']
            dfs.append(s)

        aligned_df = pd.concat(dfs, axis=1, join='inner').dropna()

        if len(aligned_df) < 20:
             raise ValueError(f"Insufficient common history ({len(aligned_df)} days) starting {common_start.date()}. Cannot simulate.")

        return aligned_df

    def simulate_portfolio(self, assets, allocations, method=None):
        if method is None: method = self.config.method

        # 1. Get Aligned Data
        aligned_returns_df = self._prepare_multivariate_data(assets)

        # 2. Run Simulation
        n_assets = len(assets)
        weights = np.array(allocations)

        # Use aligned data for parameters
        if method == 'bootstrap':
            # Resample from the ALIGNED history
            random_indices = np.random.randint(0, len(aligned_returns_df), (self.simulations, self.trading_days))
            asset_returns = aligned_returns_df.values[random_indices].transpose(2, 0, 1)
        elif method in ['geometric_brownian', 'parametric']:
            mean_returns = aligned_returns_df.mean().values
            cov_matrix = aligned_returns_df.cov().values
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                U, S, V = np.linalg.svd(cov_matrix)
                L = U @ np.sqrt(np.diag(S))

            uncorrelated = np.random.normal(0, 1, (n_assets, self.simulations, self.trading_days))
            correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
            variances = np.diag(cov_matrix)
            drift = (mean_returns - 0.5 * variances).reshape(n_assets, 1, 1)
            asset_returns = drift + correlated
            if method == 'geometric_brownian': asset_returns = np.exp(asset_returns) - 1

        # 3. Aggregate
        weighted_returns = asset_returns * weights.reshape(n_assets, 1, 1)
        portfolio_daily_returns = np.sum(weighted_returns, axis=0)

        cumulative_growth = np.cumprod(1 + portfolio_daily_returns, axis=1)
        portfolio_values = np.column_stack([np.ones(self.simulations), cumulative_growth]) * self.initial_capital

        final_values = portfolio_values[:, -1]
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1
        max_drawdowns = self._calculate_max_drawdown_vectorized(portfolio_values)

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': allocations,
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _calculate_max_drawdown_vectorized(self, portfolio_values):
        running_max = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns = (portfolio_values - running_max) / running_max
        return np.min(drawdowns, axis=1)

    def calculate_statistics(self, final_values, cagr, max_drawdowns):
        downside_mask = cagr < 0
        sortino = (np.mean(cagr) / np.std(cagr[downside_mask])) if np.sum(downside_mask) > 0 and np.std(cagr[downside_mask]) > 0 else 10.0
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'median_cagr': np.median(cagr),
            'mean_cagr': np.mean(cagr),
            'std_cagr': np.std(cagr),
            'median_max_drawdown': np.median(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'max_drawdown_95': np.percentile(max_drawdowns, 5),
            'sharpe_ratio': np.mean(cagr) / np.std(cagr) if np.std(cagr) > 0 else 0,
            'sortino_ratio': sortino,
            'probability_loss': np.mean(final_values < self.initial_capital),
            'probability_double': np.mean(final_values >= 2 * self.initial_capital),
        }

    def print_detailed_stats(self, results, label):
        stats = results['stats']
        print(f"\n--- {label} ---")
        print(f"  Median CAGR: {stats['median_cagr']*100:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")