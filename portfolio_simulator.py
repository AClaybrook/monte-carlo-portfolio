"""
Monte Carlo portfolio simulation engine.
FIXED: Global Time Alignment & Advanced Probability Calculations.
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
        self.years = sim_config.years
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

    def _prepare_multivariate_data(self, assets, start_date_override=None):
        """Aligns data, optionally forcing a specific start date."""
        dfs = []
        for asset in assets:
            s = asset['historical_returns']
            if start_date_override:
                s = s[s.index >= start_date_override]
            s.name = asset['ticker']
            dfs.append(s)

        aligned_df = pd.concat(dfs, axis=1, join='inner').dropna()

        if len(aligned_df) < 20:
             raise ValueError(f"Insufficient common history ({len(aligned_df)} days).")

        return aligned_df

    def simulate_portfolio(self, assets, allocations, method=None, start_date_override=None):
        if method is None: method = self.config.method

        # 1. Get Aligned Data (With Override)
        aligned_returns_df = self._prepare_multivariate_data(assets, start_date_override)

        # 2. Run Simulation (Standard Logic)
        n_assets = len(assets)
        weights = np.array(allocations)

        if method == 'bootstrap':
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

        weighted_returns = asset_returns * weights.reshape(n_assets, 1, 1)
        portfolio_daily_returns = np.sum(weighted_returns, axis=0)

        cumulative_growth = np.cumprod(1 + portfolio_daily_returns, axis=1)
        portfolio_values = np.column_stack([np.ones(self.simulations), cumulative_growth]) * self.initial_capital

        final_values = portfolio_values[:, -1]
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1
        max_drawdowns = self._calculate_max_drawdown_vectorized(portfolio_values)

        # 3. Calculate Advanced Probabilities
        probs = self._calculate_probabilities(portfolio_values)

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': allocations,
            'probabilities': probs, # NEW
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _calculate_probabilities(self, portfolio_values):
        """
        Calculates probability of Loss and probability of >10% return
        at every year mark (Year 1, Year 2 ... Year N).
        """
        years = np.arange(1, self.years + 1)
        # Convert years to indices (approx 252 days per year)
        indices = (years * 252).astype(int)
        indices = np.minimum(indices, portfolio_values.shape[1] - 1)

        prob_loss = []
        prob_high_return = [] # Probability of > 10% CAGR

        for year, idx in zip(years, indices):
            values_at_year = portfolio_values[:, idx]

            # 1. Probability of Loss (< Initial Capital)
            p_loss = np.mean(values_at_year < self.initial_capital)
            prob_loss.append(p_loss)

            # 2. Probability of > 10% CAGR
            # Value needed for 10% CAGR: Initial * (1.10)^Year
            target_value = self.initial_capital * (1.10 ** year)
            p_high = np.mean(values_at_year > target_value)
            prob_high_return.append(p_high)

        return {
            'years': years,
            'prob_loss': np.array(prob_loss),
            'prob_high_return': np.array(prob_high_return)
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
        # Stub for main calling
        pass