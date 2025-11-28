"""
Monte Carlo portfolio simulation engine.
FIXED: Memory Optimization (Batch Processing) to prevent crashes.
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
        """
        Runs simulation in batches to save RAM.
        """
        if method is None: method = self.config.method

        # 1. Get Aligned Data (With Override)
        aligned_returns_df = self._prepare_multivariate_data(assets, start_date_override)
        n_assets = len(assets)
        weights = np.array(allocations)

        # Pre-calculate params for GBM method if needed
        if method in ['geometric_brownian', 'parametric']:
            mean_returns = aligned_returns_df.mean().values
            cov_matrix = aligned_returns_df.cov().values
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                U, S, V = np.linalg.svd(cov_matrix)
                L = U @ np.sqrt(np.diag(S))
            variances = np.diag(cov_matrix)
            drift = (mean_returns - 0.5 * variances).reshape(n_assets, 1, 1)

        # 2. Run Simulation in Batches
        # We process e.g., 1000 simulations at a time to keep memory usage low.
        BATCH_SIZE = 1000
        all_final_values = []
        all_max_drawdowns = []

        # We need to store paths for the visualizer, but storing ALL paths for ALL sims
        # is what kills memory. We will collect them, but if memory is still tight,
        # we might need to only store percentiles. For now, we construct the full array
        # incrementally.
        portfolio_values_list = []

        total_sims = self.simulations

        for i in range(0, total_sims, BATCH_SIZE):
            # Determine size of this batch (last batch might be smaller)
            current_batch_size = min(BATCH_SIZE, total_sims - i)

            # --- A. Generate Asset Returns for Batch ---
            if method == 'bootstrap':
                # Shape: (Batch, Days)
                random_indices = np.random.randint(0, len(aligned_returns_df), (current_batch_size, self.trading_days))
                # Shape: (Assets, Batch, Days) <--- This is the heavy array we split up
                asset_returns = aligned_returns_df.values[random_indices].transpose(2, 0, 1)

            elif method in ['geometric_brownian', 'parametric']:
                uncorrelated = np.random.normal(0, 1, (n_assets, current_batch_size, self.trading_days))
                correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
                # Re-broadcast drift for current batch size
                curr_drift = drift
                # If drift shape is (n, 1, 1), it broadcasts automatically.

                batch_returns = curr_drift + correlated
                if method == 'geometric_brownian':
                    asset_returns = np.exp(batch_returns) - 1
                else:
                    asset_returns = batch_returns

            # --- B. Calculate Portfolio Value for Batch ---
            # Weighted returns: (Batch, Days)
            weighted_returns = np.sum(asset_returns * weights.reshape(n_assets, 1, 1), axis=0)

            # Clean up heavy asset_returns immediately
            del asset_returns

            # Cumulative Growth
            cumulative_growth = np.cumprod(1 + weighted_returns, axis=1)

            # Convert to Dollars
            # Shape: (Batch, Days + 1)
            batch_values = np.column_stack([np.ones(current_batch_size), cumulative_growth]) * self.initial_capital

            # --- C. Extract Stats for Batch ---
            # Store full paths
            portfolio_values_list.append(batch_values)

            # Store finals
            all_final_values.append(batch_values[:, -1])

            # Store drawdowns
            running_max = np.maximum.accumulate(batch_values, axis=1)
            drawdowns = (batch_values - running_max) / running_max
            all_max_drawdowns.append(np.min(drawdowns, axis=1))

        # 3. Consolidate Results
        portfolio_values = np.vstack(portfolio_values_list)
        final_values = np.concatenate(all_final_values)
        max_drawdowns = np.concatenate(all_max_drawdowns)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        # 4. Calculate Advanced Probabilities (Vectorized on full result)
        probs = self._calculate_probabilities(portfolio_values)

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': allocations,
            'probabilities': probs,
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