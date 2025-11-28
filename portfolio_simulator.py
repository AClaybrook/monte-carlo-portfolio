"""
Monte Carlo portfolio simulation engine.
UPDATED: Supports DCA and Dynamic Allocation Strategies via Time-Stepping.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from strategies import StaticAllocationStrategy

class PortfolioSimulator:
    def __init__(self, data_manager, sim_config):
        self.data_manager = data_manager
        self.config = sim_config
        self.initial_capital = sim_config.initial_capital
        self.simulations = sim_config.simulations
        self.years = sim_config.years
        self.trading_days = 252 * sim_config.years

        # DCA Config
        self.contrib_amount = getattr(sim_config, 'contribution_amount', 0.0)
        self.contrib_freq = getattr(sim_config, 'contribution_frequency', 21) # Default monthly

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

    def simulate_portfolio(self, assets, allocations, method=None, start_date_override=None, strategy=None):
        """
        Main entry point. Switches between fast Vectorized (Lump Sum)
        and Time-Stepped (DCA/Dynamic) automatically.
        """
        if method is None: method = self.config.method

        # 1. Prepare Data
        aligned_returns_df = self._prepare_multivariate_data(assets, start_date_override)
        weights = np.array(allocations)
        n_assets = len(assets)

        # 2. Check if we need the complex Time-Step engine
        # We use Time-Step if there is a contribution OR a custom strategy
        use_time_step = (self.contrib_amount > 0) or (strategy is not None)

        if not use_time_step:
            # Fallback to the Fast Vectorized engine (Lump Sum only)
            # This is 100x faster for simple cases
            return self._simulate_fast_vectorized(aligned_returns_df, weights, method, assets)
        else:
            # Use the Time-Step engine (DCA / Dynamic)
            if strategy is None:
                strategy = StaticAllocationStrategy()
            return self._simulate_time_stepped(aligned_returns_df, weights, method, assets, strategy)

    def _simulate_fast_vectorized(self, aligned_returns_df, weights, method, assets):
        """
        Legacy fast method for Lump Sum only.
        """
        # ... (Same setup as your original code for Drift/Cholesky) ...
        mean_returns = aligned_returns_df.mean().values
        cov_matrix = aligned_returns_df.cov().values
        n_assets = len(assets)

        if method in ['geometric_brownian', 'parametric']:
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                U, S, V = np.linalg.svd(cov_matrix)
                L = U @ np.sqrt(np.diag(S))
            variances = np.diag(cov_matrix)
            drift = (mean_returns - 0.5 * variances).reshape(n_assets, 1, 1)

        BATCH_SIZE = 1000
        total_sims = self.simulations
        portfolio_values_list = []
        all_final_values = []
        all_max_drawdowns = []

        for i in range(0, total_sims, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, total_sims - i)

            # Generate Returns
            if method == 'bootstrap':
                random_indices = np.random.randint(0, len(aligned_returns_df), (current_batch_size, self.trading_days))
                asset_returns = aligned_returns_df.values[random_indices].transpose(2, 0, 1)
            elif method in ['geometric_brownian', 'parametric']:
                uncorrelated = np.random.normal(0, 1, (n_assets, current_batch_size, self.trading_days))
                correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
                batch_returns = drift + correlated
                asset_returns = np.exp(batch_returns) - 1 if method == 'geometric_brownian' else batch_returns

            # Calculate Value
            weighted_returns = np.sum(asset_returns * weights.reshape(n_assets, 1, 1), axis=0)
            del asset_returns
            cumulative_growth = np.cumprod(1 + weighted_returns, axis=1)
            batch_values = np.column_stack([np.ones(current_batch_size), cumulative_growth]) * self.initial_capital

            portfolio_values_list.append(batch_values)
            all_final_values.append(batch_values[:, -1])

            running_max = np.maximum.accumulate(batch_values, axis=1)
            drawdowns = (batch_values - running_max) / running_max
            all_max_drawdowns.append(np.min(drawdowns, axis=1))

        # Consolidate
        portfolio_values = np.vstack(portfolio_values_list)
        final_values = np.concatenate(all_final_values)
        max_drawdowns = np.concatenate(all_max_drawdowns)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': list(weights), # return as list
            'probabilities': self._calculate_probabilities(portfolio_values),
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _simulate_time_stepped(self, aligned_returns_df, base_weights, method, assets, strategy):
        """
        Slower but flexible engine: Loops through Days to allow DCA and Logic.
        """
        mean_returns = aligned_returns_df.mean().values
        cov_matrix = aligned_returns_df.cov().values
        n_assets = len(assets)

        # Setup Drift/Cholesky
        if method in ['geometric_brownian', 'parametric']:
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                U, S, V = np.linalg.svd(cov_matrix)
                L = U @ np.sqrt(np.diag(S))
            variances = np.diag(cov_matrix)
            drift = (mean_returns - 0.5 * variances).reshape(n_assets, 1) # Note shape change

        BATCH_SIZE = 1000  # Smaller batch size might be needed if memory is tight
        total_sims = self.simulations

        all_final_values = []
        all_max_drawdowns = []
        portfolio_values_list = [] # Store full paths for visualization

        # Calculate Total Invested Capital (Initial + All DCA) for CAGR Calc
        total_invested = self.initial_capital + (self.contrib_amount * (self.trading_days // self.contrib_freq))

        print(f"  > Running Time-Stepped Simulation (DCA/Dynamic)...")
        print(f"  > Initial: ${self.initial_capital:,.0f} | DCA: ${self.contrib_amount:,.0f} every {self.contrib_freq} days")

        for i in range(0, total_sims, BATCH_SIZE):
            current_batch = min(BATCH_SIZE, total_sims - i)

            # --- 1. Pre-generate ALL random returns for this batch ---
            # We do this outside the day loop for speed.
            # Shape: (Assets, Batch, Days)
            if method == 'bootstrap':
                random_indices = np.random.randint(0, len(aligned_returns_df), (current_batch, self.trading_days))
                # (Batch, Days, Assets) -> (Assets, Batch, Days)
                asset_returns_all = aligned_returns_df.values[random_indices].transpose(2, 0, 1)

            elif method in ['geometric_brownian', 'parametric']:
                uncorrelated = np.random.normal(0, 1, (n_assets, current_batch, self.trading_days))
                correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
                # Add drift (Assets, 1) broadcast to (Assets, Batch, Days)
                batch_rets = drift.reshape(n_assets, 1, 1) + correlated
                if method == 'geometric_brownian':
                    asset_returns_all = np.exp(batch_rets) - 1
                else:
                    asset_returns_all = batch_rets

            # --- 2. Initialize Batch State ---
            # Holdings: (Batch, Assets) - Value in dollars per asset
            current_holdings = np.zeros((current_batch, n_assets))

            # Distribute Initial Capital
            # base_weights is (Assets,) -> (Batch, Assets)
            start_w = np.tile(base_weights, (current_batch, 1))
            current_holdings = start_w * self.initial_capital

            # Trackers
            batch_portfolio_values = np.zeros((current_batch, self.trading_days + 1))
            batch_portfolio_values[:, 0] = np.sum(current_holdings, axis=1)

            # Track Asset Peaks for Drawdown Logic (Batch, Assets)
            asset_peaks = np.copy(current_holdings)

            # --- 3. Time-Step Loop ---
            for day in range(self.trading_days):
                # a. Get returns for this specific day: (Assets, Batch)
                todays_returns = asset_returns_all[:, :, day].T # -> (Batch, Assets)

                # b. Apply returns to holdings
                current_holdings = current_holdings * (1 + todays_returns)

                # c. Update Portfolio Value History
                total_val = np.sum(current_holdings, axis=1)
                batch_portfolio_values[:, day + 1] = total_val

                # d. Check for DCA
                if self.contrib_amount > 0 and (day + 1) % self.contrib_freq == 0:

                    # Update Peaks for logic (only needed on contribution days for efficiency)
                    # Note: This approximates peak tracking to contribution days to save CPU.
                    # If you need daily precise drawdown logic, move this out of the 'if'.
                    asset_peaks = np.maximum(asset_peaks, current_holdings)

                    # Calculate Asset Drawdowns: (Batch, Assets)
                    # Avoid divide by zero
                    asset_drawdowns = (current_holdings - asset_peaks) / np.where(asset_peaks!=0, asset_peaks, 1)

                    # e. ASK STRATEGY for allocation weights
                    # Returns (Batch, Assets)
                    new_money_weights = strategy.get_allocation(
                        current_holdings,
                        asset_drawdowns,
                        base_weights
                    )

                    # f. Add Cash
                    # Contribution (Batch, Assets)
                    cash_injection = new_money_weights * self.contrib_amount
                    current_holdings += cash_injection

            # --- 4. Store Batch Results ---
            portfolio_values_list.append(batch_portfolio_values)
            all_final_values.append(batch_portfolio_values[:, -1])

            # Calc max drawdown for this batch
            running_max = np.maximum.accumulate(batch_portfolio_values, axis=1)
            drawdowns = (batch_portfolio_values - running_max) / running_max
            all_max_drawdowns.append(np.min(drawdowns, axis=1))

            # Cleanup heavy array
            del asset_returns_all

        # Consolidate
        portfolio_values = np.vstack(portfolio_values_list)
        final_values = np.concatenate(all_final_values)
        max_drawdowns = np.concatenate(all_max_drawdowns)

        # CAGR calc is tricky with DCA. We use IRR logic or simple approximation.
        # Simple Approx: (Final / Total_Invested)^(1/years) - 1
        # Note: This penalizes DCA performance visually compared to Lump Sum because
        # later money hasn't had time to grow.
        cagr = (final_values / total_invested) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': list(base_weights),
            'probabilities': self._calculate_probabilities(portfolio_values),
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _calculate_probabilities(self, portfolio_values):
        # (Same as before)
        years = np.arange(1, self.years + 1)
        indices = (years * 252).astype(int)
        indices = np.minimum(indices, portfolio_values.shape[1] - 1)
        prob_loss = []
        prob_high_return = []

        # Adjusted for DCA: Probability of having less than Total Invested so far
        for year, idx in zip(years, indices):
            values_at_year = portfolio_values[:, idx]

            # How much have we put in by this year?
            invested_at_year = self.initial_capital + (self.contrib_amount * (idx // self.contrib_freq))

            p_loss = np.mean(values_at_year < invested_at_year)
            prob_loss.append(p_loss)

            target_value = invested_at_year * (1.10 ** year)
            p_high = np.mean(values_at_year > target_value)
            prob_high_return.append(p_high)

        return {
            'years': years,
            'prob_loss': np.array(prob_loss),
            'prob_high_return': np.array(prob_high_return)
        }

    def calculate_statistics(self, final_values, cagr, max_drawdowns):
        # (Same as before)
        downside_mask = cagr < 0
        sortino = (np.mean(cagr) / np.std(cagr[downside_mask])) if np.sum(downside_mask) > 0 and np.std(cagr[downside_mask]) > 0 else 10.0

        # Adjusted basis for probability loss
        total_invested = self.initial_capital + (self.contrib_amount * (self.trading_days // self.contrib_freq))

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
            'probability_loss': np.mean(final_values < total_invested),
            'probability_double': np.mean(final_values >= 2 * total_invested),
        }