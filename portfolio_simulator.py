"""
Monte Carlo portfolio simulation engine.
UPDATED: Enhanced strategy support with rich MarketContext.
UPDATED: Added define_asset_from_dataframe for bulk data loading.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from strategies import StaticAllocationStrategy, MarketContext


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
        self.contrib_freq = getattr(sim_config, 'contribution_frequency', 21)

        if hasattr(sim_config, 'end_date') and sim_config.end_date:
            self.end_date = date.fromisoformat(sim_config.end_date)
        else:
            self.end_date = date.today()

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=None):
        """Original method - fetches data per ticker (can cause rate limiting)."""
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

    def define_asset_from_dataframe(self, ticker, name, df):
        """
        Build an asset dict from a pre-fetched DataFrame.
        Use this with DataManager.get_data_bulk() to avoid per-ticker requests.

        Args:
            ticker: Stock ticker symbol
            name: Display name for the asset
            df: DataFrame with 'Adj Close' column and DatetimeIndex

        Returns:
            Asset dict compatible with simulate_portfolio()
        """
        if df is None or df.empty:
            raise ValueError(f"No data provided for {ticker}")

        # Use 'Adj Close' for returns calculation (matching define_asset_from_ticker)
        if 'Adj Close' in df.columns:
            prices = df['Adj Close']
        elif 'Close' in df.columns:
            prices = df['Close']
        else:
            raise ValueError(f"No price column found for {ticker}")

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # Build asset dict matching define_asset_from_ticker structure
        return {
            'ticker': ticker,
            'name': name,
            'historical_returns': returns,
            'full_data': df,
            'daily_mean': returns.mean(),
            'daily_std': returns.std()
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

        aligned_returns_df = self._prepare_multivariate_data(assets, start_date_override)
        weights = np.array(allocations)

        # Use Time-Step if there is a contribution OR a custom strategy
        use_time_step = (self.contrib_amount > 0) or (strategy is not None)

        if not use_time_step:
            return self._simulate_fast_vectorized(aligned_returns_df, weights, method, assets)
        else:
            if strategy is None:
                strategy = StaticAllocationStrategy()
            return self._simulate_time_stepped(aligned_returns_df, weights, method, assets, strategy)

    def _simulate_fast_vectorized(self, aligned_returns_df, weights, method, assets):
        """Legacy fast method for Lump Sum only."""
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

            if method == 'bootstrap':
                random_indices = np.random.randint(0, len(aligned_returns_df), (current_batch_size, self.trading_days))
                asset_returns = aligned_returns_df.values[random_indices].transpose(2, 0, 1)
            elif method in ['geometric_brownian', 'parametric']:
                uncorrelated = np.random.normal(0, 1, (n_assets, current_batch_size, self.trading_days))
                correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
                batch_returns = drift + correlated
                asset_returns = np.exp(batch_returns) - 1 if method == 'geometric_brownian' else batch_returns

            weighted_returns = np.sum(asset_returns * weights.reshape(n_assets, 1, 1), axis=0)
            del asset_returns
            cumulative_growth = np.cumprod(1 + weighted_returns, axis=1)
            batch_values = np.column_stack([np.ones(current_batch_size), cumulative_growth]) * self.initial_capital

            portfolio_values_list.append(batch_values)
            all_final_values.append(batch_values[:, -1])

            running_max = np.maximum.accumulate(batch_values, axis=1)
            drawdowns = (batch_values - running_max) / running_max
            all_max_drawdowns.append(np.min(drawdowns, axis=1))

        portfolio_values = np.vstack(portfolio_values_list)
        final_values = np.concatenate(all_final_values)
        max_drawdowns = np.concatenate(all_max_drawdowns)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values, 'final_values': final_values,
            'cagr': cagr, 'max_drawdowns': max_drawdowns,
            'assets': assets, 'allocations': list(weights),
            'probabilities': self._calculate_probabilities(portfolio_values),
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _simulate_time_stepped(self, aligned_returns_df, base_weights, method, assets, strategy):
        """
        Enhanced time-stepped engine with rich MarketContext for strategies.
        """
        mean_returns = aligned_returns_df.mean().values
        cov_matrix = aligned_returns_df.cov().values
        n_assets = len(assets)
        asset_tickers = [a['ticker'] for a in assets]

        if method in ['geometric_brownian', 'parametric']:
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                U, S, V = np.linalg.svd(cov_matrix)
                L = U @ np.sqrt(np.diag(S))
            variances = np.diag(cov_matrix)
            drift = (mean_returns - 0.5 * variances).reshape(n_assets, 1)

        BATCH_SIZE = 1000
        total_sims = self.simulations

        all_final_values = []
        all_max_drawdowns = []
        portfolio_values_list = []
        strategy_decisions_log = []  # Track strategy decisions for analysis

        total_invested = self.initial_capital + (self.contrib_amount * (self.trading_days // self.contrib_freq))

        print(f"  > Running Time-Stepped Simulation with Strategy: {strategy.name}")
        print(f"  > Initial: ${self.initial_capital:,.0f} | DCA: ${self.contrib_amount:,.0f} every {self.contrib_freq} days")

        # Rolling window size for indicators
        indicator_window = max(21, getattr(strategy, 'lookback_days', 21))

        for batch_start in range(0, total_sims, BATCH_SIZE):
            current_batch = min(BATCH_SIZE, total_sims - batch_start)

            # Pre-generate ALL random returns
            if method == 'bootstrap':
                random_indices = np.random.randint(0, len(aligned_returns_df), (current_batch, self.trading_days))
                asset_returns_all = aligned_returns_df.values[random_indices].transpose(2, 0, 1)
            elif method in ['geometric_brownian', 'parametric']:
                uncorrelated = np.random.normal(0, 1, (n_assets, current_batch, self.trading_days))
                correlated = np.einsum('ij,jkl->ikl', L, uncorrelated)
                batch_rets = drift.reshape(n_assets, 1, 1) + correlated
                asset_returns_all = np.exp(batch_rets) - 1 if method == 'geometric_brownian' else batch_rets

            # Initialize Batch State
            current_holdings = np.zeros((current_batch, n_assets))
            start_w = np.tile(base_weights, (current_batch, 1))
            current_holdings = start_w * self.initial_capital

            batch_portfolio_values = np.zeros((current_batch, self.trading_days + 1))
            batch_portfolio_values[:, 0] = np.sum(current_holdings, axis=1)

            # Track peaks for drawdown
            asset_peaks = np.copy(current_holdings)
            portfolio_peaks = batch_portfolio_values[:, 0].copy()

            # Rolling return buffer for momentum calculation
            returns_buffer = np.zeros((current_batch, indicator_window, n_assets))

            # Time-Step Loop
            for day in range(self.trading_days):
                # Get returns for this day
                todays_returns = asset_returns_all[:, :, day].T  # (Batch, Assets)

                # Apply returns
                current_holdings = current_holdings * (1 + todays_returns)

                # Update portfolio value
                total_val = np.sum(current_holdings, axis=1)
                batch_portfolio_values[:, day + 1] = total_val

                # Update peaks
                asset_peaks = np.maximum(asset_peaks, current_holdings)
                portfolio_peaks = np.maximum(portfolio_peaks, total_val)

                # Update returns buffer (circular)
                buffer_idx = day % indicator_window
                returns_buffer[:, buffer_idx, :] = todays_returns

                # DCA Contribution Day
                if self.contrib_amount > 0 and (day + 1) % self.contrib_freq == 0:
                    # Calculate drawdowns
                    asset_drawdowns = (current_holdings - asset_peaks) / np.where(asset_peaks != 0, asset_peaks, 1)
                    portfolio_drawdown = (total_val - portfolio_peaks) / np.where(portfolio_peaks != 0, portfolio_peaks, 1)

                    # Calculate rolling indicators
                    if day >= indicator_window:
                        rolling_returns = np.mean(returns_buffer, axis=1) * 252  # Annualized
                        rolling_volatility = np.std(returns_buffer, axis=1) * np.sqrt(252)
                        momentum_score = np.sum(returns_buffer, axis=1)  # Cumulative return over window
                        rolling_sharpe = rolling_returns / (rolling_volatility + 1e-6)
                    else:
                        rolling_returns = None
                        rolling_volatility = None
                        momentum_score = None
                        rolling_sharpe = None

                    # Build MarketContext
                    context = MarketContext(
                        current_holdings=current_holdings,
                        current_drawdowns=asset_drawdowns,
                        base_allocations=base_weights,
                        asset_tickers=asset_tickers,
                        current_day=day,
                        total_days=self.trading_days,
                        rolling_returns=rolling_returns,
                        rolling_volatility=rolling_volatility,
                        rolling_sharpe=rolling_sharpe,
                        momentum_score=momentum_score,
                        portfolio_drawdown=portfolio_drawdown
                    )

                    # Get allocation from strategy
                    new_money_weights = strategy.get_allocation(context)

                    # Add contribution
                    cash_injection = new_money_weights * self.contrib_amount
                    current_holdings += cash_injection

            # Store batch results
            portfolio_values_list.append(batch_portfolio_values)
            all_final_values.append(batch_portfolio_values[:, -1])

            running_max = np.maximum.accumulate(batch_portfolio_values, axis=1)
            drawdowns = (batch_portfolio_values - running_max) / running_max
            all_max_drawdowns.append(np.min(drawdowns, axis=1))

            del asset_returns_all

        # Consolidate
        portfolio_values = np.vstack(portfolio_values_list)
        final_values = np.concatenate(all_final_values)
        max_drawdowns = np.concatenate(all_max_drawdowns)
        cagr = (final_values / total_invested) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values,
            'final_values': final_values,
            'cagr': cagr,
            'max_drawdowns': max_drawdowns,
            'assets': assets,
            'allocations': list(base_weights),
            'strategy': strategy.name,
            'strategy_config': strategy.get_config_summary(),
            'probabilities': self._calculate_probabilities(portfolio_values),
            'stats': self.calculate_statistics(final_values, cagr, max_drawdowns)
        }

    def _calculate_probabilities(self, portfolio_values):
        years = np.arange(1, self.years + 1)
        indices = (years * 252).astype(int)
        indices = np.minimum(indices, portfolio_values.shape[1] - 1)
        prob_loss = []
        prob_high_return = []

        for year, idx in zip(years, indices):
            values_at_year = portfolio_values[:, idx]
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
        downside_mask = cagr < 0
        sortino = (np.mean(cagr) / np.std(cagr[downside_mask])) if np.sum(downside_mask) > 0 and np.std(cagr[downside_mask]) > 0 else 10.0

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