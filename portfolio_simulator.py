"""
Monte Carlo portfolio simulation engine.
Generates return scenarios and calculates portfolio statistics.
"""

import numpy as np
from datetime import datetime, timedelta, date

class PortfolioSimulator:
    """Monte Carlo portfolio simulator using real historical data"""

    def __init__(self, data_manager, initial_capital=10000, years=10, simulations=10000):
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        self.years = years
        self.simulations = simulations
        self.trading_days = 252 * years

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=10):
        """Define an asset by analyzing historical data from a ticker"""
        if name is None:
            name = ticker

        end_date = date.today()
        start_date = end_date - timedelta(days=365*lookback_years)
        df = self.data_manager.get_data(ticker, start_date, end_date)

        returns = df['Adj Close'].pct_change().dropna()

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        daily_return = returns.mean()
        daily_volatility = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        data_start = df.index.min().date()
        data_end = df.index.max().date()

        asset = {
            'ticker': ticker,
            'name': name,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'daily_return': daily_return,
            'daily_volatility': daily_volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'historical_returns': returns,
            'data_points': len(returns),
            'start_date': data_start,
            'end_date': data_end
        }

        print(f"\n{name} ({ticker}) Statistics:")
        print(f"  Annual Return: {annual_return*100:.2f}%")
        print(f"  Annual Volatility: {annual_volatility*100:.2f}%")
        print(f"  Sharpe Ratio (0% rf): {annual_return/annual_volatility:.2f}")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Kurtosis: {kurtosis:.2f}")
        print(f"  Data points: {len(returns)} ({data_start} to {data_end})")

        return asset

    def generate_returns(self, asset, method='bootstrap'):
        """Generate simulated returns for an asset"""
        if method == 'bootstrap':
            historical = asset['historical_returns'].values
            if len(historical) < 100:
                print(f"  Warning: {asset['name']} has only {len(historical)} data points. Using parametric method.")
                method = 'parametric'
            else:
                indices = np.random.randint(0, len(historical), (self.simulations, self.trading_days))
                return historical[indices]

        if method == 'parametric':
            return np.random.normal(
                asset['daily_return'],
                asset['daily_volatility'],
                (self.simulations, self.trading_days)
            )
        elif method == 'geometric_brownian':
            dt = 1/252
            drift = (asset['annual_return'] - 0.5 * asset['annual_volatility']**2) * dt
            shock = asset['annual_volatility'] * np.sqrt(dt)
            random_shocks = np.random.normal(0, 1, (self.simulations, self.trading_days))
            return drift + shock * random_shocks

    def simulate_portfolio(self, assets, allocations, method='bootstrap'):
        """Run Monte Carlo simulation for a portfolio"""
        assert len(assets) == len(allocations), "Assets and allocations must match"
        assert abs(sum(allocations) - 1.0) < 0.01, f"Allocations must sum to 1.0"

        all_returns = [self.generate_returns(asset, method) for asset in assets]

        portfolio_returns = np.zeros((self.simulations, self.trading_days))
        for returns, weight in zip(all_returns, allocations):
            portfolio_returns += returns * weight

        portfolio_values = np.zeros((self.simulations, self.trading_days + 1))
        portfolio_values[:, 0] = self.initial_capital

        for day in range(self.trading_days):
            portfolio_values[:, day + 1] = portfolio_values[:, day] * (1 + portfolio_returns[:, day])

        final_values = portfolio_values[:, -1]
        total_returns = (final_values / self.initial_capital) - 1
        max_drawdowns = self.calculate_max_drawdown(portfolio_values)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'final_values': final_values,
            'total_returns': total_returns,
            'cagr': cagr,
            'max_drawdowns': max_drawdowns,
            'assets': assets,
            'allocations': allocations,
            'stats': self.calculate_statistics(final_values, total_returns, cagr, max_drawdowns)
        }

    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown for each simulation path"""
        max_drawdowns = np.zeros(self.simulations)
        for i in range(self.simulations):
            running_max = np.maximum.accumulate(portfolio_values[i, :])
            drawdown = (portfolio_values[i, :] - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)
        return max_drawdowns

    def calculate_statistics(self, final_values, total_returns, cagr, max_drawdowns):
        """Calculate comprehensive statistics"""
        downside_returns = cagr[cagr < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino_ratio = np.mean(cagr) / downside_std

        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'mean_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'mean_cagr': np.mean(cagr),
            'median_cagr': np.median(cagr),
            'std_cagr': np.std(cagr),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_loss': np.mean(final_values < self.initial_capital),
            'probability_double': np.mean(final_values >= 2 * self.initial_capital),
            'sharpe_ratio': np.mean(cagr) / np.std(cagr) if np.std(cagr) > 0 else 0,
            'sortino_ratio': sortino_ratio
        }

    def print_detailed_stats(self, results, label):
        """Print detailed statistics for a portfolio"""
        stats = results['stats']

        print(f"\n{'='*60}")
        print(f"Portfolio: {label}")
        print(f"{'='*60}")
        print(f"\nAllocation:")
        for asset, weight in zip(results['assets'], results['allocations']):
            print(f"  {asset['name']}: {weight*100:.1f}%")

        print(f"\nFinal Value Statistics:")
        print(f"  Mean: ${stats['mean_final_value']:,.2f}")
        print(f"  Median: ${stats['median_final_value']:,.2f}")
        print(f"  5th-95th Percentile: ${stats['percentile_5']:,.2f} - ${stats['percentile_95']:,.2f}")

        print(f"\nReturn Metrics:")
        print(f"  Mean CAGR: {stats['mean_cagr']*100:.2f}%")
        print(f"  Median CAGR: {stats['median_cagr']*100:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Median Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {stats['sortino_ratio']:.3f}")

        print(f"\nProbabilities:")
        print(f"  P(Loss): {stats['probability_loss']*100:.2f}%")
        print(f"  P(Double): {stats['probability_double']*100:.2f}%")

