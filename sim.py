import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta, date
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Float, Date, Integer, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
import time

Base = declarative_base()

class StockPrice(Base):
    """
    Database model for storing stock prices

    Note: Using a single table for all tickers is efficient because:
    1. Simpler queries - no need to dynamically create/query multiple tables
    2. Better indexing - composite index on (ticker, date) is very efficient
    3. Easier maintenance - one schema to manage
    4. Standard practice - most financial databases use this approach
    """
    __tablename__ = 'stock_prices'

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

    # Ensure we don't have duplicate ticker-date combinations
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='unique_ticker_date'),
    )

class DataManager:
    """Manages downloading and caching of stock data"""

    def __init__(self, db_path='stock_data.db'):
        """
        Initialize the data manager with SQLite database

        Parameters:
        -----------
        db_path : str
            Path to SQLite database file
        """
        self.db_path = db_path

        # Create engine with check_same_thread=False for SQLite
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_data(self, ticker, start_date=None, end_date=None, force_update=False):
        """
        Get stock data from database or download from Yahoo Finance

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str, date, or datetime
            Start date for data (default: 10 years ago)
        end_date : str, date, or datetime
            End date for data (default: today)
        force_update : bool
            If True, force download even if data exists

        Returns:
        --------
        pandas.DataFrame : Stock price data with DatetimeIndex
        """
        # Normalize dates to date objects
        if start_date is None:
            start_date = date.today() - timedelta(days=365*10)
        else:
            start_date = self._normalize_date(start_date)

        if end_date is None:
            end_date = date.today()
        else:
            end_date = self._normalize_date(end_date)

        # Check if data exists in database
        if not force_update:
            db_data = self._get_from_db(ticker, start_date, end_date)

            if db_data is not None and len(db_data) > 0:
                # Check if we have all the data we need
                db_start = db_data.index.min().date()
                db_end = db_data.index.max().date()

                # If database has sufficient data, return it
                # Allow 5 day buffer for recent data (weekends/holidays)
                if db_start <= start_date and db_end >= end_date - timedelta(days=5):
                    print(f"✓ Retrieved {ticker} from database ({len(db_data)} records)")
                    return db_data
                else:
                    print(f"⚠ Partial data for {ticker} in database, updating...")

        # Download from Yahoo Finance with retry logic
        print(f"↓ Downloading {ticker} from Yahoo Finance...")
        try:
            df = self._download_with_retry(ticker, start_date, end_date)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Save to database
            self._save_to_db(ticker, df)

            print(f"✓ Downloaded and saved {ticker} ({len(df)} records)")
            return df

        except Exception as e:
            print(f"✗ Error downloading {ticker}: {str(e)}")
            # Try to return whatever we have in the database
            db_data = self._get_from_db(ticker, start_date, end_date)
            if db_data is not None and len(db_data) > 0:
                print(f"⚠ Using existing database data for {ticker}")
                return db_data
            raise

    def _download_with_retry(self, ticker, start_date, end_date, max_retries=3):
        """
        Download data with retry logic and rate limit handling

        This helps with WSL rate limiting issues by:
        1. Adding delays between retries
        2. Using longer timeout
        3. Handling rate limit errors gracefully
        """
        for attempt in range(max_retries):
            try:
                # Add delay before retry attempts
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

                # Download with explicit timeout
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=False  # Keep both Close and Adj Close
                )

                if not df.empty:
                    return df

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {str(e)}")
                else:
                    raise

        raise ValueError(f"Failed to download {ticker} after {max_retries} attempts")

    def _normalize_date(self, date_input):
        """Convert various date formats to date object"""
        if isinstance(date_input, date):
            return date_input
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, str):
            return pd.to_datetime(date_input).date()
        elif isinstance(date_input, pd.Timestamp):
            return date_input.date()
        else:
            raise ValueError(f"Unsupported date type: {type(date_input)}")

    def _get_from_db(self, ticker, start_date, end_date):
        """Retrieve data from database"""
        try:
            query = self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date)

            results = query.all()

            if not results:
                return None

            # Convert to DataFrame with proper column names matching yfinance
            data = {
                'Open': [r.open for r in results],
                'High': [r.high for r in results],
                'Low': [r.low for r in results],
                'Close': [r.close for r in results],
                'Adj Close': [r.adj_close for r in results],
                'Volume': [r.volume for r in results],
            }

            # Create DatetimeIndex for consistency with yfinance
            dates = pd.DatetimeIndex([r.date for r in results])
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Date'

            return df

        except Exception as e:
            print(f"Error reading from database: {str(e)}")
            return None

    def _save_to_db(self, ticker, df):
        """Save data to database"""
        try:
            # Get date range as date objects
            dates = [self._normalize_date(d) for d in df.index]
            date_min = min(dates)
            date_max = max(dates)

            # Delete existing data for this ticker in the date range
            # Use merge to avoid duplicates
            self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= date_min,
                StockPrice.date <= date_max
            ).delete()

            # Insert new data
            records = []
            for dt, row in df.iterrows():
                record = StockPrice(
                    ticker=ticker,
                    date=self._normalize_date(dt),
                    open=float(row['Open']) if pd.notna(row['Open']) else None,
                    high=float(row['High']) if pd.notna(row['High']) else None,
                    low=float(row['Low']) if pd.notna(row['Low']) else None,
                    close=float(row['Close']) if pd.notna(row['Close']) else None,
                    adj_close=float(row['Adj Close']) if pd.notna(row['Adj Close']) else None,
                    volume=float(row['Volume']) if pd.notna(row['Volume']) else None,
                )
                records.append(record)

            # Bulk insert for better performance
            self.session.bulk_save_objects(records)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f"Error saving to database: {str(e)}")
            raise

    def get_ticker_info(self, ticker):
        """Get information about stored data for a ticker"""
        try:
            query = self.session.query(
                StockPrice.date
            ).filter(
                StockPrice.ticker == ticker
            ).order_by(StockPrice.date)

            dates = [r.date for r in query.all()]

            if not dates:
                return None

            return {
                'ticker': ticker,
                'start_date': min(dates),
                'end_date': max(dates),
                'record_count': len(dates)
            }
        except Exception as e:
            print(f"Error getting ticker info: {str(e)}")
            return None

    def list_all_tickers(self):
        """List all tickers stored in database"""
        try:
            results = self.session.query(StockPrice.ticker).distinct().all()
            return [r.ticker for r in results]
        except Exception as e:
            print(f"Error listing tickers: {str(e)}")
            return []

    def close(self):
        """Close database connection"""
        self.session.close()

class PortfolioSimulator:
    """
    Monte Carlo portfolio simulator using real historical data
    """

    def __init__(self, initial_capital=10000, years=10, simulations=10000):
        """
        Initialize the simulator

        Parameters:
        -----------
        initial_capital : float
            Starting portfolio value
        years : int
            Investment time horizon in years
        simulations : int
            Number of Monte Carlo simulations to run
        """
        self.initial_capital = initial_capital
        self.years = years
        self.simulations = simulations
        self.trading_days = 252 * years
        self.data_manager = DataManager()

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=10):
        """
        Define an asset by analyzing historical data from a ticker

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        name : str
            Display name (defaults to ticker)
        lookback_years : int
            Years of historical data to analyze

        Returns:
        --------
        dict : Asset definition with calculated statistics
        """
        if name is None:
            name = ticker

        # Get historical data using date objects
        end_date = date.today()
        start_date = end_date - timedelta(days=365*lookback_years)

        df = self.data_manager.get_data(ticker, start_date, end_date)

        # Calculate daily returns using Adj Close
        returns = df['Adj Close'].pct_change().dropna()

        # Calculate statistics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        daily_return = returns.mean()
        daily_volatility = returns.std()

        # Calculate higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Get actual date range as date objects
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
        """
        Generate simulated returns for an asset

        Parameters:
        -----------
        asset : dict
            Asset definition from define_asset_from_ticker()
        method : str
            'bootstrap' (resample historical), 'parametric' (normal dist), or 'geometric_brownian'

        Returns:
        --------
        numpy.ndarray : shape (simulations, trading_days)
        """
        if method == 'bootstrap':
            # Bootstrap from historical returns (preserves all characteristics)
            historical = asset['historical_returns'].values

            # Check if we have enough historical data
            if len(historical) < 100:
                print(f"  Warning: {asset['name']} has only {len(historical)} data points. Using parametric method instead.")
                method = 'parametric'
            else:
                # Resample with replacement from historical returns
                indices = np.random.randint(0, len(historical), (self.simulations, self.trading_days))
                daily_returns = historical[indices]
                return daily_returns

        if method == 'parametric':
            # Use fitted normal distribution
            daily_returns = np.random.normal(
                asset['daily_return'],
                asset['daily_volatility'],
                (self.simulations, self.trading_days)
            )

        elif method == 'geometric_brownian':
            # Geometric Brownian Motion
            dt = 1/252
            drift = (asset['annual_return'] - 0.5 * asset['annual_volatility']**2) * dt
            shock = asset['annual_volatility'] * np.sqrt(dt)

            random_shocks = np.random.normal(0, 1, (self.simulations, self.trading_days))
            daily_returns = drift + shock * random_shocks

        return daily_returns

    def simulate_portfolio(self, assets, allocations, method='bootstrap'):
        """
        Run Monte Carlo simulation for a portfolio

        Parameters:
        -----------
        assets : list of dict
            List of asset definitions
        allocations : list of float
            Allocation weights (must sum to 1.0)
        method : str
            Return generation method

        Returns:
        --------
        dict : Simulation results and statistics
        """
        assert len(assets) == len(allocations), "Assets and allocations must match"
        assert abs(sum(allocations) - 1.0) < 0.01, f"Allocations must sum to 1.0 (got {sum(allocations)})"

        # Generate returns for each asset
        all_returns = []
        for asset in assets:
            asset_returns = self.generate_returns(asset, method)
            all_returns.append(asset_returns)

        # Combine returns based on allocations
        portfolio_returns = np.zeros((self.simulations, self.trading_days))
        for returns, weight in zip(all_returns, allocations):
            portfolio_returns += returns * weight

        # Calculate portfolio value over time
        portfolio_values = np.zeros((self.simulations, self.trading_days + 1))
        portfolio_values[:, 0] = self.initial_capital

        for day in range(self.trading_days):
            portfolio_values[:, day + 1] = portfolio_values[:, day] * (1 + portfolio_returns[:, day])

        # Calculate statistics
        final_values = portfolio_values[:, -1]
        total_returns = (final_values / self.initial_capital) - 1

        # Calculate maximum drawdown for each simulation
        max_drawdowns = self.calculate_max_drawdown(portfolio_values)

        # CAGR (Compound Annual Growth Rate)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        results = {
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

        return results

    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown for each simulation path"""
        max_drawdowns = np.zeros(self.simulations)

        for i in range(self.simulations):
            running_max = np.maximum.accumulate(portfolio_values[i, :])
            drawdown = (portfolio_values[i, :] - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)

        return max_drawdowns

    def calculate_statistics(self, final_values, total_returns, cagr, max_drawdowns):
        """Calculate comprehensive statistics for the simulation"""
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
            'sortino_ratio': self.calculate_sortino(cagr)
        }

    def calculate_sortino(self, returns, target_return=0):
        """Calculate Sortino ratio (accounts for downside risk only)"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                return (np.mean(returns) - target_return) / downside_std
        return 0

    def plot_results(self, portfolio_configs, figsize=(18, 12)):
        """
        Create comprehensive visualization of simulation results

        Parameters:
        -----------
        portfolio_configs : list of dict
            List of portfolio configurations with 'results' and 'label' keys
        """
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Monte Carlo Portfolio Simulation Analysis (Real Historical Data)',
                     fontsize=16, fontweight='bold')

        colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

        # 1. Portfolio value trajectories (sample paths)
        ax = axes[0, 0]
        for results, label, color in zip(results_list, labels, colors):
            sample_indices = np.random.choice(self.simulations, 100, replace=False)
            for idx in sample_indices:
                ax.plot(results['portfolio_values'][idx, :], alpha=0.1, color=color)
            # Plot median path
            median_path = np.median(results['portfolio_values'], axis=0)
            ax.plot(median_path, color=color, linewidth=2, label=label)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Sample Portfolio Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Final value distribution
        ax = axes[0, 1]
        for results, label, color in zip(results_list, labels, colors):
            ax.hist(results['final_values'], bins=50, alpha=0.5, label=label, color=color)
        ax.axvline(self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        ax.set_xlabel('Final Portfolio Value ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. CAGR distribution
        ax = axes[0, 2]
        for results, label, color in zip(results_list, labels, colors):
            ax.hist(results['cagr'] * 100, bins=50, alpha=0.5, label=label, color=color)
        ax.set_xlabel('CAGR (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of CAGR')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Box plot of returns
        ax = axes[1, 0]
        data_to_plot = [results['total_returns'] * 100 for results in results_list]
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Return Distribution Comparison')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Maximum drawdown distribution
        ax = axes[1, 1]
        for results, label, color in zip(results_list, labels, colors):
            ax.hist(results['max_drawdowns'] * 100, bins=50, alpha=0.5, label=label, color=color)
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Maximum Drawdown Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Percentile fan chart
        ax = axes[1, 2]
        for results, label, color in zip(results_list, labels, colors):
            values = results['portfolio_values']
            p5 = np.percentile(values, 5, axis=0)
            p25 = np.percentile(values, 25, axis=0)
            p50 = np.percentile(values, 50, axis=0)
            p75 = np.percentile(values, 75, axis=0)
            p95 = np.percentile(values, 95, axis=0)

            days = np.arange(values.shape[1])
            ax.fill_between(days, p5, p95, alpha=0.2, color=color)
            ax.fill_between(days, p25, p75, alpha=0.3, color=color)
            ax.plot(days, p50, color=color, linewidth=2, label=label)

        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Percentile Fan Chart (5-95, 25-75)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 7. Risk-Return scatter
        ax = axes[2, 0]
        for results, label, color in zip(results_list, labels, colors):
            stats = results['stats']
            ax.scatter(stats['std_cagr'] * 100, stats['mean_cagr'] * 100,
                      s=200, alpha=0.6, color=color, label=label)
            ax.annotate(label, (stats['std_cagr'] * 100, stats['mean_cagr'] * 100),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xlabel('CAGR Volatility (%)')
        ax.set_ylabel('Mean CAGR (%)')
        ax.set_title('Risk-Return Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Statistics comparison table
        ax = axes[2, 1]
        ax.axis('tight')
        ax.axis('off')

        stats_data = []
        for results, label in zip(results_list, labels):
            stats = results['stats']
            stats_data.append([
                label,
                f"${stats['median_final_value']:,.0f}",
                f"{stats['median_cagr']*100:.1f}%",
                f"{stats['median_max_drawdown']*100:.1f}%",
                f"{stats['sharpe_ratio']:.2f}"
            ])

        table = ax.table(cellText=stats_data,
                        colLabels=['Portfolio', 'Median Value', 'Median CAGR', 'Median DD', 'Sharpe'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax.set_title('Key Statistics Comparison')

        # 9. Probability of outcomes
        ax = axes[2, 2]
        width = 0.35
        x = np.arange(len(labels))

        prob_loss = [results['stats']['probability_loss'] * 100 for results in results_list]
        prob_double = [results['stats']['probability_double'] * 100 for results in results_list]

        ax.bar(x - width/2, prob_loss, width, label='P(Loss)', alpha=0.7)
        ax.bar(x + width/2, prob_double, width, label='P(Double)', alpha=0.7)

        ax.set_ylabel('Probability (%)')
        ax.set_title('Probability of Outcomes')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

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
        print(f"  Std Dev: ${stats['std_final_value']:,.2f}")
        print(f"  5th Percentile: ${stats['percentile_5']:,.2f}")
        print(f"  95th Percentile: ${stats['percentile_95']:,.2f}")

        print(f"\nReturn Metrics:")
        print(f"  Mean Total Return: {stats['mean_total_return']*100:.2f}%")
        print(f"  Median Total Return: {stats['median_total_return']*100:.2f}%")
        print(f"  Mean CAGR: {stats['mean_cagr']*100:.2f}%")
        print(f"  Median CAGR: {stats['median_cagr']*100:.2f}%")
        print(f"  CAGR Std Dev: {stats['std_cagr']*100:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Mean Max Drawdown: {stats['mean_max_drawdown']*100:.2f}%")
        print(f"  Median Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {stats['sortino_ratio']:.3f}")

        print(f"\nProbabilities:")
        print(f"  Probability of Loss: {stats['probability_loss']*100:.2f}%")
        print(f"  Probability of Doubling: {stats['probability_double']*100:.2f}%")

    def close(self):
        """Clean up resources"""
        self.data_manager.close()


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    sim = PortfolioSimulator(initial_capital=100000, years=10, simulations=10000)

    print("="*60)
    print("Loading and analyzing historical data...")
    print("="*60)

    # Define assets from real tickers
    voo = sim.define_asset_from_ticker('VOO', name='VOO (S&P 500)')
    tqqq = sim.define_asset_from_ticker('TQQQ', name='TQQQ (3x Nasdaq)')
    bnd = sim.define_asset_from_ticker('BND', name='BND (Bonds)')
    qqq = sim.define_asset_from_ticker('QQQ', name='QQQ (Nasdaq 100)')

    print("\n" + "="*60)
    print("Running Monte Carlo simulations...")
    print("="*60)

    # Define portfolio configurations
    # Using a list of dicts to keep data together
    portfolio_configs = [
        {
            'label': '100% VOO',
            'assets': [voo],
            'allocations': [1.0],
            'results': None  # Will be filled in
        },
        {
            'label': '80% VOO / 20% TQQQ',
            'assets': [voo, tqqq],
            'allocations': [0.8, 0.2],
            'results': None
        },
        {
            'label': '50% VOO / 50% QQQ',
            'assets': [voo, qqq],
            'allocations': [0.5, 0.5],
            'results': None
        },
        {
            'label': '70% VOO / 20% QQQ / 10% BND',
            'assets': [voo, qqq, bnd],
            'allocations': [0.7, 0.2, 0.1],
            'results': None
        },
        {
            'label': '60% VOO / 30% TQQQ / 10% BND',
            'assets': [voo, tqqq, bnd],
            'allocations': [0.6, 0.3, 0.1],
            'results': None
        }
    ]

    # Run simulations for each portfolio
    for config in portfolio_configs:
        print(f"\nSimulating: {config['label']}...")
        config['results'] = sim.simulate_portfolio(
            config['assets'],
            config['allocations'],
            method='bootstrap'  # Use bootstrap to preserve real characteristics
        )

    # Print detailed statistics for each portfolio
    for config in portfolio_configs:
        sim.print_detailed_stats(config['results'], config['label'])

    # Create comprehensive visualization
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    fig = sim.plot_results(portfolio_configs)
    plt.show()

    # Show database info
    print("\n" + "="*60)
    print("Database Information")
    print("="*60)
    print(f"Database file: stock_data.db")
    print(f"\nStored tickers:")
    tickers = sim.data_manager.list_all_tickers()
    for ticker in tickers:
        info = sim.data_manager.get_ticker_info(ticker)
        if info:
            print(f"  {ticker}: {info['record_count']} records ({info['start_date']} to {info['end_date']})")

    # Clean up
    sim.close()

    print("\n✓ Analysis complete!")