"""
Data management module for downloading and caching stock data.
Handles Yahoo Finance downloads and SQLite database caching.
"""

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Float, Date, Integer, UniqueConstraint, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta, date
import time
import json

Base = declarative_base()

class StockPrice(Base):
    """Database model for storing stock prices"""
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

    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='unique_ticker_date'),
    )

class OptimizationResult(Base):
    """Database model for storing optimization results"""
    __tablename__ = 'optimization_results'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    portfolio_name = Column(String(200))
    tickers = Column(String(500))  # JSON string of tickers
    allocations = Column(String(500))  # JSON string of allocations
    optimization_score = Column(Float)
    expected_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    probability_loss = Column(Float)
    probability_double = Column(Float)
    median_cagr = Column(Float)
    optimization_params = Column(String(1000))  # JSON string of optimization parameters

class DataManager:
    """Manages downloading and caching of stock data"""

    def __init__(self, db_path='stock_data.db'):
        self.db_path = db_path
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False
        )
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_data(self, ticker, start_date=None, end_date=None, force_update=False):
        """Get stock data from database or download from Yahoo Finance"""
        if start_date is None:
            start_date = date.today() - timedelta(days=365*10)
        else:
            start_date = self._normalize_date(start_date)

        if end_date is None:
            end_date = date.today()
        else:
            end_date = self._normalize_date(end_date)

        if not force_update:
            db_data = self._get_from_db(ticker, start_date, end_date)
            if db_data is not None and len(db_data) > 0:
                db_start = db_data.index.min().date()
                db_end = db_data.index.max().date()
                print(f"{db_start = }, {db_end = }")  # Debug print
                print(f"{start_date = }, {end_date = }")  # Debug print
                if db_start <= start_date and db_end >= end_date - timedelta(days=5):
                    print(f"✓ Retrieved {ticker} from database ({len(db_data)} records)")
                    return db_data
                else:
                    print(f"⚠ Partial data for {ticker} in database, updating...")

        print(f"↓ Downloading {ticker} from Yahoo Finance...")
        try:
            df = self._download_with_retry(ticker, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            self._save_to_db(ticker, df)
            print(f"✓ Downloaded and saved {ticker} ({len(df)} records)")
            return df
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {str(e)}")
            db_data = self._get_from_db(ticker, start_date, end_date)
            if db_data is not None and len(db_data) > 0:
                print(f"⚠ Using existing database data for {ticker}")
                return db_data
            raise

    def _download_with_retry(self, ticker, start_date, end_date, max_retries=3):
        """Download data with retry logic and rate limit handling"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
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
                StockPrice.date >= start_date - timedelta(days=1),
                StockPrice.date <= end_date
            ).order_by(StockPrice.date)

            results = query.all()
            if not results:
                return None

            data = {
                'Open': [r.open for r in results],
                'High': [r.high for r in results],
                'Low': [r.low for r in results],
                'Close': [r.close for r in results],
                'Adj Close': [r.adj_close for r in results],
                'Volume': [r.volume for r in results],
            }

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
            dates = [self._normalize_date(d) for d in df.index]
            date_min = min(dates)
            date_max = max(dates)

            self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= date_min,
                StockPrice.date <= date_max
            ).delete()

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

            self.session.bulk_save_objects(records)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving to database: {str(e)}")
            raise

    def save_optimization_result(self, portfolio_name, assets, allocations, stats, optimization_params):
        """Save optimization result to database"""
        try:
            tickers = [asset['ticker'] for asset in assets]

            result = OptimizationResult(
                portfolio_name=portfolio_name,
                tickers=json.dumps(tickers),
                allocations=json.dumps(allocations),
                optimization_score=optimization_params.get('score', 0),
                expected_return=stats.get('mean_cagr', 0),
                sharpe_ratio=stats.get('sharpe_ratio', 0),
                sortino_ratio=stats.get('sortino_ratio', 0),
                max_drawdown=stats.get('median_max_drawdown', 0),
                probability_loss=stats.get('probability_loss', 0),
                probability_double=stats.get('probability_double', 0),
                median_cagr=stats.get('median_cagr', 0),
                optimization_params=json.dumps(optimization_params)
            )

            self.session.add(result)
            self.session.commit()
            print(f"✓ Saved optimization result: {portfolio_name}")
            return result.id
        except Exception as e:
            self.session.rollback()
            print(f"Error saving optimization result: {str(e)}")
            raise

    def get_optimization_results(self, limit=10):
        """Retrieve recent optimization results"""
        try:
            results = self.session.query(OptimizationResult).order_by(
                OptimizationResult.optimization_score.desc()
            ).limit(limit).all()

            return [{
                'id': r.id,
                'timestamp': r.timestamp,
                'portfolio_name': r.portfolio_name,
                'tickers': json.loads(r.tickers),
                'allocations': json.loads(r.allocations),
                'score': r.optimization_score,
                'expected_return': r.expected_return,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'max_drawdown': r.max_drawdown,
                'median_cagr': r.median_cagr
            } for r in results]
        except Exception as e:
            print(f"Error retrieving optimization results: {str(e)}")
            return []

    def get_ticker_info(self, ticker):
        """Get information about stored data for a ticker"""
        try:
            query = self.session.query(StockPrice.date).filter(
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

