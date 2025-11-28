"""
Data management module - Smart Memory & Metadata.
Now remembers when a stock didn't exist to avoid redundant downloads.
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
import random
import requests

# Try to import requests_cache
try:
    import requests_cache
    HAS_REQUESTS_CACHE = True
except ImportError:
    HAS_REQUESTS_CACHE = False

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
    __table_args__ = (UniqueConstraint('ticker', 'date', name='unique_ticker_date'),)

class TickerMetadata(Base):
    """
    NEW: Remembers metadata about the ticker to avoid redundant queries.
    Stores the 'First Valid Date' (Inception) so we don't query before it.
    """
    __tablename__ = 'ticker_metadata'
    ticker = Column(String(20), primary_key=True)
    first_valid_date = Column(Date, nullable=True) # The actual start of data (e.g. IPO date)
    last_updated = Column(DateTime, default=datetime.now)

class OptimizationResult(Base):
    __tablename__ = 'optimization_results'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    portfolio_name = Column(String(200))
    tickers = Column(String(500))
    allocations = Column(String(500))
    optimization_score = Column(Float)
    expected_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    probability_loss = Column(Float)
    probability_double = Column(Float)
    median_cagr = Column(Float)
    optimization_params = Column(String(1000))

class DataManager:
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
        self.yf_session = self._setup_session()

    def _setup_session(self):
        if HAS_REQUESTS_CACHE:
            session = requests_cache.CachedSession('yfinance_cache', expire_after=timedelta(days=1))
        else:
            session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    def get_data(self, ticker, start_date=None, end_date=None, force_update=False):
        """Get stock data with Smart Memory logic"""
        if start_date is None: start_date = date.today() - timedelta(days=365*10)
        else: start_date = self._normalize_date(start_date)

        if end_date is None: end_date = date.today()
        else: end_date = self._normalize_date(end_date)

        # 1. Check Metadata (Did we already find the inception date?)
        metadata = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
        known_inception = metadata.first_valid_date if metadata else None

        # Smart Trim: If we asked for 2000, but we know it starts in 2015,
        # treat the request as if it started in 2015.
        effective_start_date = start_date
        if known_inception and start_date < known_inception:
            # print(f"  [Smart] {ticker} inception is {known_inception}. Ignoring request for prior dates.")
            effective_start_date = known_inception

        if not force_update:
            db_data = self._get_from_db(ticker, effective_start_date, end_date)
            if db_data is not None and len(db_data) > 0:
                db_start = db_data.index.min().date()
                db_end = db_data.index.max().date()

                # Logic: We have enough data IF:
                # 1. DB Start is before requested start OR DB Start is the Known Inception
                # 2. DB End is recent enough
                start_ok = (db_start <= effective_start_date) or (known_inception and db_start <= known_inception)
                end_ok = (db_end >= end_date - timedelta(days=5))

                if start_ok and end_ok:
                    print(f"✓ Retrieved {ticker} from database ({len(db_data)} records)")
                    return db_data
                else:
                    print(f"⚠ Partial data for {ticker}. Need {effective_start_date} to {end_date}. Have {db_start} to {db_end}.")

        print(f"↓ Downloading {ticker} from Yahoo Finance...")
        try:
            # If we know the inception, don't hammer Yahoo for data before it
            download_start = effective_start_date

            df = self._smart_download(ticker, download_start, end_date)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Normalize Index
            df.index = pd.DatetimeIndex([d.date() for d in df.index])
            df.index.name = 'Date'

            actual_start = df.index.min().date()

            gap = (actual_start - download_start).days
            if gap > 7:
                self._update_metadata(ticker, actual_start)

            self._save_to_db(ticker, df)
            print(f"✓ Downloaded {ticker} ({len(df)} records). Start: {actual_start}")

            return df
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {str(e)}")
            # Fallback to DB
            db_data = self._get_from_db(ticker, effective_start_date, end_date)
            if db_data is not None:
                print(f"⚠ FALLBACK: Using existing data for {ticker}")
                return db_data
            raise

    def _update_metadata(self, ticker, first_date):
        """Update or Create metadata entry"""
        try:
            meta = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
            if not meta:
                meta = TickerMetadata(ticker=ticker)
                self.session.add(meta)

            # If we found an earlier date, update it
            if meta.first_valid_date is None or first_date < meta.first_valid_date:
                meta.first_valid_date = first_date

            meta.last_updated = datetime.now()
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Warning: Could not update metadata: {e}")

    def _smart_download(self, ticker, start, end):
        """Download with Exponential Backoff"""
        retries = 3
        delay = 2
        for i in range(retries):
            try:
                dat = yf.Ticker(ticker, session=self.yf_session)
                df = dat.history(start=start, end=end, auto_adjust=False)
                if not df.empty: return df
                time.sleep(delay)
                delay *= 2
            except Exception:
                if i == retries - 1: raise
                time.sleep(delay + random.random())
                delay *= 2
        return pd.DataFrame()

    def _normalize_date(self, date_input):
        if isinstance(date_input, date): return date_input
        if isinstance(date_input, datetime): return date_input.date()
        if isinstance(date_input, str): return pd.to_datetime(date_input).date()
        return pd.to_datetime(date_input).date()

    def _get_from_db(self, ticker, start_date, end_date):
        try:
            query = self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= start_date, # Strict filter? No, handled by logic
                StockPrice.date <= end_date
            ).order_by(StockPrice.date)
            results = query.all()
            if not results: return None

            # Convert to DF
            data = {'Adj Close': [r.adj_close for r in results], 'Close': [r.close for r in results],
                    'Open': [r.open for r in results], 'High': [r.high for r in results],
                    'Low': [r.low for r in results], 'Volume': [r.volume for r in results]}
            df = pd.DataFrame(data, index=pd.DatetimeIndex([r.date for r in results]))
            df.index.name = 'Date'
            return df
        except Exception: return None

    def _save_to_db(self, ticker, df):
        try:
            dates = [self._normalize_date(d) for d in df.index]
            self.session.query(StockPrice).filter(StockPrice.ticker==ticker, StockPrice.date>=min(dates), StockPrice.date<=max(dates)).delete()
            records = [StockPrice(ticker=ticker, date=self._normalize_date(dt),
                                  open=r.get('Open'), high=r.get('High'), low=r.get('Low'),
                                  close=r.get('Close'), adj_close=r.get('Adj Close'), volume=r.get('Volume'))
                       for dt, r in df.iterrows()]
            self.session.bulk_save_objects(records)
            self.session.commit()
        except Exception: self.session.rollback()

    def save_optimization_result(self, portfolio_name, assets, allocations, stats, optimization_params):
        try:
            tickers = [asset['ticker'] for asset in assets]
            result = OptimizationResult(
                portfolio_name=portfolio_name, tickers=json.dumps(tickers),
                allocations=json.dumps(allocations), optimization_score=optimization_params.get('score', 0),
                expected_return=stats.get('mean_cagr', 0), sharpe_ratio=stats.get('sharpe_ratio', 0),
                sortino_ratio=stats.get('sortino_ratio', 0), max_drawdown=stats.get('median_max_drawdown', 0),
                probability_loss=stats.get('probability_loss', 0), probability_double=stats.get('probability_double', 0),
                median_cagr=stats.get('median_cagr', 0), optimization_params=json.dumps(optimization_params)
            )
            self.session.add(result)
            self.session.commit()
            return result.id
        except Exception: self.session.rollback(); return None

    def get_optimization_results(self, limit=10):
        return [r.__dict__ for r in self.session.query(OptimizationResult).order_by(OptimizationResult.optimization_score.desc()).limit(limit).all()]
    def get_ticker_info(self, ticker): return None
    def list_all_tickers(self): return [r.ticker for r in self.session.query(StockPrice.ticker).distinct().all()]
    def close(self): self.session.close()