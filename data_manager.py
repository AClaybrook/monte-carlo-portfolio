"""
Data management module - Interval-Based Smart Caching with Bulk Download.

Key improvements:
1. Uses `portion` library for interval arithmetic to track known data ranges
2. Bulk downloads via yf.download() for efficiency
3. Stores ticker inception dates to avoid pre-IPO queries
4. Only fetches missing intervals, not full history
5. Detects gaps in data and handles them properly
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Float, Date, Integer, UniqueConstraint, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta, date
import time
import json
import random
import os
import requests
import portion as P
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

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
    Stores metadata about tickers to avoid redundant queries.
    - first_valid_date: IPO/inception date (don't query before this)
    - last_updated: When we last refreshed metadata
    - is_valid: Whether the ticker exists and has data
    - data_intervals: JSON string of known data intervals
    """
    __tablename__ = 'ticker_metadata'
    ticker = Column(String(20), primary_key=True)
    first_valid_date = Column(Date, nullable=True)
    last_valid_date = Column(Date, nullable=True)
    last_updated = Column(DateTime, default=datetime.now)
    is_valid = Column(Boolean, default=True)
    data_intervals_json = Column(Text, nullable=True)  # JSON representation of intervals


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


@dataclass
class TickerDataRequest:
    """Represents a request for ticker data"""
    ticker: str
    start_date: date
    end_date: date


@dataclass
class DataInterval:
    """Represents a continuous interval of data"""
    start: date
    end: date

    def to_portion(self) -> P.Interval:
        """Convert to portion interval"""
        return P.closed(self.start, self.end)


class IntervalTracker:
    """
    Manages intervals of known data for a ticker using the portion library.

    Example:
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 12, 31))
        tracker.add_interval(date(2021, 6, 1), date(2021, 12, 31))

        # Find what's missing to cover 2020-01-01 to 2022-12-31
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2022, 12, 31))
        # Returns: [(2021-01-01, 2021-05-31), (2022-01-01, 2022-12-31)]
    """

    # Maximum gap in days before considering data as separate intervals
    # Accounts for weekends (2 days) + extended holidays (5 days max)
    MAX_GAP_DAYS = 7

    def __init__(self, intervals_json: Optional[str] = None):
        self._intervals: P.Interval = P.empty()
        if intervals_json:
            self._load_from_json(intervals_json)

    def _load_from_json(self, json_str: str):
        """Load intervals from JSON string"""
        try:
            data = json.loads(json_str)
            for item in data:
                start = date.fromisoformat(item['start'])
                end = date.fromisoformat(item['end'])
                self._intervals = self._intervals | P.closed(start, end)
        except (json.JSONDecodeError, KeyError, ValueError):
            self._intervals = P.empty()

    def to_json(self) -> str:
        """Serialize intervals to JSON string"""
        intervals_list = []
        for interval in self._intervals:
            if not interval.empty:
                intervals_list.append({
                    'start': interval.lower.isoformat(),
                    'end': interval.upper.isoformat()
                })
        return json.dumps(intervals_list)

    def add_interval(self, start: date, end: date):
        """Add a new interval of known data"""
        self._intervals = self._intervals | P.closed(start, end)

    def add_dates(self, dates: List[date]):
        """Add individual dates, automatically merging nearby dates into intervals"""
        if not dates:
            return

        sorted_dates = sorted(dates)
        current_start = sorted_dates[0]
        current_end = sorted_dates[0]

        for d in sorted_dates[1:]:
            gap = (d - current_end).days
            if gap <= self.MAX_GAP_DAYS:
                # Extend current interval
                current_end = d
            else:
                # Save current interval and start new one
                self._intervals = self._intervals | P.closed(current_start, current_end)
                current_start = d
                current_end = d

        # Don't forget the last interval
        self._intervals = self._intervals | P.closed(current_start, current_end)

    def get_missing_intervals(self, start: date, end: date) -> List[Tuple[date, date]]:
        """
        Find intervals within [start, end] that we don't have data for.

        Returns list of (start, end) tuples representing missing ranges.
        """
        requested = P.closed(start, end)
        missing = requested - self._intervals

        result = []
        for interval in missing:
            if not interval.empty:
                # Handle open/closed boundaries
                lower = interval.lower if interval.left == P.CLOSED else interval.lower + timedelta(days=1)
                upper = interval.upper if interval.right == P.CLOSED else interval.upper - timedelta(days=1)
                if lower <= upper:
                    result.append((lower, upper))

        return result

    def has_data_for(self, start: date, end: date) -> bool:
        """Check if we have complete data coverage for the given range"""
        requested = P.closed(start, end)
        return requested in self._intervals

    def get_coverage_ratio(self, start: date, end: date) -> float:
        """Get the ratio of requested range that we have data for (0.0 to 1.0)"""
        requested = P.closed(start, end)
        covered = requested & self._intervals

        if requested.empty:
            return 1.0

        # Count days
        requested_days = (end - start).days + 1
        covered_days = 0

        for interval in covered:
            if not interval.empty:
                covered_days += (interval.upper - interval.lower).days + 1

        return covered_days / requested_days

    @property
    def is_empty(self) -> bool:
        return self._intervals.empty

    @property
    def bounds(self) -> Optional[Tuple[date, date]]:
        """Get the overall bounds of known data"""
        if self._intervals.empty:
            return None

        # Get the enclosure (smallest interval containing all data)
        enc = self._intervals.enclosure
        return (enc.lower, enc.upper)


class DataManager:
    """
    Enhanced data manager with interval-based caching and bulk downloads.

    Key features:
    - Tracks data intervals per ticker to avoid redundant downloads
    - Uses bulk yf.download() for multiple tickers
    - Stores ticker inception dates to avoid pre-IPO queries
    - Automatically detects and fills gaps in data
    """

    def __init__(self, db_path: str = 'stock_data.db', data_source: str = 'yfinance'):
        if data_source not in ('yfinance', 'fmp'):
            raise ValueError(f"Unknown data_source: {data_source}. Valid: yfinance, fmp")
        self.data_source = data_source
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
        # Plain HTTP session for FMP. yfinance is deliberately NOT given a session:
        # yfinance >= 0.2.58 rejects requests.Session (Yahoo requires curl_cffi
        # browser impersonation, which yfinance sets up itself).
        self.yf_session = self._setup_session()

        # Large requests are what trigger Yahoo rate limits, so downloads are
        # split into ticker batches and date chunks with a pause between each.
        self.bulk_batch_size = 5
        self.max_chunk_years = 5
        self.chunk_pause_seconds = 2.0

        # FMP API key (loaded from environment when using FMP data source)
        self.fmp_api_key = None
        if self.data_source == 'fmp':
            self.fmp_api_key = os.environ.get('FMP_API_KEY')
            if not self.fmp_api_key:
                raise ValueError(
                    "FMP_API_KEY environment variable is required when using --data-source fmp. "
                    "Get a free key at https://site.financialmodelingprep.com/developer/docs"
                )

        # In-memory cache of interval trackers
        self._interval_cache: Dict[str, IntervalTracker] = {}

        # Track failed download attempts to avoid retrying too soon
        # Key: (ticker, start_iso, end_iso), Value: datetime of last failure
        self._failed_downloads: Dict[tuple, datetime] = {}
        self.FAILED_COOLDOWN_HOURS = 1

        # Known ETF/Stock inception dates (avoid querying before these)
        self._known_inception_dates = {
            'TQQQ': date(2010, 2, 11),
            'SQQQ': date(2010, 2, 11),
            'SPXL': date(2008, 11, 5),
            'SPXS': date(2008, 11, 5),
            'TMF': date(2009, 4, 16),
            'TMV': date(2009, 4, 16),
            'UPRO': date(2009, 6, 25),
            'QQQM': date(2020, 10, 13),
            'AVUV': date(2019, 9, 24),
            'AVDV': date(2019, 9, 24),
            'VTI': date(2001, 5, 24),
            'VOO': date(2010, 9, 7),
            'BND': date(2007, 4, 3),
            'VXUS': date(2011, 1, 26),
            'BTC-USD': date(2014, 9, 17),
            'ETH-USD': date(2017, 11, 9),
        }

    def _setup_session(self) -> requests.Session:
        return requests.Session()

    def _date_chunks(self, start: date, end: date) -> List[Tuple[date, date]]:
        """Split [start, end] into consecutive chunks of at most max_chunk_years."""
        chunk_days = int(365.25 * self.max_chunk_years)
        chunks = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
            chunks.append((cur, chunk_end))
            cur = chunk_end + timedelta(days=1)
        return chunks

    def _normalize_date(self, date_input) -> date:
        """Convert various date formats to date object"""
        if isinstance(date_input, date) and not isinstance(date_input, datetime):
            return date_input
        if isinstance(date_input, datetime):
            return date_input.date()
        if isinstance(date_input, str):
            return pd.to_datetime(date_input).date()
        return pd.to_datetime(date_input).date()

    def _get_interval_tracker(self, ticker: str) -> IntervalTracker:
        """Get or create interval tracker for a ticker"""
        ticker = ticker.upper()

        if ticker not in self._interval_cache:
            # Try to load from database
            meta = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
            if meta and meta.data_intervals_json:
                self._interval_cache[ticker] = IntervalTracker(meta.data_intervals_json)
            else:
                self._interval_cache[ticker] = IntervalTracker()

        return self._interval_cache[ticker]

    def _save_interval_tracker(self, ticker: str, tracker: IntervalTracker):
        """Save interval tracker to database"""
        ticker = ticker.upper()

        meta = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
        if not meta:
            meta = TickerMetadata(ticker=ticker)
            self.session.add(meta)

        meta.data_intervals_json = tracker.to_json()
        meta.last_updated = datetime.now()

        # Update bounds
        bounds = tracker.bounds
        if bounds:
            meta.first_valid_date = bounds[0]
            meta.last_valid_date = bounds[1]

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Warning: Could not save interval tracker: {e}")

    def get_ticker_inception_date(self, ticker: str) -> Optional[date]:
        """
        Get the known inception/IPO date for a ticker.
        Returns None if unknown.
        """
        ticker = ticker.upper()

        # Check hardcoded known dates first
        if ticker in self._known_inception_dates:
            return self._known_inception_dates[ticker]

        # Check database
        meta = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
        if meta and meta.first_valid_date:
            return meta.first_valid_date

        return None

    def _is_on_cooldown(self, ticker: str, start: date, end: date) -> bool:
        """Check if a download attempt is on cooldown due to recent failure."""
        key = (ticker.upper(), start.isoformat(), end.isoformat())
        if key in self._failed_downloads:
            failed_at = self._failed_downloads[key]
            if datetime.now() - failed_at < timedelta(hours=self.FAILED_COOLDOWN_HOURS):
                return True
            del self._failed_downloads[key]
        return False

    def _record_failure(self, ticker: str, start: date, end: date):
        """Record a failed download attempt."""
        key = (ticker.upper(), start.isoformat(), end.isoformat())
        self._failed_downloads[key] = datetime.now()

    def get_data(self, ticker: str, start_date: date = None, end_date: date = None,
                 force_update: bool = False) -> pd.DataFrame:
        """
        Get stock data with smart interval-based caching.

        Only downloads missing intervals, uses cached data for known ranges.
        """
        ticker = ticker.upper()

        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        else:
            start_date = self._normalize_date(start_date)

        if end_date is None:
            end_date = date.today()
        else:
            end_date = self._normalize_date(end_date)

        # Adjust start date based on known inception
        inception = self.get_ticker_inception_date(ticker)
        if inception and start_date < inception:
            start_date = inception

        # Get interval tracker
        tracker = self._get_interval_tracker(ticker)

        if force_update:
            # Force re-download everything
            missing_intervals = [(start_date, end_date)]
        else:
            # Find what we're missing
            missing_intervals = tracker.get_missing_intervals(start_date, end_date)

        # Download missing data
        if missing_intervals:
            print(f"↓ {ticker}: Downloading {len(missing_intervals)} interval(s)...")
            for interval_start, interval_end in missing_intervals:
                if self._is_on_cooldown(ticker, interval_start, interval_end):
                    print(f"  ⏸ Skipping {ticker} [{interval_start} to {interval_end}] (failed recently, cooldown active)")
                    continue
                self._download_and_save(ticker, interval_start, interval_end, tracker)

            # Save updated tracker
            self._save_interval_tracker(ticker, tracker)
        else:
            print(f"✓ {ticker}: Using cached data (100% coverage)")

        # Retrieve from database
        return self._get_from_db(ticker, start_date, end_date)

    def _download_and_save(self, ticker: str, start: date, end: date,
                           tracker: IntervalTracker):
        """Download data for a specific interval and save to database"""
        try:
            df = self._smart_download(ticker, start, end)

            if df.empty:
                print(f"  ⚠ No data returned for {ticker} [{start} to {end}]")
                self._record_failure(ticker, start, end)
                return

            # Normalize index
            df.index = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in df.index])
            df.index.name = 'Date'

            # Update inception date if we learned something
            actual_start = df.index.min().date() if hasattr(df.index.min(), 'date') else df.index.min()
            if (start - actual_start).days < -7:
                # Data started significantly later than requested - this is likely the inception
                self._update_inception_date(ticker, actual_start)

            # Save to database
            self._save_to_db(ticker, df)

            # Update tracker with actual dates we received
            dates_received = [d.date() if hasattr(d, 'date') else d for d in df.index]
            tracker.add_dates(dates_received)

            print(f"  ✓ Downloaded {len(df)} rows for {ticker}")

        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")

    def _smart_download(self, ticker: str, start: date, end: date,
                        max_retries: int = 5) -> pd.DataFrame:
        """Download in date chunks, each with exponential backoff."""
        chunks = self._date_chunks(start, end)
        frames = []
        for i, (c_start, c_end) in enumerate(chunks):
            if i > 0:
                time.sleep(self.chunk_pause_seconds + random.random())
            if self.data_source == 'fmp':
                df = self._fmp_download(ticker, c_start, c_end, max_retries)
            else:
                df = self._yf_download_one(ticker, c_start, c_end, max_retries)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep='last')].sort_index()

    def _yf_download_one(self, ticker: str, start: date, end: date,
                         max_retries: int = 5) -> pd.DataFrame:
        """Single-ticker, single-chunk yfinance download with backoff."""

        delay = 2.0  # Start with longer delay

        for attempt in range(max_retries):
            try:
                # Add buffer day to end date for yfinance quirks
                end_buffered = end + timedelta(days=1)

                # Small delay before each request to avoid rate limits
                time.sleep(0.5 + random.random())

                data = yf.Ticker(ticker)
                df = data.history(start=start, end=end_buffered, auto_adjust=False, timeout=30)

                if not df.empty:
                    return df

                time.sleep(delay)
                delay *= 2

            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    # Rate limited - wait longer
                    wait_time = delay * 2 + random.random() * 5
                    print(f"  ⏳ Rate limited, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    delay *= 2
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(delay + random.random())
                    delay *= 2

        return pd.DataFrame()

    def _fmp_download(self, ticker: str, start: date, end: date,
                      max_retries: int = 5) -> pd.DataFrame:
        """Download from Financial Modeling Prep API with retry logic."""
        delay = 2.0

        for attempt in range(max_retries):
            try:
                time.sleep(0.5 + random.random())

                url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
                params = {
                    'symbol': ticker,
                    'from': start.isoformat(),
                    'to': end.isoformat(),
                }
                headers = {'apikey': self.fmp_api_key}

                response = self.yf_session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()

                if not data or not isinstance(data, list):
                    time.sleep(delay)
                    delay *= 2
                    continue

                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Rename columns to match yfinance format expected by _save_to_db
                column_map = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'adjClose': 'Adj Close',
                    'volume': 'Volume',
                }
                df = df.rename(columns=column_map)
                keep_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                df = df[[c for c in keep_cols if c in df.columns]]

                if not df.empty:
                    return df

                time.sleep(delay)
                delay *= 2

            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    wait_time = delay * 2 + random.random() * 5
                    print(f"  ⏳ FMP rate limited, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    delay *= 2
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(delay + random.random())
                    delay *= 2

        return pd.DataFrame()

    def _update_inception_date(self, ticker: str, inception_date: date):
        """Update the known inception date for a ticker"""
        ticker = ticker.upper()

        meta = self.session.query(TickerMetadata).filter_by(ticker=ticker).first()
        if not meta:
            meta = TickerMetadata(ticker=ticker)
            self.session.add(meta)

        if meta.first_valid_date is None or inception_date < meta.first_valid_date:
            meta.first_valid_date = inception_date
            meta.last_updated = datetime.now()

            try:
                self.session.commit()
                print(f"  📅 Updated {ticker} inception date: {inception_date}")
            except Exception:
                self.session.rollback()

    def bulk_download(self, tickers: List[str], start_date: date = None,
                      end_date: date = None, force_update: bool = False,
                      sequential: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Efficiently download data for multiple tickers.

        Uses yf.download() for bulk operations and only fetches missing data.

        Parameters:
            sequential: If True, download one at a time (slower but avoids rate limits)
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        else:
            start_date = self._normalize_date(start_date)

        if end_date is None:
            end_date = date.today()
        else:
            end_date = self._normalize_date(end_date)

        tickers = [t.upper() for t in tickers]
        results = {}

        # Categorize tickers by what we need to download
        tickers_to_download: Dict[str, List[Tuple[date, date]]] = {}
        tickers_cached: Set[str] = set()

        for ticker in tickers:
            # Adjust for inception date
            effective_start = start_date
            inception = self.get_ticker_inception_date(ticker)
            if inception and start_date < inception:
                effective_start = inception

            if force_update:
                tickers_to_download[ticker] = [(effective_start, end_date)]
            else:
                tracker = self._get_interval_tracker(ticker)
                missing = tracker.get_missing_intervals(effective_start, end_date)

                if missing:
                    tickers_to_download[ticker] = missing
                else:
                    tickers_cached.add(ticker)

        # Report status
        if tickers_cached:
            print(f"✓ Cached (no download needed): {', '.join(sorted(tickers_cached))}")

        if tickers_to_download:
            all_tickers_needing_data = list(tickers_to_download.keys())

            # Print download plan so user can see what will be fetched
            print(f"\nDownload plan:")
            for ticker in sorted(tickers_to_download.keys()):
                intervals = tickers_to_download[ticker]
                interval_strs = [f"{s} to {e}" for s, e in intervals]
                print(f"  {ticker}: {', '.join(interval_strs)}")
            print()

            # Decide: bulk yf.download() only when all tickers need the same full range
            # (e.g., force_update or fresh database). Otherwise, sequential per-ticker
            # interval-targeted downloads are more efficient and avoid wasted API calls.
            all_need_full_range = (
                self.data_source == 'yfinance'  # FMP has no bulk endpoint
                and not sequential
                and len(all_tickers_needing_data) > 1
                and all(
                    len(intervals) == 1
                    and intervals[0][0] <= start_date
                    and intervals[0][1] >= end_date
                    for intervals in tickers_to_download.values()
                )
            )

            if all_need_full_range:
                # True bulk download - all tickers need the same full range
                print(f"↓ Bulk downloading {len(all_tickers_needing_data)} tickers (full range)...")
                self._bulk_download_and_save(
                    all_tickers_needing_data, start_date, end_date,
                    ticker_intervals=tickers_to_download
                )
            else:
                # Sequential with per-ticker intervals (most efficient for catch-up)
                print(f"↓ Downloading {len(all_tickers_needing_data)} tickers (interval-targeted)...")
                self._sequential_download(all_tickers_needing_data, start_date, end_date,
                                          tickers_to_download)

        # Retrieve all data from database
        for ticker in tickers:
            effective_start = start_date
            inception = self.get_ticker_inception_date(ticker)
            if inception and start_date < inception:
                effective_start = inception

            df = self._get_from_db(ticker, effective_start, end_date)
            if df is not None and not df.empty:
                results[ticker] = df

        return results

    def _bulk_download_and_save(self, tickers: List[str], start: date, end: date,
                                ticker_intervals: Dict[str, List[Tuple[date, date]]] = None):
        """Bulk download via yf.download(), split into small ticker batches and
        date chunks so no single request is large enough to get rate limited.

        Args:
            ticker_intervals: Per-ticker missing intervals dict, used for fallback
                to sequential downloads if a batch fails.
        """
        if self.data_source != 'yfinance':
            self._sequential_download(tickers, start, end, ticker_intervals)
            return

        batches = [tickers[i:i + self.bulk_batch_size]
                   for i in range(0, len(tickers), self.bulk_batch_size)]
        chunks = self._date_chunks(start, end)
        print(f"  {len(batches)} ticker batch(es) x {len(chunks)} date chunk(s)")

        for b_idx, batch in enumerate(batches):
            failed = False
            for c_start, c_end in chunks:
                time.sleep(self.chunk_pause_seconds + random.random())
                try:
                    df = yf.download(
                        tickers=batch,
                        start=c_start,
                        end=c_end + timedelta(days=1),
                        auto_adjust=False,
                        group_by='ticker',
                        threads=False,
                        progress=False,
                        timeout=30
                    )
                except Exception as e:
                    print(f"  ⚠ Batch {batch} [{c_start} to {c_end}] failed: {e}")
                    failed = True
                    break

                if df is None or df.empty:
                    continue
                for ticker in batch:
                    ticker_df = self._extract_ticker_frame(df, ticker)
                    if ticker_df is None or ticker_df.empty:
                        continue
                    self._save_to_db(ticker, ticker_df)
                    tracker = self._get_interval_tracker(ticker)
                    tracker.add_dates([d.date() for d in ticker_df.index])
                    self._save_interval_tracker(ticker, tracker)
                    print(f"  ✓ {ticker}: {len(ticker_df)} rows [{c_start} to {c_end}]")

            if failed:
                print("  ⏳ Falling back to per-ticker downloads for this batch...")
                time.sleep(5)
                self._sequential_download(batch, start, end, ticker_intervals)

    def _sequential_download(self, tickers: List[str], start: date, end: date,
                             ticker_intervals: Dict[str, List[Tuple[date, date]]] = None):
        for ticker in tickers:
            intervals = (ticker_intervals.get(ticker, [(start, end)])
                         if ticker_intervals else [(start, end)])
            tracker = self._get_interval_tracker(ticker)
            for interval_start, interval_end in intervals:
                if self._is_on_cooldown(ticker, interval_start, interval_end):
                    print(f"    ⏸ Skipping {ticker} [{interval_start} to {interval_end}] (cooldown)")
                    continue
                time.sleep(1 + random.random())
                self._download_and_save(ticker, interval_start, interval_end, tracker)
            self._save_interval_tracker(ticker, tracker)

    @staticmethod
    def _extract_ticker_frame(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Pull one ticker's OHLCV frame out of a yf.download() result.

        Handles both flat columns (old yfinance, single ticker) and
        (ticker, field) MultiIndex columns (group_by='ticker').
        """
        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df.columns.get_level_values(0):
                return None
            out = df[ticker]
        else:
            out = df
        out = out.dropna(how='all').copy()
        if 'Adj Close' not in out.columns or out['Adj Close'].isna().all():
            return None
        out.index = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in out.index])
        out.index.name = 'Date'
        return out

    def _get_from_db(self, ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Retrieve data from database"""
        try:
            query = self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker.upper(),
                StockPrice.date >= start_date,
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

            df = pd.DataFrame(data, index=pd.DatetimeIndex([r.date for r in results]))
            df.index.name = 'Date'
            return df

        except Exception as e:
            print(f"Database read error for {ticker}: {e}")
            return None

    def _save_to_db(self, ticker: str, df: pd.DataFrame):
        """Save data to database, replacing existing records in the date range"""
        try:
            ticker = ticker.upper()
            dates = [self._normalize_date(d) for d in df.index]

            if not dates:
                return

            # Delete existing records in this range
            self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker,
                StockPrice.date >= min(dates),
                StockPrice.date <= max(dates)
            ).delete()

            # Insert new records
            records = []
            for dt, row in df.iterrows():
                dt_normalized = self._normalize_date(dt)
                records.append(StockPrice(
                    ticker=ticker,
                    date=dt_normalized,
                    open=row.get('Open'),
                    high=row.get('High'),
                    low=row.get('Low'),
                    close=row.get('Close'),
                    adj_close=row.get('Adj Close'),
                    volume=row.get('Volume')
                ))

            self.session.bulk_save_objects(records)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f"Database write error for {ticker}: {e}")

    def get_data_coverage_report(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Generate a report showing data coverage for tickers.

        Returns DataFrame with columns:
        - ticker: Ticker symbol
        - first_date: Earliest data available
        - last_date: Latest data available
        - days_of_data: Total trading days
        - inception_known: Whether we know the ticker's inception date
        - coverage_intervals: Number of separate intervals (1 = continuous)
        """
        if tickers is None:
            # Get all tickers from database
            tickers = self.list_all_tickers()

        report_data = []

        for ticker in tickers:
            ticker = ticker.upper()
            tracker = self._get_interval_tracker(ticker)

            bounds = tracker.bounds
            inception = self.get_ticker_inception_date(ticker)

            # Count records
            count = self.session.query(StockPrice).filter(
                StockPrice.ticker == ticker
            ).count()

            # Count intervals (from JSON)
            try:
                intervals_json = tracker.to_json()
                interval_count = len(json.loads(intervals_json))
            except:
                interval_count = 0

            report_data.append({
                'ticker': ticker,
                'first_date': bounds[0] if bounds else None,
                'last_date': bounds[1] if bounds else None,
                'days_of_data': count,
                'inception_known': inception is not None,
                'known_inception': inception,
                'coverage_intervals': interval_count
            })

        return pd.DataFrame(report_data)

    def find_gaps(self, ticker: str, start_date: date = None,
                  end_date: date = None, max_gap_days: int = 7) -> List[Tuple[date, date, int]]:
        """
        Find gaps in data for a ticker.

        Returns list of (gap_start, gap_end, gap_days) tuples.
        """
        ticker = ticker.upper()

        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        if end_date is None:
            end_date = date.today()

        # Get all dates from database
        results = self.session.query(StockPrice.date).filter(
            StockPrice.ticker == ticker,
            StockPrice.date >= start_date,
            StockPrice.date <= end_date
        ).order_by(StockPrice.date).all()

        if not results:
            return [(start_date, end_date, (end_date - start_date).days)]

        dates = sorted([r.date for r in results])
        gaps = []

        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > max_gap_days:
                gaps.append((dates[i-1], dates[i], gap))

        return gaps

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
            return result.id
        except Exception:
            self.session.rollback()
            return None

    def get_optimization_results(self, limit=10):
        """Get recent optimization results"""
        return [
            r.__dict__ for r in
            self.session.query(OptimizationResult)
            .order_by(OptimizationResult.optimization_score.desc())
            .limit(limit).all()
        ]

    def get_latest_cached_date(self, tickers: List[str] = None) -> Optional[date]:
        """Get the latest date for which we have cached data across given tickers.

        Returns the minimum of the latest dates across tickers (so all tickers
        have data through this date), or None if no data exists.
        """
        if tickers is None:
            tickers = self.list_all_tickers()

        if not tickers:
            return None

        latest_dates = []
        for ticker in tickers:
            tracker = self._get_interval_tracker(ticker.upper())
            bounds = tracker.bounds
            if bounds:
                latest_dates.append(bounds[1])

        if not latest_dates:
            return None

        return min(latest_dates)

    def list_all_tickers(self) -> List[str]:
        """List all tickers in database"""
        return [r.ticker for r in self.session.query(StockPrice.ticker).distinct().all()]

    def clear_ticker_data(self, ticker: str):
        """Remove all data for a ticker (useful for corrupted data)"""
        ticker = ticker.upper()

        self.session.query(StockPrice).filter(StockPrice.ticker == ticker).delete()
        self.session.query(TickerMetadata).filter(TickerMetadata.ticker == ticker).delete()

        if ticker in self._interval_cache:
            del self._interval_cache[ticker]

        self.session.commit()
        print(f"✓ Cleared all data for {ticker}")

    def close(self):
        """Close database connection"""
        self.session.close()


# Convenience function for quick data access
def get_stock_data(tickers: List[str], start_date: date = None,
                   end_date: date = None, db_path: str = 'stock_data.db',
                   data_source: str = 'yfinance') -> Dict[str, pd.DataFrame]:
    """
    Quick helper to get data for multiple tickers.

    Example:
        data = get_stock_data(['VOO', 'QQQ', 'BND'], start_date=date(2015, 1, 1))
        voo_prices = data['VOO']
    """
    dm = DataManager(db_path, data_source=data_source)
    try:
        return dm.bulk_download(tickers, start_date, end_date)
    finally:
        dm.close()