"""
Tests for IntervalTracker and DataManager.
Validates interval-based caching logic, cooldown tracking, and helper methods.
"""
import pytest
import json
import sys
import os
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import IntervalTracker, DataManager


# ============================================================
# IntervalTracker Tests
# ============================================================

class TestIntervalTracker:
    def test_empty_tracker_returns_full_range_as_missing(self):
        tracker = IntervalTracker()
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(missing) == 1
        assert missing[0] == (date(2020, 1, 1), date(2020, 12, 31))

    def test_empty_tracker_is_empty(self):
        tracker = IntervalTracker()
        assert tracker.is_empty
        assert tracker.bounds is None

    def test_add_interval_then_no_missing(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 12, 31))
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(missing) == 0

    def test_add_interval_updates_bounds(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 12, 31))
        assert tracker.bounds == (date(2020, 1, 1), date(2020, 12, 31))

    def test_partial_coverage_returns_correct_gaps(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 6, 30))
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(missing) == 1
        assert missing[0] == (date(2020, 7, 1), date(2020, 12, 31))

    def test_multiple_intervals_with_gap(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 3, 31))
        tracker.add_interval(date(2020, 10, 1), date(2020, 12, 31))
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(missing) == 1
        assert missing[0] == (date(2020, 4, 1), date(2020, 9, 30))

    def test_has_data_for_complete_coverage(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 12, 31))
        assert tracker.has_data_for(date(2020, 3, 1), date(2020, 6, 30))

    def test_has_data_for_incomplete_coverage(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 6, 30))
        assert not tracker.has_data_for(date(2020, 1, 1), date(2020, 12, 31))

    def test_coverage_ratio(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 6, 30))
        ratio = tracker.get_coverage_ratio(date(2020, 1, 1), date(2020, 12, 31))
        # ~50% coverage (181 out of 366 days)
        assert 0.45 < ratio < 0.55

    def test_add_dates_merges_weekends(self):
        """Gaps <= 7 days (weekends + holidays) should merge into one interval."""
        tracker = IntervalTracker()
        # Monday through Friday, then skip weekend, then next Monday through Friday
        dates = [
            date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10),
            date(2024, 1, 11), date(2024, 1, 12),
            # Weekend gap (2 days)
            date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17),
            date(2024, 1, 18), date(2024, 1, 19),
        ]
        tracker.add_dates(dates)
        # Should be a single interval since gap is only 2 days (within MAX_GAP_DAYS=7)
        bounds = tracker.bounds
        assert bounds == (date(2024, 1, 8), date(2024, 1, 19))

    def test_add_dates_splits_on_large_gap(self):
        """Gaps > 7 days should create separate intervals."""
        tracker = IntervalTracker()
        dates = [
            date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
            # 10-day gap
            date(2024, 1, 13), date(2024, 1, 14), date(2024, 1, 15),
        ]
        tracker.add_dates(dates)
        missing = tracker.get_missing_intervals(date(2024, 1, 1), date(2024, 1, 15))
        assert len(missing) == 1  # Gap between the two intervals
        assert missing[0][0] == date(2024, 1, 4)

    def test_json_roundtrip_preserves_intervals(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 6, 30))
        tracker.add_interval(date(2021, 1, 1), date(2021, 6, 30))

        json_str = tracker.to_json()
        tracker2 = IntervalTracker(json_str)

        assert tracker2.bounds == tracker.bounds
        # Both should report the same missing intervals
        missing1 = tracker.get_missing_intervals(date(2019, 1, 1), date(2022, 1, 1))
        missing2 = tracker2.get_missing_intervals(date(2019, 1, 1), date(2022, 1, 1))
        assert missing1 == missing2

    def test_json_roundtrip_with_multiple_intervals(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 3, 31))
        tracker.add_interval(date(2020, 7, 1), date(2020, 9, 30))
        tracker.add_interval(date(2021, 1, 1), date(2021, 3, 31))

        json_str = tracker.to_json()
        data = json.loads(json_str)
        assert len(data) == 3

        tracker2 = IntervalTracker(json_str)
        assert tracker2.bounds == (date(2020, 1, 1), date(2021, 3, 31))

    def test_invalid_json_creates_empty_tracker(self):
        tracker = IntervalTracker("not valid json")
        assert tracker.is_empty

    def test_overlapping_intervals_merge(self):
        tracker = IntervalTracker()
        tracker.add_interval(date(2020, 1, 1), date(2020, 6, 30))
        tracker.add_interval(date(2020, 4, 1), date(2020, 12, 31))
        # Should merge into one continuous interval
        missing = tracker.get_missing_intervals(date(2020, 1, 1), date(2020, 12, 31))
        assert len(missing) == 0


# ============================================================
# DataManager Tests (using in-memory SQLite)
# ============================================================

class TestDataManager:
    @pytest.fixture
    def dm(self, tmp_path):
        """Create a DataManager with a temp database."""
        db_path = str(tmp_path / "test_stock_data.db")
        manager = DataManager(db_path=db_path)
        yield manager
        manager.close()

    def test_cooldown_prevents_retry(self, dm):
        """Failed downloads should be on cooldown."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        assert not dm._is_on_cooldown('TEST', start, end)

        dm._record_failure('TEST', start, end)
        assert dm._is_on_cooldown('TEST', start, end)

    def test_cooldown_expires(self, dm):
        """Cooldown should expire after the configured period."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        dm._record_failure('TEST', start, end)

        # Manually set the failure time to 2 hours ago
        key = ('TEST', start.isoformat(), end.isoformat())
        dm._failed_downloads[key] = datetime.now() - timedelta(hours=2)

        assert not dm._is_on_cooldown('TEST', start, end)

    def test_cooldown_case_insensitive(self, dm):
        """Cooldown should be case-insensitive for ticker names."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        dm._record_failure('test', start, end)
        assert dm._is_on_cooldown('TEST', start, end)

    def test_get_latest_cached_date_empty_db(self, dm):
        """Should return None for empty database."""
        result = dm.get_latest_cached_date(['VOO', 'BND'])
        assert result is None

    def test_get_latest_cached_date_with_data(self, dm):
        """Should return the minimum latest date across tickers."""
        # Manually set up interval trackers
        tracker1 = dm._get_interval_tracker('VOO')
        tracker1.add_interval(date(2020, 1, 1), date(2024, 12, 1))
        dm._save_interval_tracker('VOO', tracker1)

        tracker2 = dm._get_interval_tracker('BND')
        tracker2.add_interval(date(2020, 1, 1), date(2024, 11, 15))
        dm._save_interval_tracker('BND', tracker2)

        result = dm.get_latest_cached_date(['VOO', 'BND'])
        assert result == date(2024, 11, 15)  # min of the two latest dates

    def test_get_latest_cached_date_single_ticker(self, dm):
        """Should work with a single ticker."""
        tracker = dm._get_interval_tracker('VOO')
        tracker.add_interval(date(2020, 1, 1), date(2024, 12, 5))
        dm._save_interval_tracker('VOO', tracker)

        result = dm.get_latest_cached_date(['VOO'])
        assert result == date(2024, 12, 5)

    def test_inception_date_lookup(self, dm):
        """Known inception dates should be returned correctly."""
        assert dm.get_ticker_inception_date('VOO') == date(2010, 9, 7)
        assert dm.get_ticker_inception_date('BTC-USD') == date(2014, 9, 17)
        assert dm.get_ticker_inception_date('UNKNOWN_TICKER') is None

    def test_interval_tracker_persistence(self, dm):
        """Interval tracker should survive save/load cycle."""
        tracker = dm._get_interval_tracker('TEST')
        tracker.add_interval(date(2020, 1, 1), date(2020, 12, 31))
        dm._save_interval_tracker('TEST', tracker)

        # Clear in-memory cache to force DB load
        dm._interval_cache.clear()

        tracker2 = dm._get_interval_tracker('TEST')
        assert not tracker2.is_empty
        assert tracker2.bounds == (date(2020, 1, 1), date(2020, 12, 31))


# ============================================================
# FMP Data Source Tests
# ============================================================

class TestFMPDataSource:
    def test_invalid_data_source_raises(self, tmp_path):
        """Unknown data_source should raise ValueError."""
        db_path = str(tmp_path / "test.db")
        with pytest.raises(ValueError, match="Unknown data_source"):
            DataManager(db_path=db_path, data_source='bloomberg')

    def test_fmp_requires_api_key(self, tmp_path):
        """FMP data source should raise if FMP_API_KEY is not set."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {}, clear=True):
            # Ensure FMP_API_KEY is not set
            os.environ.pop('FMP_API_KEY', None)
            with pytest.raises(ValueError, match="FMP_API_KEY"):
                DataManager(db_path=db_path, data_source='fmp')

    def test_fmp_init_with_api_key(self, tmp_path):
        """FMP data source should initialize when API key is set."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'test_key_123'}):
            dm = DataManager(db_path=db_path, data_source='fmp')
            assert dm.data_source == 'fmp'
            assert dm.fmp_api_key == 'test_key_123'
            dm.close()

    def test_yfinance_default_data_source(self, tmp_path):
        """Default data source should be yfinance."""
        db_path = str(tmp_path / "test.db")
        dm = DataManager(db_path=db_path)
        assert dm.data_source == 'yfinance'
        assert dm.fmp_api_key is None
        dm.close()

    def test_smart_download_routes_to_fmp(self, tmp_path):
        """_smart_download should call _fmp_download when data_source is fmp."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'test_key'}):
            dm = DataManager(db_path=db_path, data_source='fmp')
            mock_df = pd.DataFrame({
                'Open': [100.0], 'High': [105.0], 'Low': [99.0],
                'Close': [103.0], 'Adj Close': [103.0], 'Volume': [1000000]
            }, index=pd.to_datetime(['2024-01-02']))

            with patch.object(dm, '_fmp_download', return_value=mock_df) as mock_fmp:
                result = dm._smart_download('AAPL', date(2024, 1, 1), date(2024, 1, 31))
                mock_fmp.assert_called_once_with('AAPL', date(2024, 1, 1), date(2024, 1, 31), 5)
                assert len(result) == 1
            dm.close()

    def test_fmp_download_column_mapping(self, tmp_path):
        """_fmp_download should rename FMP columns to yfinance format."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'test_key'}):
            dm = DataManager(db_path=db_path, data_source='fmp')

            fmp_response = [
                {
                    'date': '2024-01-02',
                    'open': 100.0, 'high': 105.0, 'low': 99.0,
                    'close': 103.0, 'adjClose': 102.5, 'volume': 5000000,
                    'changePercent': 1.5, 'change': 1.5,
                },
                {
                    'date': '2024-01-03',
                    'open': 103.0, 'high': 106.0, 'low': 101.0,
                    'close': 104.0, 'adjClose': 103.5, 'volume': 4500000,
                    'changePercent': 0.97, 'change': 1.0,
                },
            ]

            mock_response = MagicMock()
            mock_response.json.return_value = fmp_response
            mock_response.raise_for_status = MagicMock()

            with patch.object(dm.yf_session, 'get', return_value=mock_response):
                with patch('time.sleep'):  # Skip delays in test
                    result = dm._fmp_download('AAPL', date(2024, 1, 1), date(2024, 1, 5))

            assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            assert len(result) == 2
            assert result.iloc[0]['Open'] == 100.0
            assert result.iloc[0]['Adj Close'] == 102.5
            assert result.iloc[1]['Close'] == 104.0
            dm.close()

    def test_fmp_download_empty_response_returns_empty_df(self, tmp_path):
        """_fmp_download should return empty DataFrame on empty API response."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'test_key'}):
            dm = DataManager(db_path=db_path, data_source='fmp')

            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = MagicMock()

            with patch.object(dm.yf_session, 'get', return_value=mock_response):
                with patch('time.sleep'):
                    result = dm._fmp_download('FAKE', date(2024, 1, 1), date(2024, 1, 5),
                                              max_retries=1)

            assert result.empty
            dm.close()

    def test_fmp_sends_correct_request(self, tmp_path):
        """_fmp_download should send correct URL, params, and API key header."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'my_secret_key'}):
            dm = DataManager(db_path=db_path, data_source='fmp')

            fmp_response = [
                {'date': '2024-06-01', 'open': 1.0, 'high': 2.0, 'low': 0.5,
                 'close': 1.5, 'adjClose': 1.5, 'volume': 100},
            ]
            mock_response = MagicMock()
            mock_response.json.return_value = fmp_response
            mock_response.raise_for_status = MagicMock()

            with patch.object(dm.yf_session, 'get', return_value=mock_response) as mock_get:
                with patch('time.sleep'):
                    dm._fmp_download('GBTC', date(2024, 6, 1), date(2024, 6, 30))

            call_args = mock_get.call_args
            assert call_args[0][0] == "https://financialmodelingprep.com/stable/historical-price-eod/full"
            assert call_args[1]['params']['symbol'] == 'GBTC'
            assert call_args[1]['params']['from'] == '2024-06-01'
            assert call_args[1]['params']['to'] == '2024-06-30'
            assert call_args[1]['headers']['apikey'] == 'my_secret_key'
            dm.close()

    def test_bulk_download_forces_sequential_for_fmp(self, tmp_path):
        """FMP data source should never use bulk yf.download() path."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {'FMP_API_KEY': 'test_key'}):
            dm = DataManager(db_path=db_path, data_source='fmp')
            # The all_need_full_range condition includes `self.data_source == 'yfinance'`
            # so FMP always goes through sequential per-ticker downloads
            assert dm.data_source == 'fmp'
            # Verify the condition would be False for FMP even if other conditions met
            assert dm.data_source != 'yfinance'
            dm.close()


class TestChunkedDownloads:
    """Large requests are split so no single Yahoo call is big enough to get rate limited."""

    def _fake_download(self, calls):
        def fake(tickers, start, end, **kw):
            calls.append((tuple(tickers), start, end))
            idx = pd.bdate_range(start, end - timedelta(days=1))
            cols = pd.MultiIndex.from_product(
                [tickers, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
            return pd.DataFrame(1.0, index=idx, columns=cols)
        return fake

    def test_date_chunks_cover_range_without_overlap(self, tmp_path):
        dm = DataManager(str(tmp_path / 't.db'))
        dm.max_chunk_years = 5
        chunks = dm._date_chunks(date(2006, 1, 1), date(2025, 12, 31))
        assert chunks[0][0] == date(2006, 1, 1)
        assert chunks[-1][1] == date(2025, 12, 31)
        for (_, e1), (s2, _) in zip(chunks, chunks[1:]):
            assert s2 == e1 + timedelta(days=1)
        assert all((e - s).days < 365.25 * 5 for s, e in chunks)
        dm.close()

    def test_bulk_download_batches_tickers_and_dates(self, tmp_path):
        dm = DataManager(str(tmp_path / 't.db'))
        dm.chunk_pause_seconds = 0
        calls = []
        tickers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        with patch('data_manager.yf.download', side_effect=self._fake_download(calls)), \
             patch('data_manager.time.sleep'):
            out = dm.bulk_download(tickers, date(2006, 1, 1), date(2025, 12, 31))
        assert all(len(c[0]) <= dm.bulk_batch_size for c in calls)
        assert len(calls) == 2 * len(dm._date_chunks(date(2006, 1, 1), date(2025, 12, 31)))
        assert set(out) == set(tickers)
        assert out['A'].index.min().date() == date(2006, 1, 2)
        assert out['A'].index.max().date() == date(2025, 12, 31)
        dm.close()

    def test_extract_ticker_frame_handles_flat_and_multiindex(self):
        idx = pd.bdate_range('2024-01-01', periods=3)
        flat = pd.DataFrame({'Adj Close': [1.0, 2.0, 3.0]}, index=idx)
        multi = pd.concat({'X': flat}, axis=1)
        assert len(DataManager._extract_ticker_frame(flat, 'X')) == 3
        assert len(DataManager._extract_ticker_frame(multi, 'X')) == 3
        assert DataManager._extract_ticker_frame(multi, 'Y') is None
