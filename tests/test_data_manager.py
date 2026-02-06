"""
Tests for IntervalTracker and DataManager.
Validates interval-based caching logic, cooldown tracking, and helper methods.
"""
import pytest
import json
import sys
import os
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
