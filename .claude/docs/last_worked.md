# Last Worked Notes

## 2026-02-06: Data Manager Caching Fixes

### Problem
yfinance rate-limits aggressively on WSL/Linux. The bulk download path in `data_manager.py` was discarding per-ticker missing intervals and re-requesting full date ranges for every ticker, wasting API calls.

### What Was Fixed

1. **bulk_download() now uses per-ticker intervals** (`data_manager.py`)
   - Previously: bulk path passed global `start_date`/`end_date` to `_bulk_download_and_save()`, ignoring the per-ticker missing intervals it had just computed
   - Now: only uses `yf.download()` bulk path when ALL tickers need the same full range (force_update / fresh DB). Otherwise uses sequential per-ticker interval-targeted downloads
   - Added download plan summary printed before any API calls begin

2. **_bulk_download_and_save() fallback uses per-ticker intervals** (`data_manager.py`)
   - Previously: fallback to sequential used the global date range
   - Now: accepts `ticker_intervals` dict and uses per-ticker ranges in fallback

3. **Offline mode dynamic date detection** (`main.py`)
   - Previously: hardcoded `end_date = date(2025, 12, 5)`
   - Now: uses `DataManager.get_latest_cached_date()` to dynamically find the latest cached date

4. **Failed download cooldown** (`data_manager.py`)
   - Added in-memory cooldown tracking (1 hour) so failed intervals aren't retried within the same session
   - Resets on process restart (intentional - user re-running is explicitly choosing to retry)

5. **Column type fix** (`data_manager.py`, `db_scripts/migrate_db.py`)
   - Changed `data_intervals_json` from `String(2000)` to `Text`
   - SQLite treats these identically, but prevents issues on other DB engines

### Data Fetching Architecture (How It Works)

```
User Request (start_date, end_date, tickers)
    │
    ▼
bulk_download() — for each ticker:
    │
    ├─ get_ticker_inception_date() → adjust start_date
    │
    ├─ _get_interval_tracker() → load IntervalTracker from DB metadata
    │
    ├─ tracker.get_missing_intervals(start, end) → list of (start, end) gaps
    │
    ├─ If no gaps → "Using cached data (100% coverage)"
    │
    └─ If gaps exist:
         │
         ├─ All tickers need full range? → yf.download() bulk
         │
         └─ Otherwise → sequential per-ticker downloads for each gap
              │
              ├─ _is_on_cooldown()? → skip
              │
              └─ _smart_download() → yfinance with retries + backoff
                   │
                   └─ _save_to_db() + tracker.add_dates()
    │
    ▼
_get_from_db() → return data from SQLite (always from DB, never raw yfinance)
```

### Key Classes

- **IntervalTracker** (`data_manager.py:106`): Uses `portion` library for interval arithmetic. Tracks exactly which date ranges are cached per ticker. `MAX_GAP_DAYS = 7` (bridges weekends + holidays).
- **DataManager** (`data_manager.py:236`): Orchestrates fetch/cache/retrieve. In-memory interval cache + SQLite persistence.
- **TickerMetadata** (`data_manager.py:52`): DB model storing `data_intervals_json`, `first_valid_date`, `last_valid_date`.

### Useful Debugging Commands

```bash
# Check what data is cached
python data_utils.py coverage

# See detailed info for a specific ticker
python data_utils.py info VOO

# Find data gaps
python data_utils.py gaps VOO --max-gap 5

# Preview what a sync would download (no API calls)
python data_utils.py sync --dry-run

# Force sequential downloads (avoids rate limits)
python data_utils.py sync --sequential

# Run offline with cached data only
python main.py --offline

# Force re-download everything
python main.py --force-download
```

### Known Quirks

- yfinance returns data with timezone-aware timestamps that need `.date()` normalization
- `yf.download()` with `group_by='ticker'` returns multi-level columns for multiple tickers but flat columns for single tickers — handled with `len(tickers) == 1` check
- Weekend/holiday gaps up to 7 days are automatically merged by IntervalTracker
- The `requests_cache` optional dependency caches HTTP responses for 6 hours
