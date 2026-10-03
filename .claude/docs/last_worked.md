# Last Worked Notes

## 2026-10 (cont.): Research tools

- `sweeps.py` + `SweepConfig`: 1-2 parameter strategy grids, heatmaps vs the no-strategy baseline,
  optional Monte Carlo per cell with a shared seed (common random numbers).
- `walk_forward.py`: optimizer refit on the trailing `train_years`, traded for `test_years`, compared
  with full-history weights over the same window. Uses `ScheduledWeightsStrategy` through the engine.
- Optimizer data cache now keys on each asset's data span (it reused stale data for sliced windows).
- `PortfolioOptimizer.efficient_frontier` + report chart.
- `examples/timing_analysis.py` uses calendar dates and `quant_analytics`.
- Skipped on purpose: moving modules into a package (would break `from run_config import ...` in
  personal configs for no functional gain).

## 2026-10: Engine, metrics and report overhaul

Why: report numbers were often wrong, and the same portfolio produced different
results depending on code path.

Bugs fixed:
- Simulator joined *returns* across assets, dropping crypto weekend moves (BTC mean
  understated by ~1/3). Prices are now aligned first (`engine.align_asset_prices`).
- The static backtest and fast MC rebalanced daily (labelled "Buy and Hold"); the strategy
  paths never rebalanced. Replaced by one engine with explicit `RebalanceConfig`.
- DCA contributions were counted as returns (inflated vol/best year); DCA CAGR used a made-up
  formula. Metrics now use the time-weighted index; IRR (XIRR) is separate.
- Sharpe had no risk-free rate, Sortino used std of negative days, Best/Worst Year were rolling
  252-day sums, annualization used rows/252. All rewritten in `quant_analytics.py` (PV conventions).
- MC "Sharpe/Sortino" were cross-sectional stats of CAGR; the risk-return chart plotted terminal
  CAGR dispersion as "volatility". Removed/replaced with per-path realized vol and percentiles.
- Optimizer rows used 1000-sim draft results and lump-sum backtests; now re-evaluated with the
  full simulation, same cash flows, rebalancing and benchmark as every row.
- Strategies: indicators from prices not holdings; `drawdown_protection` hysteresis on the
  base-weight portfolio's drawdown (a portfolio sitting in bonds never "recovers");
  `dual_momentum` lookback honoured; `volatility_target` uses realized portfolio vol + `safe_ticker`.
- PV links: rounded weights now sum to 100; dates/amounts come from the run.

New: `StrategyConfig.apply_to='rebalance'` (signal-driven conditional rebalancing), drift bands,
transaction costs, `block_bootstrap`, `seed`, `inflation_rate`, `risk_free_rate`,
`contribution_frequency='monthly'`, `--synthetic`, `--embed-plotlyjs`, rebuilt report.

Data: bulk downloads split into 5-ticker × 5-year chunks; yfinance no longer receives a
custom `requests.Session` (rejected by yfinance >= 0.2.58).

Not verified against live data yet (developed on synthetic data). Run
`python -m pytest tests/test_pv_benchmark.py -v -s` with a populated `stock_data.db`:
`TestBacktesterMatchesPV` compares the new metrics with saved PV reference values.

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
         ├─ All tickers need full range? → yf.download() in 5-ticker × 5-year chunks
         │
         └─ Otherwise → sequential per-ticker downloads for each gap
              │
              ├─ _is_on_cooldown()? → skip
              │
              └─ _smart_download() → 5-year chunks, each with retries + backoff
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
