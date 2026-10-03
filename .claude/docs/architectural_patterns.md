# Architectural Patterns

## Dataclass configuration

All configuration lives in `run_config.py` as dataclasses validated in `__post_init__` (raise `ValueError` with a clear message). Shorthands are coerced there too: a rebalance frequency string becomes a `RebalanceConfig`. Keep new fields optional with backward-compatible defaults; users keep personal configs outside git.

## One engine, two drivers

`engine.run_engine` is the only place portfolio mechanics live. `Backtester` feeds it one historical path (`HistoricalReturns`) with calendar schedules; `PortfolioSimulator` feeds it `SimulatedReturns` with evenly spaced schedules. New mechanics (a new rebalance trigger, a cost model) go into the engine so both drivers get them.

Per-day work must stay vectorized over paths and cheap: row sums use `holdings @ ones_k` (much faster than `.sum(axis=1)` for few assets), and indicator bookkeeping runs only when a strategy is present.

## Strategy pattern + registry

`AllocationStrategy.get_allocation(context) -> (paths, assets)` weights. Stateful strategies (e.g. drawdown hysteresis) keep per-path arrays and clear them in `reset(n_paths, tickers)`.

- Read price-based context: `current_drawdowns`, `reference_drawdown`, `trailing_return(n)`, `trailing_portfolio_volatility(n)`.
- Set `lookback_days` so the engine keeps a long enough return buffer.
- Set `uses_rolling_stats = False` unless the strategy reads `rolling_*`/`momentum_score`/`portfolio_volatility` (they are expensive on daily checks).
- Register config names in `STRATEGY_BUILDERS`; `StrategyConfig` validates against it.

## Metrics are computed once

`quant_analytics.compute_performance(balance, twr_index, contributions, benchmark_index, ...)` produces every headline number. The backtester attaches it; the report only formats it. Conventions (calendar-span CAGR, monthly-return risk stats, risk-free-adjusted Sharpe/Sortino, TWR vs IRR) are documented at the top of that module.

## Data access

`DataManager` wraps the API clients and the SQLite cache. `IntervalTracker` (portion intervals) records which dates are cached per ticker so only gaps are downloaded. Large downloads are batched and chunked. `SyntheticDataManager` implements the subset main.py needs, so the whole pipeline runs offline.

## Report

`PortfolioVisualizer` builds Plotly figures with light colors and a light→dark color map; a small script swaps colors for dark mode and handles per-portfolio selectors (traces carry `meta`, shapes carry `name`). Tables are the accessible twin of every chart. Text is escaped with `html.escape`; only cells built by the module itself are raw HTML.
