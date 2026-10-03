# Monte Carlo Portfolio Simulator

Portfolio analysis tool inspired by [PortfolioVisualizer.com](https://www.portfoliovisualizer.com): historical backtests, Monte Carlo simulation, dynamic/conditional rebalancing strategies, and SciPy optimization, compared in one interactive HTML report.

## Tech Stack

- Python 3.11, dataclass configuration
- pandas / numpy / scipy; Plotly for the report
- yfinance (default) or Financial Modeling Prep, cached in SQLite (`stock_data.db`) via SQLAlchemy

## Project Structure

```
main.py                  Pipeline: data -> evaluate_portfolio() per portfolio -> optimize -> report
run_config.py            RunConfig, PortfolioConfig, SimulationConfig, RebalanceConfig, StrategyConfig, ...
engine.py                Shared day-step engine (run_engine), price alignment, schedules, return generators
backtester.py            Backtester.run_backtest: one historical path through the engine + metrics
portfolio_simulator.py   PortfolioSimulator.simulate_portfolio: N simulated paths through the engine
quant_analytics.py       Every metric (compute_performance, CAGR, Sharpe, drawdowns, XIRR, ...)
strategies.py            AllocationStrategy subclasses + STRATEGY_BUILDERS registry
portfolio_optimizer.py   SLSQP optimizers (max Sharpe, min vol, risk parity, Sortino, custom) + efficient_frontier
sweeps.py                run_sweep: strategy parameter grid (SweepConfig) -> backtest (+MC) per cell
walk_forward.py          run_walk_forward: refit on trailing window, trade out of sample (ScheduledWeightsStrategy)
visualizations.py        PortfolioVisualizer.generate_html_report (presentation only)
data_manager.py          Downloads + interval-tracked SQLite cache; data_utils.py is its CLI
synthetic_data.py        Deterministic generated prices (SyntheticDataManager) for offline runs/tests
pv_compat.py             Portfolio Visualizer CSV/URL export
config/                  example_config.py, strategy_example.py, synthetic_demo.py (others git-ignored)
examples/                compare_strategies.py, timing_analysis.py
db_scripts/              DB repair/migration one-offs
tests/                   pytest suite
```

## Essential Commands

```bash
source venv/bin/activate
pip install -r requirements.txt

python main.py                                  # config/my_portfolios.py, else example_config.py
python main.py config/example_config.py
python main.py config/synthetic_demo.py --synthetic   # offline, ~5s, fake prices
python main.py --offline                        # cached data only
python main.py --no-optimize --force-download --coverage-report --embed-plotlyjs
python main.py --data-source fmp                # needs FMP_API_KEY

python data_utils.py list | coverage | sync --since 2024-12-05 | download VOO,QQQ | info VOO
python examples/compare_strategies.py --synthetic --test broad

python -m pytest tests -q
```

## Key Design Rules

- **One engine.** Backtests and Monte Carlo both go through `engine.run_engine`; never add a separate fast path with different math.
- **One metrics module.** Every reported number comes from `quant_analytics`. The report never recomputes financial metrics.
- **Align prices, then compute returns** (`engine.align_asset_prices`), so crypto weekend moves land in Monday's return.
- **Time-weighted vs money-weighted.** Return metrics use the TWR index (contributions removed); IRR is reported separately.
- **Annualize by calendar span** (`quant_analytics.years_between` / `infer_periods_per_year`), never rows / 252.
- **Strategy indicators come from prices**, not holdings. Strategies are vectorized over paths and get `reset()` at the start of every run.
- Library defaults are buy-and-hold (`rebalance=None`); configs default to annual rebalancing (`SimulationConfig.rebalance`).

## Configuration Highlights

- `SimulationConfig`: `start_date`/`end_date` or `lookback_years`; `contribution_amount` + `contribution_frequency` (int trading days or `'monthly'`/`'quarterly'`/`'annual'`); `rebalance`; `risk_free_rate`; `method` (`bootstrap`, `block_bootstrap`, `geometric_brownian`, `parametric`); `block_size`; `seed`; `inflation_rate`.
- `PortfolioConfig.rebalance` overrides the default: a `RebalanceConfig(frequency, threshold, transaction_cost_bps)` or a frequency string.
- `StrategyConfig(type, params, apply_to='contributions'|'rebalance'|'both', check_frequency='monthly')`. Types are the keys of `strategies.STRATEGY_BUILDERS`.
- `RunConfig.benchmark_ticker` (defaults to `optimization.benchmark_ticker`).
- `RunConfig.sweeps`: list of `SweepConfig(name, allocations, strategy, grid={1-2 params: values}, rebalance, simulations)`.
- `OptimizationConfig.walk_forward` / `train_years` / `test_years` / `efficient_frontier`.
- `VisualizationConfig.embed_plotlyjs` for offline reports.

## Data Notes

- Yahoo rate-limits large requests (worse on WSL). `bulk_download` splits work into `bulk_batch_size` tickers × `max_chunk_years` date chunks with `chunk_pause_seconds` pauses; per-ticker downloads are chunked too.
- yfinance must NOT be given a `requests.Session` (yfinance >= 0.2.58 requires its own curl_cffi session).
- Failed intervals have a 1-hour in-memory cooldown. `db_scripts/repair_metadata.py` fixes interval metadata drift.
- For reproducible comparisons, set an explicit `end_date` that your cache covers (`python data_utils.py coverage`).

## Testing

- `tests/test_performance.py`: golden metric values; `test_engine.py`: rebalancing, signals, costs, alignment, MC/backtest consistency; `test_monte_carlo.py`: generators, seeds, inflation; `test_sweeps.py`, `test_walk_forward.py`, `test_frontier.py`; `test_report.py`: end-to-end synthetic run.
- `test_pv_benchmark.py` / `test_historical_validation.py` compare against Portfolio Visualizer reference numbers and need `stock_data.db`; they skip otherwise.
- Test data that means "trading days" should use business-day dates (`freq='B'`); calendar-day dates are annualized as 365/yr.

## Additional Documentation

- [.claude/docs/architecture_diagram.md](docs/architecture_diagram.md): pipeline and module map
- [.claude/docs/architectural_patterns.md](docs/architectural_patterns.md): conventions
- [.claude/docs/last_worked.md](docs/last_worked.md): recent change log and data-fetching notes
