# Monte Carlo Portfolio Simulator

Backtest portfolios and dynamic strategies on historical data, simulate their range of
outcomes with Monte Carlo, optimize weights, and compare everything in one interactive
HTML report modelled on [Portfolio Visualizer](https://www.portfoliovisualizer.com).

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Offline demo on generated prices (no network, numbers are NOT real)
python main.py config/synthetic_demo.py --synthetic

# Your own portfolios
cp config/example_config.py config/my_portfolios.py   # git-ignored
python main.py                                         # uses config/my_portfolios.py if present
```

Reports are written to `output/<timestamp>_<name>.html`. Add `--embed-plotlyjs` for a
report that opens without internet.

## Command line

| Command | What it does |
|---|---|
| `python main.py [config]` | Download/cache data, backtest, simulate, optimize, write the report |
| `--no-optimize` | Skip the optimization step |
| `--offline` | Use cached data only (no API calls) |
| `--force-download` | Re-download everything |
| `--data-source fmp` | Use Financial Modeling Prep (`FMP_API_KEY` env var) instead of yfinance |
| `--synthetic` | Use deterministic generated prices (offline testing) |
| `--coverage-report` | Show cached data coverage for the config's tickers |
| `python data_utils.py list / coverage / sync / download VOO,QQQ / info VOO` | Cache management |
| `python examples/compare_strategies.py [--synthetic]` | Strategy comparison scenarios |
| `python examples/timing_analysis.py --ticker VOO [--synthetic]` | Entry-point sensitivity and lump sum vs DCA |
| `python -m pytest tests -q` | Test suite |

## Configuration

Configs are Python files defining `config = RunConfig(...)`. Everything is a validated dataclass
in [`run_config.py`](run_config.py).

```python
from run_config import (RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig,
                        StrategyConfig, RebalanceConfig)

config = RunConfig(
    name="My analysis",
    benchmark_ticker='VOO',              # benchmark row + beta/alpha/capture ratios
    portfolios=[
        PortfolioConfig(name='60/40', allocations={'VOO': 0.6, 'BND': 0.4}),
        PortfolioConfig(name='Growth + BTC', allocations={'VOO': 0.5, 'QQQ': 0.4, 'BTC-USD': 0.1},
                        rebalance=RebalanceConfig('none', threshold=0.05)),   # 5% drift band
        PortfolioConfig(
            name='Drawdown protected',
            allocations={'VOO': 0.8, 'BND': 0.2},
            strategy=StrategyConfig(
                type='drawdown_protection',
                apply_to='rebalance',          # trade holdings, not just new cash
                check_frequency='daily',
                params={'threshold': 0.15, 'recovery_threshold': 0.05,
                        'risk_off_allocation': {'VOO': 0.3, 'BND': 0.7}})),
    ],
    simulation=SimulationConfig(
        initial_capital=10000,
        start_date='2015-01-01', end_date='2025-12-31',   # or lookback_years=10
        contribution_amount=500, contribution_frequency='monthly',   # or N trading days
        rebalance='annual',                # default for portfolios without their own
        risk_free_rate=0.02,
        years=10, simulations=10000,
        method='block_bootstrap',          # bootstrap | block_bootstrap | geometric_brownian | parametric
        seed=42, inflation_rate=0.0,
    ),
    optimization=OptimizationConfig(assets=['VOO', 'QQQ', 'BND'],
                                    active_strategies=['max_sharpe', 'min_volatility']),
)
```

### Rebalancing

`RebalanceConfig(frequency, threshold=None, transaction_cost_bps=0)`, or just the frequency
string: `'none'` (buy and hold), `'daily'`, `'weekly'`, `'monthly'`, `'quarterly'`, `'annual'`.
Historical runs rebalance on the first trading day of each period. A `threshold` trades
whenever any weight drifts that far from target, with or without a calendar.

### Dynamic strategies

| `type` | Behaviour | Key params |
|---|---|---|
| `static` | Base weights | |
| `buy_the_dip` | Overweight `target_ticker` while its price is > `threshold` below its peak | `target_ticker`, `threshold`, `aggressive_weight` |
| `crypto_opportunistic` | Crypto at `normal_weight` (or base), `dip_weight` during a dip | `crypto_ticker`, `dip_threshold`, `normal_weight`, `dip_weight` |
| `momentum` | Tilt toward trailing winners | `lookback`, `tilt_strength`, `min_weight` |
| `relative_value` | Tilt toward the most beaten-down assets | `threshold`, `max_tilt` |
| `volatility_target` | Scale risky weights to a target realized vol, rest to `safe_ticker` | `target_vol`, `lookback`, `equity_tickers`, `safe_ticker` |
| `drawdown_protection` | Switch to `risk_off_allocation` in a drawdown, back on recovery | `threshold`, `recovery_threshold`, `risk_off_allocation` |
| `dual_momentum` | Hold base weights while equity beats the safe asset over `lookback`, else 100% safe | `equity_ticker`, `safe_ticker`, `lookback` |

`apply_to` controls what a strategy steers:
- `'contributions'` (default): only new cash is split by the strategy. Pair it with `rebalance='none'`, or calendar rebalancing will undo the tilt.
- `'rebalance'`: holdings are traded to the strategy's target whenever it changes (checked every `check_frequency`) and on calendar rebalance dates.
- `'both'`: both of the above.

Indicators are computed from prices, never from holdings, so contributions can't fake a drawdown or a recovery.

### Parameter sweeps

Grid-search one or two strategy parameters on a fixed allocation. Every cell is a full backtest
(optionally plus a Monte Carlo) with the run's contributions and rebalancing, shown as a heatmap
against the same allocation without the strategy:

```python
from run_config import SweepConfig
config = RunConfig(
    ...,
    sweeps=[SweepConfig(
        name='Drawdown protection thresholds',
        allocations={'VOO': 0.8, 'BND': 0.2},
        strategy=StrategyConfig('drawdown_protection', apply_to='rebalance', check_frequency='daily',
                                params={'risk_off_allocation': {'BND': 1.0}}),
        grid={'threshold': [0.08, 0.12, 0.16, 0.20], 'recovery_threshold': [0.0, 0.03, 0.06]},
        simulations=300,     # optional: Monte Carlo per cell, same seed for every cell
    )],
)
```

Cells are fit to the same history they're scored on: prefer parameters inside a broad good region over the single best cell.

### Optimizer checks

With `optimization` configured, the report also includes:
- **Walk-forward:** at each refit the optimizer sees only the previous `train_years` (default 3), and its weights are traded for the next `test_years` (default 1). This is compared with the full-history optimum over the same window, which shows how much of the optimized rows' performance is hindsight. Turn it off with `walk_forward=False`.
- **Efficient frontier:** the long-only mean-variance frontier of the optimization assets, with every fixed-weight portfolio built from them placed on the chart (`efficient_frontier=False` to skip).

## How the numbers are computed

One engine ([`engine.py`](engine.py)) runs both the historical backtest (one path) and the
Monte Carlo (many paths), so a portfolio's rules behave identically in both. All report
metrics come from [`quant_analytics.py`](quant_analytics.py):

- **CAGR** is time-weighted and annualized by calendar span. Contributions never count as returns; **IRR** is the money-weighted return including them.
- **Stdev, Sharpe, Sortino, beta, capture ratios, VaR** use monthly returns, matching Portfolio Visualizer. Sharpe and Sortino subtract `risk_free_rate`.
- **Max drawdown** uses daily values, so it can be deeper than PV's month-end figure.
- **Best/Worst year** are calendar-year returns over full years.
- Prices are aligned across assets **before** returns are computed, so mixing 24/7 crypto with exchange-traded funds keeps weekend moves.
- Simulated years have as many trading days as the history has per year (252 for ETFs, 365 for crypto-only).
- **block_bootstrap** resamples runs of consecutive days (`block_size`, default 21), keeping trends and volatility clustering. Prefer it when testing drawdown or momentum strategies; plain `bootstrap` scrambles those patterns away.

## Data

yfinance (default) or FMP downloads are cached in `stock_data.db` (SQLite) with interval
tracking, so only missing ranges are fetched. Large requests are split into 5-ticker batches
and 5-year chunks with pauses between them, because big single requests are what trigger
Yahoo's rate limits. Tune with `DataManager.bulk_batch_size`, `max_chunk_years` and
`chunk_pause_seconds`.

## Project layout

```
main.py                  pipeline: data -> backtest + Monte Carlo per portfolio -> optimize -> report
run_config.py            config dataclasses (RunConfig, PortfolioConfig, SimulationConfig, ...)
engine.py                shared day-step engine, alignment, schedules, return generators
backtester.py            historical runs on the engine + metrics
portfolio_simulator.py   Monte Carlo runs on the engine + percentile stats
quant_analytics.py       every performance metric
strategies.py            dynamic strategies + registry
portfolio_optimizer.py   SciPy SLSQP optimization (in-sample) + efficient frontier
sweeps.py                strategy parameter grid search
walk_forward.py          out-of-sample refit check for the optimizers
visualizations.py        HTML report
data_manager.py          download + SQLite cache;  data_utils.py: cache CLI
synthetic_data.py        generated prices for offline runs/tests
pv_compat.py             Portfolio Visualizer CSV/URL export
config/                  example configs (my_*.py and other personal configs are git-ignored)
examples/                standalone analysis scripts
tests/                   pytest suite (PV comparison tests need stock_data.db)
```
