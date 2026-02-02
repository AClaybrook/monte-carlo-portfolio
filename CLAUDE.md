# Monte Carlo Portfolio Simulator

Portfolio analysis tool inspired by [PortfolioVisualizer.com](https://www.portfoliovisualizer.com). Runs Monte Carlo simulations, historical backtests, and portfolio optimization to compare investment strategies.

## Tech Stack

- **Python 3.11** with type-safe dataclass configuration
- **Data**: pandas, numpy, scipy for numerical operations
- **Market Data**: yfinance with SQLite caching via SQLAlchemy
- **Visualization**: Plotly for interactive HTML dashboards
- **Database**: SQLite (`stock_data.db`) for price caching and results

## Project Structure

```
├── main.py                    # Entry point - orchestrates analysis pipeline
├── run_config.py              # Dataclass config definitions (RunConfig, PortfolioConfig, etc.)
├── data_manager.py            # Data fetching, caching, interval-based smart downloads
├── portfolio_simulator.py     # Monte Carlo simulation engine
├── portfolio_optimizer.py     # SciPy-based portfolio optimization
├── backtester.py              # Historical backtesting with strategy support
├── strategies.py              # Dynamic allocation strategies (buy-the-dip, momentum, etc.)
├── visualizations.py          # Plotly HTML report generation
├── config/                    # Portfolio configuration files
│   ├── example_config.py      # Example with optimization
│   └── strategy_example.py    # Example with dynamic strategies
├── output/                    # Generated HTML reports
└── stock_data.db              # SQLite cache for market data
```

## Essential Commands



```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with default config (config/my_portfolios.py or config/example_config.py)
python main.py

# Run with specific config
python main.py config/example_config.py

# Force re-download all market data
python main.py --force-download

# Check data coverage for configured tickers
python main.py --coverage-report

# Skip optimization step
python main.py --no-optimize

# Data management utility (data_utils.py)
python data_utils.py list                      # List all tickers in DB
python data_utils.py coverage                  # Show data coverage report
python data_utils.py sync --since 2024-12-05   # Update all tickers stale since Dec 5
python data_utils.py sync --dry-run            # Preview what would be updated
python data_utils.py download VOO,QQQ,BND      # Bulk download specific tickers
python data_utils.py info VOO                  # Show ticker info and gaps
```

## Configuration

All configuration uses type-safe dataclasses. Create configs in `config/my_*.py` (gitignored).

Key config classes ([run_config.py:8-129](run_config.py#L8-L129)):
- `RunConfig` - Top-level container
- `PortfolioConfig` - Portfolio name, allocations, optional strategy
- `SimulationConfig` - Capital, years, simulation count, method, **start_date/end_date**
- `OptimizationConfig` - Assets to optimize, strategies (max_sharpe, min_volatility, etc.)
- `StrategyConfig` - Dynamic allocation strategy type and params

### SimulationConfig Date Options

```python
# Option 1: Explicit date range (most control)
SimulationConfig(start_date='2020-01-01', end_date='2024-12-31')

# Option 2: End date with lookback period
SimulationConfig(end_date='2024-12-31', lookback_years=5)

# Option 3: Lookback from today (default behavior)
SimulationConfig(lookback_years=10)
```

## Key Module Entry Points

| Module | Main Class/Function | Purpose |
|--------|---------------------|---------|
| [main.py:47](main.py#L47) | `main()` | Orchestrates full analysis pipeline |
| [data_manager.py:237](data_manager.py#L237) | `DataManager` | Market data with interval caching |
| [portfolio_simulator.py:12](portfolio_simulator.py#L12) | `PortfolioSimulator` | Monte Carlo engine |
| [portfolio_optimizer.py:17](portfolio_optimizer.py#L17) | `PortfolioOptimizer` | SciPy optimization |
| [backtester.py:10](backtester.py#L10) | `Backtester` | Historical backtesting |
| [strategies.py:61](strategies.py#L61) | `AllocationStrategy` | Base class for strategies |
| [visualizations.py:12](visualizations.py#L12) | `PortfolioVisualizer` | HTML report generation |

## Simulation Methods

- **bootstrap** (default): Resamples historical returns, preserves fat tails
- **geometric_brownian**: Industry-standard GBM model
- **parametric**: Simple normal distribution

## Dynamic Strategies

Strategies modify DCA allocation based on market conditions ([strategies.py:61-622](strategies.py#L61-L622)):
- `static` - Fixed allocation
- `buy_the_dip` - Increase allocation when target asset drops
- `crypto_opportunistic` - Buy more crypto during drawdowns
- `momentum` - Tilt toward positive momentum assets
- `volatility_target` - Adjust to maintain target volatility
- `drawdown_protection` - Shift to defensive allocation during crashes
- `relative_value` - Buy most beaten-down assets

## Output

Reports saved to `output/` as interactive HTML dashboards with:
- Monte Carlo probability distributions
- Historical backtest equity curves
- Drawdown analysis
- Rolling returns
- Risk-return scatter plots

## Additional Documentation

See `.claude/docs/` for detailed patterns:
- [architectural_patterns.md](.claude/docs/architectural_patterns.md) - Design patterns and code conventions

## Recent Improvements (2026-02)

### Data Management
- Added `sync` command to `data_utils.py` for bulk updating stale data
- Added `start_date` and `end_date` to `SimulationConfig` for explicit date ranges
- Use `--offline` flag to skip yfinance calls and use cached data only

### Performance & Stability
- Fixed overflow in `portfolio_optimizer.py` custom_objective using log-returns for drawdown
- Fixed overflow in `portfolio_simulator.py` using float64 for cumprod operations
- yfinance rate limiting is aggressive - use `--sequential` flag or try during off-peak hours

### Known Issues
- yfinance may heavily rate limit downloads, especially on WSL/Linux
- For best results, use explicit `end_date` matching your cached data (check with `python data_utils.py coverage`)

### Tests
- Run: python -m pytest tests/ -v
- Run tests after code changes to python files
