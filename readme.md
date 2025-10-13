# Monte Carlo Portfolio Simulator

Monte Carlo simulation tool for portfolio analysis with optimization and configuration-based portfolio definitions.

## Features

- 10,000 Monte Carlo simulations (industry standard)
- Geometric Brownian Motion & Bootstrap methods
- Portfolio optimization with custom objectives
- Interactive Plotly visualizations with synchronized legends
- SQLite database for caching historical data
- **Configuration-based portfolio definitions** (git-ignored for privacy)

## Installation

```bash
uv pip install -r requirements.txt
```

## Quick Start

### 1. Create Your Configuration

```bash
# Create config directory
mkdir -p config
touch config/__init__.py

# Copy example config
cp config/example_config.py config/my_portfolios.py

# Edit your portfolios
nano config/my_portfolios.py
```

### 2. Run Simulation

```bash
# Use your personal config (config/my_portfolios.py)
python main.py

# Or specify a config file
python main.py --config config/my_portfolios.py

# List available configs
python main.py --list-configs
```

## Configuration

All portfolio definitions live in Python configuration files in the `config/` directory.

### Personal Config (Git-Ignored)

Create `config/my_portfolios.py` - this file is automatically git-ignored:

```python
SIMULATION_CONFIG = {
    'initial_capital': 100000,
    'years': 10,
    'simulations': 10000,
    'method': 'bootstrap'  # or 'geometric_brownian'
}

ASSETS = {
    'voo': {
        'ticker': 'VOO',
        'name': 'VOO (S&P 500)',
        'lookback_years': 10
    },
    'qqq': {
        'ticker': 'QQQ',
        'name': 'QQQ (Nasdaq)',
        'lookback_years': 10
    },
}

PORTFOLIOS = [
    {
        'name': 'My Portfolio',
        'description': 'Custom allocation',
        'allocations': {
            'voo': 0.6,
            'qqq': 0.4
        }
    }
]

OPTIMIZATION_CONFIG = {
    'enabled': True,
    'method': 'grid_search',
    'objective_weights': {
        'return': 0.50,
        'sharpe': 0.20,
        'drawdown': 0.30
    },
    'optimize_assets': ['voo', 'qqq']
}
```

### Git-Ignored Patterns

Files matching these patterns are automatically ignored:
- `config/my_*.py`
- `config/personal_*.py`
- `config/*_private.py`

The `config/example_config.py` template **IS** committed to the repo.

## File Structure

```
├── config/
│   ├── __init__.py
│   ├── example_config.py      # ✅ Committed (template)
│   └── my_portfolios.py        # ❌ Git-ignored (your portfolios)
├── config_loader.py
├── data_manager.py
├── portfolio_simulator.py
├── portfolio_optimizer.py
├── visualizations.py
├── main.py
├── .gitignore
└── README.md
```

## Validation

Compare results with:
- [Portfolio Visualizer](https://www.portfoliovisualizer.com/monte-carlo-simulation)
- [Bogleheads Tools](https://www.bogleheads.org/wiki/Using_open_source_software_for_portfolio_analysis)

## Best Practices Implemented

✓ 10,000 simulations (industry standard)
✓ Geometric Brownian Motion (recommended)
✓ Bootstrap resampling (preserves distribution)
✓ 10-year lookback period
✓ Adjusted close prices
✓ Full configuration tracking
✓ Synchronized interactive visualizations

## Key Features

### 1. Synchronized Legends
- Click any legend item → ALL 9 charts update together
- Double-click to isolate a single portfolio
- All subplots respond to legend interactions

### 2. Portfolio Optimization
- Grid search for 2-3 assets
- Random search for 4+ assets
- Custom objective functions
- Results saved to database

### 3. Database Storage
- Stock price data cached in SQLite
- Optimization results with timestamps
- Data quality metrics tracked

### 4. Interactive Visualizations
- Sample trajectories
- Return distributions
- Risk metrics (Sharpe, Sortino)
- Percentile fan charts
- Risk-return profiles
- Probability analysis

## Example Usage

```bash
# First run - downloads data and saves to database
python main.py

# Subsequent runs - instant loading from database
python main.py

# Try different config
python main.py --config config/aggressive.py
```

## Output Files

- `portfolio_dashboard.html` - Interactive dashboard
- `stock_data.db` - SQLite database with cached data
- Your `config/my_portfolios.py` stays private (git-ignored)

## Privacy

Your personal portfolios in `config/my_portfolios.py` will **NEVER** be committed to git! The `.gitignore` protects your privacy while allowing you to share the tool.

## License

MIT
