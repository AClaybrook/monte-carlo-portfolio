# Monte Carlo Portfolio Simulator

Type-safe, dataclass-based configuration system for portfolio analysis.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your config
cp config/example_simple.py config/my_portfolios.py

# 3. Edit config/my_portfolios.py with your portfolios

# 4. Run!
python main.py
```

## Configuration System

All configuration uses **type-safe dataclasses** with clear documentation and validation.

### Simple Example

```python
# config/my_portfolios.py
from run_config import RunConfig, AssetConfig, PortfolioConfig

config = RunConfig(
    name="My Analysis",

    # Step 1: Define assets
    assets=[
        AssetConfig(ticker='VOO', name='S&P 500'),
        AssetConfig(ticker='BND', name='Bonds'),
    ],

    # Step 2: Define portfolios to test
    portfolios=[
        PortfolioConfig(
            name='80/20',
            allocations={'voo': 0.8, 'bnd': 0.2}
        ),
    ]
)
```

### With Optimization

```python
# config/my_portfolios.py
from run_config import (
    RunConfig, AssetConfig, PortfolioConfig,
    SimulationConfig, OptimizationConfig
)

config = RunConfig(
    name="Optimized Portfolio Analysis",

    assets=[
        AssetConfig(ticker='VOO'),
        AssetConfig(ticker='QQQ'),
        AssetConfig(ticker='BND'),
    ],

    portfolios=[
        PortfolioConfig(name='Balanced', allocations={'voo': 0.5, 'qqq': 0.3, 'bnd': 0.2}),
    ],

    # Customize simulation
    simulation=SimulationConfig(
        initial_capital=100000,
        years=10,
        simulations=10000,
        method='bootstrap'  # or 'geometric_brownian'
    ),

    # Enable optimization
    optimization=OptimizationConfig(
        assets=['voo', 'qqq', 'bnd'],
        method='grid_search',
        objective_weights={
            'return': 0.50,    # 50% weight on returns
            'sharpe': 0.20,    # 20% weight on risk-adjusted returns
            'drawdown': 0.30   # 30% weight on limiting losses
        }
    )
)
```

## Configuration Classes

### AssetConfig
```python
AssetConfig(
    ticker='VOO',              # Required: Yahoo Finance ticker
    name='S&P 500',           # Optional: Display name
    lookback_years=10         # Optional: Years of history (default 10)
)
```

### PortfolioConfig
```python
PortfolioConfig(
    name='My Portfolio',       # Required: Portfolio name
    allocations={              # Required: Asset weights (must sum to 1.0)
        'voo': 0.6,
        'bnd': 0.4
    },
    description='60/40'       # Optional: Description
)
```

### SimulationConfig
```python
SimulationConfig(
    initial_capital=100000,    # Starting portfolio value
    years=10,                  # Investment horizon
    simulations=10000,         # Number of Monte Carlo runs
    method='bootstrap'         # 'bootstrap', 'geometric_brownian', 'parametric'
)
```

### OptimizationConfig
```python
OptimizationConfig(
    assets=['voo', 'qqq'],     # Assets to optimize
    method='grid_search',      # 'grid_search' or 'random_search'
    objective_weights={        # What you care about (must sum to 1.0)
        'return': 0.50,
        'sharpe': 0.20,
        'drawdown': 0.30
    },
    grid_points=6,             # Granularity for grid_search
    n_iterations=50,           # Iterations for random_search
    top_n=5                    # Number of results to keep
)
```

## Usage

```bash
# Use personal config
python main.py

# Use specific config file
python main.py config/example_simple.py
python main.py config/aggressive.py

# Dry run (show config without running)
python main.py --dry-run
```

## Understanding the Output

### Simulation Results
- **Median Final Value**: Middle outcome across 10,000 simulations
- **Median CAGR**: Compound annual growth rate
- **Median Max Drawdown**: Worst loss from peak
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Downside risk-adjusted return (higher is better)
- **P(Loss)**: Probability of losing money
- **P(Double)**: Probability of doubling investment

### Interactive Dashboard
- Click legend items → ALL 9 charts update together
- Double-click → Isolate single portfolio
- Hover → Detailed information
- Zoom/pan → Explore time periods

## File Structure

```
├── config/
│   ├── __init__.py
│   ├── example_simple.py          # ✅ Simple template (committed)
│   ├── example_with_optimization.py # ✅ Full example (committed)
│   └── my_portfolios.py           # ❌ Your config (git-ignored)
├── run_config.py                  # Dataclass definitions
├── config_loader.py               # Legacy loader (deprecated)
├── data_manager.py                # Data downloading/caching
├── portfolio_simulator.py         # Monte Carlo engine
├── portfolio_optimizer.py         # Optimization algorithms
├── visualizations.py              # Plotly dashboards
├── main.py                        # Main script
└── requirements.txt
```

## Best Practices

### Simulation Methods

**Bootstrap (Recommended)**
- Resamples actual historical returns
- Preserves real characteristics (fat tails, skewness)
- Best for assets with 10+ years of data

**Geometric Brownian Motion**
- Industry standard mathematical model
- Good for theoretical analysis
- Assumes normal distribution (may underestimate tail risk)

**Parametric**
- Simplest method
- Uses normal distribution
- Least realistic, use only for testing

### Optimization Objective Weights

Common combinations:
```python
# Balanced
{'return': 0.50, 'sharpe': 0.20, 'drawdown': 0.30}

# Return-focused
{'return': 0.70, 'sharpe': 0.20, 'drawdown': 0.10}

# Risk-focused
{'return': 0.30, 'sharpe': 0.30, 'drawdown': 0.40}

# Sharpe-focused
{'return': 0.30, 'sharpe': 0.50, 'drawdown': 0.20}
```

## Validation

Compare results with:
- [Portfolio Visualizer](https://www.portfoliovisualizer.com/monte-carlo-simulation) (industry standard)
- [Bogleheads Tools](https://www.bogleheads.org/wiki/Using_open_source_software_for_portfolio_analysis)

### Expected Ranges (10-year horizon)
- S&P 500 (VOO): 8-12% CAGR, 15-20% volatility
- Nasdaq (QQQ): 10-15% CAGR, 20-25% volatility
- Bonds (BND): 3-5% CAGR, 5-8% volatility
- 60/40 Stock/Bond: 6-9% CAGR, 10-12% volatility

## Privacy

Your `config/my_portfolios.py` is **automatically git-ignored**. Your personal allocations never leave your machine.

Git-ignored patterns:
- `config/my_*.py`
- `config/personal_*.py`
- `config/*_private.py`

## Examples

### Example 1: Simple Comparison
```python
from run_config import RunConfig, AssetConfig, PortfolioConfig

config = RunConfig(
    name="S&P 500 vs 60/40",
    assets=[
        AssetConfig(ticker='VOO'),
        AssetConfig(ticker='BND'),
    ],
    portfolios=[
        PortfolioConfig(name='100% Stocks', allocations={'voo': 1.0}),
        PortfolioConfig(name='60/40', allocations={'voo': 0.6, 'bnd': 0.4}),
    ]
)
```

### Example 2: Testing Leverage
```python
config = RunConfig(
    name="Leverage Analysis",
    assets=[
        AssetConfig(ticker='VOO', name='S&P 500'),
        AssetConfig(ticker='TQQQ', name='3x Nasdaq'),
        AssetConfig(ticker='BND', name='Bonds'),
    ],
    portfolios=[
        PortfolioConfig(name='No Leverage', allocations={'voo': 0.8, 'bnd': 0.2}),
        PortfolioConfig(name='10% Leverage', allocations={'voo': 0.7, 'tqqq': 0.1, 'bnd': 0.2}),
        PortfolioConfig(name='20% Leverage', allocations={'voo': 0.6, 'tqqq': 0.2, 'bnd': 0.2}),
    ]
)
```

### Example 3: Find Optimal Allocation
```python
config = RunConfig(
    name="Optimize 3-Fund Portfolio",
    assets=[
        AssetConfig(ticker='VOO'),
        AssetConfig(ticker='VTI'),
        AssetConfig(ticker='BND'),
    ],
    portfolios=[],  # No manual portfolios, just optimize
    optimization=OptimizationConfig(
        assets=['voo', 'vti', 'bnd'],
        method='grid_search',
        objective_weights={'return': 0.4, 'sharpe': 0.3, 'drawdown': 0.3},
        grid_points=6,  # Test 6^3 = 216 combinations
        top_n=10        # Keep top 10 results
    )
)
```

## Troubleshooting

### "Config file must define a 'config' variable"
Make sure your config file has:
```python
config = RunConfig(...)  # Must be named 'config'
```

### "Allocations must sum to 1.0"
Check your portfolio allocations:
```python
PortfolioConfig(
    name='Test',
    allocations={'voo': 0.6, 'bnd': 0.4}  # Must equal 1.0
)
```

### "Portfolio references unknown asset"
Asset names in allocations must match ticker names:
```python
assets=[AssetConfig(ticker='VOO')],  # Use lowercase in allocations
portfolios=[
    PortfolioConfig(allocations={'voo': 1.0})  # Match with lowercase
]
```

### "Not enough historical data"
Some ETFs are newer. Reduce `lookback_years`:
```python
AssetConfig(ticker='TQQQ', lookback_years=8)  # TQQQ launched in 2010
```

## License

MIT
name='60/40',
            allocations={'voo': 0.6, 'bnd': 0.4}
        ),
        PortfolioConfig(
