# Architecture Diagram

## High-Level Pipeline

```mermaid
flowchart LR
    Config["config/*.py\n(RunConfig)"] --> Main["main.py\n(Orchestrator)"]
    Main --> DM["data_manager.py\n(DataManager)"]
    DM --> |"Asset Map\n(DataFrames)"| Sim["portfolio_simulator.py\n(Monte Carlo)"]
    DM --> |"Asset Map\n(DataFrames)"| BT["backtester.py\n(Backtest)"]
    DM --> |"Asset Map\n(DataFrames)"| Opt["portfolio_optimizer.py\n(Optimizer)"]
    Sim --> Results["Portfolio Results"]
    BT --> Results
    Opt --> |"Optimized\nAllocations"| Sim
    Opt --> |"Optimized\nAllocations"| BT
    Results --> Viz["visualizations.py\n(PortfolioVisualizer)"]
    Viz --> HTML["output/*.html\n(Interactive Dashboard)"]

    style Config fill:#4a9,color:#fff
    style Main fill:#369,color:#fff
    style DM fill:#e83,color:#fff
    style Sim fill:#96c,color:#fff
    style BT fill:#96c,color:#fff
    style Opt fill:#96c,color:#fff
    style Viz fill:#c63,color:#fff
    style HTML fill:#999,color:#fff
    style Results fill:#666,color:#fff
```

## Module Dependency Map

```mermaid
flowchart TD
    main["main.py"]
    rc["run_config.py"]
    dm["data_manager.py"]
    ps["portfolio_simulator.py"]
    po["portfolio_optimizer.py"]
    bt["backtester.py"]
    st["strategies.py"]
    vis["visualizations.py"]
    du["data_utils.py"]

    main --> rc
    main --> dm
    main --> ps
    main --> po
    main --> bt
    main --> st
    main --> vis

    ps --> dm
    ps --> st
    po --> ps
    po --> dm
    bt --> dm
    bt --> st

    du --> dm
    du --> rc

    subgraph External
        yf["yfinance"]
        sa["SQLAlchemy"]
        pd["pandas / numpy / scipy"]
        pl["plotly"]
        por["portion"]
    end

    dm --> yf
    dm --> sa
    dm --> por
    ps --> pd
    po --> pd
    bt --> pd
    vis --> pl
```

## Data Flow

```mermaid
flowchart TD
    subgraph "Data Layer"
        API["yfinance API"] --> |"OHLCV prices"| SM["_smart_download()"]
        SM --> |"DataFrame"| DB[("SQLite\nstock_data.db")]
        DB --> |"Cached prices"| GM["get_data()"]
        IT["IntervalTracker\n(portion library)"] --> |"Missing\nintervals"| SM
        DB --> |"data_intervals_json"| IT
    end

    subgraph "Config Layer"
        CF["config/*.py"] --> |"Python module"| LC["load_config_from_file()"]
        LC --> RC["RunConfig"]
        RC --> PC["PortfolioConfig[]"]
        RC --> SC["SimulationConfig"]
        RC --> OC["OptimizationConfig"]
    end

    subgraph "Processing Layer"
        GM --> |"Asset Map\n{ticker: DataFrame}"| SIM["PortfolioSimulator"]
        GM --> |"Asset Map"| OPT["PortfolioOptimizer"]
        GM --> |"Asset Map"| BACK["Backtester"]

        SIM --> |"simulate_portfolio()"| PR["portfolio_values\nfinal_values\nCAGR, Sharpe\nprobabilities"]
        BACK --> |"run_backtest()"| BR["equity_curve\ndrawdowns\nmetrics"]
        OPT --> |"optimize_*()"| OR["optimal_allocations"]
        OR --> SIM
        OR --> BACK

        STR["strategies.py\nAllocationStrategy"] --> |"get_allocation()\nMarketContext"| SIM
        STR --> |"get_allocation()\nMarketContext"| BACK
    end

    subgraph "Output Layer"
        PR --> VIZ["PortfolioVisualizer"]
        BR --> VIZ
        VIZ --> |"Plotly"| HTML["Interactive HTML\nDashboard"]
    end

    PC --> SIM
    SC --> SIM
    OC --> OPT
```

## Execution Pipeline (main.py)

```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant C as RunConfig
    participant D as DataManager
    participant S as PortfolioSimulator
    participant B as Backtester
    participant O as PortfolioOptimizer
    participant V as PortfolioVisualizer

    U->>M: python main.py config/example_config.py
    M->>C: load_config_from_file()
    C-->>M: RunConfig

    M->>D: bulk_download(all_tickers, date_range)
    Note over D: IntervalTracker computes gaps<br/>Only fetches missing data
    D-->>M: Asset Map {ticker: DataFrame}

    loop Each PortfolioConfig
        M->>S: simulate_portfolio(assets, allocations, strategy)
        S-->>M: MC results (distributions, stats)
        M->>B: run_backtest(assets, allocations, strategy)
        B-->>M: Backtest results (equity curve, metrics)
    end

    opt OptimizationConfig enabled
        loop Each optimization strategy
            M->>O: optimize_*(assets)
            O-->>M: Optimal allocations
            M->>S: simulate_portfolio(optimal_allocs)
            M->>B: run_backtest(optimal_allocs)
        end
    end

    M->>V: generate_html_report(all_results)
    V-->>M: output/{timestamp}.html
    M-->>U: Report saved
```

## Main Features

### Monte Carlo Simulation
| Method | Description |
|--------|-------------|
| **Bootstrap** (default) | Resamples historical daily returns; preserves fat tails and real market behavior |
| **Geometric Brownian Motion** | Industry-standard GBM with drift and volatility from historical data |
| **Parametric** | Normal distribution sampling from historical mean/std |

- Vectorized NumPy with 2000-sim batch processing
- Supports DCA contributions with configurable frequency
- Calculates probability of loss, doubling, and custom targets

### Portfolio Optimization (SciPy SLSQP)
| Strategy | Objective |
|----------|-----------|
| **Max Sharpe** | Maximize risk-adjusted return |
| **Min Volatility** | Minimize portfolio variance |
| **Max Sortino** | Maximize downside-risk-adjusted return |
| **Risk Parity** | Equalize risk contribution across assets |
| **Custom Weighted** | Blend of objectives with user-defined weights |

- Per-asset min/max weight bounds
- Concentrated starting points for better convergence
- Data caching across optimization strategies

### Dynamic Allocation Strategies
| Strategy | Behavior |
|----------|----------|
| **Static** | Fixed allocation (baseline) |
| **Buy the Dip** | Increase target asset weight during drawdowns |
| **Momentum** | Tilt toward assets with positive momentum |
| **Volatility Target** | Adjust exposure to maintain target portfolio volatility |
| **Drawdown Protection** | Shift to defensive assets during market crashes |
| **Relative Value** | Overweight the most beaten-down assets |
| **Composite** | Blend multiple strategies with weights |
| **Conditional** | Switch strategy based on market regime |

- Rich `MarketContext` dataclass provides rolling stats, drawdowns, regime detection
- Used in both Monte Carlo simulation and backtesting

### Historical Backtesting
- Static (fast vectorized) and dynamic (time-stepped with strategies) modes
- DCA contribution modeling
- Metrics: CAGR, max drawdown, Sharpe ratio, Sortino ratio
- `StrategyComparison` class for side-by-side strategy evaluation

### Smart Data Caching
- `IntervalTracker` using `portion` library for precise date-range tracking
- Per-ticker gap detection — only fetches missing data from yfinance
- Failed download cooldown (1 hour) to prevent API thrashing
- Bulk download when all tickers need the same range; sequential otherwise
- `data_utils.py` CLI for coverage reports, sync, and manual downloads

### Visualization
- Interactive Plotly HTML dashboards
- Monte Carlo distribution curves with percentile bands
- Backtest equity curves indexed to 100
- Drawdown analysis and rolling returns
- Allocation tables with performance metrics
