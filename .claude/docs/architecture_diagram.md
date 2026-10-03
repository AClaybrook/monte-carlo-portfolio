# Architecture

## Pipeline

```mermaid
flowchart LR
    Config["config/*.py<br/>RunConfig"] --> Main["main.py"]
    Main --> DM["data_manager.py<br/>(or synthetic_data.py)"]
    DM -->|aligned prices| Eval["evaluate_portfolio()"]
    Eval --> BT["backtester.py<br/>1 historical path"]
    Eval --> MC["portfolio_simulator.py<br/>N simulated paths"]
    BT --> Engine["engine.run_engine"]
    MC --> Engine
    Engine --> Strat["strategies.py"]
    BT --> QA["quant_analytics.py<br/>compute_performance"]
    Opt["portfolio_optimizer.py"] -->|weights| Eval
    BT --> Viz["visualizations.py"]
    MC --> Viz
    Viz --> HTML["output/*.html"]
```

## Engine step (per day, vectorized over paths)

1. `holdings *= 1 + r` for every asset
2. Update price indexes/peaks (asset drawdowns), the base-weight reference index, trailing return buffers
3. On decision days (contribution days, or `check_frequency`/rebalance days for `apply_to='rebalance'`), build a `MarketContext` and ask the strategy for target weights
4. Add the contribution, split by strategy weights (`apply_to` includes contributions) or base weights
5. Trade to target on calendar rebalance days, drift-band breaches, or strategy target changes; charge transaction costs
6. Day's time-weighted return = (value after trades − contribution) / previous value − 1

Historical runs schedule events on the first trading day of each calendar period; simulated runs space them evenly with `days_per_year` taken from the history.

## Monte Carlo return generators (`engine.SimulatedReturns`)

| method | draws |
|---|---|
| `bootstrap` | i.i.d. historical days (all assets from the same day) |
| `block_bootstrap` | circular blocks of `block_size` consecutive days |
| `geometric_brownian` | multivariate normal fitted to log returns, exponentiated |
| `parametric` | multivariate normal fitted to simple returns |

Inflation (`inflation_rate`) deflates every simulated return, so results are in today's dollars.

## Module dependencies

```mermaid
flowchart TD
    main --> run_config & data_manager & synthetic_data & portfolio_simulator & backtester & portfolio_optimizer & visualizations & pv_compat
    backtester --> engine & quant_analytics & run_config
    portfolio_simulator --> engine & quant_analytics & run_config
    portfolio_optimizer --> portfolio_simulator & quant_analytics
    engine --> strategies
    run_config --> strategies
    visualizations --> quant_analytics & pv_compat & engine
    data_utils --> data_manager
```
