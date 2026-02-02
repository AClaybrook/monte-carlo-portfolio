# Architectural Patterns

Design patterns and conventions used throughout the codebase.

## 1. Dataclass-Based Configuration

All configuration uses Python dataclasses with validation in `__post_init__`.

**Pattern**: Hierarchical config objects with type hints and runtime validation.

**Implementation**:
- [run_config.py:8-16](../run_config.py#L8-L16) - `AssetConfig` with ticker validation
- [run_config.py:56-66](../run_config.py#L56-L66) - `PortfolioConfig` validates allocations sum to 1.0
- [run_config.py:104-129](../run_config.py#L104-L129) - `RunConfig` auto-discovers tickers from portfolios

**Convention**: Always validate in `__post_init__`, raise `ValueError` with descriptive messages.

```python
@dataclass
class PortfolioConfig:
    name: str
    allocations: Dict[str, float]

    def __post_init__(self):
        total = sum(self.allocations.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")
```

## 2. Strategy Pattern

Dynamic allocation strategies use abstract base class with polymorphic dispatch.

**Pattern**: Abstract base `AllocationStrategy` with concrete implementations.

**Implementation**:
- [strategies.py:61-93](../strategies.py#L61-L93) - `AllocationStrategy` abstract base
- [strategies.py:95-104](../strategies.py#L95-L104) - `StaticAllocationStrategy` (default)
- [strategies.py:106-169](../strategies.py#L106-L169) - `BuyTheDipStrategy`
- [strategies.py:172-218](../strategies.py#L172-L218) - `MomentumStrategy`

**Key Method**: `get_allocation(context: MarketContext) -> np.ndarray`

**MarketContext** ([strategies.py:24-58](../strategies.py#L24-L58)): Rich context dataclass providing:
- Current holdings and drawdowns
- Rolling returns, volatility, momentum
- Asset tickers and base allocations

**Composite Strategies**:
- [strategies.py:384-416](../strategies.py#L384-L416) - `CompositeStrategy` combines multiple strategies
- [strategies.py:419-458](../strategies.py#L419-L458) - `ConditionalStrategy` switches based on conditions

## 3. Factory Pattern

Config-to-object mapping via factory functions and registry.

**Pattern**: Registry dict maps type strings to classes, factory function instantiates.

**Implementation**:
- [strategies.py:592-599](../strategies.py#L592-L599) - `STRATEGY_REGISTRY` dict
- [strategies.py:602-622](../strategies.py#L602-L622) - `create_strategy_from_config()`
- [run_config.py:154-221](../run_config.py#L154-L221) - `create_strategy_from_config()` in run_config

**Convention**: Factory handles parameter extraction and defaults:
```python
def create_strategy_from_config(strategy_config: StrategyConfig):
    params = strategy_config.params
    if strategy_config.type == 'buy_the_dip':
        return BuyTheDipStrategy(
            target_ticker=params.get('target_ticker', 'VOO'),
            threshold=params.get('threshold', 0.10),
            aggressive_weight=params.get('aggressive_weight', 0.80)
        )
```

## 4. Manager Pattern (Data Access Layer)

Centralized data access through `DataManager` class.

**Pattern**: Single class encapsulates database, API, and caching logic.

**Implementation**:
- [data_manager.py:237-260](../data_manager.py#L237-L260) - `DataManager.__init__` sets up SQLAlchemy
- [data_manager.py:360-406](../data_manager.py#L360-L406) - `get_data()` with interval-based caching
- [data_manager.py:497-575](../data_manager.py#L497-L575) - `bulk_download()` for efficiency

**Key Features**:
- Interval tracking avoids redundant downloads ([data_manager.py:107-234](../data_manager.py#L107-L234))
- Known inception dates prevent pre-IPO queries ([data_manager.py:264-282](../data_manager.py#L264-L282))
- Rate limit handling with exponential backoff ([data_manager.py:440-476](../data_manager.py#L440-L476))

**Dependency Injection**: `DataManager` passed to `PortfolioSimulator`, `Backtester`, `PortfolioOptimizer`:
```python
data_manager = DataManager(db_path=config.database.path)
sim = PortfolioSimulator(data_manager, config.simulation)
optimizer = PortfolioOptimizer(sim, data_manager)
```

## 5. Batch/Vectorized Processing

Monte Carlo simulations use NumPy vectorization with batch processing.

**Pattern**: Process simulations in batches to manage memory while maintaining vectorized speed.

**Implementation**:
- [portfolio_simulator.py:133-174](../portfolio_simulator.py#L133-L174) - `_simulate_fast_vectorized()` with BATCH_SIZE=1000
- [portfolio_simulator.py:176-325](../portfolio_simulator.py#L176-L325) - `_simulate_time_stepped()` for DCA/strategies

**Vectorization Techniques**:
- `np.einsum` for correlated returns: [portfolio_simulator.py:148](../portfolio_simulator.py#L148)
- `np.cumprod` for cumulative growth: [portfolio_simulator.py:153](../portfolio_simulator.py#L153)
- `np.maximum.accumulate` for running max/drawdowns: [portfolio_simulator.py:159](../portfolio_simulator.py#L159)

**Bulk Data Operations**:
- [data_manager.py:577-649](../data_manager.py#L577-L649) - `_bulk_download_and_save()` uses `yf.download()` for multiple tickers

## 6. Consistent Results Structure

All analysis components return dicts with standardized keys.

**Pattern**: Uniform result structure enables consistent visualization handling.

**Standard Keys**:
```python
{
    'portfolio_values': np.ndarray,  # (simulations, days)
    'final_values': np.ndarray,      # (simulations,)
    'cagr': np.ndarray,
    'max_drawdowns': np.ndarray,
    'assets': List[dict],
    'allocations': List[float],
    'probabilities': dict,
    'stats': dict                    # Computed statistics
}
```

**Implementation**:
- [portfolio_simulator.py:168-174](../portfolio_simulator.py#L168-L174) - Simulator results
- [portfolio_optimizer.py:284-290](../portfolio_optimizer.py#L284-L290) - Optimizer results
- [backtester.py:108-116](../backtester.py#L108-L116) - Backtest results

**Stats Dict** (computed by `calculate_statistics`):
- `mean_final_value`, `median_final_value`
- `median_cagr`, `mean_cagr`, `std_cagr`
- `sharpe_ratio`, `sortino_ratio`
- `probability_loss`, `probability_double`

## 7. SQLAlchemy ORM Models

Database persistence uses declarative base pattern.

**Implementation**:
- [data_manager.py:38-50](../data_manager.py#L38-L50) - `StockPrice` model with unique constraint
- [data_manager.py:53-67](../data_manager.py#L53-L67) - `TickerMetadata` for caching metadata
- [data_manager.py:70-86](../data_manager.py#L70-L86) - `OptimizationResult` for saved results

**Convention**: Use `UniqueConstraint` for natural keys, explicit indexes on query columns.

## 8. Visualization Composition

Report generation uses Plotly subplots with linked legends.

**Pattern**: Build figures with `make_subplots`, use `legendgroup` for linked toggling.

**Implementation**:
- [visualizations.py:111-185](../visualizations.py#L111-L185) - Monte Carlo plots
- [visualizations.py:187-213](../visualizations.py#L187-L213) - Backtest plots
- [visualizations.py:215-314](../visualizations.py#L215-L314) - HTML report assembly

**Legend Linking**:
```python
for idx, item in enumerate(portfolio_results):
    grp = f"mc_{idx}"
    fig.add_trace(..., legendgroup=grp, showlegend=True, ...)
    fig.add_trace(..., legendgroup=grp, showlegend=False, ...)
```
