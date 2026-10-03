"""
Configuration system - Updated with Strategy Support.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any, Union

CALENDAR_FREQUENCIES = ('daily', 'weekly', 'monthly', 'quarterly', 'annual')


@dataclass
class RebalanceConfig:
    """
    When holdings are traded back to target weights.

    frequency: 'none' (buy and hold), 'daily', 'weekly', 'monthly', 'quarterly', 'annual'.
        Historical runs rebalance on the first trading day of each new period.
    threshold: Optional absolute drift band, e.g. 0.05 trades whenever any
        asset's weight is more than 5 percentage points from its target.
        Works alone (frequency='none') or alongside a calendar.
    transaction_cost_bps: Cost charged on traded dollars (one-way), in basis points.
    """
    frequency: Literal['none', 'daily', 'weekly', 'monthly', 'quarterly', 'annual'] = 'annual'
    threshold: Optional[float] = None
    transaction_cost_bps: float = 0.0

    def __post_init__(self):
        if self.frequency not in ('none',) + CALENDAR_FREQUENCIES:
            raise ValueError(f"Unknown rebalance frequency: {self.frequency}")
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError("rebalance threshold must be between 0 and 1")

    @staticmethod
    def coerce(value: Union['RebalanceConfig', str, None]) -> Optional['RebalanceConfig']:
        if value is None or isinstance(value, RebalanceConfig):
            return value
        return RebalanceConfig(frequency=value)

@dataclass
class AssetConfig:
    ticker: str
    name: Optional[str] = None
    lookback_years: int = 10
    def __post_init__(self):
        if self.name is None: self.name = self.ticker
        if self.lookback_years < 1: raise ValueError("lookback_years must be at least 1")

@dataclass
class StrategyConfig:
    """
    Configuration for dynamic allocation strategies.

    Example configurations:

    # Simple buy-the-dip
    StrategyConfig(
        type='buy_the_dip',
        params={'target_ticker': 'BTC-USD', 'threshold': 0.20, 'aggressive_weight': 0.50}
    )

    # Crypto opportunistic
    StrategyConfig(
        type='crypto_opportunistic',
        params={'crypto_ticker': 'BTC-USD', 'equity_ticker': 'VOO',
                'dip_threshold': 0.25, 'normal_weight': 0.10, 'dip_weight': 0.40}
    )

    # Momentum tilt
    StrategyConfig(
        type='momentum',
        params={'tilt_strength': 0.5, 'min_weight': 0.05}
    )

    # Conditional rebalancing: move holdings to bonds in a 15% drawdown
    StrategyConfig(
        type='drawdown_protection', apply_to='rebalance', check_frequency='daily',
        params={'threshold': 0.15, 'recovery_threshold': 0.05,
                'risk_off_allocation': {'VOO': 0.3, 'BND': 0.7}}
    )

    apply_to: what the strategy's weights steer.
        'contributions' - only new cash is split by the strategy (default)
        'rebalance'     - holdings are traded to the strategy's target whenever it
                          changes (checked every check_frequency) and on calendar
                          rebalance dates; contributions use base weights
        'both'          - both of the above
    check_frequency: how often the strategy is evaluated for 'rebalance'/'both'.
    """
    type: str  # Strategy type from registry
    params: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # Override auto-generated name
    apply_to: Literal['contributions', 'rebalance', 'both'] = 'contributions'
    check_frequency: Literal['daily', 'weekly', 'monthly', 'quarterly', 'annual'] = 'monthly'

    def __post_init__(self):
        from strategies import STRATEGY_BUILDERS
        if self.type not in STRATEGY_BUILDERS:
            raise ValueError(f"Unknown strategy type: {self.type}. Valid: {sorted(STRATEGY_BUILDERS)}")
        if self.apply_to not in ('contributions', 'rebalance', 'both'):
            raise ValueError(f"Unknown apply_to: {self.apply_to}")
        if self.check_frequency not in CALENDAR_FREQUENCIES:
            raise ValueError(f"Unknown check_frequency: {self.check_frequency}")

@dataclass
class PortfolioConfig:
    name: str
    allocations: Dict[str, float]
    description: Optional[str] = None
    strategy: Optional[StrategyConfig] = None
    # None -> SimulationConfig.rebalance. Accepts a RebalanceConfig or a frequency string.
    rebalance: Optional[Union[RebalanceConfig, str]] = None

    def __post_init__(self):
        self.rebalance = RebalanceConfig.coerce(self.rebalance)
        total = sum(self.allocations.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    assets: List[str]
    active_strategies: List[str] = field(default_factory=lambda: ['max_sharpe', 'min_volatility'])
    benchmark_ticker: str = 'VFINX'
    method: str = 'scipy'
    objective_weights: Dict[str, float] = field(default_factory=dict)
    # Walk-forward check: refit every `test_years` on the previous `train_years`
    # and trade those weights out of sample, to show how much of the in-sample
    # result was hindsight.
    walk_forward: bool = True
    train_years: int = 3
    test_years: int = 1
    # Plot the mean-variance efficient frontier of these assets
    efficient_frontier: bool = True

    def __post_init__(self):
        if len(self.assets) < 2:
            raise ValueError("Need at least 2 assets for optimization")
        if self.train_years < 1 or self.test_years < 1:
            raise ValueError("train_years and test_years must be at least 1")

@dataclass
class SimulationConfig:
    """
    Configuration for Monte Carlo simulations and historical data.

    Date range options (in order of precedence):
    1. start_date + end_date: Explicit date range (format: 'YYYY-MM-DD')
    2. end_date + lookback_years: End date with lookback period
    3. lookback_years only: Uses today as end_date with lookback period

    Examples:
        # Use explicit date range
        SimulationConfig(start_date='2020-01-01', end_date='2024-12-31')

        # Use lookback from specific end date
        SimulationConfig(end_date='2024-12-31', lookback_years=5)

        # Use lookback from today (default behavior)
        SimulationConfig(lookback_years=10)
    """
    initial_capital: float = 100000
    years: int = 10
    simulations: int = 10000
    # bootstrap: i.i.d. resampled historical days
    # block_bootstrap: resampled runs of `block_size` consecutive days (keeps
    #   trends, volatility clustering and drawdown shapes that strategies react to)
    # geometric_brownian: multivariate lognormal fitted to historical log returns
    # parametric: multivariate normal fitted to historical simple returns
    method: Literal['bootstrap', 'block_bootstrap', 'geometric_brownian', 'parametric'] = 'bootstrap'
    block_size: int = 21
    seed: Optional[int] = None
    inflation_rate: float = 0.0       # >0 reports Monte Carlo results in today's dollars

    # Date range options
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = None    # Format: 'YYYY-MM-DD', defaults to today
    lookback_years: int = 10          # Used if start_date not specified

    contribution_amount: float = 0.0
    # int = every N trading days, or 'monthly' / 'quarterly' / 'annual'
    contribution_frequency: Union[int, str] = 21

    # Withdrawals (retirement / decumulation). Sold pro-rata; a path that runs out is "depleted".
    withdrawal_amount: float = 0.0               # taken out each period, as a positive number
    withdrawal_frequency: Union[int, str] = 'monthly'
    withdrawal_start_years: float = 0.0          # e.g. 20 = contribute first, retire after 20 years
    contribution_years: Optional[float] = None   # stop contributions after this many years
    # Annual growth applied to every contribution and withdrawal, e.g. 0.03 to keep pace with
    # inflation in nominal dollars. Leave 0 when inflation_rate > 0 (Monte Carlo is then in
    # today's dollars, so constant amounts are already inflation-adjusted).
    cash_flow_growth: float = 0.0

    # Default rebalancing for portfolios that do not set their own (PV default: annual)
    rebalance: Union[RebalanceConfig, str] = field(default_factory=RebalanceConfig)
    risk_free_rate: float = 0.02      # Annual, used for Sharpe/Sortino/alpha

    def __post_init__(self):
        from datetime import date as dt_date

        self.rebalance = RebalanceConfig.coerce(self.rebalance)
        if isinstance(self.contribution_frequency, str) and \
                self.contribution_frequency not in CALENDAR_FREQUENCIES:
            raise ValueError(f"Unknown contribution_frequency: {self.contribution_frequency}")
        if isinstance(self.withdrawal_frequency, str) and \
                self.withdrawal_frequency not in CALENDAR_FREQUENCIES:
            raise ValueError(f"Unknown withdrawal_frequency: {self.withdrawal_frequency}")
        if self.withdrawal_amount < 0 or self.contribution_amount < 0:
            raise ValueError("contribution_amount and withdrawal_amount must be >= 0")
        if self.method not in ('bootstrap', 'block_bootstrap', 'geometric_brownian', 'parametric'):
            raise ValueError(f"Unknown simulation method: {self.method}")

        # Validate dates if provided
        if self.start_date:
            try:
                dt_date.fromisoformat(self.start_date)
            except ValueError:
                raise ValueError(f"start_date must be YYYY-MM-DD format, got: {self.start_date}")

        if self.end_date:
            try:
                dt_date.fromisoformat(self.end_date)
            except ValueError:
                raise ValueError(f"end_date must be YYYY-MM-DD format, got: {self.end_date}")

        # Validate start < end if both provided
        if self.start_date and self.end_date:
            start = dt_date.fromisoformat(self.start_date)
            end = dt_date.fromisoformat(self.end_date)
            if start >= end:
                raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")

    def cash_flow_settings(self) -> dict:
        """Keyword arguments for Backtester.run_backtest describing every cash flow."""
        return dict(contribution_amount=self.contribution_amount,
                    contribution_frequency=self.contribution_frequency,
                    withdrawal_amount=self.withdrawal_amount,
                    withdrawal_frequency=self.withdrawal_frequency,
                    withdrawal_start_years=self.withdrawal_start_years,
                    contribution_years=self.contribution_years,
                    cash_flow_growth=self.cash_flow_growth)

    def get_date_range(self) -> tuple:
        """
        Calculate the effective start and end dates.
        Returns (start_date, end_date) as date objects.
        """
        from datetime import date as dt_date, timedelta

        # Determine end date
        if self.end_date:
            end = dt_date.fromisoformat(self.end_date)
        else:
            end = dt_date.today()

        # Determine start date
        if self.start_date:
            start = dt_date.fromisoformat(self.start_date)
        else:
            start = end - timedelta(days=365 * self.lookback_years)

        return (start, end)

@dataclass
class VisualizationConfig:
    save_html: bool = True
    show_browser: bool = True
    output_filename: str = 'portfolio_dashboard.html'
    embed_plotlyjs: bool = False  # True: report works offline (+~4.5 MB)

@dataclass
class DatabaseConfig:
    path: str = 'stock_data.db'
    save_results: bool = True

@dataclass
class SweepConfig:
    """
    Grid search over one or two strategy parameters on a fixed allocation.

    Every cell is backtested with the run's cash flows and rebalancing and compared
    with the same allocation without a strategy. The best cell is fit to this
    history (in-sample), so prefer broad plateaus over a single bright cell.

    Example:
        SweepConfig(
            name='TQQQ dip buyer',
            allocations={'VOO': 0.8, 'TQQQ': 0.2},
            strategy=StrategyConfig('buy_the_dip', {'target_ticker': 'TQQQ'}),
            grid={'threshold': [0.1, 0.2, 0.3, 0.4], 'aggressive_weight': [0.4, 0.6, 0.8]},
            simulations=500,   # optional Monte Carlo per cell
        )
    """
    name: str
    allocations: Dict[str, float]
    strategy: StrategyConfig
    grid: Dict[str, List[Any]]
    rebalance: Optional[Union[RebalanceConfig, str]] = None   # None -> SimulationConfig.rebalance
    simulations: int = 0       # >0 adds a Monte Carlo per cell (same seed for every cell)

    def __post_init__(self):
        self.rebalance = RebalanceConfig.coerce(self.rebalance)
        if not 1 <= len(self.grid) <= 2:
            raise ValueError(f"Sweep '{self.name}': grid must have 1 or 2 parameters")
        if any(len(v) == 0 for v in self.grid.values()):
            raise ValueError(f"Sweep '{self.name}': every grid parameter needs at least one value")
        total = sum(self.allocations.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Sweep '{self.name}': allocations must sum to 1.0, got {total}")


@dataclass
class RunConfig:
    name: str
    portfolios: List[PortfolioConfig]
    assets: Optional[Dict[str, AssetConfig]] = None
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: Optional[OptimizationConfig] = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    # Benchmark for relative metrics and the benchmark row; defaults to
    # optimization.benchmark_ticker when optimization is configured.
    benchmark_ticker: Optional[str] = None
    sweeps: List[SweepConfig] = field(default_factory=list)

    def __post_init__(self):
        if self.benchmark_ticker is None and self.optimization:
            self.benchmark_ticker = self.optimization.benchmark_ticker
        discovered_tickers = set()
        if self.benchmark_ticker:
            discovered_tickers.add(self.benchmark_ticker)
        for p in self.portfolios:
            discovered_tickers.update(p.allocations.keys())
        for sw in self.sweeps:
            discovered_tickers.update(sw.allocations.keys())
        if self.optimization:
            discovered_tickers.update(self.optimization.assets)
            discovered_tickers.add(self.optimization.benchmark_ticker)

        if self.assets is None:
            self.assets = {}
        self.assets = {k.upper(): v for k, v in self.assets.items()}

        for ticker in discovered_tickers:
            t_up = ticker.upper()
            if t_up not in self.assets:
                self.assets[t_up] = AssetConfig(ticker=t_up)


def load_config_from_file(filepath: str) -> RunConfig:
    """Load a RunConfig from a Python file."""
    import importlib.util
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    spec = importlib.util.spec_from_file_location("user_config", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'config'):
        raise ValueError(f"Config file {filepath} must define a 'config' variable")

    config = module.config
    if not isinstance(config, RunConfig):
        raise ValueError(f"'config' must be a RunConfig instance, got {type(config)}")

    return config


def create_strategy_from_config(strategy_config: StrategyConfig):
    """Build the AllocationStrategy described by a StrategyConfig."""
    from strategies import create_strategy
    return create_strategy(strategy_config.type, strategy_config.params, strategy_config.name)
