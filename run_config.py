"""
Configuration system - Updated with Strategy Support.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any

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
    """
    type: str  # Strategy type from registry
    params: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # Override auto-generated name

    def __post_init__(self):
        valid_types = [
            'static', 'buy_the_dip', 'momentum', 'volatility_target',
            'drawdown_protection', 'relative_value', 'crypto_opportunistic',
            'dual_momentum'
        ]
        if self.type not in valid_types:
            raise ValueError(f"Unknown strategy type: {self.type}. Valid: {valid_types}")

@dataclass
class PortfolioConfig:
    name: str
    allocations: Dict[str, float]
    description: Optional[str] = None
    strategy: Optional[StrategyConfig] = None  # NEW: Optional strategy

    def __post_init__(self):
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

    def __post_init__(self):
        if len(self.assets) < 2:
            raise ValueError("Need at least 2 assets for optimization")

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
    method: Literal['bootstrap', 'geometric_brownian', 'parametric'] = 'bootstrap'

    # Date range options
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = None    # Format: 'YYYY-MM-DD', defaults to today
    lookback_years: int = 10          # Used if start_date not specified

    contribution_amount: float = 0.0
    contribution_frequency: int = 21  # trading days (~monthly)

    def __post_init__(self):
        from datetime import date as dt_date

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

@dataclass
class DatabaseConfig:
    path: str = 'stock_data.db'
    save_results: bool = True

@dataclass
class RunConfig:
    name: str
    portfolios: List[PortfolioConfig]
    assets: Optional[Dict[str, AssetConfig]] = None
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: Optional[OptimizationConfig] = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def __post_init__(self):
        discovered_tickers = set()
        for p in self.portfolios:
            discovered_tickers.update(p.allocations.keys())
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
    """
    Factory function to create strategy instances from StrategyConfig.
    """
    from strategies import (
        StaticAllocationStrategy, BuyTheDipStrategy, MomentumStrategy,
        VolatilityTargetStrategy, DrawdownProtectionStrategy, RelativeValueStrategy,
        create_crypto_opportunistic_strategy, create_dual_momentum_strategy
    )

    params = strategy_config.params

    if strategy_config.type == 'static':
        return StaticAllocationStrategy()

    elif strategy_config.type == 'buy_the_dip':
        return BuyTheDipStrategy(
            target_ticker=params.get('target_ticker', 'VOO'),
            threshold=params.get('threshold', 0.10),
            aggressive_weight=params.get('aggressive_weight', 0.80)
        )

    elif strategy_config.type == 'momentum':
        return MomentumStrategy(
            momentum_lookback=params.get('lookback', 63),
            tilt_strength=params.get('tilt_strength', 0.5),
            min_weight=params.get('min_weight', 0.05)
        )

    elif strategy_config.type == 'volatility_target':
        return VolatilityTargetStrategy(
            target_vol=params.get('target_vol', 0.15),
            vol_lookback=params.get('lookback', 21),
            equity_tickers=params.get('equity_tickers', []),
            safe_ticker=params.get('safe_ticker', 'BND')
        )

    elif strategy_config.type == 'drawdown_protection':
        return DrawdownProtectionStrategy(
            dd_threshold=params.get('threshold', 0.15),
            risk_off_allocation=params.get('risk_off_allocation', {}),
            recovery_threshold=params.get('recovery_threshold', 0.05)
        )

    elif strategy_config.type == 'relative_value':
        return RelativeValueStrategy(
            rebalance_threshold=params.get('threshold', 0.10),
            max_tilt=params.get('max_tilt', 0.50)
        )

    elif strategy_config.type == 'crypto_opportunistic':
        return create_crypto_opportunistic_strategy(
            crypto_ticker=params.get('crypto_ticker', 'BTC-USD'),
            equity_ticker=params.get('equity_ticker', 'VOO'),
            crypto_dip_threshold=params.get('dip_threshold', 0.25),
            normal_crypto_weight=params.get('normal_weight', 0.10),
            dip_crypto_weight=params.get('dip_weight', 0.40)
        )

    elif strategy_config.type == 'dual_momentum':
        return create_dual_momentum_strategy(
            equity_ticker=params.get('equity_ticker', 'VOO'),
            safe_ticker=params.get('safe_ticker', 'BND'),
            lookback=params.get('lookback', 126)
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_config.type}")