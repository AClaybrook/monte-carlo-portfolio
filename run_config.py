"""
Configuration system - Updated for Selectable Strategies & Benchmarks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

@dataclass
class AssetConfig:
    ticker: str
    name: Optional[str] = None
    lookback_years: int = 10
    def __post_init__(self):
        if self.name is None: self.name = self.ticker
        if self.lookback_years < 1: raise ValueError("lookback_years must be at least 1")

@dataclass
class PortfolioConfig:
    name: str
    allocations: Dict[str, float]
    description: Optional[str] = None
    def __post_init__(self):
        total = sum(self.allocations.values())
        if abs(total - 1.0) > 0.01: raise ValueError(f"Allocations must sum to 1.0, got {total}")

@dataclass
class OptimizationConfig:
    """
    Configuration for portfolio optimization.

    active_strategies: List of strategies to run. Options:
        - 'max_sharpe': Maximize Sharpe Ratio
        - 'min_volatility': Minimize Standard Deviation
        - 'risk_parity': Equal Risk Contribution
        - 'max_sortino': Maximize Sortino Ratio
        - 'max_return': Maximize CAGR (High Risk)
    """
    assets: List[str]
    active_strategies: List[str] = field(default_factory=lambda: ['max_sharpe', 'min_volatility'])
    benchmark_ticker: str = 'VFINX'  # Always include this benchmark

    # Legacy fields (kept for compatibility but can be ignored)
    method: str = 'scipy'
    objective_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.assets) < 2: raise ValueError("Need at least 2 assets for optimization")

@dataclass
class SimulationConfig:
    initial_capital: float = 100000
    years: int = 10
    simulations: int = 10000
    method: Literal['bootstrap', 'geometric_brownian', 'parametric'] = 'bootstrap'
    end_date: Optional[str] = None
    lookback_years: int = 10

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
        for p in self.portfolios: discovered_tickers.update(p.allocations.keys())
        if self.optimization:
            discovered_tickers.update(self.optimization.assets)
            # Add Benchmark Ticker
            discovered_tickers.add(self.optimization.benchmark_ticker)

        if self.assets is None: self.assets = {}
        self.assets = {k.upper(): v for k, v in self.assets.items()}

        for ticker in discovered_tickers:
            t_up = ticker.upper()
            if t_up not in self.assets:
                self.assets[t_up] = AssetConfig(ticker=t_up)

# Helper function to load config from a Python file
def load_config_from_file(filepath: str) -> RunConfig:
    """
    Load a RunConfig from a Python file.

    The file should define a variable called 'config' that is a RunConfig instance.

    Example file content:
        from run_config import RunConfig, AssetConfig, PortfolioConfig

        config = RunConfig(
            name="My Config",
            assets=[...],
            portfolios=[...]
        )
    """
    import importlib.util
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    # Load the module
    spec = importlib.util.spec_from_file_location("user_config", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the config object
    if not hasattr(module, 'config'):
        raise ValueError(f"Config file {filepath} must define a 'config' variable")

    config = module.config
    if not isinstance(config, RunConfig):
        raise ValueError(f"'config' must be a RunConfig instance, got {type(config)}")

    return config