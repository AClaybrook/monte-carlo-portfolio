"""
Configuration system using dataclasses for type safety and validation.

This makes it clear what options are available and what they do.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from pathlib import Path

@dataclass
class AssetConfig:
    """
    Configuration for a single asset/ticker.

    Example:
        AssetConfig(
            ticker='VOO',
            name='S&P 500 ETF',
            lookback_years=10
        )
    """
    ticker: str  # Yahoo Finance ticker symbol (e.g., 'VOO', 'QQQQQ', 'BND')
    name: Optional[str] = None  # Display name (defaults to ticker)
    lookback_years: int = 10  # Years of historical data to use

    def __post_init__(self):
        """Validate and set defaults"""
        if self.name is None:
            self.name = self.ticker
        if self.lookback_years < 1:
            raise ValueError("lookback_years must be at least 1")

@dataclass
class PortfolioConfig:
    """
    Configuration for a portfolio with asset allocations.

    The allocations dict maps asset names to weights (0.0 to 1.0).
    Weights must sum to 1.0.

    Example:
        PortfolioConfig(
            name='Balanced',
            allocations={'voo': 0.6, 'bnd': 0.4},
            description='60/40 stocks/bonds'
        )
    """
    name: str  # Portfolio name for display
    allocations: Dict[str, float]  # Asset name -> weight (must sum to 1.0)
    description: Optional[str] = None  # Optional description

    def __post_init__(self):
        """Validate allocations"""
        total = sum(self.allocations.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")

        for asset, weight in self.allocations.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for {asset} must be between 0 and 1, got {weight}")

@dataclass
class OptimizationConfig:
    """
    Configuration for portfolio optimization.

    The optimizer will test different allocations of the specified assets
    to find portfolios that maximize a custom score based on your preferences.

    Objective weights determine what you care about:
    - return: Higher expected returns
    - sharpe: Better risk-adjusted returns (return/volatility)
    - sortino: Better downside risk-adjusted returns
    - drawdown: Smaller maximum losses

    Example:
        OptimizationConfig(
            assets=['voo', 'qqq', 'bnd'],
            method='grid_search',
            objective_weights={'return': 0.5, 'sharpe': 0.2, 'drawdown': 0.3}
        )
    """
    assets: List[str]  # Asset names to optimize over
    method: Literal['grid_search', 'random_search'] = 'grid_search'
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'return': 0.50,
        'sharpe': 0.20,
        'drawdown': 0.30
    })
    grid_points: int = 6  # For grid_search: granularity (higher = slower but more thorough)
    n_iterations: int = 50  # For random_search: number of portfolios to test
    top_n: int = 5  # Number of top results to keep

    def __post_init__(self):
        """Validate optimization config"""
        if len(self.assets) < 2:
            raise ValueError("Need at least 2 assets for optimization")

        # Validate objective weights
        total = sum(self.objective_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Objective weights must sum to 1.0, got {total}")

@dataclass
class SimulationConfig:
    """
    Monte Carlo simulation settings.

    Methods:
    - 'bootstrap': Resample from historical data (preserves actual characteristics)
    - 'geometric_brownian': Use geometric brownian motion (industry standard)
    - 'parametric': Use normal distribution (simplest, least realistic)

    Example:
        SimulationConfig(
            initial_capital=100000,
            years=10,
            simulations=10000,
            method='bootstrap'
        )
    """
    initial_capital: float = 100000  # Starting portfolio value in dollars
    years: int = 10  # Investment time horizon
    simulations: int = 10000  # Number of Monte Carlo runs (more = more accurate)
    method: Literal['bootstrap', 'geometric_brownian', 'parametric'] = 'bootstrap'
    end_date: Optional[str] = None  # End date for historical data (YYYY-MM-DD), defaults to today
    lookback_years: int = 10  # Years of historical data to use if available

    def __post_init__(self):
        """Validate simulation config"""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.years < 1:
            raise ValueError("years must be at least 1")
        if self.simulations < 100:
            raise ValueError("simulations must be at least 100")

@dataclass
class VisualizationConfig:
    """
    Settings for output visualizations.

    Example:
        VisualizationConfig(
            save_html=True,
            show_browser=True,
            output_filename='my_portfolio.html'
        )
    """
    save_html: bool = True  # Save dashboard to HTML file
    show_browser: bool = True  # Open in web browser automatically
    output_filename: str = 'portfolio_dashboard.html'  # Output file name

@dataclass
class DatabaseConfig:
    """
    Database settings for caching data.

    Example:
        DatabaseConfig(
            path='stock_data.db',
            save_results=True
        )
    """
    path: str = 'stock_data.db'  # SQLite database file path
    save_results: bool = True  # Save simulation results to database

@dataclass
class RunConfig:
    """
    Main configuration class that ties everything together.

    Assets are automatically discovered from your portfolio allocations!
    You only need to specify them if you want custom settings.

    Example - Simple (assets auto-discovered):

        config = RunConfig(
            name="Simple Analysis",
            portfolios=[
                PortfolioConfig(
                    name='60/40',
                    allocations={'VOO': 0.6, 'BND': 0.4}  # ← Assets discovered here
                ),
            ]
        )

    Example - Custom asset settings:

        config = RunConfig(
            name="Custom Settings",
            portfolios=[
                PortfolioConfig(name='My Portfolio', allocations={'VOO': 0.6, 'TQQQ': 0.4}),
            ],
            assets={
                'TQQQ': AssetConfig(lookback_years=8, name='3x Nasdaq')  # Custom settings
            }
        )
    """
    name: str  # Name for this configuration/run
    portfolios: List[PortfolioConfig]  # Portfolios to simulate

    # Optional: Only specify if you need custom settings
    assets: Optional[Dict[str, AssetConfig]] = None  # ticker -> custom config

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimization: Optional[OptimizationConfig] = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def __post_init__(self):
        """Validate and auto-discover assets"""
        # Discover all tickers from portfolios and optimization
        discovered_tickers = set()

        for portfolio in self.portfolios:
            discovered_tickers.update(portfolio.allocations.keys())

        if self.optimization:
            discovered_tickers.update(self.optimization.assets)

        # Create default AssetConfig for any ticker without custom config
        if self.assets is None:
            self.assets = {}

        # Normalize ticker case (store as uppercase)
        self.assets = {k.upper(): v for k, v in self.assets.items()}

        # Fill in missing assets with defaults
        for ticker in discovered_tickers:
            ticker_upper = ticker.upper()
            if ticker_upper not in self.assets:
                self.assets[ticker_upper] = AssetConfig(ticker=ticker_upper)

        # Validate optimization assets if present
        if self.optimization:
            for asset_name in self.optimization.assets:
                if asset_name.upper() not in self.assets:
                    # This shouldn't happen after auto-discovery, but just in case
                    self.assets[asset_name.upper()] = AssetConfig(ticker=asset_name.upper())

    def get_asset_config(self, ticker: str) -> AssetConfig:
        """Get asset config for a ticker (case-insensitive)"""
        return self.assets[ticker.upper()]

    def summary(self) -> str:
        """Get a human-readable summary of this configuration"""
        lines = [
            f"Configuration: {self.name}",
            f"",
            f"Assets ({len(self.assets)}):",
        ]
        for ticker, asset in sorted(self.assets.items()):
            custom = " (custom settings)" if asset.name != ticker or asset.lookback_years != 10 else ""
            lines.append(f"  - {ticker}: {asset.name}{custom}")

        lines.append(f"")
        lines.append(f"Portfolios ({len(self.portfolios)}):")
        for portfolio in self.portfolios:
            alloc_str = ', '.join([f"{k}={v*100:.0f}%" for k, v in portfolio.allocations.items()])
            lines.append(f"  - {portfolio.name}: {alloc_str}")

        lines.append(f"")
        lines.append(f"Simulation:")
        lines.append(f"  - Capital: ${self.simulation.initial_capital:,}")
        lines.append(f"  - Years: {self.simulation.years}")
        lines.append(f"  - Simulations: {self.simulation.simulations:,}")
        lines.append(f"  - Method: {self.simulation.method}")

        if self.optimization:
            lines.append(f"")
            lines.append(f"Optimization:")
            lines.append(f"  - Assets: {', '.join(self.optimization.assets)}")
            lines.append(f"  - Method: {self.optimization.method}")
            weights_str = ', '.join([f"{k}={v*100:.0f}%" for k, v in self.optimization.objective_weights.items()])
            lines.append(f"  - Objective: {weights_str}")

        return '\n'.join(lines)


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