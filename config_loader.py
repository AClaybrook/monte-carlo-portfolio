"""
Configuration loader that reads portfolio definitions from config files.
"""

import os
import sys
import importlib.util
from pathlib import Path

class ConfigLoader:
    """Loads portfolio configurations from Python files"""
    
    def __init__(self, config_dir='config'):
        self.config_dir = Path(config_dir)
        self.config = None
        
    def load_config(self, config_file=None):
        """
        Load configuration from a Python file
        
        Priority order:
        1. Explicitly specified config_file
        2. config/my_portfolios.py (personal config)
        3. config/example_config.py (template)
        
        Parameters:
        -----------
        config_file : str or Path, optional
            Path to configuration file
            
        Returns:
        --------
        dict : Configuration dictionary
        """
        if config_file:
            config_path = Path(config_file)
        else:
            # Try personal config first
            config_path = self.config_dir / 'my_portfolios.py'
            if not config_path.exists():
                print("⚠ Personal config not found, using example_config.py")
                print(f"  Create {config_path} to customize your portfolios")
                config_path = self.config_dir / 'example_config.py'
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Create config/my_portfolios.py or use config/example_config.py as template"
            )
        
        print(f"✓ Loading configuration from: {config_path}")
        
        # Load the Python module
        spec = importlib.util.spec_from_file_location("portfolio_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract configuration
        self.config = {
            'simulation': getattr(config_module, 'SIMULATION_CONFIG', {}),
            'assets': getattr(config_module, 'ASSETS', {}),
            'portfolios': getattr(config_module, 'PORTFOLIOS', []),
            'optimization': getattr(config_module, 'OPTIMIZATION_CONFIG', {}),
            'visualization': getattr(config_module, 'VISUALIZATION_CONFIG', {}),
            'database': getattr(config_module, 'DATABASE_CONFIG', {})
        }
        
        self._validate_config()
        return self.config
    
    def _validate_config(self):
        """Validate configuration structure"""
        if not self.config['assets']:
            raise ValueError("No assets defined in configuration")
        
        if not self.config['portfolios']:
            raise ValueError("No portfolios defined in configuration")
        
        # Validate portfolio allocations reference existing assets
        for portfolio in self.config['portfolios']:
            for asset_key in portfolio['allocations'].keys():
                if asset_key not in self.config['assets']:
                    raise ValueError(
                        f"Portfolio '{portfolio['name']}' references unknown asset '{asset_key}'"
                    )
            
            # Check allocations sum to ~1.0
            total = sum(portfolio['allocations'].values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Portfolio '{portfolio['name']}' allocations sum to {total}, must be 1.0"
                )
        
        print(f"✓ Configuration validated:")
        print(f"  - {len(self.config['assets'])} assets defined")
        print(f"  - {len(self.config['portfolios'])} portfolios defined")
        if self.config['optimization'].get('enabled'):
            print(f"  - Optimization enabled")
    
    def list_available_configs(self):
        """List all configuration files in config directory"""
        if not self.config_dir.exists():
            return []
        
        configs = []
        for file in self.config_dir.glob('*.py'):
            if file.name != '__init__.py':
                configs.append(file)
        return configs
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration"""
        if not self.config:
            print("No configuration loaded")
            return
        
        print("\n" + "="*60)
        print("Configuration Summary")
        print("="*60)
        
        print("\nSimulation Settings:")
        sim = self.config['simulation']
        print(f"  Initial Capital: ${sim.get('initial_capital', 100000):,}")
        print(f"  Time Horizon: {sim.get('years', 10)} years")
        print(f"  Simulations: {sim.get('simulations', 10000):,}")
        print(f"  Method: {sim.get('method', 'geometric_brownian')}")
        
        print("\nAssets:")
        for key, asset in self.config['assets'].items():
            print(f"  {key}: {asset['ticker']} - {asset['name']}")
        
        print("\nPortfolios:")
        for portfolio in self.config['portfolios']:
            print(f"  {portfolio['name']}")
            for asset_key, weight in portfolio['allocations'].items():
                print(f"    - {asset_key}: {weight*100:.1f}%")
        
        if self.config['optimization'].get('enabled'):
            print("\nOptimization:")
            opt = self.config['optimization']
            print(f"  Method: {opt.get('method', 'grid_search')}")
            print(f"  Objective weights: {opt.get('objective_weights', {})}")
        
        print("="*60 + "\n")
