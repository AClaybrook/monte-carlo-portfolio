"""
COMPLETE MONTE CARLO PORTFOLIO SIMULATOR WITH OPTIMIZATION

Installation:
-------------
pip install yfinance sqlalchemy pandas numpy plotly

File Structure:
--------------
1. data_manager.py - Handles data downloading and database operations
2. portfolio_simulator.py - Monte Carlo simulation engine
3. portfolio_optimizer.py - Portfolio optimization algorithms
4. visualizations.py - Interactive Plotly dashboards
5. main.py - Main execution script

Key Features:
------------
1. SYNCHRONIZED LEGENDS
   - Click any legend item to toggle that portfolio across ALL charts
   - Double-click to isolate a single portfolio
   - All 9 subplots respond to legend interactions

2. PORTFOLIO OPTIMIZATION
   - Grid search for 2-3 assets (exhaustive)
   - Random search for 4+ assets (efficient)
   - Custom objective function with configurable weights:
     * Expected return (default 50%)
     * Sharpe ratio (default 20%)
     * Max drawdown (default 30%)
   - Results saved to database for historical tracking

3. DATABASE STORAGE
   - Stock price data cached in SQLite
   - Optimization results stored with:
     * Portfolio allocations
     * Performance metrics
     * Optimization scores
     * Timestamps for tracking
   - Retrieve and compare past optimizations

4. INTERACTIVE VISUALIZATIONS
   - 9 synchronized charts showing:
     * Sample trajectories
     * Return distributions
     * Risk metrics
     * Percentile fan charts
     * Risk-return profiles
     * Statistics table (includes Sortino ratio)
     * Probability analysis
   - Hover for detailed information
   - Zoom and pan capabilities
   - Export to PNG

5. COMPREHENSIVE METRICS
   - Expected return (CAGR)
   - Sharpe ratio
   - Sortino ratio (added to table)
   - Maximum drawdown
   - Probability of loss/doubling
   - Percentile analysis

Customization Examples:
----------------------

# Change optimization objective:
objective_weights = {
    'return': 0.40,
    'sharpe': 0.30,
    'sortino': 0.20,
    'drawdown': 0.10
}

# Test different asset combinations:
assets = [voo, qqq, tqqq, bnd, spy, iwm]  # Add more ETFs

# Adjust simulation parameters:
sim = PortfolioSimulator(
    initial_capital=100000,
    years=15,  # Longer horizon
    simulations=20000  # More precision
)

Output Files:
------------
- portfolio_dashboard.html - Main interactive dashboard
- stock_data.db - SQLite database with:
  * Historical price data
  * Optimization results

Database Tables:
---------------
1. stock_prices - Historical OHLCV data
2. optimization_results - Optimized portfolio configurations

The optimization feature finds portfolios that balance return, risk, and
drawdown according to your preferences, then stores them for future reference.
"""