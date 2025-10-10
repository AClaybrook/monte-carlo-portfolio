
# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
To use this multi-file structure:

1. Create separate files:
   - data_manager.py
   - portfolio_simulator.py
   - visualizations.py
   - main.py

2. Install required packages:
   pip install yfinance sqlalchemy pandas numpy plotly

3. Run the main script:
   python main.py

4. Output:
   - portfolio_dashboard.html (main interactive dashboard)
   - portfolio_*.html (individual portfolio analyses)
   - stock_data.db (SQLite database with cached data)

The code will:
- Download data from Yahoo Finance (first run)
- Cache data in SQLite database (subsequent runs are instant)
- Run 10,000 Monte Carlo simulations
- Generate interactive Plotly visualizations
- Open dashboard in your web browser

Interactive Features:
- Hover over plots for detailed information
- Zoom in/out on any chart
- Pan across time series
- Click legend to show/hide portfolios
- Export plots as PNG images

File Structure Benefits:
- Modular and maintainable
- Easy to extend with new features
- Separate concerns (data/simulation/visualization)
- Reusable components

Customization:
- Modify portfolio_configs in main.py to add portfolios
- Change simulation parameters in PortfolioSimulator()
- Adjust visualization settings in PortfolioVisualizer
- Add new analysis methods in any module
"""