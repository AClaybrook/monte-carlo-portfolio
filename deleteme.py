# File structure:
# ├── data_manager.py           - Database and data downloading
# ├── portfolio_simulator.py    - Monte Carlo simulation engine
# ├── portfolio_optimizer.py    - Portfolio optimization algorithms
# ├── visualizations.py          - Interactive Plotly visualizations
# └── main.py                    - Main script to run simulations

# =============================================================================
# data_manager.py
# =============================================================================

# =============================================================================
# portfolio_simulator.py
# =============================================================================
"""
Monte Carlo portfolio simulation engine.
Generates return scenarios and calculates portfolio statistics.
"""

import numpy as np
from datetime import datetime, timedelta, date

class PortfolioSimulator:
    """Monte Carlo portfolio simulator using real historical data"""

    def __init__(self, data_manager, initial_capital=10000, years=10, simulations=10000):
        self.data_manager = data_manager
        self.initial_capital = initial_capital
        self.years = years
        self.simulations = simulations
        self.trading_days = 252 * years

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=10):
        """Define an asset by analyzing historical data from a ticker"""
        if name is None:
            name = ticker

        end_date = date.today()
        start_date = end_date - timedelta(days=365*lookback_years)
        df = self.data_manager.get_data(ticker, start_date, end_date)

        returns = df['Adj Close'].pct_change().dropna()

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        daily_return = returns.mean()
        daily_volatility = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        data_start = df.index.min().date()
        data_end = df.index.max().date()

        asset = {
            'ticker': ticker,
            'name': name,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'daily_return': daily_return,
            'daily_volatility': daily_volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'historical_returns': returns,
            'data_points': len(returns),
            'start_date': data_start,
            'end_date': data_end
        }

        print(f"\n{name} ({ticker}) Statistics:")
        print(f"  Annual Return: {annual_return*100:.2f}%")
        print(f"  Annual Volatility: {annual_volatility*100:.2f}%")
        print(f"  Sharpe Ratio (0% rf): {annual_return/annual_volatility:.2f}")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Kurtosis: {kurtosis:.2f}")
        print(f"  Data points: {len(returns)} ({data_start} to {data_end})")

        return asset

    def generate_returns(self, asset, method='bootstrap'):
        """Generate simulated returns for an asset"""
        if method == 'bootstrap':
            historical = asset['historical_returns'].values
            if len(historical) < 100:
                print(f"  Warning: {asset['name']} has only {len(historical)} data points. Using parametric method.")
                method = 'parametric'
            else:
                indices = np.random.randint(0, len(historical), (self.simulations, self.trading_days))
                return historical[indices]

        if method == 'parametric':
            return np.random.normal(
                asset['daily_return'],
                asset['daily_volatility'],
                (self.simulations, self.trading_days)
            )
        elif method == 'geometric_brownian':
            dt = 1/252
            drift = (asset['annual_return'] - 0.5 * asset['annual_volatility']**2) * dt
            shock = asset['annual_volatility'] * np.sqrt(dt)
            random_shocks = np.random.normal(0, 1, (self.simulations, self.trading_days))
            return drift + shock * random_shocks

    def simulate_portfolio(self, assets, allocations, method='bootstrap'):
        """Run Monte Carlo simulation for a portfolio"""
        assert len(assets) == len(allocations), "Assets and allocations must match"
        assert abs(sum(allocations) - 1.0) < 0.01, f"Allocations must sum to 1.0"

        all_returns = [self.generate_returns(asset, method) for asset in assets]

        portfolio_returns = np.zeros((self.simulations, self.trading_days))
        for returns, weight in zip(all_returns, allocations):
            portfolio_returns += returns * weight

        portfolio_values = np.zeros((self.simulations, self.trading_days + 1))
        portfolio_values[:, 0] = self.initial_capital

        for day in range(self.trading_days):
            portfolio_values[:, day + 1] = portfolio_values[:, day] * (1 + portfolio_returns[:, day])

        final_values = portfolio_values[:, -1]
        total_returns = (final_values / self.initial_capital) - 1
        max_drawdowns = self.calculate_max_drawdown(portfolio_values)
        cagr = (final_values / self.initial_capital) ** (1 / self.years) - 1

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'final_values': final_values,
            'total_returns': total_returns,
            'cagr': cagr,
            'max_drawdowns': max_drawdowns,
            'assets': assets,
            'allocations': allocations,
            'stats': self.calculate_statistics(final_values, total_returns, cagr, max_drawdowns)
        }

    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown for each simulation path"""
        max_drawdowns = np.zeros(self.simulations)
        for i in range(self.simulations):
            running_max = np.maximum.accumulate(portfolio_values[i, :])
            drawdown = (portfolio_values[i, :] - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)
        return max_drawdowns

    def calculate_statistics(self, final_values, total_returns, cagr, max_drawdowns):
        """Calculate comprehensive statistics"""
        downside_returns = cagr[cagr < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino_ratio = np.mean(cagr) / downside_std

        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'mean_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'mean_cagr': np.mean(cagr),
            'median_cagr': np.median(cagr),
            'std_cagr': np.std(cagr),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_loss': np.mean(final_values < self.initial_capital),
            'probability_double': np.mean(final_values >= 2 * self.initial_capital),
            'sharpe_ratio': np.mean(cagr) / np.std(cagr) if np.std(cagr) > 0 else 0,
            'sortino_ratio': sortino_ratio
        }

    def print_detailed_stats(self, results, label):
        """Print detailed statistics for a portfolio"""
        stats = results['stats']

        print(f"\n{'='*60}")
        print(f"Portfolio: {label}")
        print(f"{'='*60}")
        print(f"\nAllocation:")
        for asset, weight in zip(results['assets'], results['allocations']):
            print(f"  {asset['name']}: {weight*100:.1f}%")

        print(f"\nFinal Value Statistics:")
        print(f"  Mean: ${stats['mean_final_value']:,.2f}")
        print(f"  Median: ${stats['median_final_value']:,.2f}")
        print(f"  5th-95th Percentile: ${stats['percentile_5']:,.2f} - ${stats['percentile_95']:,.2f}")

        print(f"\nReturn Metrics:")
        print(f"  Mean CAGR: {stats['mean_cagr']*100:.2f}%")
        print(f"  Median CAGR: {stats['median_cagr']*100:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Median Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {stats['sortino_ratio']:.3f}")

        print(f"\nProbabilities:")
        print(f"  P(Loss): {stats['probability_loss']*100:.2f}%")
        print(f"  P(Double): {stats['probability_double']*100:.2f}%")


# =============================================================================
# portfolio_optimizer.py
# =============================================================================
"""
Portfolio optimization algorithms.
Finds optimal asset allocations based on custom objective functions.
"""

import numpy as np
from itertools import product
import random

class PortfolioOptimizer:
    """Optimizes portfolio allocations"""

    def __init__(self, simulator, data_manager):
        self.simulator = simulator
        self.data_manager = data_manager

    def custom_objective(self, stats, weights):
        """
        Custom objective function for portfolio optimization

        Parameters:
        -----------
        stats : dict
            Portfolio statistics
        weights : dict
            Weights for each metric in objective function
            Example: {'return': 0.5, 'sharpe': 0.2, 'drawdown': 0.3}

        Returns:
        --------
        float : Optimization score (higher is better)
        """
        score = 0

        # Expected return component (normalized to 0-1 range, assuming 0-30% annual return)
        if 'return' in weights:
            return_score = min(stats['mean_cagr'] / 0.30, 1.0)
            score += weights['return'] * return_score

        # Sharpe ratio component (normalized, assuming 0-3 range)
        if 'sharpe' in weights:
            sharpe_score = min(stats['sharpe_ratio'] / 3.0, 1.0)
            score += weights['sharpe'] * sharpe_score

        # Sortino ratio component (normalized, assuming 0-4 range)
        if 'sortino' in weights:
            sortino_score = min(stats['sortino_ratio'] / 4.0, 1.0)
            score += weights['sortino'] * sortino_score

        # Drawdown component (1 - abs(drawdown), since drawdown is negative)
        if 'drawdown' in weights:
            drawdown_score = 1 + stats['median_max_drawdown']  # Convert to 0-1 range
            score += weights['drawdown'] * drawdown_score

        # Probability of doubling component
        if 'prob_double' in weights:
            score += weights['prob_double'] * stats['probability_double']

        # Penalty for high probability of loss
        if 'prob_loss_penalty' in weights:
            score -= weights['prob_loss_penalty'] * stats['probability_loss']

        return score

    def grid_search(self, assets, objective_weights, grid_points=5, top_n=10):
        """
        Grid search optimization over asset allocations

        Parameters:
        -----------
        assets : list
            List of asset definitions
        objective_weights : dict
            Weights for objective function
        grid_points : int
            Number of points per dimension (more = finer but slower)
        top_n : int
            Number of top portfolios to return

        Returns:
        --------
        list : Top N portfolio configurations with scores
        """
        print(f"\n{'='*60}")
        print(f"Running Grid Search Optimization")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Grid points per asset: {grid_points}")
        print(f"Objective weights: {objective_weights}")

        n_assets = len(assets)

        # Generate allocation grid (ensuring they sum to 1.0)
        allocations = np.linspace(0, 1, grid_points)

        # Generate all combinations
        all_combinations = []
        for combo in product(allocations, repeat=n_assets):
            if abs(sum(combo) - 1.0) < 0.01:  # Allow small floating point errors
                # Normalize to exactly 1.0
                normalized = np.array(combo) / sum(combo)
                all_combinations.append(normalized.tolist())

        print(f"Testing {len(all_combinations)} portfolio combinations...")

        results = []
        for i, allocation in enumerate(all_combinations):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(all_combinations)}")

            try:
                sim_results = self.simulator.simulate_portfolio(assets, allocation)
                stats = sim_results['stats']
                score = self.custom_objective(stats, objective_weights)

                results.append({
                    'allocations': allocation,
                    'score': score,
                    'stats': stats,
                    'results': sim_results
                })
            except Exception as e:
                print(f"  Error with allocation {allocation}: {str(e)}")
                continue

        # Sort by score and return top N
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_n]

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")

        return top_results

    def random_search(self, assets, objective_weights, n_iterations=1000, top_n=10):
        """
        Random search optimization (useful for many assets)

        Parameters:
        -----------
        assets : list
            List of asset definitions
        objective_weights : dict
            Weights for objective function
        n_iterations : int
            Number of random portfolios to test
        top_n : int
            Number of top portfolios to return

        Returns:
        --------
        list : Top N portfolio configurations with scores
        """
        print(f"\n{'='*60}")
        print(f"Running Random Search Optimization")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Iterations: {n_iterations}")
        print(f"Objective weights: {objective_weights}")

        n_assets = len(assets)
        results = []

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_iterations}")

            # Generate random allocation
            allocation = np.random.dirichlet(np.ones(n_assets))

            try:
                sim_results = self.simulator.simulate_portfolio(assets, allocation.tolist())
                stats = sim_results['stats']
                score = self.custom_objective(stats, objective_weights)

                results.append({
                    'allocations': allocation.tolist(),
                    'score': score,
                    'stats': stats,
                    'results': sim_results
                })
            except Exception as e:
                print(f"  Error with allocation {allocation}: {str(e)}")
                continue

        # Sort by score and return top N
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_n]

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")

        return top_results

    def save_optimized_portfolios(self, optimization_results, prefix="Optimized"):
        """Save optimization results to database"""
        for i, result in enumerate(optimization_results):
            portfolio_name = f"{prefix}_#{i+1}_Score_{result['score']:.4f}"

            self.data_manager.save_optimization_result(
                portfolio_name=portfolio_name,
                assets=result['results']['assets'],
                allocations=result['allocations'],
                stats=result['stats'],
                optimization_params={
                    'score': result['score'],
                    'rank': i + 1
                }
            )

        print(f"\n✓ Saved {len(optimization_results)} optimized portfolios to database")


# =============================================================================
# visualizations.py (Part 1)
# =============================================================================
"""
Interactive Plotly visualizations for portfolio analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class PortfolioVisualizer:
    """Creates interactive Plotly visualizations with synchronized legend"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def create_dashboard(self, portfolio_configs):
        """Create comprehensive interactive dashboard with synchronized legends"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Sample Portfolio Trajectories',
                'Distribution of Final Values',
                'CAGR Distribution',
                'Return Distribution Comparison',
                'Maximum Drawdown Distribution',
                'Percentile Fan Chart',
                'Risk-Return Profile',
                'Key Statistics',
                'Probability of Outcomes'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'box'}, {'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'table'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Use legendgroup to synchronize legend clicks across all subplots
        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            legendgroup = f"group{idx}"

            # 1. Sample trajectories - add to first subplot only for legend
            sample_indices = np.random.choice(self.simulator.simulations, 50, replace=False)
            for sample_idx in sample_indices[:3]:
                fig.add_trace(
                    go.Scatter(
                        y=results['portfolio_values'][sample_idx, :],
                        mode='lines',
                        line=dict(color=color, width=0.5),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=legendgroup,
                        hovertemplate='Day: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )

            median_path = np.median(results['portfolio_values'], axis=0)
            fig.add_trace(
                go.Scatter(
                    y=median_path,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=3),
                    legendgroup=legendgroup,
                    showlegend=True,
                    hovertemplate='Day: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Final value distribution
            fig.add_trace(
                go.Histogram(
                    x=results['final_values'],
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. CAGR distribution
            fig.add_trace(
                go.Histogram(
                    x=results['cagr'] * 100,
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='CAGR: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=3
            )

            # 4. Box plot
            fig.add_trace(
                go.Box(
                    y=results['total_returns'] * 100,
                    name=label,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )

            # 5. Max drawdown
            fig.add_trace(
                go.Histogram(
                    x=results['max_drawdowns'] * 100,
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Max DD: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )

            # 6. Percentile fan chart
            values = results['portfolio_values']
            p50 = np.percentile(values, 50, axis=0)
            p5 = np.percentile(values, 5, axis=0)
            p95 = np.percentile(values, 95, axis=0)

            days = np.arange(values.shape[1])

            # Add shaded area
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p95,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=legendgroup,
                    hoverinfo='skip'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p5,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    showlegend=False,
                    legendgroup=legendgroup,
                    hovertemplate='Day: %{x}<br>5th-95th: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p50,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=3
            )

            # 7. Risk-return scatter
            stats = results['stats']
            fig.add_trace(
                go.Scatter(
                    x=[stats['std_cagr'] * 100],
                    y=[stats['mean_cagr'] * 100],
                    mode='markers+text',
                    name=label,
                    marker=dict(size=15, color=color),
                    text=[label],
                    textposition='top center',
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )

        # 8. Statistics table
        table_data = []
        for results, label in zip(results_list, labels):
            stats = results['stats']
            table_data.append([
                label,
                f"${stats['median_final_value']:,.0f}",
                f"{stats['median_cagr']*100:.1f}%",
                f"{stats['median_max_drawdown']*100:.1f}%",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['sortino_ratio']:.2f}"
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Portfolio', 'Median Value', 'Median CAGR', 'Median DD', 'Sharpe', 'Sortino'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=10, color='black')
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=9)
                )
            ),
            row=3, col=2
        )

        # 9. Probability outcomes
        prob_loss = [r['stats']['probability_loss'] * 100 for r in results_list]
        prob_double = [r['stats']['probability_double'] * 100 for r in results_list]

        # Use same legendgroup for bars
        for idx, label in enumerate(labels):
            legendgroup = f"group{idx}"
            color = self.colors[idx % len(self.colors)]

            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[prob_loss[idx]],
                    name=f'{label} - Loss',
                    marker_color=color,
                    opacity=0.5,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f'{label}<br>P(Loss): %{{y:.1f}}%<extra></extra>'
                ),
                row=3, col=3
            )

            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[prob_double[idx]],
                    name=f'{label} - Double',
                    marker_color=color,
                    opacity=0.8,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f'{label}<br>P(Double): %{{y:.1f}}%<extra></extra>'
                ),
                row=3, col=3
            )

        # Update axes labels
        fig.update_xaxes(title_text="Trading Days", row=1, col=1)
        fig.update_xaxes(title_text="Final Value ($)", row=1, col=2)
        fig.update_xaxes(title_text="CAGR (%)", row=1, col=3)
        fig.update_xaxes(title_text="Portfolio", row=2, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)
        fig.update_xaxes(title_text="Trading Days", row=2, col=3)
        fig.update_xaxes(title_text="CAGR Volatility (%)", row=3, col=1)
        fig.update_xaxes(title_text="", row=3, col=3)

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=3)
        fig.update_yaxes(title_text="Mean CAGR (%)", row=3, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=3, col=3)

        fig.update_layout(
            height=1400,
            title_text="Monte Carlo Portfolio Simulation Dashboard - Click Legend to Toggle Portfolios",
            title_font_size=18,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='closest',
            barmode='group'
        )

        return fig


# =============================================================================
# main.py
# =============================================================================
"""
Main script to run Monte Carlo portfolio simulations with optimization.
"""

from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer

def main():
    # Initialize
    data_manager = DataManager(db_path='stock_data.db')
    sim = PortfolioSimulator(
        data_manager=data_manager,
        initial_capital=100000,
        years=10,
        simulations=10000
    )
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)

    print("="*60)
    print("Monte Carlo Portfolio Simulator with Optimization")
    print("="*60)
    print("Loading and analyzing historical data...")
    print("="*60)

    # Define assets
    voo = sim.define_asset_from_ticker('VOO', name='VOO (S&P 500)')
    tqqq = sim.define_asset_from_ticker('TQQQ', name='TQQQ (3x Nasdaq)')
    bnd = sim.define_asset_from_ticker('BND', name='BND (Bonds)')
    qqq = sim.define_asset_from_ticker('QQQ', name='QQQ (Nasdaq 100)')

    # =================================================================
    # PART 1: Standard Portfolio Analysis
    # =================================================================
    print("\n" + "="*60)
    print("PART 1: Standard Portfolio Analysis")
    print("="*60)

    portfolio_configs = [
        {
            'label': '100% VOO',
            'assets': [voo],
            'allocations': [1.0],
            'results': None
        },
        {
            'label': '80% VOO / 20% TQQQ',
            'assets': [voo, tqqq],
            'allocations': [0.8, 0.2],
            'results': None
        },
        {
            'label': '50% VOO / 50% QQQ',
            'assets': [voo, qqq],
            'allocations': [0.5, 0.5],
            'results': None
        },
        {
            'label': '70% VOO / 20% QQQ / 10% BND',
            'assets': [voo, qqq, bnd],
            'allocations': [0.7, 0.2, 0.1],
            'results': None
        },
        {
            'label': '60% VOO / 30% TQQQ / 10% BND',
            'assets': [voo, tqqq, bnd],
            'allocations': [0.6, 0.3, 0.1],
            'results': None
        }
    ]

    # Run simulations
    for config in portfolio_configs:
        print(f"\nSimulating: {config['label']}...")
        config['results'] = sim.simulate_portfolio(
            config['assets'],
            config['allocations'],
            method='bootstrap'
        )

    # Print statistics
    for config in portfolio_configs:
        sim.print_detailed_stats(config['results'], config['label'])

    # =================================================================
    # PART 2: Portfolio Optimization
    # =================================================================
    print("\n" + "="*60)
    print("PART 2: Portfolio Optimization")
    print("="*60)

    # Define objective function weights
    objective_weights = {
        'return': 0.50,      # 50% weight on expected return
        'sharpe': 0.20,      # 20% weight on Sharpe ratio
        'drawdown': 0.30     # 30% weight on (1 - max drawdown)
    }

    # Run optimization on 3-asset portfolio
    optimization_assets = [voo, qqq, bnd]

    # Use grid search for 3 assets (manageable)
    top_portfolios = optimizer.grid_search(
        assets=optimization_assets,
        objective_weights=objective_weights,
        grid_points=6,  # 6^3 = 216 combinations
        top_n=5
    )

    # Save optimization results to database
    optimizer.save_optimized_portfolios(top_portfolios, prefix="GridSearch_3Asset")

    # Add top optimized portfolios to visualization
    for i, opt_result in enumerate(top_portfolios[:3]):  # Add top 3
        alloc_str = ' / '.join([
            f"{int(a*100)}% {asset['ticker']}"
            for a, asset in zip(opt_result['allocations'], optimization_assets)
        ])

        portfolio_configs.append({
            'label': f"Optimized #{i+1}: {alloc_str}",
            'assets': optimization_assets,
            'allocations': opt_result['allocations'],
            'results': opt_result['results']
        })

    # Print optimization results
    print("\n" + "="*60)
    print("Top Optimized Portfolios:")
    print("="*60)
    for i, opt_result in enumerate(top_portfolios):
        print(f"\nRank #{i+1} - Score: {opt_result['score']:.4f}")
        print("Allocation:")
        for asset, alloc in zip(optimization_assets, opt_result['allocations']):
            print(f"  {asset['ticker']}: {alloc*100:.1f}%")
        stats = opt_result['stats']
        print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {stats['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

    # =================================================================
    # PART 3: Advanced Optimization - Random Search with 4 Assets
    # =================================================================
    print("\n" + "="*60)
    print("PART 3: Advanced Optimization (4 Assets - Random Search)")
    print("="*60)

    # For 4+ assets, use random search (grid search becomes too slow)
    optimization_assets_4 = [voo, qqq, tqqq, bnd]

    top_portfolios_4 = optimizer.random_search(
        assets=optimization_assets_4,
        objective_weights=objective_weights,
        n_iterations=500,  # Test 500 random portfolios
        top_n=3
    )

    # Save to database
    optimizer.save_optimized_portfolios(top_portfolios_4, prefix="RandomSearch_4Asset")

    # Add to visualization
    for i, opt_result in enumerate(top_portfolios_4):
        alloc_str = ' / '.join([
            f"{int(a*100)}% {asset['ticker']}"
            for a, asset in zip(opt_result['allocations'], optimization_assets_4)
        ])

        portfolio_configs.append({
            'label': f"4-Asset Opt #{i+1}: {alloc_str}",
            'assets': optimization_assets_4,
            'allocations': opt_result['allocations'],
            'results': opt_result['results']
        })

    print("\n" + "="*60)
    print("Top 4-Asset Optimized Portfolios:")
    print("="*60)
    for i, opt_result in enumerate(top_portfolios_4):
        print(f"\nRank #{i+1} - Score: {opt_result['score']:.4f}")
        print("Allocation:")
        for asset, alloc in zip(optimization_assets_4, opt_result['allocations']):
            print(f"  {asset['ticker']}: {alloc*100:.1f}%")
        stats = opt_result['stats']
        print(f"Expected Return: {stats['mean_cagr']*100:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {stats['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {stats['median_max_drawdown']*100:.2f}%")

    # =================================================================
    # PART 4: Visualizations
    # =================================================================
    print("\n" + "="*60)
    print("Generating interactive visualizations...")
    print("="*60)

    # Create main dashboard with all portfolios (standard + optimized)
    dashboard = visualizer.create_dashboard(portfolio_configs)
    dashboard.write_html('portfolio_dashboard.html')
    print("✓ Saved dashboard to: portfolio_dashboard.html")

    # Show main dashboard in browser
    dashboard.show()

    # =================================================================
    # Database Information
    # =================================================================
    print("\n" + "="*60)
    print("Database Information")
    print("="*60)
    print(f"Database file: stock_data.db")

    print(f"\nStored tickers:")
    tickers = data_manager.list_all_tickers()
    for ticker in tickers:
        info = data_manager.get_ticker_info(ticker)
        if info:
            print(f"  {ticker}: {info['record_count']} records ({info['start_date']} to {info['end_date']})")

    print(f"\nStored optimization results:")
    opt_results = data_manager.get_optimization_results(limit=10)
    for result in opt_results:
        print(f"  {result['portfolio_name']}")
        print(f"    Score: {result['score']:.4f}, CAGR: {result['median_cagr']*100:.2f}%, Sharpe: {result['sharpe_ratio']:.3f}")

    # Clean up
    data_manager.close()
    print("\n✓ Analysis complete!")
    print("\nInteractive Features:")
    print("  - Click legend items to show/hide portfolios across ALL charts")
    print("  - Hover over any point for detailed information")
    print("  - Zoom and pan on any chart")
    print("  - Double-click legend to isolate a single portfolio")

if __name__ == "__main__":
    main()


# =============================================================================
# USAGE INSTRUCTIONS & FEATURES
# =============================================================================
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