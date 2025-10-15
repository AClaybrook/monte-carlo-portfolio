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

        if 'worst_max_drawdown' in weights:
            max_dd_score = 1 + stats['worst_max_drawdown']  # Convert to 0-1 range
            score += weights['worst_max_drawdown'] * max_dd_score

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

