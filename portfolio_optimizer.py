"""
Portfolio optimization algorithms - Memory efficient version.
Handles thousands of portfolios without memory issues.
"""

import numpy as np
from itertools import product
import heapq
import random

class PortfolioOptimizer:
    """Optimizes portfolio allocations with memory-efficient processing"""

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

        Returns:
        --------
        float : Optimization score (higher is better)
        """
        score = 0

        # Expected return component (normalized so that 15% has a score of 1)
        if 'return' in weights:
            return_score = stats['mean_cagr'] / 0.15
            score += weights['return'] * return_score

        # Sharpe ratio component (normalized so that 3 has a score of 1)
        if 'sharpe' in weights:
            sharpe_score = stats['sharpe_ratio'] / 3.0
            score += weights['sharpe'] * sharpe_score

        # Sortino ratio component (normalized so that 3 has a score of 1)
        if 'sortino' in weights:
            sortino_score = stats['sortino_ratio'] / 3.0
            score += weights['sortino'] * sortino_score

        # Drawdown component (1 - abs(drawdown))
        if 'drawdown' in weights:
            drawdown_score = stats['median_max_drawdown']
            score += weights['drawdown'] * drawdown_score

        if 'drawdown_95' in weights:
            drawdown_95_score = stats['max_drawdown_95']
            score += weights['drawdown_95'] * drawdown_95_score

        if 'worst_max_drawdown' in weights:
            max_dd_score = stats['worst_max_drawdown']
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
        Memory-efficient grid search using a heap to keep only top N results

        Parameters:
        -----------
        assets : list
            List of asset definitions
        objective_weights : dict
            Weights for objective function
        grid_points : int
            Number of points per dimension
        top_n : int
            Number of top portfolios to return

        Returns:
        --------
        list : Top N portfolio configurations with scores
        """
        print(f"\n{'='*60}")
        print(f"Running Memory-Efficient Grid Search")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Grid points per asset: {grid_points}")
        print(f"Objective weights: {objective_weights}")

        n_assets = len(assets)
        allocations = np.linspace(0, 1, grid_points)

        # Generate valid combinations (sum to 1.0)
        all_combinations = []
        for combo in product(allocations, repeat=n_assets):
            if abs(sum(combo) - 1.0) < 0.01:
                normalized = np.array(combo) / sum(combo)
                all_combinations.append(normalized.tolist())

        total_combos = len(all_combinations)
        print(f"Testing {total_combos} portfolio combinations...")
        print(f"Keeping only top {top_n} results in memory")

        # Use a min-heap to keep only top N results
        # Heap stores tuples: (-score, index, allocation, stats)
        # Negative score because heapq is a min-heap
        top_heap = []

        for i, allocation in enumerate(all_combinations):
            if (i + 1) % 100 == 0:
                current_worst = -top_heap[0][0] if top_heap else float('inf')
                print(f"  Progress: {i+1}/{total_combos} | Current top-{top_n} cutoff: {current_worst:.4f}")

            try:
                # Run simulation - only keep stats, not full results
                sim_results = self.simulator.simulate_portfolio(assets, allocation)
                stats = sim_results['stats']
                score = self.custom_objective(stats, objective_weights)

                # Only store if it's in top N
                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, (-score, i, allocation, stats))
                elif score > -top_heap[0][0]:  # Better than worst in heap
                    heapq.heapreplace(top_heap, (-score, i, allocation, stats))

                # Clear sim_results to free memory
                del sim_results

            except Exception as e:
                print(f"  Error with allocation {allocation}: {str(e)}")
                continue

        # Convert heap to sorted list (best first)
        top_results = []
        while top_heap:
            neg_score, idx, allocation, stats = heapq.heappop(top_heap)
            score = -neg_score

            # Re-run simulation for top results to get full data
            sim_results = self.simulator.simulate_portfolio(assets, allocation)

            top_results.append({
                'allocations': allocation,
                'score': score,
                'stats': stats,
                'results': sim_results
            })

        # Reverse to get best-first order
        top_results.reverse()

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")
        print(f"  Peak memory usage: Only {top_n} portfolios kept in memory")

        return top_results

    def random_search(self, assets, objective_weights, n_iterations=1000, top_n=10):
        """
        Memory-efficient random search using a heap

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
        print(f"Running Memory-Efficient Random Search")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Iterations: {n_iterations}")
        print(f"Objective weights: {objective_weights}")

        n_assets = len(assets)
        top_heap = []

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                current_worst = -top_heap[0][0] if top_heap else float('inf')
                print(f"  Progress: {i+1}/{n_iterations} | Current top-{top_n} cutoff: {current_worst:.4f}")

            # Generate random allocation using Dirichlet distribution
            allocation = np.random.dirichlet(np.ones(n_assets))

            try:
                sim_results = self.simulator.simulate_portfolio(assets, allocation.tolist())
                stats = sim_results['stats']
                score = self.custom_objective(stats, objective_weights)

                # Only store if it's in top N
                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, (-score, i, allocation.tolist(), stats))
                elif score > -top_heap[0][0]:
                    heapq.heapreplace(top_heap, (-score, i, allocation.tolist(), stats))

                del sim_results

            except Exception as e:
                print(f"  Error with allocation {allocation}: {str(e)}")
                continue

        # Convert heap to sorted list
        top_results = []
        while top_heap:
            neg_score, idx, allocation, stats = heapq.heappop(top_heap)
            score = -neg_score

            # Re-run simulation for top results
            sim_results = self.simulator.simulate_portfolio(assets, allocation)

            top_results.append({
                'allocations': allocation,
                'score': score,
                'stats': stats,
                'results': sim_results
            })

        top_results.reverse()

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")

        return top_results

    def grid_search_batched(self, assets, objective_weights, grid_points=5, top_n=10, batch_size=500):
        """
        Alternative: Process grid search in batches for even better memory control

        Useful for very large grid searches (e.g., 10+ assets)
        """
        print(f"\n{'='*60}")
        print(f"Running Batched Grid Search")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Grid points: {grid_points}, Batch size: {batch_size}")

        n_assets = len(assets)
        allocations = np.linspace(0, 1, grid_points)

        # Generate combinations in batches
        all_combinations = []
        for combo in product(allocations, repeat=n_assets):
            if abs(sum(combo) - 1.0) < 0.01:
                normalized = np.array(combo) / sum(combo)
                all_combinations.append(normalized.tolist())

        total_combos = len(all_combinations)
        print(f"Total combinations: {total_combos}")
        print(f"Processing in batches of {batch_size}")

        top_heap = []

        # Process in batches
        for batch_start in range(0, total_combos, batch_size):
            batch_end = min(batch_start + batch_size, total_combos)
            batch = all_combinations[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end})...")

            for i, allocation in enumerate(batch):
                global_idx = batch_start + i

                if (global_idx + 1) % 100 == 0:
                    current_worst = -top_heap[0][0] if top_heap else float('inf')
                    print(f"  Progress: {global_idx+1}/{total_combos} | Cutoff: {current_worst:.4f}")

                try:
                    sim_results = self.simulator.simulate_portfolio(assets, allocation)
                    stats = sim_results['stats']
                    score = self.custom_objective(stats, objective_weights)

                    if len(top_heap) < top_n:
                        heapq.heappush(top_heap, (-score, global_idx, allocation, stats))
                    elif score > -top_heap[0][0]:
                        heapq.heapreplace(top_heap, (-score, global_idx, allocation, stats))

                    del sim_results

                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue

        # Convert heap to results
        top_results = []
        while top_heap:
            neg_score, idx, allocation, stats = heapq.heappop(top_heap)
            sim_results = self.simulator.simulate_portfolio(assets, allocation)
            top_results.append({
                'allocations': allocation,
                'score': -neg_score,
                'stats': stats,
                'results': sim_results
            })

        top_results.reverse()
        print(f"\n✓ Batched optimization complete!")
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