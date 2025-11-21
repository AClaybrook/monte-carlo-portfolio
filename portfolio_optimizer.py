"""
Portfolio optimization algorithms - Memory efficient version with enhanced debugging.

FIXED:
- Better objective function balance
- Detailed debug output to identify scoring issues
- Warnings for anomalous portfolios
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

    def custom_objective(self, stats, weights, debug=False):
        """
        Custom objective function for portfolio optimization

        Enhanced with better balance and debug output

        Parameters:
        -----------
        stats : dict
            Portfolio statistics
        weights : dict
            Weights for each metric in objective function
        debug : bool
            If True, print detailed score breakdown

        Returns:
        --------
        float : Optimization score (higher is better)
        """
        score = 0
        components = {}

        # Expected return component (normalized so that 15% has a score of 1)
        if 'return' in weights:
            # Use more conservative normalization to prevent extreme portfolios
            return_normalized = stats['mean_cagr'] / 0.15
            return_score = min(return_normalized, 10)  # Cap at 5x instead of 1.5x
            score += weights['return'] * return_score
            components['return'] = weights['return'] * return_score

        # Sharpe ratio component (normalized so that 2 has a score of 1)
        # Changed from 3 to 2 for better balance
        if 'sharpe' in weights:
            sharpe_normalized = stats['sharpe_ratio'] / 3.0
            sharpe_score = min(sharpe_normalized, 10.0)
            score += weights['sharpe'] * sharpe_score
            components['sharpe'] = weights['sharpe'] * sharpe_score

        # Sortino ratio component (normalized so that 2 has a score of 1)
        # Changed from 3 to 2 for better balance
        if 'sortino' in weights:
            sortino_normalized = stats['sortino_ratio'] / 3.0
            sortino_score = min(sortino_normalized, 10.0)
            score += weights['sortino'] * sortino_score
            components['sortino'] = weights['sortino'] * sortino_score

        # Drawdown components (convert negative to positive)
        if 'drawdown' in weights:
            drawdown_score = 1 + stats['median_max_drawdown']
            score += weights['drawdown'] * max(drawdown_score, 0)
            components['drawdown'] = weights['drawdown'] * max(drawdown_score, 0)

        if 'drawdown_95' in weights:
            drawdown_95_score = 1 + stats['max_drawdown_95']
            score += weights['drawdown_95'] * max(drawdown_95_score, 0)
            components['drawdown_95'] = weights['drawdown_95'] * max(drawdown_95_score, 0)

        if 'worst_max_drawdown' in weights:
            max_dd_score = 1 + stats['worst_max_drawdown']
            score += weights['worst_max_drawdown'] * max(max_dd_score, 0)
            components['worst_max_drawdown'] = weights['worst_max_drawdown'] * max(max_dd_score, 0)

        # Probability components
        if 'prob_double' in weights:
            score += weights['prob_double'] * stats['probability_double']
            components['prob_double'] = weights['prob_double'] * stats['probability_double']

        if 'prob_loss_penalty' in weights:
            penalty = weights['prob_loss_penalty'] * stats['probability_loss']
            score -= penalty
            components['prob_loss_penalty'] = -penalty

        if debug:
            print(f"\n  Score Breakdown (Total: {score:.4f}):")
            for key, value in components.items():
                print(f"    {key:20s}: {value:7.4f}")

        return score

    def grid_search(self, assets, objective_weights, grid_points=5, top_n=10):
        """
        Memory-efficient grid search with enhanced debugging

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
        list : Top N portfolio configurations with scores (sorted best-first)
        """
        print(f"\n{'='*60}")
        print(f"Running Memory-Efficient Grid Search")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Grid points per asset: {grid_points}")
        print(f"Objective weights: {objective_weights}")

        # Validate weights sum to 1.0
        weight_sum = sum(objective_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            print(f"\n⚠ WARNING: Objective weights sum to {weight_sum:.3f}, not 1.0!")
            print(f"  This may cause unexpected scoring behavior.")

        n_assets = len(assets)
        allocations = np.linspace(0, 1, grid_points)

        # Generate valid combinations
        all_combinations = []
        for combo in product(allocations, repeat=n_assets):
            if abs(sum(combo) - 1.0) < 0.01:
                normalized = np.array(combo) / sum(combo)
                all_combinations.append(normalized.tolist())

        total_combos = len(all_combinations)
        print(f"Testing {total_combos} portfolio combinations...")
        print(f"Keeping only top {top_n} results in memory")

        top_heap = []

        # Track score statistics
        all_scores = []

        for i, allocation in enumerate(all_combinations):
            if (i + 1) % 100 == 0:
                current_worst = -top_heap[0][0] if top_heap else float('inf')
                if len(all_scores) > 0:
                    print(f"  Progress: {i+1}/{total_combos} | "
                          f"Top-{top_n} cutoff: {current_worst:.4f} | "
                          f"Scores: min={min(all_scores):.3f}, max={max(all_scores):.3f}, "
                          f"mean={np.mean(all_scores):.3f}")
                else:
                    print(f"  Progress: {i+1}/{total_combos}")

            try:
                sim_results = self.simulator.simulate_portfolio(assets, allocation)
                stats = sim_results['stats']

                # Calculate score with debug for first 3
                debug = (i < 3)
                score = self.custom_objective(stats, objective_weights, debug=debug)
                all_scores.append(score)

                # Debug first few portfolios
                if i < 3:
                    alloc_str = ' / '.join([f"{a*100:.0f}% {asset['ticker']}"
                                           for a, asset in zip(allocation, assets)])
                    print(f"\n  Portfolio {i}: {alloc_str}")
                    print(f"    Return: {stats['mean_cagr']*100:.2f}%, "
                          f"Sharpe: {stats['sharpe_ratio']:.3f}, "
                          f"Sortino: {stats['sortino_ratio']:.3f}")
                    print(f"    Median DD: {stats['median_max_drawdown']*100:.2f}%, "
                          f"Worst DD: {stats['worst_max_drawdown']*100:.2f}%")
                    print(f"    Final Score: {score:.4f}")

                # Check for anomalies
                if score > 2.0:
                    alloc_str = ' / '.join([f"{a*100:.0f}% {asset['ticker']}"
                                           for a, asset in zip(allocation, assets)])
                    print(f"\n  ⚠ High score detected: {score:.4f} for {alloc_str}")
                    print(f"    Return: {stats['mean_cagr']*100:.2f}%, "
                          f"Sharpe: {stats['sharpe_ratio']:.3f}, "
                          f"Sortino: {stats['sortino_ratio']:.3f}")

                # Store in heap
                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, (-score, i, allocation))
                elif score > -top_heap[0][0]:
                    heapq.heapreplace(top_heap, (-score, i, allocation))

                del sim_results

            except Exception as e:
                print(f"  Error with allocation {allocation}: {str(e)}")
                continue

        print(f"\n{'='*60}")
        print(f"Score Statistics Across All {len(all_scores)} Portfolios:")
        print(f"{'='*60}")
        print(f"  Min:    {min(all_scores):.4f}")
        print(f"  Max:    {max(all_scores):.4f}")
        print(f"  Mean:   {np.mean(all_scores):.4f}")
        print(f"  Median: {np.median(all_scores):.4f}")
        print(f"  Std:    {np.std(all_scores):.4f}")

        print(f"\n{'='*60}")
        print("Re-simulating top portfolios to get final results...")
        print(f"{'='*60}")

        # Re-simulate and recalculate
        top_results = []
        temp_results = []

        while top_heap:
            neg_score, idx, allocation = heapq.heappop(top_heap)
            temp_results.append((allocation, -neg_score))

        for i, (allocation, old_score) in enumerate(temp_results):
            sim_results = self.simulator.simulate_portfolio(assets, allocation)
            stats = sim_results['stats']
            new_score = self.custom_objective(stats, objective_weights, debug=False)

            alloc_str = ' / '.join([f"{a*100:.0f}% {asset['ticker']}"
                                   for a, asset in zip(allocation, assets)])

            print(f"\n  Re-simulated #{i+1}: {alloc_str}")
            print(f"    Old score: {old_score:.4f}, New score: {new_score:.4f} "
                  f"(diff: {new_score - old_score:+.4f})")

            top_results.append({
                'allocations': allocation,
                'score': new_score,
                'stats': stats,
                'results': sim_results
            })

        # Sort by score
        top_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")
        print(f"  Score range: {top_results[-1]['score']:.4f} to {top_results[0]['score']:.4f}")

        print(f"\n  Final Ranking:")
        for i, result in enumerate(top_results):
            alloc_str = ' / '.join([f"{a*100:.0f}% {asset['ticker']}"
                                   for a, asset in zip(result['allocations'], assets)])
            stats = result['stats']
            print(f"    #{i+1}: Score={result['score']:.4f} | {alloc_str}")
            print(f"        Return: {stats['mean_cagr']*100:.2f}%, "
                  f"Sharpe: {stats['sharpe_ratio']:.3f}, "
                  f"Sortino: {stats['sortino_ratio']:.3f}, "
                  f"DD: {stats['median_max_drawdown']*100:.2f}%")

        return top_results

    def random_search(self, assets, objective_weights, n_iterations=1000, top_n=10):
        """Memory-efficient random search with debugging"""
        print(f"\n{'='*60}")
        print(f"Running Memory-Efficient Random Search")
        print(f"{'='*60}")
        print(f"Assets: {[a['ticker'] for a in assets]}")
        print(f"Iterations: {n_iterations}")
        print(f"Objective weights: {objective_weights}")

        n_assets = len(assets)
        top_heap = []
        all_scores = []

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                current_worst = -top_heap[0][0] if top_heap else float('inf')
                if len(all_scores) > 0:
                    print(f"  Progress: {i+1}/{n_iterations} | "
                          f"Cutoff: {current_worst:.4f} | "
                          f"Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")

            allocation = np.random.dirichlet(np.ones(n_assets))

            try:
                sim_results = self.simulator.simulate_portfolio(assets, allocation.tolist())
                stats = sim_results['stats']
                score = self.custom_objective(stats, objective_weights)
                all_scores.append(score)

                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, (-score, i, allocation.tolist()))
                elif score > -top_heap[0][0]:
                    heapq.heapreplace(top_heap, (-score, i, allocation.tolist()))

                del sim_results

            except Exception as e:
                print(f"  Error: {str(e)}")
                continue

        print(f"\nScore statistics: min={min(all_scores):.4f}, "
              f"max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")

        # Re-simulate
        top_results = []
        temp_results = []

        while top_heap:
            neg_score, idx, allocation = heapq.heappop(top_heap)
            temp_results.append((allocation, -neg_score))

        for allocation, old_score in temp_results:
            sim_results = self.simulator.simulate_portfolio(assets, allocation)
            stats = sim_results['stats']
            new_score = self.custom_objective(stats, objective_weights)

            top_results.append({
                'allocations': allocation,
                'score': new_score,
                'stats': stats,
                'results': sim_results
            })

        top_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n✓ Optimization complete!")
        print(f"  Top score: {top_results[0]['score']:.4f}")

        return top_results

    def save_optimized_portfolios(self, optimization_results, prefix="Optimized"):
        """Save optimization results to database"""
        print(f"\n{'='*60}")
        print(f"Saving {len(optimization_results)} portfolios to database...")
        print(f"{'='*60}")

        for i, result in enumerate(optimization_results):
            portfolio_name = f"{prefix}_#{i+1}_Score_{result['score']:.4f}"

            print(f"  Saving Rank #{i+1}: {portfolio_name}")

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