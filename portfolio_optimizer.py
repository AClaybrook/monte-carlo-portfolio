"""
Portfolio optimization using SciPy with Enhanced Methods (Risk Parity, Sortino).
FIXED: Handled NaNs and insufficient data.
"""
import numpy as np
import scipy.optimize as sco
from portfolio_simulator import PortfolioSimulator

class PortfolioOptimizer:
    def __init__(self, simulator: PortfolioSimulator, data_manager):
        self.simulator = simulator
        self.data_manager = data_manager

    def _get_data(self, assets):
        """Helper to get aligned covariance and returns"""
        try:
            df = self.simulator._prepare_multivariate_data(assets)
        except ValueError as e:
            print(f"Optimization Skipped: {e}")
            return None, None, None

        # Calculate returns
        returns = df.pct_change().dropna()

        # CLEAN DATA CHECK
        if len(returns) < 50:
            print(f"Optimization Skipped: Insufficient data points ({len(returns)}) after alignment.")
            return None, None, None

        if not np.isfinite(returns.values).all():
             print(f"Optimization Skipped: Data contains NaNs or Infinite values.")
             return None, None, None

        return returns, returns.mean(), returns.cov()

    def _minimize(self, fun, assets, args, label):
        """Generic minimizer with RANDOM RESTARTS"""
        # Data validation check
        if args[0] is None:
            return self._package_fail(assets, label, "Bad Data")

        num_assets = len(assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        initial_guess = np.array(num_assets * [1. / num_assets,])

        # Try finding a solution
        try:
            result = sco.minimize(fun, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

            # Random Restarts if stuck
            best_result = result
            if not result.success or np.allclose(result.x, initial_guess):
                for _ in range(5):
                    rand_guess = np.random.random(num_assets)
                    rand_guess /= np.sum(rand_guess)
                    res = sco.minimize(fun, rand_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                    if res.fun < best_result.fun:
                        best_result = res

            return self._package_result(best_result, assets, label)

        except Exception as e:
            print(f"Optimizer Error ({label}): {e}")
            return self._package_fail(assets, label, str(e))

    def _package_fail(self, assets, label, reason):
        """Return a safe 'failed' result so the script continues"""
        num = len(assets)
        alloc = np.array([1/num]*num) # Default to equal weight
        return {
            'label': f"{label} (FAILED: {reason})",
            'score': 0,
            'allocations': alloc,
            'stats': self.simulator.simulate_portfolio(assets, alloc)['stats'],
            'results': self.simulator.simulate_portfolio(assets, alloc)
        }

    # --- Optimizers ---

    def optimize_sharpe_ratio(self, assets, risk_free_rate=0.04):
        returns, mean_rets, cov_mat = self._get_data(assets)
        if returns is None: return self._package_fail(assets, "Max Sharpe", "No Data")

        def neg_sharpe(weights, mean_rets, cov_mat, rf):
            p_ret = np.sum(mean_rets * weights) * 252
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
            if p_vol == 0: return 0
            return - (p_ret - rf) / p_vol

        return self._minimize(neg_sharpe, assets, args=(mean_rets, cov_mat, risk_free_rate), label="Max Sharpe Ratio")

    def optimize_min_volatility(self, assets):
        returns, mean_rets, cov_mat = self._get_data(assets)
        if returns is None: return self._package_fail(assets, "Min Volatility", "No Data")

        def port_vol(weights, cov_mat):
            return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)

        return self._minimize(port_vol, assets, args=(cov_mat,), label="Min Volatility")

    def optimize_sortino_ratio(self, assets, risk_free_rate=0.04):
        returns, mean_rets, cov_mat = self._get_data(assets)
        if returns is None: return self._package_fail(assets, "Max Sortino", "No Data")

        def neg_sortino(weights, returns, rf):
            p_daily_rets = returns.dot(weights)
            ann_ret = np.mean(p_daily_rets) * 252
            downside = p_daily_rets[p_daily_rets < 0]
            downside_std = np.std(downside) * np.sqrt(252)
            if downside_std == 0: return -10
            return - (ann_ret - rf) / downside_std

        return self._minimize(neg_sortino, assets, args=(returns, risk_free_rate), label="Max Sortino (Growth)")

    def optimize_risk_parity(self, assets):
        returns, mean_rets, cov_mat = self._get_data(assets)
        if returns is None: return self._package_fail(assets, "Risk Parity", "No Data")

        def risk_parity_objective(weights, cov_mat):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
            mrc = weights * np.dot(cov_mat, weights) / portfolio_volatility
            target_risk = portfolio_volatility / len(weights)
            return np.sum(np.square(mrc - target_risk))

        return self._minimize(risk_parity_objective, assets, args=(cov_mat,), label="Risk Parity")

    def _package_result(self, scipy_result, assets, label):
        allocations = scipy_result.x / np.sum(scipy_result.x)
        sim_results = self.simulator.simulate_portfolio(assets, allocations)
        return {
            'label': label, 'score': 0, 'allocations': allocations,
            'stats': sim_results['stats'], 'results': sim_results
        }