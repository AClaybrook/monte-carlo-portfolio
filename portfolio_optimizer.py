"""
Portfolio optimization using SciPy with Enhanced Methods (Risk Parity, Sortino).
FIXED: Robust handling of NaNs/Infinites AND Consistent Timeframe Alignment.
"""
import numpy as np
import scipy.optimize as sco
from portfolio_simulator import PortfolioSimulator

class PortfolioOptimizer:
    def __init__(self, simulator: PortfolioSimulator, data_manager):
        self.simulator = simulator
        self.data_manager = data_manager

    def _get_data(self, assets, start_date_override=None):
        """
        Helper to get aligned covariance and returns with strict cleaning.
        UPDATED: Accepts start_date_override to align optimization timeframe.
        """
        try:
            # Pass the override down to the simulator's data preparer
            df = self.simulator._prepare_multivariate_data(assets, start_date_override=start_date_override)
        except ValueError as e:
            print(f"Optimization Skipped: {e}")
            return None, None, None

        # 1. Forward Fill to handle missing days (e.g. holidays differ)
        df = df.ffill()

        # 2. Calculate Returns
        returns = df.pct_change()

        # 3. CLEANING: Replace Infinites (divide by zero) and Drop NaNs
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        # 4. Check for Empty or Bad Data
        if len(returns) < 50:
            print(f"Optimization Skipped: Insufficient data points ({len(returns)}) after cleaning.")
            return None, None, None

        if not np.isfinite(returns.values).all():
             print(f"Optimization Skipped: Data still contains NaNs or Infinite values after cleaning.")
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

    def optimize_sharpe_ratio(self, assets, risk_free_rate=0.04, start_date_override=None):
        """Maximize Sharpe Ratio"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Max Sharpe", "No Data")

        def neg_sharpe(weights, mean_rets, cov_mat, rf):
            p_ret = np.sum(mean_rets * weights) * 252
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
            if p_vol == 0: return 0
            return - (p_ret - rf) / p_vol

        return self._minimize(neg_sharpe, assets, args=(mean_rets, cov_mat, risk_free_rate), label="Max Sharpe Ratio")

    def optimize_min_volatility(self, assets, start_date_override=None):
        """Minimize Volatility"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Min Volatility", "No Data")

        def port_vol(weights, cov_mat):
            return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)

        return self._minimize(port_vol, assets, args=(cov_mat,), label="Min Volatility")

    def optimize_sortino_ratio(self, assets, risk_free_rate=0.04, start_date_override=None):
        """Maximize Sortino Ratio"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Max Sortino", "No Data")

        def neg_sortino(weights, returns, rf):
            p_daily_rets = returns.dot(weights)
            ann_ret = np.mean(p_daily_rets) * 252
            downside = p_daily_rets[p_daily_rets < 0]
            downside_std = np.std(downside) * np.sqrt(252)
            if downside_std == 0: return -10
            return - (ann_ret - rf) / downside_std

        return self._minimize(neg_sortino, assets, args=(returns, risk_free_rate), label="Max Sortino (Growth)")

    def optimize_risk_parity(self, assets, start_date_override=None):
        """Equal Risk Contribution"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Risk Parity", "No Data")

        def risk_parity_objective(weights, cov_mat):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
            # Marginal Risk Contribution
            mrc = weights * np.dot(cov_mat, weights) / portfolio_volatility
            target_risk = portfolio_volatility / len(weights)
            return np.sum(np.square(mrc - target_risk))

        return self._minimize(risk_parity_objective, assets, args=(cov_mat,), label="Risk Parity")

    def optimize_custom_weighted(self, assets, weights_config, risk_free_rate=0.04, start_date_override=None):
        """
        Optimize based on a user-defined weighted sum of metrics.
        Minimizes: sum(weight * metric_penalty)
        """
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)
        if returns is None: return self._package_fail(assets, "Custom Objective", "No Data")

        # Normalize weights to ensure they sum to 1 (optional, but good practice)
        total_w = sum(weights_config.values())
        obj_weights = {k: v/total_w for k, v in weights_config.items()}

        def custom_objective(w, returns, mean_rets, cov_mat, rf, obj_weights):
            # 1. Calculate Portfolio Metrics
            # Annualized Return
            p_ret = np.sum(mean_rets * w) * 252

            # Volatility
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(252)

            # Sharpe
            p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0

            score = 0.0

            # 2. Add Weighted Penalties (We minimize the score)

            # Maximize Return -> Minimize Negative Return
            if 'return' in obj_weights:
                score += obj_weights['return'] * (-p_ret)

            # Maximize Sharpe -> Minimize Negative Sharpe
            if 'sharpe' in obj_weights:
                score += obj_weights['sharpe'] * (-p_sharpe)

            # Minimize Volatility -> Add Volatility
            if 'volatility' in obj_weights:
                score += obj_weights['volatility'] * p_vol

            # Minimize Drawdown (Computationally expensive, but included per request)
            if 'drawdown' in obj_weights or 'sortino' in obj_weights:
                # Generate series
                p_daily = returns.dot(w)

                if 'drawdown' in obj_weights:
                    # Quick approximate Max DD calculation
                    cum_ret = np.cumsum(p_daily)
                    running_max = np.maximum.accumulate(cum_ret)
                    # We use log returns approximation or simple cumulative for speed in optimizer
                    # Let's use simple cumulative sum as proxy for price for speed
                    dd = np.max(running_max - cum_ret)
                    score += obj_weights['drawdown'] * dd

                if 'sortino' in obj_weights:
                    downside = p_daily[p_daily < 0]
                    downside_std = np.std(downside) * np.sqrt(252)
                    sortino = (p_ret - rf) / downside_std if downside_std > 0 else 0
                    score += obj_weights['sortino'] * (-sortino)

            return score

        return self._minimize(
            custom_objective,
            assets,
            args=(returns, mean_rets, cov_mat, risk_free_rate, obj_weights),
            label="Custom Weighted"
        )

    def _package_result(self, scipy_result, assets, label):
        """Package results and run a final simulation check"""
        allocations = scipy_result.x / np.sum(scipy_result.x)
        # Note: We do NOT pass start_date_override here because the main loop
        # will run its own verification simulation/backtest with the correct date.
        sim_results = self.simulator.simulate_portfolio(assets, allocations)
        return {
            'label': label, 'score': 0, 'allocations': allocations,
            'stats': sim_results['stats'], 'results': sim_results
        }