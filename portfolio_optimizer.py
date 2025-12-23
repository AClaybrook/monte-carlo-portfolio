"""
Portfolio optimization using SciPy with Enhanced Methods.
FIXED:
- Min volatility now excludes leveraged ETFs by default
- Custom weighted objective normalized properly
- Better constraint handling
"""
import numpy as np
import scipy.optimize as sco
from portfolio_simulator import PortfolioSimulator

# Leveraged ETFs that shouldn't dominate "min volatility" portfolios
LEVERAGED_ETFS = {'TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'UPRO', 'TMF', 'TMV', 'UDOW', 'SDOW',
                  'QLD', 'QID', 'SSO', 'SDS', 'UVXY', 'SVXY', 'SOXL', 'SOXS'}
LEVERAGED_ETFS = {}

class PortfolioOptimizer:
    def __init__(self, simulator: PortfolioSimulator, data_manager):
        self.simulator = simulator
        self.data_manager = data_manager

    def _get_data(self, assets, start_date_override=None):
        """
        Helper to get aligned covariance and returns with strict cleaning.
        """
        try:
            df = self.simulator._prepare_multivariate_data(assets, start_date_override=start_date_override)
        except ValueError as e:
            print(f"Optimization Skipped: {e}")
            return None, None, None

        df = df.ffill()
        returns = df.pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 50:
            print(f"Optimization Skipped: Insufficient data points ({len(returns)}) after cleaning.")
            return None, None, None

        if not np.isfinite(returns.values).all():
             print(f"Optimization Skipped: Data still contains NaNs or Infinite values after cleaning.")
             return None, None, None

        return returns, returns.mean(), returns.cov()

    def _minimize(self, fun, assets, args, label, bounds=None, min_weight=0.0, max_weight=1.0):
        """Generic minimizer with RANDOM RESTARTS and configurable bounds"""
        if args[0] is None:
            return self._package_fail(assets, label, "Bad Data")

        num_assets = len(assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        if bounds is None:
            bounds = tuple((min_weight, max_weight) for _ in range(num_assets))

        initial_guess = np.array(num_assets * [1. / num_assets,])

        try:
            result = sco.minimize(fun, initial_guess, args=args, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'ftol': 1e-9, 'maxiter': 1000})

            best_result = result
            # Random restarts
            for _ in range(10):
                rand_guess = np.random.random(num_assets)
                rand_guess /= np.sum(rand_guess)
                res = sco.minimize(fun, rand_guess, args=args, method='SLSQP',
                                   bounds=bounds, constraints=constraints,
                                   options={'ftol': 1e-9, 'maxiter': 1000})
                if res.success and res.fun < best_result.fun:
                    best_result = res

            return self._package_result(best_result, assets, label)

        except Exception as e:
            print(f"Optimizer Error ({label}): {e}")
            return self._package_fail(assets, label, str(e))

    def _package_fail(self, assets, label, reason):
        """Return a safe 'failed' result so the script continues"""
        num = len(assets)
        alloc = np.array([1/num]*num)
        return {
            'label': f"{label} (FAILED: {reason})",
            'score': 0,
            'allocations': alloc,
            'stats': self.simulator.simulate_portfolio(assets, alloc)['stats'],
            'results': self.simulator.simulate_portfolio(assets, alloc)
        }

    def optimize_sharpe_ratio(self, assets, risk_free_rate=0.04, start_date_override=None):
        """Maximize Sharpe Ratio"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Max Sharpe", "No Data")

        def neg_sharpe(weights, mean_rets, cov_mat, rf):
            p_ret = np.sum(mean_rets * weights) * 252
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
            if p_vol == 0: return 0
            return - (p_ret - rf) / p_vol

        return self._minimize(neg_sharpe, assets, args=(mean_rets, cov_mat, risk_free_rate),
                              label="Max Sharpe Ratio")

    def optimize_min_volatility(self, assets, start_date_override=None,
                                 exclude_leveraged=True, max_leveraged_weight=0.10):
        """
        Minimize Volatility

        Parameters:
            exclude_leveraged: If True, cap leveraged ETF weights (they shouldn't dominate min vol)
            max_leveraged_weight: Maximum weight for any single leveraged ETF
        """
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Min Volatility", "No Data")

        def port_vol(weights, cov_mat):
            return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)

        # Build custom bounds - cap leveraged ETFs
        bounds = []
        for asset in assets:
            ticker = asset['ticker'].upper()
            if exclude_leveraged and ticker in LEVERAGED_ETFS:
                bounds.append((0.0, max_leveraged_weight))
            else:
                bounds.append((0.0, 1.0))

        result = self._minimize(port_vol, assets, args=(cov_mat,), label="Min Volatility",
                                bounds=tuple(bounds))

        # Verify result makes sense
        allocations = result['allocations']
        for i, asset in enumerate(assets):
            if asset['ticker'].upper() in LEVERAGED_ETFS and allocations[i] > 0.15:
                print(f"  ⚠ Warning: {asset['ticker']} at {allocations[i]*100:.1f}% in Min Vol portfolio")

        return result

    def optimize_sortino_ratio(self, assets, risk_free_rate=0.04, start_date_override=None):
        """Maximize Sortino Ratio"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Max Sortino", "No Data")

        def neg_sortino(weights, returns, rf):
            p_daily_rets = returns.dot(weights)
            ann_ret = np.mean(p_daily_rets) * 252
            downside = p_daily_rets[p_daily_rets < 0]

            # Need enough downside days for meaningful calculation
            if len(downside) < 20:
                # Penalize assets with too few negative days (like cash)
                # Can't calculate meaningful Sortino
                return 100  # Return LARGE positive = bad score for minimizer

            downside_std = np.std(downside) * np.sqrt(252)
            if downside_std < 0.001:
                # Near-zero downside vol means we can't trust Sortino
                return 100

            sortino = (ann_ret - rf) / downside_std
            return -sortino  # Negative because we're minimizing

        return self._minimize(neg_sortino, assets, args=(returns, risk_free_rate),
                              label="Max Sortino (Growth)")

    def optimize_risk_parity(self, assets, start_date_override=None):
        """Equal Risk Contribution"""
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)

        if returns is None: return self._package_fail(assets, "Risk Parity", "No Data")

        def risk_parity_objective(weights, cov_mat):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
            if portfolio_volatility == 0:
                return 1e10
            mrc = weights * np.dot(cov_mat, weights) / portfolio_volatility
            target_risk = portfolio_volatility / len(weights)
            return np.sum(np.square(mrc - target_risk))

        return self._minimize(risk_parity_objective, assets, args=(cov_mat,), label="Risk Parity")

    def optimize_custom_weighted(self, assets, weights_config, risk_free_rate=0.04,
                                  start_date_override=None):
        """
        Optimize based on a user-defined weighted sum of metrics.

        FIXED: Proper normalization of each metric to similar scales before combining.
        """
        returns, mean_rets, cov_mat = self._get_data(assets, start_date_override=start_date_override)
        if returns is None: return self._package_fail(assets, "Custom Objective", "No Data")

        # Normalize weights
        total_w = sum(weights_config.values())
        if total_w == 0:
            total_w = 1
        obj_weights = {k: v/total_w for k, v in weights_config.items()}

        # Pre-compute baseline metrics for normalization
        equal_w = np.ones(len(assets)) / len(assets)
        baseline_ret = np.sum(mean_rets * equal_w) * 252
        baseline_vol = np.sqrt(np.dot(equal_w.T, np.dot(cov_mat, equal_w))) * np.sqrt(252)

        def custom_objective(w, returns, mean_rets, cov_mat, rf, obj_weights, baseline_ret, baseline_vol):
            # Calculate metrics
            p_ret = np.sum(mean_rets * w) * 252
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(252)

            if p_vol < 1e-6:
                p_vol = 1e-6

            p_sharpe = (p_ret - rf) / p_vol

            score = 0.0

            # NORMALIZED contributions (all roughly same scale)

            # Return: higher is better -> minimize negative normalized return
            if 'return' in obj_weights and obj_weights['return'] > 0:
                # Normalize to baseline (score of 1.0 = equal weight return)
                norm_ret = p_ret / max(abs(baseline_ret), 0.01)
                score += obj_weights['return'] * (-norm_ret)

            # Sharpe: higher is better
            if 'sharpe' in obj_weights and obj_weights['sharpe'] > 0:
                score += obj_weights['sharpe'] * (-p_sharpe)

            # Volatility: lower is better
            if 'volatility' in obj_weights and obj_weights['volatility'] > 0:
                norm_vol = p_vol / max(baseline_vol, 0.01)
                score += obj_weights['volatility'] * norm_vol

            # Drawdown & Sortino (need daily returns)
            if ('drawdown' in obj_weights and obj_weights['drawdown'] > 0) or \
               ('sortino' in obj_weights and obj_weights['sortino'] > 0):
                p_daily = returns.dot(w)

                if 'drawdown' in obj_weights and obj_weights['drawdown'] > 0:
                    cum_ret = (1 + p_daily).cumprod()
                    running_max = np.maximum.accumulate(cum_ret)
                    dd = (cum_ret - running_max) / running_max
                    max_dd = abs(dd.min())
                    # Normalize: typical max DD is 0.1-0.5
                    score += obj_weights['drawdown'] * (max_dd * 5)  # Scale to ~1 for 20% DD

                if 'sortino' in obj_weights and obj_weights['sortino'] > 0:
                    downside = p_daily[p_daily < 0]
                    if len(downside) > 0:
                        downside_std = np.std(downside) * np.sqrt(252)
                        sortino = (p_ret - rf) / downside_std if downside_std > 0 else 0
                    else:
                        sortino = 5.0  # Very good
                    score += obj_weights['sortino'] * (-sortino / 2)  # Normalize ~1-3 range

            return score

        return self._minimize(
            custom_objective,
            assets,
            args=(returns, mean_rets, cov_mat, risk_free_rate, obj_weights, baseline_ret, baseline_vol),
            label="Custom Weighted"
        )

    def _package_result(self, scipy_result, assets, label):
        """Package results and run a final simulation check"""
        allocations = scipy_result.x / np.sum(scipy_result.x)

        # Clean tiny allocations
        allocations[allocations < 0.001] = 0
        allocations = allocations / np.sum(allocations)

        sim_results = self.simulator.simulate_portfolio(assets, allocations)

        # Print allocation
        alloc_str = " | ".join([f"{a['ticker']}: {w*100:.1f}%"
                                for a, w in zip(assets, allocations) if w > 0.001])
        print(f"  → {label}: {alloc_str}")

        return {
            'label': label,
            'score': -scipy_result.fun if scipy_result.success else 0,
            'allocations': allocations,
            'stats': sim_results['stats'],
            'results': sim_results
        }