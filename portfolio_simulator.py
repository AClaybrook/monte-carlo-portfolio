"""
Monte Carlo simulator: runs the shared engine over many simulated return paths.
"""
from datetime import date, timedelta

import numpy as np

import quant_analytics as qa
from engine import (SimulatedReturns, aligned_returns, build_cash_flows,
                    run_engine, simulated_schedule, vectorized_irr)
from run_config import RebalanceConfig

PERCENTILES = (10, 25, 50, 75, 90)


class PortfolioSimulator:
    def __init__(self, data_manager, sim_config):
        self.data_manager = data_manager
        self.config = sim_config
        self.initial_capital = sim_config.initial_capital
        self.simulations = sim_config.simulations
        self.years = sim_config.years
        self.contrib_amount = getattr(sim_config, 'contribution_amount', 0.0)
        self.contrib_freq = getattr(sim_config, 'contribution_frequency', 21)

        if getattr(sim_config, 'end_date', None):
            self.end_date = date.fromisoformat(sim_config.end_date)
        else:
            self.end_date = date.today()

    def _rng(self) -> np.random.Generator:
        seed = getattr(self.config, 'seed', None)
        if seed is None:
            # Draw from the legacy global RNG so np.random.seed() still makes runs reproducible
            seed = np.random.randint(0, 2 ** 31 - 1)
        return np.random.default_rng(seed)

    def define_asset_from_ticker(self, ticker, name=None, lookback_years=None):
        lookback_years = lookback_years or self.config.lookback_years
        start = self.end_date - timedelta(days=365 * lookback_years)
        df = self.data_manager.get_data(ticker, start, self.end_date)
        return self.define_asset_from_dataframe(ticker, name or ticker, df)

    def define_asset_from_dataframe(self, ticker, name, df):
        if df is None or df.empty:
            raise ValueError(f"No data provided for {ticker}")
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        returns = df[col].pct_change().dropna()
        return {'ticker': ticker, 'name': name, 'historical_returns': returns, 'full_data': df,
                'daily_mean': returns.mean(), 'daily_std': returns.std()}

    def _prepare_multivariate_data(self, assets, start_date_override=None):
        """Daily returns on common dates, computed from aligned prices."""
        aligned = aligned_returns(assets, start_date_override).dropna()
        if len(aligned) < 20:
            raise ValueError(f"Insufficient common history ({len(aligned)} days).")
        return aligned

    def simulate_portfolio(self, assets, allocations, method=None, start_date_override=None,
                           strategy=None, rebalance=None, apply_to='contributions',
                           check_frequency='monthly'):
        """
        Simulate `years` forward from resampled/fitted history of the assets.

        rebalance: RebalanceConfig or frequency string; None = buy and hold.
        Simulated years have as many days as the history's observed rows/year
        (252 for exchange-traded mixes, 365 for crypto-only portfolios).
        """
        method = method or self.config.method
        rebalance = RebalanceConfig.coerce(rebalance) or RebalanceConfig(frequency='none')
        hist = self._prepare_multivariate_data(assets, start_date_override)
        tickers = [a['ticker'] for a in assets]

        dpy = int(round(qa.infer_periods_per_year(hist.index)))
        n_days = self.years * dpy
        inflation = getattr(self.config, 'inflation_rate', 0.0) or 0.0
        source = SimulatedReturns(
            hist.values, method, self.simulations, self._rng(),
            block_size=getattr(self.config, 'block_size', 21),
            inflation_per_day=(1 + inflation) ** (1 / dpy) - 1,
        )
        cfg = self.config
        flows = build_cash_flows(
            n_days, np.arange(1, n_days + 1) / dpy, self.contrib_amount, self.contrib_freq,
            getattr(cfg, 'withdrawal_amount', 0.0), getattr(cfg, 'withdrawal_frequency', 'monthly'),
            getattr(cfg, 'withdrawal_start_years', 0.0), getattr(cfg, 'contribution_years', None),
            getattr(cfg, 'cash_flow_growth', 0.0), days_per_year=dpy)
        record_steps = np.unique(np.round(np.linspace(0, n_days, self.years * 12 + 1)).astype(int))

        res = run_engine(
            source, n_days, allocations, tickers, self.initial_capital,
            cash_flows=flows,
            rebalance_days=simulated_schedule(n_days, rebalance.frequency, dpy),
            rebalance_threshold=rebalance.threshold,
            transaction_cost_bps=rebalance.transaction_cost_bps,
            strategy=strategy, apply_to=apply_to,
            check_days=simulated_schedule(n_days, check_frequency, dpy) if strategy else None,
            periods_per_year=dpy, record_steps=record_steps,
        )

        final_values = res.values[:, -1]
        twr_final = np.maximum(res.twr[:, -1], 1e-12)
        cagr = twr_final ** (1 / self.years) - 1
        total_invested = self.initial_capital + float(flows[flows > 0].sum())
        if len(res.flow_steps):
            irr = vectorized_irr(self.initial_capital, res.flow_steps / dpy, res.flow_amounts,
                                 final_values, self.years)
        else:
            irr = cagr
        has_withdrawals = bool((flows < 0).any())
        withdrawn = -np.minimum(res.flow_amounts, 0).sum(axis=1) if has_withdrawals else np.zeros(len(cagr))

        record_years = record_steps / dpy
        # Money in the portfolio from outside, at each record point (for the report)
        cum_in = np.concatenate([[0.0], np.cumsum(np.maximum(flows, 0))])[record_steps]
        cum_net = np.concatenate([[0.0], np.cumsum(flows)])[record_steps]
        out = {
            'portfolio_values': res.values,
            'twr_paths': res.twr,
            'record_years': record_years,
            'final_values': final_values,
            'cagr': cagr,
            'irr': irr,
            'max_drawdowns': res.max_drawdown,
            'volatility': res.volatility,
            'assets': assets,
            'allocations': list(np.asarray(allocations, dtype=float)),
            'strategy': strategy.name if strategy else None,
            'strategy_config': strategy.get_config_summary() if strategy else None,
            'method': method,
            'days_per_year': dpy,
            'total_invested': total_invested,
            'invested_curve': self.initial_capital + cum_in,
            'net_invested_curve': self.initial_capital + cum_net,
            'has_withdrawals': has_withdrawals,
            'withdrawn': withdrawn,
            'depleted_at_years': np.where(res.depleted_at >= 0, res.depleted_at / dpy, np.nan),
            'real_dollars': inflation > 0,
            'history_start': hist.index[0].date(),
            'history_end': hist.index[-1].date(),
        }
        out['probabilities'] = self._calculate_probabilities(res, flows, dpy)
        out['stats'] = self.calculate_statistics(final_values, cagr, res.max_drawdown,
                                                 irr=irr, volatility=res.volatility,
                                                 total_invested=total_invested)
        if has_withdrawals:
            depleted = res.depleted_at >= 0
            out['stats'].update({
                'success_rate': float(np.mean(~depleted)),
                'median_withdrawn': float(np.median(withdrawn)),
                'median_depletion_year': (float(np.median(res.depleted_at[depleted] / dpy))
                                          if depleted.any() else None),
            })
        return out

    def _calculate_probabilities(self, res, flows, dpy):
        years = np.arange(1, self.years + 1)
        pos = np.searchsorted(res.record_steps, years * dpy)
        added = np.cumsum(np.maximum(flows, 0))
        prob_loss, prob_high, survival = [], [], []
        for y, p in zip(years, pos):
            invested = self.initial_capital + added[y * dpy - 1]
            prob_loss.append(np.mean(res.values[:, p] < invested))
            prob_high.append(np.mean(res.twr[:, p] ** (1 / y) - 1 > 0.10))
            survival.append(np.mean((res.depleted_at < 0) | (res.depleted_at > y * dpy)))
        return {'years': years, 'prob_loss': np.array(prob_loss),
                'prob_high_return': np.array(prob_high), 'survival': np.array(survival)}

    def calculate_statistics(self, final_values, cagr, max_drawdowns, irr=None,
                             volatility=None, total_invested=None):
        if total_invested is None:
            total_invested = self.initial_capital
        irr = cagr if irr is None else irr

        def pct(a):
            return {p: float(np.percentile(a, p)) for p in PERCENTILES}

        stats = {
            'total_invested': total_invested,
            'mean_final_value': float(np.mean(final_values)),
            'median_final_value': float(np.median(final_values)),
            'median_cagr': float(np.median(cagr)),
            'mean_cagr': float(np.mean(cagr)),
            'std_cagr': float(np.std(cagr)),
            'median_irr': float(np.median(irr)),
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'worst_max_drawdown': float(np.min(max_drawdowns)),
            'max_drawdown_95': float(np.percentile(max_drawdowns, 5)),
            'probability_loss': float(np.mean(final_values < total_invested)),
            'probability_double': float(np.mean(final_values >= 2 * total_invested)),
            'percentiles': {
                'final_value': pct(final_values),
                'cagr': pct(cagr),
                'irr': pct(irr),
                'max_drawdown': pct(max_drawdowns),
            },
        }
        if volatility is not None:
            stats['median_volatility'] = float(np.median(volatility))
            stats['percentiles']['volatility'] = pct(volatility)
        return stats
