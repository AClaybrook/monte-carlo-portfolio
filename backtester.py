"""
Historical backtester: runs the shared engine over one real price path.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import quant_analytics as qa
from engine import (HistoricalReturns, align_asset_prices, calendar_schedule,
                    contribution_schedule, run_engine)
from run_config import RebalanceConfig


def describe_policy(strategy, rebalance: RebalanceConfig, apply_to: str = 'contributions') -> str:
    parts = []
    if rebalance.frequency != 'none':
        parts.append(f"{rebalance.frequency.capitalize()} rebalance")
    if rebalance.threshold:
        parts.append(f"{rebalance.threshold:.0%} band")
    if not parts:
        parts.append("Buy and hold")
    if strategy is not None and strategy.name != 'Static':
        parts.append(f"{strategy.name} [{apply_to}]")
    return " + ".join(parts)


class Backtester:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager

    def run_backtest(self, assets, allocations, initial_capital=10000,
                     benchmark_ticker=None, start_date_override=None,
                     strategy=None, contribution_amount=0.0, contribution_frequency=21,
                     rebalance=None, apply_to: str = 'contributions',
                     check_frequency: str = 'monthly', risk_free_rate: float = 0.0,
                     benchmark: Optional[dict] = None, end_date=None) -> Dict:
        """
        Backtest one portfolio over its assets' common history.

        rebalance: RebalanceConfig or frequency string; None = buy and hold.
        benchmark: asset dict used for relative metrics. If omitted and
            benchmark_ticker is one of the assets, that asset is used.
        """
        rebalance = RebalanceConfig.coerce(rebalance) or RebalanceConfig(frequency='none')
        prices = align_asset_prices(assets, start_date_override, end_date)
        if len(prices) < 2:
            raise ValueError(f"Not enough overlapping history ({len(prices)} rows)")

        dates = prices.index
        returns = prices.pct_change().iloc[1:].values
        n_days = len(returns)
        tickers = [a['ticker'] for a in assets]
        ppy = qa.infer_periods_per_year(dates)

        contrib_days = (contribution_schedule(contribution_frequency, n_days, dates=dates)
                        if contribution_amount > 0 else np.zeros(n_days, dtype=bool))
        res = run_engine(
            HistoricalReturns(returns), n_days, allocations, tickers, initial_capital,
            contribution_amount=contribution_amount,
            contribution_days=contrib_days,
            rebalance_days=calendar_schedule(dates, rebalance.frequency),
            rebalance_threshold=rebalance.threshold,
            transaction_cost_bps=rebalance.transaction_cost_bps,
            strategy=strategy, apply_to=apply_to,
            check_days=calendar_schedule(dates, check_frequency) if strategy else None,
            periods_per_year=ppy, dates=dates,
        )

        balance = pd.Series(res.values[0], index=dates)
        twr = pd.Series(res.twr[0], index=dates)
        contributions = pd.Series(contribution_amount, index=dates[res.contribution_steps],
                                  dtype=float)

        bench_index, bench_name = self._benchmark_index(prices, benchmark, benchmark_ticker, tickers)
        metrics = qa.compute_performance(balance, twr, contributions, bench_index,
                                         bench_name, risk_free_rate)
        metrics['Transaction Costs'] = float(res.costs[0])
        metrics['Rebalances'] = int(res.n_trades[0])

        events = [dict(e, date=dates[e['step']]) for e in res.events]
        return {
            'dates': dates,
            'values': balance.values,
            'balance': balance,
            'twr': twr,
            'contributions': contributions,
            'total_invested': initial_capital + float(contributions.sum()),
            'drawdowns': qa.drawdown_series(twr),
            'rolling_1y': qa.rolling_annualized_return(twr, 1),
            'rolling_3y': qa.rolling_annualized_return(twr, 3),
            'rolling_5y': qa.rolling_annualized_return(twr, 5),
            'annual_returns': qa.annual_returns(twr),
            'monthly_table': qa.monthly_returns_table(twr),
            'drawdown_periods': qa.drawdown_periods(twr, top=10, min_depth=0.01),
            'weights': pd.DataFrame(res.weights[0], index=dates, columns=tickers),
            'events': events,
            'asset_prices': prices,
            'metrics': metrics,
            'strategy': describe_policy(strategy, rebalance, apply_to),
            'strategy_config': strategy.get_config_summary() if strategy else None,
            'benchmark_index': bench_index,
        }

    @staticmethod
    def _benchmark_index(prices, benchmark, benchmark_ticker, tickers):
        if benchmark is not None:
            b = align_asset_prices([benchmark], prices.index[0], prices.index[-1]).iloc[:, 0]
            return b / b.iloc[0], benchmark['ticker']
        if benchmark_ticker is None:
            benchmark_ticker = tickers[0]
        if benchmark_ticker in prices.columns:
            b = prices[benchmark_ticker]
            return b / b.iloc[0], benchmark_ticker
        return None, None


class StrategyComparison:
    """Run several strategies on the same assets and contribution plan."""

    def __init__(self, data_manager=None):
        self.backtester = Backtester(data_manager)

    def compare_strategies(self, assets, base_allocations, strategies: List,
                           initial_capital=10000, start_date_override=None,
                           contribution_amount=0.0, contribution_frequency=21,
                           rebalance=None, apply_to='contributions', risk_free_rate=0.0) -> Dict:
        results = {}
        for strategy in strategies:
            bt = self.backtester.run_backtest(
                assets, base_allocations, initial_capital,
                start_date_override=start_date_override, strategy=strategy,
                contribution_amount=contribution_amount,
                contribution_frequency=contribution_frequency,
                rebalance=rebalance, apply_to=apply_to, risk_free_rate=risk_free_rate)
            results[strategy.name] = bt
            m = bt['metrics']
            print(f"  {strategy.name:<40} CAGR {m['CAGR']:7.2%}  IRR {m['IRR']:7.2%}  "
                  f"MaxDD {m['Max Drawdown']:7.2%}  Sharpe {m['Sharpe']:5.2f}")
        return results
