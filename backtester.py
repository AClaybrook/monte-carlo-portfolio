"""
Historical Backtester with Rolling Metrics.
FIXED: Uses full available history from asset objects instead of re-fetching default 10y.
"""
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def run_backtest(self, assets, allocations, initial_capital=10000, benchmark_ticker=None, start_date_override=None):
        # ... inside loop ...
        if start_date_override:
            s = s[s.index >= start_date_override]
        # 1. Align Data (Use full_data from assets to ensure max history)
        dfs = []
        for asset in assets:
            # Use the already loaded full_data if available, otherwise fetch
            if 'full_data' in asset and not asset['full_data'].empty:
                s = asset['full_data']['Adj Close']
            else:
                # Fallback (should not happen usually)
                df = self.data_manager.get_data(asset['ticker'])
                s = df['Adj Close']

            s.name = asset['ticker']
            dfs.append(s)

        # Join Inner to find common history
        prices = pd.concat(dfs, axis=1, join='inner').dropna()
        returns = prices.pct_change().fillna(0)

        # 2. Portfolio Construction
        portfolio_daily_rets = (returns * allocations).sum(axis=1)
        cumulative_growth = (1 + portfolio_daily_rets).cumprod()
        portfolio_value = cumulative_growth * initial_capital

        # 3. Drawdown Series
        running_max = portfolio_value.cummax()
        drawdown_series = (portfolio_value - running_max) / running_max

        # 4. Rolling Returns (Annualized)
        roll_3y = cumulative_growth.rolling(window=756).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/3) - 1 if x.iloc[0] > 0 else 0
        )

        roll_5y = cumulative_growth.rolling(window=1260).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/5) - 1 if x.iloc[0] > 0 else 0
        )

        # 5. Benchmark & Metrics
        if benchmark_ticker is None: benchmark_ticker = assets[0]['ticker']
        # Handle benchmark alignment
        if benchmark_ticker in returns.columns:
            bench_rets = returns[benchmark_ticker]
        else:
            bench_rets = portfolio_daily_rets # Fallback

        days = len(portfolio_value)
        years = days / 252
        cagr = (portfolio_value.iloc[-1] / initial_capital) ** (1/years) - 1 if years > 0 else 0
        ann_std = portfolio_daily_rets.std() * np.sqrt(252)

        active_rets = portfolio_daily_rets - bench_rets
        tracking_error = active_rets.std() * np.sqrt(252)
        info_ratio = (active_rets.mean() * 252 / tracking_error) if tracking_error > 0 else 0

        downside = portfolio_daily_rets[portfolio_daily_rets < 0]
        sortino = (cagr / (downside.std() * np.sqrt(252))) if downside.std() > 0 else 0

        return {
            'dates': portfolio_value.index,
            'values': portfolio_value.values,
            'drawdowns': drawdown_series,
            'rolling_3y': roll_3y,
            'rolling_5y': roll_5y,
            'metrics': {
                'Start Balance': initial_capital,
                'End Balance': portfolio_value.iloc[-1],
                'CAGR': cagr,
                'Stdev': ann_std,
                'Best Year': portfolio_daily_rets.rolling(252).sum().max(),
                'Worst Year': portfolio_daily_rets.rolling(252).sum().min(),
                'Max Drawdown': drawdown_series.min(),
                'Sharpe': cagr / ann_std if ann_std > 0 else 0,
                'Sortino': sortino,
                'Active Return': active_rets.mean() * 252,
                'Tracking Error': tracking_error,
                'Info Ratio': info_ratio,
                'Correlation': portfolio_daily_rets.corr(bench_rets),
                'Benchmark': benchmark_ticker
            }
        }