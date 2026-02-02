"""
Historical Backtester with Strategy Support.
Now supports dynamic allocation strategies, not just buy-and-hold.
"""
import pandas as pd
import numpy as np
from strategies import StaticAllocationStrategy, MarketContext


class Backtester:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def run_backtest(self, assets, allocations, initial_capital=10000,
                     benchmark_ticker=None, start_date_override=None,
                     strategy=None, contribution_amount=0.0, contribution_frequency=21):
        """
        Run historical backtest with optional dynamic strategy.

        Parameters:
        -----------
        assets : list
            List of asset dictionaries with 'ticker' and 'full_data'
        allocations : list
            Target allocation weights
        initial_capital : float
            Starting portfolio value
        benchmark_ticker : str
            Ticker for benchmark comparison
        start_date_override : date
            Start date for backtest
        strategy : AllocationStrategy, optional
            Dynamic allocation strategy (None = buy and hold)
        contribution_amount : float
            DCA amount per contribution
        contribution_frequency : int
            Trading days between contributions
        """

        # Prepare price data
        dfs = []
        for asset in assets:
            if 'full_data' in asset and not asset['full_data'].empty:
                s = asset['full_data']['Adj Close']
            else:
                df = self.data_manager.get_data(asset['ticker'])
                s = df['Adj Close']

            if start_date_override:
                s = s[s.index >= start_date_override]

            s.name = asset['ticker']
            dfs.append(s)

        prices = pd.concat(dfs, axis=1, join='inner').dropna()

        # Decide which backtest mode to use
        use_strategy = (strategy is not None) or (contribution_amount > 0)

        if not use_strategy:
            return self._run_static_backtest(
                prices, allocations, initial_capital,
                benchmark_ticker, assets
            )
        else:
            if strategy is None:
                strategy = StaticAllocationStrategy()
            return self._run_dynamic_backtest(
                prices, allocations, initial_capital,
                benchmark_ticker, assets, strategy,
                contribution_amount, contribution_frequency
            )

    def _run_static_backtest(self, prices, allocations, initial_capital,
                             benchmark_ticker, assets):
        """Original buy-and-hold backtest (fast)"""
        returns = prices.pct_change().fillna(0)

        # Portfolio construction
        allocations = np.array(allocations)
        portfolio_daily_rets = (returns * allocations).sum(axis=1)
        cumulative_growth = (1 + portfolio_daily_rets).cumprod()
        portfolio_value = cumulative_growth * initial_capital

        # Drawdown
        running_max = portfolio_value.cummax()
        drawdown_series = (portfolio_value - running_max) / running_max

        # Rolling returns
        roll_3y = cumulative_growth.rolling(window=756).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/3) - 1 if x.iloc[0] > 0 else 0
        )
        roll_5y = cumulative_growth.rolling(window=1260).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/5) - 1 if x.iloc[0] > 0 else 0
        )

        # Benchmark
        if benchmark_ticker is None:
            benchmark_ticker = assets[0]['ticker']
        bench_rets = returns[benchmark_ticker] if benchmark_ticker in returns.columns else portfolio_daily_rets

        # Metrics
        metrics = self._calculate_metrics(
            portfolio_value, portfolio_daily_rets, bench_rets,
            initial_capital, benchmark_ticker
        )

        return {
            'dates': portfolio_value.index,
            'values': portfolio_value.values,
            'drawdowns': drawdown_series,
            'rolling_3y': roll_3y,
            'rolling_5y': roll_5y,
            'metrics': metrics,
            'strategy': 'Buy and Hold'
        }

    def _run_dynamic_backtest(self, prices, allocations, initial_capital,
                               benchmark_ticker, assets, strategy,
                               contribution_amount, contribution_frequency):
        """
        Backtest with dynamic strategy - day-by-day simulation.
        """
        n_days = len(prices)
        n_assets = len(assets)
        asset_tickers = [a['ticker'] for a in assets]
        base_weights = np.array(allocations)

        # Initialize state
        # Holdings in dollars per asset
        holdings = base_weights * initial_capital

        # Tracking arrays
        portfolio_values = np.zeros(n_days)
        portfolio_values[0] = holdings.sum()

        # Peak tracking
        asset_peaks = holdings.copy()
        portfolio_peak = portfolio_values[0]

        # Indicator window
        indicator_window = max(21, getattr(strategy, 'lookback_days', 21))
        returns_buffer = np.zeros((indicator_window, n_assets))

        # Total invested (for metrics)
        total_invested = initial_capital

        # Contribution tracking
        contribution_dates = []
        allocation_history = []  # Track how allocations changed

        print(f"  > Running Historical Backtest with Strategy: {strategy.name}")
        if contribution_amount > 0:
            print(f"  > DCA: ${contribution_amount:,.0f} every {contribution_frequency} days")

        # Calculate daily returns
        price_array = prices.values
        daily_returns = np.zeros((n_days, n_assets))
        daily_returns[1:] = price_array[1:] / price_array[:-1] - 1

        for day in range(1, n_days):
            # Apply returns to holdings
            day_return = daily_returns[day]
            holdings = holdings * (1 + day_return)

            # Update portfolio value
            portfolio_values[day] = holdings.sum()

            # Update peaks
            asset_peaks = np.maximum(asset_peaks, holdings)
            portfolio_peak = max(portfolio_peak, portfolio_values[day])

            # Update returns buffer
            buffer_idx = day % indicator_window
            returns_buffer[buffer_idx] = day_return

            # Contribution day check
            if contribution_amount > 0 and day % contribution_frequency == 0:
                # Calculate indicators
                asset_drawdowns = (holdings - asset_peaks) / np.where(asset_peaks != 0, asset_peaks, 1)
                portfolio_drawdown = (portfolio_values[day] - portfolio_peak) / portfolio_peak if portfolio_peak > 0 else 0

                if day >= indicator_window:
                    rolling_returns = np.mean(returns_buffer, axis=0) * 252
                    rolling_volatility = np.std(returns_buffer, axis=0) * np.sqrt(252)
                    momentum_score = np.sum(returns_buffer, axis=0)
                    rolling_sharpe = rolling_returns / (rolling_volatility + 1e-6)
                else:
                    rolling_returns = None
                    rolling_volatility = None
                    momentum_score = None
                    rolling_sharpe = None

                # Build context (batch size = 1 for historical)
                context = MarketContext(
                    current_holdings=holdings.reshape(1, -1),
                    current_drawdowns=asset_drawdowns.reshape(1, -1),
                    base_allocations=base_weights,
                    asset_tickers=asset_tickers,
                    current_day=day,
                    total_days=n_days,
                    rolling_returns=rolling_returns.reshape(1, -1) if rolling_returns is not None else None,
                    rolling_volatility=rolling_volatility.reshape(1, -1) if rolling_volatility is not None else None,
                    rolling_sharpe=rolling_sharpe.reshape(1, -1) if rolling_sharpe is not None else None,
                    momentum_score=momentum_score.reshape(1, -1) if momentum_score is not None else None,
                    portfolio_drawdown=np.array([portfolio_drawdown])
                )

                # Get allocation from strategy
                new_weights = strategy.get_allocation(context).flatten()

                # Add contribution
                holdings += new_weights * contribution_amount
                total_invested += contribution_amount
                portfolio_values[day] = holdings.sum()

                # Log allocation decision
                contribution_dates.append(prices.index[day])
                allocation_history.append({
                    'date': prices.index[day],
                    'weights': new_weights.copy(),
                    'portfolio_dd': portfolio_drawdown,
                    'asset_dds': asset_drawdowns.copy()
                })

        # Build output series
        portfolio_series = pd.Series(portfolio_values, index=prices.index)

        # Drawdown series
        running_max = portfolio_series.cummax()
        drawdown_series = (portfolio_series - running_max) / running_max

        # Rolling returns
        cumulative_growth = portfolio_series / portfolio_series.iloc[0]
        roll_3y = cumulative_growth.rolling(window=756).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/3) - 1 if x.iloc[0] > 0 else 0
        )
        roll_5y = cumulative_growth.rolling(window=1260).apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (1/5) - 1 if x.iloc[0] > 0 else 0
        )

        # Benchmark
        returns = prices.pct_change().fillna(0)
        portfolio_daily_rets = portfolio_series.pct_change().fillna(0)

        if benchmark_ticker is None:
            benchmark_ticker = assets[0]['ticker']
        bench_rets = returns[benchmark_ticker] if benchmark_ticker in returns.columns else portfolio_daily_rets

        # Metrics (adjusted for contributions)
        metrics = self._calculate_metrics_with_dca(
            portfolio_series, portfolio_daily_rets, bench_rets,
            initial_capital, total_invested, benchmark_ticker
        )

        return {
            'dates': portfolio_series.index,
            'values': portfolio_series.values,
            'drawdowns': drawdown_series,
            'rolling_3y': roll_3y,
            'rolling_5y': roll_5y,
            'metrics': metrics,
            'strategy': strategy.name,
            'strategy_config': strategy.get_config_summary(),
            'allocation_history': allocation_history,
            'total_invested': total_invested
        }

    def _calculate_metrics(self, portfolio_value, portfolio_daily_rets, bench_rets,
                           initial_capital, benchmark_ticker):
        """Calculate standard performance metrics"""
        days = len(portfolio_value)
        years = days / 252

        cagr = (portfolio_value.iloc[-1] / initial_capital) ** (1/years) - 1 if years > 0 else 0
        ann_std = portfolio_daily_rets.std() * np.sqrt(252)

        active_rets = portfolio_daily_rets - bench_rets
        tracking_error = active_rets.std() * np.sqrt(252)
        info_ratio = (active_rets.mean() * 252 / tracking_error) if tracking_error > 0 else 0

        downside = portfolio_daily_rets[portfolio_daily_rets < 0]
        sortino = (cagr / (downside.std() * np.sqrt(252))) if downside.std() > 0 else 0

        running_max = portfolio_value.cummax()
        drawdown_series = (portfolio_value - running_max) / running_max

        return {
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
            'Correlation': portfolio_daily_rets.corr(bench_rets) if ann_std > 0 else 0.0,
            'Benchmark': benchmark_ticker
        }

    def _calculate_metrics_with_dca(self, portfolio_value, portfolio_daily_rets, bench_rets,
                                     initial_capital, total_invested, benchmark_ticker):
        """Calculate metrics accounting for DCA contributions"""
        days = len(portfolio_value)
        years = days / 252

        # Time-weighted return (approximation)
        # This is tricky with DCA - we use a simple approximation
        final_value = portfolio_value.iloc[-1]

        # Money-weighted return (IRR approximation)
        # Simple version: (Final - Total Invested) / Average Investment
        avg_invested = (initial_capital + total_invested) / 2
        simple_return = (final_value - total_invested) / avg_invested
        cagr = (1 + simple_return) ** (1/years) - 1 if years > 0 else simple_return

        ann_std = portfolio_daily_rets.std() * np.sqrt(252)

        active_rets = portfolio_daily_rets - bench_rets
        tracking_error = active_rets.std() * np.sqrt(252)
        info_ratio = (active_rets.mean() * 252 / tracking_error) if tracking_error > 0 else 0

        downside = portfolio_daily_rets[portfolio_daily_rets < 0]
        sortino = (cagr / (downside.std() * np.sqrt(252))) if downside.std() > 0 else 0

        running_max = portfolio_value.cummax()
        drawdown_series = (portfolio_value - running_max) / running_max

        return {
            'Start Balance': initial_capital,
            'Total Invested': total_invested,
            'End Balance': final_value,
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
            'Correlation': portfolio_daily_rets.corr(bench_rets) if ann_std > 0 else 0.0,
            'Benchmark': benchmark_ticker
        }


class StrategyComparison:
    """
    Utility class to compare multiple strategies on the same portfolio.
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.backtester = Backtester(data_manager)

    def compare_strategies(self, assets, base_allocations, strategies,
                           initial_capital=10000, start_date_override=None,
                           contribution_amount=0.0, contribution_frequency=21):
        """
        Run multiple strategies on the same portfolio and compare results.

        Parameters:
        -----------
        assets : list
            List of asset dictionaries
        base_allocations : list
            Base allocation weights
        strategies : list
            List of AllocationStrategy objects to compare

        Returns:
        --------
        dict : Comparison results with metrics for each strategy
        """
        results = {}

        print("\n" + "="*60)
        print("Strategy Comparison")
        print("="*60)

        for strategy in strategies:
            print(f"\nTesting: {strategy.name}")

            bt_result = self.backtester.run_backtest(
                assets=assets,
                allocations=base_allocations,
                initial_capital=initial_capital,
                start_date_override=start_date_override,
                strategy=strategy,
                contribution_amount=contribution_amount,
                contribution_frequency=contribution_frequency
            )

            results[strategy.name] = bt_result

            # Print summary
            m = bt_result['metrics']
            print(f"  CAGR: {m['CAGR']*100:.2f}%")
            print(f"  Max DD: {m['Max Drawdown']*100:.2f}%")
            print(f"  Sharpe: {m['Sharpe']:.2f}")

        return results

    def generate_comparison_report(self, results, output_path="strategy_comparison.html"):
        """Generate HTML report comparing strategies"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Value Over Time',
                'Drawdowns',
                'Rolling 3-Year Returns',
                'Risk-Return Comparison'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        metrics_table = []

        for idx, (name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]

            # Portfolio value
            fig.add_trace(go.Scatter(
                x=result['dates'], y=result['values'],
                name=name, line=dict(color=color),
                legendgroup=name
            ), row=1, col=1)

            # Drawdowns
            fig.add_trace(go.Scatter(
                x=result['dates'], y=result['drawdowns'] * 100,
                name=name, line=dict(color=color, width=1),
                fill='tozeroy', showlegend=False, legendgroup=name
            ), row=1, col=2)

            # Rolling returns
            fig.add_trace(go.Scatter(
                x=result['dates'], y=result['rolling_3y'] * 100,
                name=name, line=dict(color=color),
                showlegend=False, legendgroup=name
            ), row=2, col=1)

            # Risk-return scatter
            m = result['metrics']
            fig.add_trace(go.Scatter(
                x=[m['Stdev'] * 100], y=[m['CAGR'] * 100],
                mode='markers+text',
                text=[name],
                textposition='top center',
                marker=dict(size=15, color=color),
                showlegend=False
            ), row=2, col=2)

            metrics_table.append({
                'Strategy': name,
                'CAGR': f"{m['CAGR']*100:.2f}%",
                'Volatility': f"{m['Stdev']*100:.2f}%",
                'Max DD': f"{m['Max Drawdown']*100:.2f}%",
                'Sharpe': f"{m['Sharpe']:.2f}",
                'Sortino': f"{m['Sortino']:.2f}",
                'End Value': f"${m['End Balance']:,.0f}"
            })

        fig.update_yaxes(type="log", title="Value ($)", row=1, col=1)
        fig.update_yaxes(title="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title="3Y Ann. Return (%)", row=2, col=1)
        fig.update_xaxes(title="Volatility (%)", row=2, col=2)
        fig.update_yaxes(title="CAGR (%)", row=2, col=2)

        fig.update_layout(height=900, template='plotly_white')

        # Build HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
                th {{ background: #f5f5f5; }}
                td:first-child {{ text-align: left; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Strategy Comparison Report</h1>
            <h2>Performance Metrics</h2>
            <table>
                <tr>{''.join(f'<th>{k}</th>' for k in metrics_table[0].keys())}</tr>
                {''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row.values()) + '</tr>' for row in metrics_table)}
            </table>
            {pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"\n✓ Comparison report saved to: {output_path}")
        return metrics_table