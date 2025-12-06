"""
Visualization engine - With Volatility, DCA Indicator, and MC/Historical Alignment
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import pandas as pd
from scipy import stats

class PortfolioVisualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def _hex_to_rgba(self, hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    def _get_kde_curve(self, data, clip_percentile=1.0):
        try:
            min_val = np.percentile(data, clip_percentile)
            max_val = np.percentile(data, 100 - clip_percentile)
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(min_val, max_val, 200)
            y_range = kde(x_range)
            return x_range, y_range
        except:
            return [], []

    def _get_contribution_info(self, item) -> str:
        """Extract contribution info from portfolio result"""
        results = item.get('results', {})
        backtest = item.get('backtest', {})

        # Check for total invested (indicates DCA)
        total_invested = backtest.get('metrics', {}).get('Total Invested', None)
        if total_invested is None:
            total_invested = results.get('total_invested', None)

        start_balance = backtest.get('metrics', {}).get('Start Balance',
                                                         results.get('stats', {}).get('initial_capital', 10000))

        if total_invested and total_invested > start_balance * 1.01:
            contributions = total_invested - start_balance
            return f"DCA: ${contributions:,.0f}"
        else:
            return "Lump Sum"

    def _get_strategy_info(self, item) -> str:
        """Extract strategy name from portfolio result"""
        results = item.get('results', {})
        backtest = item.get('backtest', {})

        strategy = backtest.get('strategy', results.get('strategy', 'Buy and Hold'))
        return strategy if strategy else 'Buy and Hold'

    def create_allocation_table_html(self, portfolio_results):
        """Creates a formatted HTML table showing allocations for ALL portfolios."""
        all_tickers = set()
        portfolio_rows = []

        for item in portfolio_results:
            p_name = item['label']
            res = item['results']
            allocations = res['allocations']
            assets = res['assets']

            row_data = {'Portfolio': p_name}

            for asset, weight in zip(assets, allocations):
                ticker = asset['ticker'].upper()
                all_tickers.add(ticker)
                row_data[ticker] = weight

            portfolio_rows.append(row_data)

        sorted_tickers = sorted(list(all_tickers))

        html = '<div style="overflow-x: auto;"><table class="allocation-table">'
        html += '<thead><tr><th style="text-align:left;">Portfolio</th>'
        for ticker in sorted_tickers:
            html += f'<th>{ticker}</th>'
        html += '</tr></thead>'

        html += '<tbody>'
        for row_dict in portfolio_rows:
            p_name = row_dict['Portfolio']
            html += f'<tr><td style="font-weight:bold; text-align:left; background-color: #f8f9fa;">{p_name}</td>'

            for ticker in sorted_tickers:
                val = row_dict.get(ticker, 0.0)

                if val > 0.001:
                    display_val = f"{val*100:.1f}%"
                    style = 'style="font-weight:500;"'
                else:
                    display_val = "-"
                    style = 'style="color: #ccc;"'

                html += f'<td {style}>{display_val}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'

        return html

    def create_monte_carlo_plot(self, portfolio_results):
        """Monte Carlo visualization with probability plots"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('95% Confidence Trajectories', 'Final Value Density', 'CAGR Density',
                            'Risk-Return Profile', 'Risk of Loss over Time', 'Prob. of >10% Annual Return'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15, horizontal_spacing=0.08
        )

        for idx, item in enumerate(portfolio_results):
            label = item['label']
            res = item['results']
            stats_data = res['stats']
            probs = res.get('probabilities', {})

            color = self.colors[idx % len(self.colors)]
            fill_color = self._hex_to_rgba(color, 0.15)
            grp = f"mc_{idx}"

            # 1. Trajectory
            p5 = np.percentile(res['portfolio_values'], 5, axis=0)
            p50 = np.median(res['portfolio_values'], axis=0)
            p95 = np.percentile(res['portfolio_values'], 95, axis=0)
            days = np.arange(len(p50))

            fig.add_trace(go.Scatter(x=days, y=p95, mode='lines', line=dict(width=0), showlegend=False, legendgroup=grp, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=days, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fill_color, showlegend=False, legendgroup=grp, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=days, y=p50, mode='lines', line=dict(color=color, width=2), name=label, legendgroup=grp, hovertemplate=f"<b>{label}</b><br>Median: $%{{y:,.0f}}"), row=1, col=1)

            # 2. Final Value KDE
            x_kde, y_kde = self._get_kde_curve(res['final_values'])
            fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode='lines', line=dict(color=color, width=1), fill='tozeroy', fillcolor=fill_color, showlegend=False, legendgroup=grp, name=label, hovertemplate=f"<b>{label}</b><br>Value: $%{{x:,.0f}}<br>Density: %{{y:.2e}}"), row=1, col=2)

            # 3. CAGR KDE
            x_cagr, y_cagr = self._get_kde_curve(res['cagr'] * 100)
            fig.add_trace(go.Scatter(x=x_cagr, y=y_cagr, mode='lines', line=dict(color=color, width=1), fill='tozeroy', fillcolor=fill_color, showlegend=False, legendgroup=grp, name=label, hovertemplate=f"<b>{label}</b><br>CAGR: %{{x:.1f}}%"), row=1, col=3)

            # 4. Risk Return
            fig.add_trace(go.Scatter(x=[stats_data['std_cagr']*100], y=[stats_data['mean_cagr']*100], mode='markers', marker=dict(color=color, size=14, line=dict(width=1, color='black')), showlegend=False, legendgroup=grp, name=label, hovertemplate=f"<b>{label}</b><br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%"), row=2, col=1)

            # 5. Risk of Loss Over Time
            if 'years' in probs:
                fig.add_trace(go.Scatter(
                    x=probs['years'], y=probs['prob_loss']*100,
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, legendgroup=grp, name=label,
                    hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Prob Loss: %{{y:.1f}}%"
                ), row=2, col=2)

            # 6. Prob of High Return
            if 'years' in probs:
                fig.add_trace(go.Scatter(
                    x=probs['years'], y=probs['prob_high_return']*100,
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, legendgroup=grp, name=label,
                    hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Prob >10%: %{{y:.1f}}%"
                ), row=2, col=3)

        fig.update_yaxes(tickformat="$,.0f", title="Portfolio Value", row=1, col=1)
        fig.update_xaxes(tickformat="$,.0s", title="Final Value", row=1, col=2)
        fig.update_xaxes(tickformat=".1f", title="CAGR (%)", row=1, col=3)

        fig.update_xaxes(tickformat=".1f", title="Volatility (%)", row=2, col=1)
        fig.update_yaxes(tickformat=".1f", title="Return (%)", row=2, col=1)

        fig.update_yaxes(tickformat=".0f", title="Probability (%)", range=[0, 100], row=2, col=2)
        fig.update_xaxes(title="Years Invested", row=2, col=2)

        fig.update_yaxes(tickformat=".0f", title="Probability (%)", range=[0, 100], row=2, col=3)
        fig.update_xaxes(title="Years Invested", row=2, col=3)

        fig.update_layout(height=900, title_text="", template='plotly_white')
        return fig

    def create_backtest_plot(self, portfolio_results):
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Growth of $10k (Log Scale)', 'Historical Drawdowns',
                            'Rolling 3-Year Annualized Return', 'Rolling 5-Year Annualized Return'),
            vertical_spacing=0.08, shared_xaxes=True
        )

        for idx, item in enumerate(portfolio_results):
            if 'backtest' not in item: continue
            bt = item['backtest']
            label = item['label']
            color = self.colors[idx % len(self.colors)]
            grp = f"bt_{idx}"

            fig.add_trace(go.Scatter(x=bt['dates'], y=bt['values'], name=label, line=dict(color=color), legendgroup=grp, showlegend=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=bt['dates'], y=bt['drawdowns']*100, name=label, line=dict(width=1, color=color), fill='tozeroy', legendgroup=grp, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=bt['dates'], y=bt['rolling_3y']*100, name=label, line=dict(width=1.5, color=color), legendgroup=grp, showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=bt['dates'], y=bt['rolling_5y']*100, name=label, line=dict(width=1.5, color=color, dash='dot'), legendgroup=grp, showlegend=False), row=4, col=1)

        fig.update_yaxes(type="log", tickformat="$,.0f", title="Value ($)", row=1, col=1)
        fig.update_yaxes(tickformat=".1f", title="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(tickformat=".1f", title="CAGR (%)", row=3, col=1)
        fig.update_yaxes(tickformat=".1f", title="CAGR (%)", row=4, col=1)

        fig.update_layout(height=1200, template='plotly_white', hovermode='x unified')
        return fig

    def generate_html_report(self, portfolio_results, filename, start_date=None, end_date=None):
        mc_fig = self.create_monte_carlo_plot(portfolio_results)
        bt_fig = self.create_backtest_plot(portfolio_results)

        allocation_table_html = self.create_allocation_table_html(portfolio_results)

        date_str = ""
        if start_date and end_date:
            date_str = f"<p style='color: #666; margin-top: -15px;'>Analysis Period: <b>{start_date}</b> to <b>{end_date}</b></p>"

        # Build performance metrics table with VOLATILITY and STRATEGY columns
        table_html = f"""
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f9f9f9; color: #333; }}
            details {{ background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #eaeaea; }}
            summary {{ cursor: pointer; font-weight: 600; font-size: 1.1em; padding-bottom: 5px; outline: none; list-style: none; }}
            summary:after {{ content: "+"; float: right; font-weight: bold; }}
            details[open] summary:after {{ content: "-"; }}

            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.85em; }}
            th, td {{ border-bottom: 1px solid #eee; padding: 10px 6px; text-align: right; }}
            th {{ background-color: #f8f9fa; color: #666; font-weight: 600; text-align: center; }}
            td:first-child {{ text-align: left; font-weight: 600; color: #2c3e50; }}

            .allocation-table th {{ background-color: #e3f2fd; color: #1565c0; }}
            .allocation-table tr:hover {{ background-color: #f5f5f5; }}

            .pos-val {{ color: #27ae60; }}
            .neg-val {{ color: #c0392b; }}
            .warn-val {{ color: #f39c12; }}
            .strategy-col {{ font-size: 0.8em; color: #666; }}
        </style>
        <h1>Portfolio Analysis Report</h1>
        {date_str}
        <details open><summary>Performance Metrics Summary</summary>
        <table>
        <tr>
            <th>Portfolio</th>
            <th>Type</th>
            <th>Sim CAGR</th>
            <th>Hist CAGR</th>
            <th>Volatility</th>
            <th>Sharpe</th>
            <th>Sortino</th>
            <th>Max DD</th>
            <th>Best Yr</th>
            <th>Worst Yr</th>
        </tr>
        """

        for item in portfolio_results:
            m = item['backtest']['metrics']
            sim_stats = item['results']['stats']

            sim_cagr = sim_stats.get('median_cagr', sim_stats.get('mean_cagr', 0))
            hist_cagr = m['CAGR']
            volatility = m['Stdev']

            # Contribution type
            contrib_type = self._get_contribution_info(item)
            strategy = self._get_strategy_info(item)

            # Combine into type column
            if strategy != 'Buy and Hold' and strategy != 'Static DCA':
                type_str = f"{contrib_type}<br><span class='strategy-col'>{strategy}</span>"
            else:
                type_str = contrib_type

            # Calculate delta and flag large discrepancies
            delta = sim_cagr - hist_cagr
            if abs(delta) > 0.05:
                delta_class = "warn-val"
            elif delta > 0:
                delta_class = "pos-val"
            else:
                delta_class = ""

            table_html += f"""<tr>
                <td>{item['label']}</td>
                <td style="text-align:center;">{type_str}</td>
                <td>{sim_cagr*100:.2f}%</td>
                <td>{hist_cagr*100:.2f}%</td>
                <td>{volatility*100:.2f}%</td>
                <td>{m['Sharpe']:.2f}</td>
                <td>{m['Sortino']:.2f}</td>
                <td class='neg-val'>{m['Max Drawdown']*100:.2f}%</td>
                <td class='pos-val'>{m['Best Year']*100:.2f}%</td>
                <td class='neg-val'>{m['Worst Year']*100:.2f}%</td>
            </tr>"""

        table_html += "</table>"

        # Add MC vs Historical explanation
        table_html += """
        <p style="font-size: 0.8em; color: #888; margin-top: 15px;">
            <b>Note:</b> Sim CAGR = Monte Carlo median. Hist CAGR = actual backtest.
            Large differences may indicate: different time periods, DCA effects not captured in MC, or strategy-specific behavior.
            Volatility = annualized standard deviation of returns.
        </p>
        </details>
        """

        html_content = f"{table_html}" \
                       f"<details open><summary>Asset Allocation Details</summary>{allocation_table_html}</details>" \
                       f"<details open><summary>Historical Backtest</summary>{pio.to_html(bt_fig, full_html=False, include_plotlyjs='cdn')}</details>" \
                       f"<details open><summary>Monte Carlo Simulation</summary>{pio.to_html(mc_fig, full_html=False, include_plotlyjs=False)}</details>"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)