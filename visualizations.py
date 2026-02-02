"""
Visualization engine - With Volatility, DCA Indicator, and MC/Historical Alignment
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import pandas as pd
from scipy import stats
import base64

from pv_compat import export_portfolio_csv, save_portfolio_csv, generate_pv_url, generate_mc_url

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

    def _get_type_info(self, item) -> str:
        """Get combined contribution + strategy info (single line)"""
        results = item.get('results', {})
        backtest = item.get('backtest', {})

        # Check for DCA
        total_invested = backtest.get('metrics', {}).get('Total Invested', None)
        if total_invested is None:
            total_invested = results.get('total_invested', None)

        start_balance = backtest.get('metrics', {}).get('Start Balance',
                                                         results.get('stats', {}).get('initial_capital', 10000))

        if total_invested and total_invested > start_balance * 1.01:
            contrib_str = f"DCA: ${total_invested - start_balance:,.0f}"
        else:
            contrib_str = "Lump Sum"

        # Get strategy (but avoid redundancy)
        strategy = backtest.get('strategy', results.get('strategy', ''))

        # Skip generic/redundant strategy names
        if strategy and strategy not in ['Buy and Hold', 'Static DCA', '', None]:
            return f"{contrib_str}<br><small>{strategy}</small>"

        return contrib_str

    def create_allocation_table_html(self, portfolio_results):
        """Creates a compact allocation table with PV export buttons"""
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
                if weight > 0.001:  # Only include non-zero allocations
                    all_tickers.add(ticker)
                    row_data[ticker] = weight

            portfolio_rows.append(row_data)

        sorted_tickers = sorted(list(all_tickers))

        # Button styles
        btn_style = '''
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 500;
            text-decoration: none;
            border: 1px solid;
            cursor: pointer;
            transition: all 0.2s;
        '''
        csv_btn_style = f'{btn_style} background: #f8f9fa; color: #495057; border-color: #dee2e6;'
        backtest_btn_style = f'{btn_style} background: #e3f2fd; color: #1565c0; border-color: #90caf9;'
        mc_btn_style = f'{btn_style} background: #f3e5f5; color: #7b1fa2; border-color: #ce93d8;'

        # Compact table with smaller cells and export column
        html = f'''<div style="overflow-x: auto; font-size: 0.85em;">
        <table style="border-collapse: collapse; width: auto;">
        <thead><tr><th style="text-align:left; padding: 6px 10px;">Portfolio</th>'''

        for ticker in sorted_tickers:
            html += f'<th style="padding: 6px 8px; min-width: 50px;">{ticker}</th>'
        html += '<th style="padding: 6px 12px;">Export</th></tr></thead>'

        html += '<tbody>'
        for row_dict in portfolio_rows:
            p_name = row_dict['Portfolio']
            html += f'<tr><td style="font-weight:600; text-align:left; padding: 5px 10px; white-space: nowrap;">{p_name}</td>'

            for ticker in sorted_tickers:
                val = row_dict.get(ticker, 0.0)
                if val > 0.001:
                    html += f'<td style="padding: 5px 8px; text-align: center;">{val*100:.1f}%</td>'
                else:
                    html += '<td style="padding: 5px 8px; text-align: center; color: #ddd;">-</td>'

            # Build allocations dict for export (only non-zero weights)
            row_allocations = {t: row_dict.get(t, 0) for t in sorted_tickers if row_dict.get(t, 0) > 0.001}

            # Save CSV to output/portfolios/
            try:
                save_portfolio_csv(row_allocations, p_name)
            except Exception:
                pass  # Don't fail if save doesn't work

            # Generate CSV data URI for browser download
            csv_content = export_portfolio_csv(row_allocations, p_name)
            csv_b64 = base64.b64encode(csv_content.encode()).decode()
            data_uri = f"data:text/csv;base64,{csv_b64}"

            # Generate PV URLs
            pv_url = generate_pv_url(row_allocations)
            mc_url = generate_mc_url(row_allocations)

            # Safe filename
            safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in p_name)

            html += f'''<td style="padding: 5px 8px; text-align: center; white-space: nowrap;">
                <a href="{data_uri}" download="{safe_name}.csv"
                   style="{csv_btn_style}" title="Download CSV for Portfolio Visualizer import">CSV</a>
                <a href="{pv_url}" target="_blank"
                   style="{backtest_btn_style}" title="Open Backtest in Portfolio Visualizer">Backtest</a>
                <a href="{mc_url}" target="_blank"
                   style="{mc_btn_style}" title="Open Monte Carlo in Portfolio Visualizer">MC Sim</a>
            </td>'''
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

            # 1. Trajectory (downsample for faster percentile computation)
            pv = res['portfolio_values']
            n_days = pv.shape[1]
            # Sample every 5 days for charts (still 500+ points for smooth lines)
            step = max(1, n_days // 500)
            sample_idx = np.arange(0, n_days, step)
            if sample_idx[-1] != n_days - 1:
                sample_idx = np.append(sample_idx, n_days - 1)  # Always include last day
            pv_sampled = pv[:, sample_idx]
            p5 = np.percentile(pv_sampled, 5, axis=0)
            p50 = np.median(pv_sampled, axis=0)
            p95 = np.percentile(pv_sampled, 95, axis=0)
            days = sample_idx

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

        # Build performance metrics table with VOLATILITY column
        table_html = f"""
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 16px 20px; background: #f9f9f9; color: #333; }}
            details {{ background: white; padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #eaeaea; }}
            summary {{ cursor: pointer; font-weight: 600; font-size: 1em; outline: none; list-style: none; }}
            summary::-webkit-details-marker {{ display: none; }}
            summary:after {{ content: " ▶"; font-size: 0.75em; color: #888; }}
            details[open] summary:after {{ content: " ▼"; }}

            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.82em; }}
            th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: right; }}
            th {{ background-color: #f8f9fa; color: #555; font-weight: 600; text-align: center; font-size: 0.85em; }}
            td:first-child {{ text-align: left; font-weight: 600; color: #2c3e50; }}
            small {{ color: #888; font-weight: normal; }}

            .pos-val {{ color: #27ae60; }}
            .neg-val {{ color: #c0392b; }}
            .warn-val {{ color: #f39c12; }}

            .header-row {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px; flex-wrap: wrap; }}
            .header-row h1 {{ margin: 0; font-size: 1.5em; }}
            .header-row p {{ margin: 0; color: #666; font-size: 0.9em; }}
        </style>
        <div class="header-row">
            <h1>Portfolio Analysis Report</h1>
            {f"<p>Analysis Period: <b>{start_date}</b> to <b>{end_date}</b></p>" if start_date and end_date else ""}
        </div>
        <details open><summary>Performance Metrics Summary</summary>
        <table>
        <tr>
            <th style="text-align:left;">Portfolio</th>
            <th>Type</th>
            <th>Sim CAGR</th>
            <th>Hist CAGR</th>
            <th>Vol</th>
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

            # Single combined type column
            type_str = self._get_type_info(item)

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
        <p style="font-size: 0.75em; color: #999; margin: 8px 0 0 0;">
            <b>Note:</b> Sim CAGR = Monte Carlo median. Hist CAGR = actual backtest. Vol = annualized stdev.
        </p>
        </details>
        """

        html_content = f"{table_html}" \
                       f"<details><summary>Asset Allocation Details</summary>{allocation_table_html}</details>" \
                       f"<details open><summary>Historical Backtest</summary>{pio.to_html(bt_fig, full_html=False, include_plotlyjs='cdn')}</details>" \
                       f"<details open><summary>Monte Carlo Simulation</summary>{pio.to_html(mc_fig, full_html=False, include_plotlyjs=False)}</details>"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)