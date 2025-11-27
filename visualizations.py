"""
Visualization engine with KDE Smoothing, Ribbon Charts, and Rolling Metrics.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
from scipy import stats  # Required for KDE smoothing

class PortfolioVisualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def _hex_to_rgba(self, hex_color, alpha):
        """Helper to create transparent colors"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    def _get_kde_curve(self, data, clip_percentile=1.0):
        """
        Generates smooth KDE curve coordinates (Seaborn-style).
        Trims extreme top/bottom 1% for visualization clarity if requested.
        """
        # Trim data for plotting range only (keeps analytics accurate, makes plot readable)
        min_val = np.percentile(data, clip_percentile)
        max_val = np.percentile(data, 100 - clip_percentile)

        # Calculate KDE
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min_val, max_val, 200)
        y_range = kde(x_range)
        return x_range, y_range

    def create_monte_carlo_plot(self, portfolio_results):
        """
        Row 1: 95% Confidence Ribbons (Trajectories) | Final Value Dist (KDE) | CAGR Dist (KDE)
        Row 2: Risk-Return Scatter | Max Drawdown Dist (KDE) | Outcome Probabilities
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('95% Confidence Trajectories', 'Final Value Density', 'CAGR Density',
                            'Risk-Return Profile', 'Max Drawdown Density', 'Probabilities'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.15, horizontal_spacing=0.08
        )

        for idx, item in enumerate(portfolio_results):
            label = item['label']
            res = item['results']
            stats_data = res['stats']
            color = self.colors[idx % len(self.colors)]
            fill_color = self._hex_to_rgba(color, 0.15) # Transparent fill
            grp = f"g{idx}"

            # --- 1. Confidence Ribbons (Trajectory) ---
            # Calculate percentiles across all simulations per day
            p5 = np.percentile(res['portfolio_values'], 5, axis=0)
            p50 = np.median(res['portfolio_values'], axis=0)
            p95 = np.percentile(res['portfolio_values'], 95, axis=0)
            days = np.arange(len(p50))

            # Upper Bound (Invisible line)
            fig.add_trace(go.Scatter(x=days, y=p95, mode='lines', line=dict(width=0), showlegend=False, legendgroup=grp, hoverinfo='skip'), row=1, col=1)
            # Lower Bound (Fill to Upper)
            fig.add_trace(go.Scatter(
                x=days, y=p5, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=fill_color,
                name=f"{label} (95% CI)", showlegend=False, legendgroup=grp, hoverinfo='skip'
            ), row=1, col=1)
            # Median (Solid Line)
            fig.add_trace(go.Scatter(
                x=days, y=p50, mode='lines', line=dict(color=color, width=2),
                name=label, legendgroup=grp,
                hovertemplate=f"<b>{label}</b><br>Median: $%{{y:,.0f}}"
            ), row=1, col=1)

            # --- 2. Final Value KDE (Smooth Curve) ---
            x_kde, y_kde = self._get_kde_curve(res['final_values'])
            fig.add_trace(go.Scatter(
                x=x_kde, y=y_kde, mode='lines', line=dict(color=color, width=1),
                fill='tozeroy', fillcolor=fill_color,
                name=label, showlegend=False, legendgroup=grp,
                hovertemplate=f"<b>{label}</b><br>Value: $%{{x:,.0f}}<br>Density: %{{y:.2e}}"
            ), row=1, col=2)

            # --- 3. CAGR KDE (Smooth Curve) ---
            x_cagr, y_cagr = self._get_kde_curve(res['cagr'] * 100)
            fig.add_trace(go.Scatter(
                x=x_cagr, y=y_cagr, mode='lines', line=dict(color=color, width=1),
                fill='tozeroy', fillcolor=fill_color,
                name=label, showlegend=False, legendgroup=grp,
                hovertemplate=f"<b>{label}</b><br>CAGR: %{{x:.1f}}%"
            ), row=1, col=3)

            # --- 4. Risk Return Scatter ---
            fig.add_trace(go.Scatter(
                x=[stats_data['std_cagr']*100], y=[stats_data['mean_cagr']*100],
                mode='markers', marker=dict(color=color, size=14, line=dict(width=1, color='black')),
                name=label, showlegend=False, legendgroup=grp
            ), row=2, col=1)

            # --- 5. Drawdown KDE ---
            # Note: Drawdowns are negative, we plot them as positive percentages or keep negative?
            # Let's plot actual negative values but trimmed
            x_dd, y_dd = self._get_kde_curve(res['max_drawdowns'] * 100)
            fig.add_trace(go.Scatter(
                x=x_dd, y=y_dd, mode='lines', line=dict(color=color, width=1),
                fill='tozeroy', fillcolor=fill_color,
                name=label, showlegend=False, legendgroup=grp,
                hovertemplate=f"<b>{label}</b><br>Drawdown: %{{x:.1f}}%"
            ), row=2, col=2)

            # --- 6. Probabilities ---
            fig.add_trace(go.Bar(
                x=['Loss', 'Double'], y=[stats_data['probability_loss']*100, stats_data['probability_double']*100],
                marker_color=color, showlegend=False, legendgroup=grp
            ), row=2, col=3)

        fig.update_layout(height=900, title_text="", template='plotly_white')
        return fig

    def create_backtest_plot(self, portfolio_results):
        """
        Row 1: Equity Curve (Log)
        Row 2: Drawdowns
        Row 3: Rolling 3Y
        Row 4: Rolling 5Y
        """
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

            # 1. Equity
            fig.add_trace(go.Scatter(x=bt['dates'], y=bt['values'], name=label, line=dict(color=color)), row=1, col=1)

            # 2. Drawdowns
            fig.add_trace(go.Scatter(
                x=bt['dates'], y=bt['drawdowns']*100, name=label,
                line=dict(width=1, color=color), fill='tozeroy', showlegend=False,
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>DD: %{{y:.1f}}%"
            ), row=2, col=1)

            # 3. Rolling 3Y
            fig.add_trace(go.Scatter(
                x=bt['dates'], y=bt['rolling_3y']*100, name=label,
                line=dict(width=1.5, color=color), showlegend=False,
                hovertemplate=f"<b>{label}</b><br>3Y Avg: %{{y:.1f}}%"
            ), row=3, col=1)

            # 4. Rolling 5Y
            fig.add_trace(go.Scatter(
                x=bt['dates'], y=bt['rolling_5y']*100, name=label,
                line=dict(width=1.5, color=color, dash='dot'), showlegend=False,
                hovertemplate=f"<b>{label}</b><br>5Y Avg: %{{y:.1f}}%"
            ), row=4, col=1)

        fig.update_yaxes(type="log", row=1, col=1, title="Value ($)")
        fig.update_yaxes(title="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title="CAGR (%)", row=3, col=1)
        fig.update_yaxes(title="CAGR (%)", row=4, col=1)

        fig.update_layout(height=1200, template='plotly_white', hovermode='x unified')
        return fig

    def generate_html_report(self, portfolio_results, filename):
        mc_fig = self.create_monte_carlo_plot(portfolio_results)
        bt_fig = self.create_backtest_plot(portfolio_results)

        table_html = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #f9f9f9; color: #333; }
            details { background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #eaeaea; }
            summary { cursor: pointer; font-weight: 600; font-size: 1.1em; padding-bottom: 5px; outline: none; list-style: none; }
            summary::-webkit-details-marker { display: none; }
            summary:after { content: "+"; float: right; font-weight: bold; }
            details[open] summary:after { content: "-"; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
            th, td { border-bottom: 1px solid #eee; padding: 12px 8px; text-align: right; }
            th { background-color: #f8f9fa; color: #666; font-weight: 600; text-align: center; }
            td:first-child { text-align: left; font-weight: 600; color: #2c3e50; }
            .pos-val { color: #27ae60; }
            .neg-val { color: #c0392b; }
        </style>
        <h1>Portfolio Analysis Report</h1>
        <details open>
            <summary>Performance Metrics Summary</summary>
            <table>
                <tr>
                    <th>Portfolio</th>
                    <th>Sim CAGR</th>
                    <th>Hist CAGR</th>
                    <th>Delta</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Best Year</th>
                    <th>Worst Year</th>
                    <th>Tracking Err</th>
                </tr>
        """

        for item in portfolio_results:
            m = item['backtest']['metrics']
            sim_cagr = item['results']['stats']['median_cagr']
            hist_cagr = m['CAGR']
            delta = sim_cagr - hist_cagr
            delta_class = "neg-val" if delta > 0.05 else "pos-val" # Red if sim was > 5% overly optimistic

            table_html += f"""
                <tr>
                    <td>{item['label']}</td>
                    <td>{sim_cagr*100:.2f}%</td>
                    <td>{hist_cagr*100:.2f}%</td>
                    <td class="{delta_class}">{delta*100:+.2f}%</td>
                    <td>{m['Sharpe']:.2f}</td>
                    <td class="neg-val">{m['Max Drawdown']*100:.2f}%</td>
                    <td class="pos-val">{m['Best Year']*100:.2f}%</td>
                    <td class="neg-val">{m['Worst Year']*100:.2f}%</td>
                    <td>{m['Tracking Error']*100:.2f}%</td>
                </tr>
            """

        table_html += "</table></details>"

        html_content = f"""
        {table_html}
        <details open>
            <summary>Historical Backtest (Drawdowns & Rolling Returns)</summary>
            {pio.to_html(bt_fig, full_html=False, include_plotlyjs='cdn')}
        </details>
        <details open>
            <summary>Monte Carlo Simulation (Distributions & Projections)</summary>
            {pio.to_html(mc_fig, full_html=False, include_plotlyjs=False)}
        </details>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)