"""
Enhanced Interactive Plotly visualizations for portfolio analysis.
Creates a comprehensive dashboard similar to Portfolio Visualizer.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PortfolioVisualizer:
    """Creates comprehensive interactive Plotly visualizations"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def create_dashboard(self, portfolio_configs):
        """Create comprehensive interactive dashboard with multiple pages"""

        # Create main overview dashboard
        overview_fig = self._create_overview_dashboard(portfolio_configs)

        # Create detailed statistics page
        stats_fig = self._create_statistics_dashboard(portfolio_configs)

        # Create risk analysis page
        risk_fig = self._create_risk_dashboard(portfolio_configs)

        # Combine into single HTML with tabs
        return self._create_multi_page_dashboard(
            portfolio_configs,
            overview_fig,
            stats_fig,
            risk_fig
        )

    def _create_overview_dashboard(self, portfolio_configs):
        """Main overview dashboard with key visualizations"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        # Create 4x3 grid for more comprehensive view
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Portfolio Growth Trajectories (Median + 5th-95th Percentile)',
                'Final Value Distribution',
                'CAGR Distribution',
                'Cumulative Returns Comparison',
                'Maximum Drawdown Over Time',
                'Rolling 1-Year Returns',
                'Risk-Return Scatter',
                'Annual Returns Heatmap Placeholder',
                'Probability Analysis',
                'Portfolio Allocations',
                'Key Statistics Comparison',
                'Drawdown Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'table'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.10,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            legendgroup = f"group{idx}"

            # 1. Growth trajectories with percentile bands
            values = results['portfolio_values']
            days = np.arange(values.shape[1])
            p50 = np.percentile(values, 50, axis=0)
            p5 = np.percentile(values, 5, axis=0)
            p95 = np.percentile(values, 95, axis=0)
            p25 = np.percentile(values, 25, axis=0)
            p75 = np.percentile(values, 75, axis=0)

            # Outer band (5th-95th)
            fig.add_trace(
                go.Scatter(
                    x=days, y=p95, mode='lines', line=dict(width=0),
                    showlegend=False, legendgroup=legendgroup,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=days, y=p5, mode='lines', line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
                    showlegend=False, legendgroup=legendgroup,
                    name=f'{label} (5-95%)',
                    hovertemplate='Day %{x}<br>5th: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Inner band (25th-75th)
            fig.add_trace(
                go.Scatter(
                    x=days, y=p75, mode='lines', line=dict(width=0),
                    showlegend=False, legendgroup=legendgroup,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=days, y=p25, mode='lines', line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',
                    showlegend=False, legendgroup=legendgroup,
                    hovertemplate='Day %{x}<br>25th: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Median line
            fig.add_trace(
                go.Scatter(
                    x=days, y=p50, mode='lines',
                    name=label, line=dict(color=color, width=3),
                    legendgroup=legendgroup, showlegend=True,
                    hovertemplate='Day %{x}<br>Median: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Final value distribution
            fig.add_trace(
                go.Histogram(
                    x=results['final_values'],
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. CAGR distribution
            fig.add_trace(
                go.Histogram(
                    x=results['cagr'] * 100,
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='CAGR: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=3
            )

            # 4. Cumulative returns over time
            cumulative_returns = (p50 / self.simulator.initial_capital - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=days, y=cumulative_returns,
                    mode='lines', name=label,
                    line=dict(color=color, width=2.5),
                    legendgroup=legendgroup, showlegend=False,
                    hovertemplate='Day %{x}<br>Return: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )

            # 5. Maximum drawdown over time
            running_max = np.maximum.accumulate(p50)
            drawdown = (p50 - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(
                    x=days, y=drawdown,
                    mode='lines', name=label,
                    line=dict(color=color, width=2),
                    legendgroup=legendgroup, showlegend=False,
                    fill='tozeroy',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',
                    hovertemplate='Day %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=2
            )

            # 6. Rolling 1-year returns (252 trading days)
            if len(p50) > 252:
                rolling_returns = []
                rolling_days = []
                for i in range(252, len(p50)):
                    ret = (p50[i] / p50[i-252] - 1) * 100
                    rolling_returns.append(ret)
                    rolling_days.append(days[i])

                fig.add_trace(
                    go.Scatter(
                        x=rolling_days, y=rolling_returns,
                        mode='lines', name=label,
                        line=dict(color=color, width=2),
                        legendgroup=legendgroup, showlegend=False,
                        hovertemplate='Day %{x}<br>1Y Return: %{y:.1f}%<extra></extra>'
                    ),
                    row=2, col=3
                )

            # 7. Risk-return scatter
            stats = results['stats']
            fig.add_trace(
                go.Scatter(
                    x=[stats['std_cagr'] * 100],
                    y=[stats['mean_cagr'] * 100],
                    mode='markers+text',
                    name=label,
                    marker=dict(size=20, color=color, line=dict(width=2, color='white')),
                    text=[label.split(':')[0]],  # Shortened label
                    textposition='top center',
                    legendgroup=legendgroup, showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f'{stats["sharpe_ratio"]:.2f}<extra></extra>'
                ),
                row=3, col=1
            )

            # 9. Probability outcomes
            prob_loss = stats['probability_loss'] * 100
            prob_double = stats['probability_double'] * 100

            fig.add_trace(
                go.Bar(
                    x=['Loss', 'Double'],
                    y=[prob_loss, prob_double],
                    name=label,
                    marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    text=[f'{prob_loss:.1f}%', f'{prob_double:.1f}%'],
                    textposition='outside',
                    hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=3
            )

            # 10. Portfolio allocations (pie chart substitute - bar chart)
            if 'allocations' in results:
                asset_names = [asset['ticker'] for asset in results['assets']]
                allocations = results['allocations']

                fig.add_trace(
                    go.Bar(
                        x=asset_names,
                        y=[a*100 for a in allocations],
                        name=label,
                        marker_color=color,
                        legendgroup=legendgroup, showlegend=False,
                        text=[f'{a*100:.1f}%' for a in allocations],
                        textposition='outside',
                        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
                    ),
                    row=4, col=1
                )

            # 12. Drawdown distribution
            fig.add_trace(
                go.Histogram(
                    x=results['max_drawdowns'] * 100,
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='Max DD: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=4, col=3
            )

        # 8. Annual returns table (placeholder - showing key stats)
        self._add_annual_returns_table(fig, results_list, labels, row=3, col=2)

        # 11. Key statistics comparison table
        self._add_statistics_table(fig, results_list, labels, row=4, col=2)

        # Update axis labels
        fig.update_xaxes(title_text="Trading Days", row=1, col=1)
        fig.update_xaxes(title_text="Final Value ($)", row=1, col=2)
        fig.update_xaxes(title_text="CAGR (%)", row=1, col=3)
        fig.update_xaxes(title_text="Trading Days", row=2, col=1)
        fig.update_xaxes(title_text="Trading Days", row=2, col=2)
        fig.update_xaxes(title_text="Trading Days", row=2, col=3)
        fig.update_xaxes(title_text="Volatility (Std Dev %)", row=3, col=1)
        fig.update_xaxes(title_text="Asset", row=4, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=4, col=3)

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
        fig.update_yaxes(title_text="1-Year Return (%)", row=2, col=3)
        fig.update_yaxes(title_text="Expected CAGR (%)", row=3, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=3, col=3)
        fig.update_yaxes(title_text="Allocation (%)", row=4, col=1)
        fig.update_yaxes(title_text="Frequency", row=4, col=3)

        fig.update_layout(
            height=1800,
            title_text="<b>Portfolio Analysis Dashboard - Monte Carlo Simulation</b><br><sub>Click legend items to show/hide portfolios across all charts</sub>",
            title_font_size=20,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='closest',
            barmode='group',
            template='plotly_white',
            font=dict(size=11)
        )

        return fig

    def _add_statistics_table(self, fig, results_list, labels, row, col):
        """Add comprehensive statistics comparison table"""
        headers = ['Portfolio', 'Median Value', 'CAGR', 'Volatility',
                   'Sharpe', 'Sortino', 'Max DD', 'P(Loss)', 'P(2x)']

        table_data = []
        for results, label in zip(results_list, labels):
            stats = results['stats']
            table_data.append([
                label[:30],  # Truncate long names
                f"${stats['median_final_value']:,.0f}",
                f"{stats['median_cagr']*100:.2f}%",
                f"{stats['std_cagr']*100:.2f}%",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['sortino_ratio']:.2f}",
                f"{stats['median_max_drawdown']*100:.2f}%",
                f"{stats['probability_loss']*100:.1f}%",
                f"{stats['probability_double']*100:.1f}%"
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color='#1f77b4',
                    align='left',
                    font=dict(size=11, color='white'),
                    height=30
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#f0f0f0', 'white'] * len(table_data)],
                    align='left',
                    font=dict(size=10),
                    height=25
                ),
                columnwidth=[3, 1.5, 1, 1, 0.8, 0.8, 1, 0.8, 0.8]
            ),
            row=row, col=col
        )

    def _add_annual_returns_table(self, fig, results_list, labels, row, col):
        """Add year-by-year returns table"""
        # Since we're working with Monte Carlo simulations, show percentile statistics
        headers = ['Portfolio', '5th %ile', '25th %ile', 'Median', '75th %ile', '95th %ile']

        table_data = []
        for results, label in zip(results_list, labels):
            final_values = results['final_values']

            table_data.append([
                label[:30],
                f"${np.percentile(final_values, 5):,.0f}",
                f"${np.percentile(final_values, 25):,.0f}",
                f"${np.percentile(final_values, 50):,.0f}",
                f"${np.percentile(final_values, 75):,.0f}",
                f"${np.percentile(final_values, 95):,.0f}"
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color='#2ca02c',
                    align='left',
                    font=dict(size=11, color='white'),
                    height=30
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#f0f0f0', 'white'] * len(table_data)],
                    align='left',
                    font=dict(size=10),
                    height=25
                ),
                columnwidth=[3, 1.5, 1.5, 1.5, 1.5, 1.5]
            ),
            row=row, col=col
        )

    def _create_statistics_dashboard(self, portfolio_configs):
        """Detailed statistics and metrics dashboard"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Return Statistics Distribution',
                'Risk Metrics Comparison',
                'Percentile Analysis',
                'Correlation of Outcomes',
                'Best/Worst Case Scenarios',
                'Time to Recovery Analysis'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'table'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            stats = results['stats']

            # 1. Return distribution box plot
            fig.add_trace(
                go.Box(
                    y=results['cagr'] * 100,
                    name=label,
                    marker_color=color,
                    boxmean='sd',
                    hovertemplate='%{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Risk metrics bar chart
            risk_metrics = {
                'Max DD': abs(stats['median_max_drawdown']) * 100,
                '95% DD': abs(stats['max_drawdown_95']) * 100,
                'Worst DD': abs(stats['worst_max_drawdown']) * 100,
                'Volatility': stats['std_cagr'] * 100
            }

            fig.add_trace(
                go.Bar(
                    x=list(risk_metrics.keys()),
                    y=list(risk_metrics.values()),
                    name=label,
                    marker_color=color,
                    text=[f'{v:.1f}%' for v in risk_metrics.values()],
                    textposition='outside',
                    hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. Percentile analysis
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            final_values = [np.percentile(results['final_values'], p) for p in percentiles]

            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=final_values,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    hovertemplate='%{x}th percentile: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )

        # 5. Best/Worst case table
        self._add_best_worst_table(fig, results_list, labels, row=3, col=1)

        # Update axes
        fig.update_xaxes(title_text="Portfolio", row=1, col=1)
        fig.update_xaxes(title_text="Risk Metric", row=1, col=2)
        fig.update_xaxes(title_text="Percentile", row=2, col=1)

        fig.update_yaxes(title_text="CAGR (%)", row=1, col=1)
        fig.update_yaxes(title_text="Value (%)", row=1, col=2)
        fig.update_yaxes(title_text="Final Value ($)", row=2, col=1)

        fig.update_layout(
            height=1400,
            title_text="<b>Detailed Portfolio Statistics</b>",
            title_font_size=18,
            showlegend=True,
            template='plotly_white',
            barmode='group'
        )

        return fig

    def _add_best_worst_table(self, fig, results_list, labels, row, col):
        """Add best/worst case scenarios table"""
        headers = ['Portfolio', 'Best Case (95%)', 'Expected (50%)', 'Worst Case (5%)', 'Range']

        table_data = []
        for results, label in zip(results_list, labels):
            final_values = results['final_values']
            best = np.percentile(final_values, 95)
            median = np.percentile(final_values, 50)
            worst = np.percentile(final_values, 5)
            range_val = best - worst

            table_data.append([
                label[:30],
                f"${best:,.0f}",
                f"${median:,.0f}",
                f"${worst:,.0f}",
                f"${range_val:,.0f}"
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color='#d62728',
                    align='left',
                    font=dict(size=11, color='white'),
                    height=30
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#f0f0f0', 'white'] * len(table_data)],
                    align='left',
                    font=dict(size=10),
                    height=25
                )
            ),
            row=row, col=col
        )

    def _create_risk_dashboard(self, portfolio_configs):
        """Risk-focused analysis dashboard"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Value at Risk (VaR) Analysis',
                'Downside Deviation',
                'Recovery Time Distribution',
                'Risk-Adjusted Return Metrics'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            stats = results['stats']

            # 1. VaR analysis
            var_95 = (self.simulator.initial_capital - stats['percentile_5']) / self.simulator.initial_capital * 100
            var_99 = (self.simulator.initial_capital - np.percentile(results['final_values'], 1)) / self.simulator.initial_capital * 100

            fig.add_trace(
                go.Bar(
                    x=['95% VaR', '99% VaR'],
                    y=[var_95, var_99],
                    name=label,
                    marker_color=color,
                    text=[f'{var_95:.1f}%', f'{var_99:.1f}%'],
                    textposition='outside',
                    hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Downside deviation
            negative_returns = results['cagr'][results['cagr'] < 0]
            if len(negative_returns) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=negative_returns * 100,
                        name=label,
                        opacity=0.6,
                        marker_color=color,
                        nbinsx=30,
                        hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )

            # 4. Risk-adjusted metrics
            metrics = {
                'Sharpe': stats['sharpe_ratio'],
                'Sortino': stats['sortino_ratio'],
                'Calmar': stats['mean_cagr'] / abs(stats['median_max_drawdown']) if stats['median_max_drawdown'] != 0 else 0
            }

            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    name=label,
                    marker_color=color,
                    text=[f'{v:.2f}' for v in metrics.values()],
                    textposition='outside',
                    hovertemplate='%{x}: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )

        # Update axes
        fig.update_xaxes(title_text="VaR Metric", row=1, col=1)
        fig.update_xaxes(title_text="Negative CAGR (%)", row=1, col=2)
        fig.update_xaxes(title_text="Metric", row=2, col=2)

        fig.update_yaxes(title_text="Potential Loss (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)

        fig.update_layout(
            height=1000,
            title_text="<b>Risk Analysis Dashboard</b>",
            title_font_size=18,
            showlegend=True,
            template='plotly_white',
            barmode='group'
        )

        return fig

    def _create_multi_page_dashboard(self, portfolio_configs, overview_fig, stats_fig, risk_fig):
        """Combine multiple dashboards into single HTML with navigation"""

        # Generate HTML with tabs/sections
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Portfolio Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .tabs {{
            display: flex;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .tab {{
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
            font-weight: 500;
        }}
        .tab:hover {{
            background-color: #f0f0f0;
        }}
        .tab.active {{
            border-bottom-color: #667eea;
            color: #667eea;
            background-color: #f8f9ff;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            animation: fadeIn 0.3s;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .info-box {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info-box h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h4 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            font-weight: normal;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .footer {{
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Portfolio Analysis Dashboard</h1>
        <p>Monte Carlo Simulation Results - {date}</p>
    </div>

    {summary_stats}

    <div class="tabs">
        <div class="tab active" onclick="showTab('overview')">Overview</div>
        <div class="tab" onclick="showTab('statistics')">Detailed Statistics</div>
        <div class="tab" onclick="showTab('risk')">Risk Analysis</div>
    </div>

    <div id="overview" class="tab-content active">
        <div id="overview-chart"></div>
    </div>

    <div id="statistics" class="tab-content">
        <div id="stats-chart"></div>
    </div>

    <div id="risk" class="tab-content">
        <div id="risk-chart"></div>
    </div>

    <div class="footer">
        <p>Generated with Monte Carlo Portfolio Simulator | Initial Capital: ${initial_capital:,} | Simulations: {simulations:,} | Time Horizon: {years} years</p>
    </div>

    <script>
        {overview_json}
        {stats_json}
        {risk_json}

        Plotly.newPlot('overview-chart', overviewData.data, overviewData.layout, {{responsive: true}});
        Plotly.newPlot('stats-chart', statsData.data, statsData.layout, {{responsive: true}});
        Plotly.newPlot('risk-chart', riskData.data, riskData.layout, {{responsive: true}});

        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

        # Generate summary statistics
        summary_html = self._generate_summary_stats(portfolio_configs)

        # Convert figures to JSON
        overview_json = f"var overviewData = {overview_fig.to_json()};"
        stats_json = f"var statsData = {stats_fig.to_json()};"
        risk_json = f"var riskData = {risk_fig.to_json()};"

        # Fill template
        html_content = html_template.format(
            date=datetime.now().strftime("%B %d, %Y"),
            summary_stats=summary_html,
            initial_capital=self.simulator.initial_capital,
            simulations=self.simulator.simulations,
            years=self.simulator.years,
            overview_json=overview_json,
            stats_json=stats_json,
            risk_json=risk_json
        )

        # Create a custom HTML object
        class CustomHTML:
            def __init__(self, html_content):
                self.html_content = html_content

            def write_html(self, filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.html_content)

            def show(self):
                import tempfile
                import webbrowser
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
                    f.write(self.html_content)
                    webbrowser.open('file://' + f.name)

        return CustomHTML(html_content)

    def _generate_summary_stats(self, portfolio_configs):
        """Generate summary statistics cards"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        # Find best portfolio by Sharpe ratio
        best_sharpe_idx = max(range(len(results_list)),
                             key=lambda i: results_list[i]['stats']['sharpe_ratio'])
        best_return_idx = max(range(len(results_list)),
                             key=lambda i: results_list[i]['stats']['median_cagr'])
        best_risk_idx = min(range(len(results_list)),
                           key=lambda i: abs(results_list[i]['stats']['median_max_drawdown']))

        html = '<div class="stats-grid">'
        html += f'''
        <div class="stat-card">
            <h4>Best Risk-Adjusted Return</h4>
            <div class="value">{labels[best_sharpe_idx][:20]}</div>
            <p style="margin: 5px 0 0 0; color: #666;">Sharpe: {results_list[best_sharpe_idx]['stats']['sharpe_ratio']:.2f}</p>
        </div>
        <div class="stat-card">
            <h4>Highest Expected Return</h4>
            <div class="value">{labels[best_return_idx][:20]}</div>
            <p style="margin: 5px 0 0 0; color: #666;">CAGR: {results_list[best_return_idx]['stats']['median_cagr']*100:.2f}%</p>
        </div>
        <div class="stat-card">
            <h4>Lowest Risk</h4>
            <div class="value">{labels[best_risk_idx][:20]}</div>
            <p style="margin: 5px 0 0 0; color: #666;">Max DD: {results_list[best_risk_idx]['stats']['median_max_drawdown']*100:.2f}%</p>
        </div>
        <div class="stat-card">
            <h4>Portfolios Analyzed</h4>
            <div class="value">{len(results_list)}</div>
            <p style="margin: 5px 0 0 0; color: #666;">{self.simulator.simulations:,} simulations each</p>
        </div>
        '''
        html += '</div>'

        return html