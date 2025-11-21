"""
Enhanced Interactive Plotly visualizations for portfolio analysis.
Creates a comprehensive dashboard similar to Portfolio Visualizer.
IMPROVED VERSION - Better spacing, formatting, and data handling
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

        # Create 4x3 grid with BETTER spacing and sizing
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Portfolio Growth (Median + Percentile Bands)',
                'Final Value Distribution',
                'CAGR Distribution',
                'Cumulative Returns Over Time',
                'Maximum Drawdown Over Time',
                'Rolling 1-Year Returns',
                'Risk vs Return Analysis',
                'Value Percentiles',
                'Probability Outcomes',
                'Portfolio Allocations',
                'Performance Statistics',
                'Max Drawdown Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.12,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            legendgroup = f"group{idx}"

            # 1. Growth trajectories with percentile bands
            values = results['portfolio_values']
            days = np.arange(values.shape[1])

            # Calculate percentiles properly
            p50 = np.percentile(values, 50, axis=0)
            p5 = np.percentile(values, 5, axis=0)
            p95 = np.percentile(values, 95, axis=0)
            p25 = np.percentile(values, 25, axis=0)
            p75 = np.percentile(values, 75, axis=0)

            # Convert color hex to RGB
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)

            # Outer band (5th-95th)
            fig.add_trace(
                go.Scatter(
                    x=days, y=p95, mode='lines', line=dict(width=0),
                    showlegend=False, legendgroup=legendgroup,
                    hoverinfo='skip', name=''
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=days, y=p5, mode='lines', line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({r}, {g}, {b}, 0.15)',
                    showlegend=False, legendgroup=legendgroup,
                    name=f'{label} (5-95%)',
                    hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Inner band (25th-75th)
            fig.add_trace(
                go.Scatter(
                    x=days, y=p75, mode='lines', line=dict(width=0),
                    showlegend=False, legendgroup=legendgroup,
                    hoverinfo='skip', name=''
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=days, y=p25, mode='lines', line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({r}, {g}, {b}, 0.3)',
                    showlegend=False, legendgroup=legendgroup,
                    name=f'{label} (25-75%)',
                    hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Median line
            fig.add_trace(
                go.Scatter(
                    x=days, y=p50, mode='lines',
                    name=label, line=dict(color=color, width=3),
                    legendgroup=legendgroup, showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>Median: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Final value distribution
            final_vals = results['final_values']
            fig.add_trace(
                go.Histogram(
                    x=final_vals,
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='<b>%{fullData.name}</b><br>Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. CAGR distribution
            cagr_vals = results['cagr'] * 100
            fig.add_trace(
                go.Histogram(
                    x=cagr_vals,
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='<b>%{fullData.name}</b><br>CAGR: %{x:.2f}%<br>Count: %{y}<extra></extra>'
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
                    hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>Return: %{y:.2f}%<extra></extra>'
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
                    fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                    hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
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
                        hovertemplate='<b>%{fullData.name}</b><br>Day %{x}<br>1Y Return: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=3
                )

            # 7. Risk-return scatter
            stats = results['stats']
            risk = stats['std_cagr'] * 100
            ret = stats['mean_cagr'] * 100
            sharpe = stats['sharpe_ratio']

            # Shorten label for display
            short_label = label.split(':')[0] if ':' in label else label[:15]

            fig.add_trace(
                go.Scatter(
                    x=[risk],
                    y=[ret],
                    mode='markers+text',
                    name=label,
                    marker=dict(size=25, color=color, line=dict(width=3, color='white')),
                    text=[short_label],
                    textposition='top center',
                    textfont=dict(size=10, color='black', family='Arial Black'),
                    legendgroup=legendgroup, showlegend=False,
                    hovertemplate=f'<b>{label}</b><br>Risk (StdDev): %{{x:.2f}}%<br>Return (Mean): %{{y:.2f}}%<br>Sharpe Ratio: {sharpe:.2f}<extra></extra>'
                ),
                row=3, col=1
            )

            # 8. Value percentiles over simulation count
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = [np.percentile(final_vals, p) for p in percentiles]

            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=percentile_values,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=10, color=color, line=dict(width=2, color='white')),
                    legendgroup=legendgroup, showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>%{x}th Percentile<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=3, col=2
            )

            # 9. Probability outcomes
            prob_loss = stats['probability_loss'] * 100
            prob_double = stats['probability_double'] * 100

            fig.add_trace(
                go.Bar(
                    x=['Loss', '2x Gain'],
                    y=[prob_loss, prob_double],
                    name=label,
                    marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    text=[f'{prob_loss:.1f}%', f'{prob_double:.1f}%'],
                    textposition='outside',
                    hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=3
            )

            # 10. Portfolio allocations
            if 'allocations' in results and 'assets' in results:
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
                        hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.1f}%<extra></extra>'
                    ),
                    row=4, col=1
                )

            # 12. Drawdown distribution
            max_dds = results['max_drawdowns'] * 100
            fig.add_trace(
                go.Histogram(
                    x=max_dds,
                    name=label, opacity=0.6, marker_color=color,
                    legendgroup=legendgroup, showlegend=False,
                    nbinsx=50,
                    hovertemplate='<b>%{fullData.name}</b><br>Max DD: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                ),
                row=4, col=3
            )

        # 11. Key statistics comparison table
        self._add_statistics_table(fig, results_list, labels, row=4, col=2)

        # Update axis labels with better formatting
        fig.update_xaxes(title_text="Trading Days", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Final Value ($)", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="CAGR (%)", row=1, col=3, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Trading Days", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Trading Days", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Trading Days", row=2, col=3, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Risk - Volatility (%)", row=3, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Percentile", row=3, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Outcome", row=3, col=3)
        fig.update_xaxes(title_text="Asset", row=4, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=4, col=3, showgrid=True, gridcolor='lightgray')

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Frequency", row=1, col=3, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Rolling 1Y Return (%)", row=2, col=3, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Expected Return (%)", row=3, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Final Value ($)", row=3, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Probability (%)", row=3, col=3)
        fig.update_yaxes(title_text="Allocation (%)", row=4, col=1)
        fig.update_yaxes(title_text="Frequency", row=4, col=3, showgrid=True, gridcolor='lightgray')

        # Format all axes with better number formatting
        for i in range(1, 5):
            for j in range(1, 4):
                fig.update_xaxes(tickformat=',', row=i, col=j)
                fig.update_yaxes(tickformat=',', row=i, col=j)

        fig.update_layout(
            height=2200,  # Increased from 1800
            title_text="<b>Portfolio Analysis Dashboard - Monte Carlo Simulation</b><br><sub>Interactive charts: Click legend to show/hide portfolios • Hover for details • Drag to zoom</sub>",
            title_font_size=22,
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#333',
                borderwidth=2,
                font=dict(size=12)
            ),
            hovermode='closest',
            barmode='group',
            template='plotly_white',
            font=dict(size=11, family='Arial, sans-serif'),
            margin=dict(t=150, b=50, l=60, r=60)
        )

        return fig

    def _add_statistics_table(self, fig, results_list, labels, row, col):
        """Add comprehensive statistics comparison table with better formatting"""
        headers = ['Portfolio', 'Median Value', 'CAGR', 'Volatility',
                   'Sharpe', 'Sortino', 'Max DD', 'Loss %', '2x %']

        table_data = []
        for results, label in zip(results_list, labels):
            stats = results['stats']
            # Truncate label but keep meaningful part
            display_label = label[:35] + '...' if len(label) > 35 else label

            table_data.append([
                display_label,
                f"${stats['median_final_value']:,.0f}",
                f"{stats['median_cagr']*100:.2f}%",
                f"{stats['std_cagr']*100:.2f}%",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['sortino_ratio']:.2f}",
                f"{stats['median_max_drawdown']*100:.2f}%",
                f"{stats['probability_loss']*100:.1f}%",
                f"{stats['probability_double']*100:.1f}%"
            ])

        # Transpose data for plotly table format
        cell_values = list(zip(*table_data))

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color='#1f77b4',
                    align='left',
                    font=dict(size=12, color='white', family='Arial Black'),
                    height=35
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=[['#f8f9fa', 'white'] * (len(table_data) // 2 + 1)],
                    align='left',
                    font=dict(size=11, family='Arial'),
                    height=30
                ),
                columnwidth=[4, 2, 1.5, 1.5, 1, 1, 1.5, 1, 1]
            ),
            row=row, col=col
        )

    def _create_statistics_dashboard(self, portfolio_configs):
        """Detailed statistics and metrics dashboard with improved layout"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Return Statistics Distribution',
                'Risk Metrics Comparison',
                'Percentile Analysis - Final Values',
                'Rolling Correlation Matrix',
                'Best/Worst/Expected Case Scenarios',
                'Year-over-Year Statistics'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'table'}],
                [{'type': 'table'}, {'type': 'table'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            stats = results['stats']
            cagr_pct = results['cagr'] * 100

            # 1. Return distribution box plot
            fig.add_trace(
                go.Box(
                    y=cagr_pct,
                    name=label,
                    marker_color=color,
                    boxmean='sd',
                    boxpoints='outliers',
                    hovertemplate='<b>%{fullData.name}</b><br>CAGR: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Risk metrics bar chart
            risk_metrics = {
                'Median DD': abs(stats['median_max_drawdown']) * 100,
                '95th DD': abs(stats['max_drawdown_95']) * 100,
                'Worst DD': abs(stats['worst_max_drawdown']) * 100,
                'Volatility': stats['std_cagr'] * 100
            }

            fig.add_trace(
                go.Bar(
                    x=list(risk_metrics.keys()),
                    y=list(risk_metrics.values()),
                    name=label,
                    marker_color=color,
                    text=[f'{v:.2f}%' for v in risk_metrics.values()],
                    textposition='outside',
                    hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. Percentile analysis
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            final_values = [np.percentile(results['final_values'], p) for p in percentiles]

            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=final_values,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=10, color=color, line=dict(width=2, color='white')),
                    hovertemplate='<b>%{fullData.name}</b><br>%{x}th percentile<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )

        # 4. Correlation placeholder (show as info table for now)
        self._add_correlation_info_table(fig, results_list, labels, row=2, col=2)

        # 5. Best/Worst case table
        self._add_best_worst_table(fig, results_list, labels, row=3, col=1)

        # 6. Year statistics table
        self._add_year_stats_table(fig, results_list, labels, row=3, col=2)

        # Update axes
        fig.update_xaxes(title_text="Portfolio", row=1, col=1)
        fig.update_xaxes(title_text="Risk Metric", row=1, col=2)
        fig.update_xaxes(title_text="Percentile", row=2, col=1, showgrid=True, gridcolor='lightgray')

        fig.update_yaxes(title_text="CAGR (%)", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Value (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Final Value ($)", row=2, col=1, showgrid=True, gridcolor='lightgray')

        fig.update_layout(
            height=1600,  # Increased height
            title_text="<b>Detailed Portfolio Statistics</b><br><sub>Comprehensive analysis of returns, risks, and distributions</sub>",
            title_font_size=20,
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#333',
                borderwidth=2
            ),
            template='plotly_white',
            barmode='group',
            font=dict(size=11, family='Arial, sans-serif'),
            margin=dict(t=120, b=50, l=60, r=60)
        )

        return fig

    def _add_correlation_info_table(self, fig, results_list, labels, row, col):
        """Add portfolio correlation information table"""
        headers = ['Metric', 'Value']

        # Calculate some cross-portfolio statistics
        if len(results_list) >= 2:
            # Get CAGRs from first two portfolios
            cagr1 = results_list[0]['cagr']
            cagr2 = results_list[1]['cagr']
            correlation = np.corrcoef(cagr1, cagr2)[0, 1]

            info_data = [
                ['Portfolios Analyzed', str(len(results_list))],
                ['Simulations per Portfolio', f"{len(cagr1):,}"],
                ['Sample Correlation (1-2)', f"{correlation:.3f}"],
                ['Data Points Total', f"{len(results_list) * len(cagr1):,}"]
            ]
        else:
            info_data = [
                ['Portfolios Analyzed', str(len(results_list))],
                ['Simulations', f"{len(results_list[0]['cagr']):,}"],
                ['Data Type', 'Monte Carlo'],
                ['Status', 'Complete']
            ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color='#9467bd',
                    align='left',
                    font=dict(size=12, color='white', family='Arial Black'),
                    height=35
                ),
                cells=dict(
                    values=list(zip(*info_data)),
                    fill_color=[['#f8f9fa', 'white'] * len(info_data)],
                    align='left',
                    font=dict(size=11, family='Arial'),
                    height=30
                )
            ),
            row=row, col=col
        )

    def _add_best_worst_table(self, fig, results_list, labels, row, col):
        """Add best/worst case scenarios table"""
        headers = ['Portfolio', 'Best (95%)', 'Expected (50%)', 'Worst (5%)', 'Range']

        table_data = []
        for results, label in zip(results_list, labels):
            final_values = results['final_values']
            best = np.percentile(final_values, 95)
            median = np.percentile(final_values, 50)
            worst = np.percentile(final_values, 5)
            range_val = best - worst

            display_label = label[:30] + '...' if len(label) > 30 else label

            table_data.append([
                display_label,
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
                    font=dict(size=12, color='white', family='Arial Black'),
                    height=35
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#f8f9fa', 'white'] * (len(table_data) // 2 + 1)],
                    align='left',
                    font=dict(size=11, family='Arial'),
                    height=30
                ),
                columnwidth=[3, 2, 2, 2, 2]
            ),
            row=row, col=col
        )

    def _add_year_stats_table(self, fig, results_list, labels, row, col):
        """Add year-by-year returns table"""
        headers = ['Portfolio', '5th %ile', '25th %ile', 'Median', '75th %ile', '95th %ile']

        table_data = []
        for results, label in zip(results_list, labels):
            final_values = results['final_values']
            display_label = label[:30] + '...' if len(label) > 30 else label

            table_data.append([
                display_label,
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
                    font=dict(size=12, color='white', family='Arial Black'),
                    height=35
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#f8f9fa', 'white'] * (len(table_data) // 2 + 1)],
                    align='left',
                    font=dict(size=11, family='Arial'),
                    height=30
                ),
                columnwidth=[3, 2, 2, 2, 2, 2]
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
                'Downside Risk Distribution',
                'Risk-Adjusted Performance Metrics',
                'Drawdown Severity Analysis'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'box'}]
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.15
        )

        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            stats = results['stats']
            final_values = results['final_values']

            # 1. VaR analysis
            initial = self.simulator.initial_capital
            var_95 = (initial - stats['percentile_5']) / initial * 100
            var_99 = (initial - np.percentile(final_values, 1)) / initial * 100
            var_90 = (initial - np.percentile(final_values, 10)) / initial * 100

            fig.add_trace(
                go.Bar(
                    x=['90% VaR', '95% VaR', '99% VaR'],
                    y=[max(0, var_90), max(0, var_95), max(0, var_99)],
                    name=label,
                    marker_color=color,
                    text=[f'{max(0, var_90):.1f}%', f'{max(0, var_95):.1f}%', f'{max(0, var_99):.1f}%'],
                    textposition='outside',
                    hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Downside deviation - only negative returns
            cagr_vals = results['cagr']
            negative_returns = cagr_vals[cagr_vals < 0] * 100

            if len(negative_returns) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=negative_returns,
                        name=label,
                        opacity=0.6,
                        marker_color=color,
                        nbinsx=40,
                        hovertemplate='<b>%{fullData.name}</b><br>Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )

            # 3. Risk-adjusted metrics
            sharpe = stats['sharpe_ratio']
            sortino = stats['sortino_ratio']

            # Calmar ratio (return / max drawdown)
            calmar = (stats['mean_cagr'] / abs(stats['median_max_drawdown'])) if stats['median_max_drawdown'] != 0 else 0

            # Omega ratio approximation (gains/losses)
            positive_returns = cagr_vals[cagr_vals > 0]
            omega = len(positive_returns) / len(cagr_vals) if len(cagr_vals) > 0 else 0

            fig.add_trace(
                go.Bar(
                    x=['Sharpe', 'Sortino', 'Calmar', 'Omega'],
                    y=[sharpe, sortino, min(calmar, 10), omega],  # Cap Calmar at 10 for display
                    name=label,
                    marker_color=color,
                    text=[f'{sharpe:.2f}', f'{sortino:.2f}', f'{min(calmar, 10):.2f}', f'{omega:.2f}'],
                    textposition='outside',
                    hovertemplate='<b>%{fullData.name}</b><br>%{x} Ratio: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )

            # 4. Drawdown severity distribution
            max_dds = results['max_drawdowns'] * 100
            fig.add_trace(
                go.Box(
                    y=max_dds,
                    name=label,
                    marker_color=color,
                    boxmean='sd',
                    boxpoints='outliers',
                    hovertemplate='<b>%{fullData.name}</b><br>Max DD: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )

        # Update axes
        fig.update_xaxes(title_text="VaR Confidence Level", row=1, col=1)
        fig.update_xaxes(title_text="Negative CAGR (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text="Risk Metric", row=2, col=1)
        fig.update_xaxes(title_text="Portfolio", row=2, col=2)

        fig.update_yaxes(title_text="Potential Loss (%)", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Ratio Value", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Maximum Drawdown (%)", row=2, col=2, showgrid=True, gridcolor='lightgray')

        fig.update_layout(
            height=1200,  # Increased height
            title_text="<b>Risk Analysis Dashboard</b><br><sub>Comprehensive risk metrics and downside analysis</sub>",
            title_font_size=20,
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#333',
                borderwidth=2
            ),
            template='plotly_white',
            barmode='group',
            font=dict(size=11, family='Arial, sans-serif'),
            margin=dict(t=120, b=50, l=60, r=60)
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        .header p {{
            margin: 12px 0 0 0;
            font-size: 16px;
            opacity: 0.95;
            font-weight: 300;
        }}

        .tabs {{
            display: flex;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
            border-bottom: 1px solid #e1e8ed;
        }}

        .tab {{
            flex: 1;
            padding: 18px 15px;
            text-align: center;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 15px;
            color: #536471;
        }}

        .tab:hover {{
            background-color: #f7f9fc;
            color: #667eea;
        }}

        .tab.active {{
            border-bottom-color: #667eea;
            color: #667eea;
            background-color: #f8f9ff;
        }}

        .tab-content {{
            display: none;
            padding: 25px 20px;
            animation: fadeIn 0.4s ease-in;
            max-width: 1800px;
            margin: 0 auto;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 0 auto 30px;
            max-width: 1400px;
        }}

        .stat-card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 1px solid #e1e8ed;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }}

        .stat-card h4 {{
            margin: 0 0 12px 0;
            color: #536471;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 8px;
        }}

        .stat-card .subvalue {{
            font-size: 13px;
            color: #657786;
            margin: 0;
        }}

        .footer {{
            background-color: #2c3e50;
            color: white;
            padding: 25px 20px;
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
        }}

        .footer p {{
            margin: 5px 0;
        }}

        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 28px;
            }}

            .tab {{
                padding: 15px 10px;
                font-size: 13px;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}

            .stat-card .value {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Portfolio Analysis Dashboard</h1>
        <p>Monte Carlo Simulation Results • {date}</p>
    </div>

    {summary_stats}

    <div class="tabs">
        <div class="tab active" onclick="showTab(event, 'overview')">📈 Overview</div>
        <div class="tab" onclick="showTab(event, 'statistics')">📊 Detailed Statistics</div>
        <div class="tab" onclick="showTab(event, 'risk')">⚠️ Risk Analysis</div>
    </div>

    <div id="overview" class="tab-content active">
        <div class="chart-container">
            <div id="overview-chart"></div>
        </div>
    </div>

    <div id="statistics" class="tab-content">
        <div class="chart-container">
            <div id="stats-chart"></div>
        </div>
    </div>

    <div id="risk" class="tab-content">
        <div class="chart-container">
            <div id="risk-chart"></div>
        </div>
    </div>

    <div class="footer">
        <p><strong>Generated with Monte Carlo Portfolio Simulator</strong></p>
        <p>Initial Capital: ${initial_capital:,} • Simulations: {simulations:,} • Time Horizon: {years} years</p>
        <p style="margin-top: 10px; font-size: 12px; opacity: 0.8;">Interactive charts: Click legend items to show/hide • Hover for details • Double-click to reset zoom</p>
    </div>

    <script>
        {overview_json}
        {stats_json}
        {risk_json}

        // Plot configuration
        const config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }};

        // Initialize plots
        Plotly.newPlot('overview-chart', overviewData.data, overviewData.layout, config);
        Plotly.newPlot('stats-chart', statsData.data, statsData.layout, config);
        Plotly.newPlot('risk-chart', riskData.data, riskData.layout, config);

        function showTab(event, tabName) {{
            // Hide all tabs
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(tab => tab.classList.remove('active'));

            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');

            // Relayout plotly charts when tab becomes visible
            setTimeout(() => {{
                if (tabName === 'overview') {{
                    Plotly.Plots.resize('overview-chart');
                }} else if (tabName === 'statistics') {{
                    Plotly.Plots.resize('stats-chart');
                }} else if (tabName === 'risk') {{
                    Plotly.Plots.resize('risk-chart');
                }}
            }}, 50);
        }}

        // Handle window resize
        window.addEventListener('resize', () => {{
            Plotly.Plots.resize('overview-chart');
            Plotly.Plots.resize('stats-chart');
            Plotly.Plots.resize('risk-chart');
        }});
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
            date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
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
                import os
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
                    f.write(self.html_content)
                    temp_path = f.name
                webbrowser.open('file://' + os.path.abspath(temp_path))

        return CustomHTML(html_content)

    def _generate_summary_stats(self, portfolio_configs):
        """Generate summary statistics cards with better formatting"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        # Find best portfolio by different metrics
        best_sharpe_idx = max(range(len(results_list)),
                             key=lambda i: results_list[i]['stats']['sharpe_ratio'])
        best_return_idx = max(range(len(results_list)),
                             key=lambda i: results_list[i]['stats']['median_cagr'])
        best_risk_idx = min(range(len(results_list)),
                           key=lambda i: abs(results_list[i]['stats']['median_max_drawdown']))

        # Truncate labels for display
        def truncate_label(label, max_len=25):
            return label[:max_len] + '...' if len(label) > max_len else label

        html = '<div class="stats-grid">'
        html += f'''
        <div class="stat-card">
            <h4>🏆 Best Risk-Adjusted</h4>
            <div class="value">{truncate_label(labels[best_sharpe_idx])}</div>
            <p class="subvalue">Sharpe Ratio: {results_list[best_sharpe_idx]['stats']['sharpe_ratio']:.3f}</p>
        </div>
        <div class="stat-card">
            <h4>📈 Highest Return</h4>
            <div class="value">{truncate_label(labels[best_return_idx])}</div>
            <p class="subvalue">Median CAGR: {results_list[best_return_idx]['stats']['median_cagr']*100:.2f}%</p>
        </div>
        <div class="stat-card">
            <h4>🛡️ Lowest Risk</h4>
            <div class="value">{truncate_label(labels[best_risk_idx])}</div>
            <p class="subvalue">Max Drawdown: {results_list[best_risk_idx]['stats']['median_max_drawdown']*100:.2f}%</p>
        </div>
        <div class="stat-card">
            <h4>📊 Analysis Scale</h4>
            <div class="value">{len(results_list)} Portfolios</div>
            <p class="subvalue">{self.simulator.simulations:,} simulations each</p>
        </div>
        '''
        html += '</div>'

        return html