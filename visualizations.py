"""
Interactive Plotly visualizations for portfolio analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class PortfolioVisualizer:
    """Creates interactive Plotly visualizations with synchronized legend"""

    def __init__(self, simulator):
        self.simulator = simulator
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def create_dashboard(self, portfolio_configs):
        """Create comprehensive interactive dashboard with synchronized legends"""
        results_list = [config['results'] for config in portfolio_configs]
        labels = [config['label'] for config in portfolio_configs]

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Sample Portfolio Trajectories',
                'Distribution of Final Values',
                'CAGR Distribution',
                'Return Distribution Comparison',
                'Maximum Drawdown Distribution',
                'Percentile Fan Chart',
                'Risk-Return Profile',
                'Key Statistics',
                'Probability of Outcomes'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'box'}, {'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'table'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Use legendgroup to synchronize legend clicks across all subplots
        for idx, (results, label) in enumerate(zip(results_list, labels)):
            color = self.colors[idx % len(self.colors)]
            legendgroup = f"group{idx}"

            # 1. Sample trajectories - add to first subplot only for legend
            sample_indices = np.random.choice(self.simulator.simulations, 50, replace=False)
            for sample_idx in sample_indices[:3]:
                fig.add_trace(
                    go.Scatter(
                        y=results['portfolio_values'][sample_idx, :],
                        mode='lines',
                        line=dict(color=color, width=0.5),
                        opacity=0.3,
                        showlegend=False,
                        legendgroup=legendgroup,
                        hovertemplate='Day: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )

            median_path = np.median(results['portfolio_values'], axis=0)
            fig.add_trace(
                go.Scatter(
                    y=median_path,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=3),
                    legendgroup=legendgroup,
                    showlegend=True,
                    hovertemplate='Day: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # 2. Final value distribution
            fig.add_trace(
                go.Histogram(
                    x=results['final_values'],
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )

            # 3. CAGR distribution
            fig.add_trace(
                go.Histogram(
                    x=results['cagr'] * 100,
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='CAGR: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=3
            )

            # 4. Box plot
            fig.add_trace(
                go.Box(
                    y=results['total_returns'] * 100,
                    name=label,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )

            # 5. Max drawdown
            fig.add_trace(
                go.Histogram(
                    x=results['max_drawdowns'] * 100,
                    name=label,
                    opacity=0.6,
                    marker_color=color,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Max DD: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )

            # 6. Percentile fan chart
            values = results['portfolio_values']
            p50 = np.percentile(values, 50, axis=0)
            p5 = np.percentile(values, 5, axis=0)
            p95 = np.percentile(values, 95, axis=0)

            days = np.arange(values.shape[1])

            # Add shaded area
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p95,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    legendgroup=legendgroup,
                    hoverinfo='skip'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p5,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    showlegend=False,
                    legendgroup=legendgroup,
                    hovertemplate='Day: %{x}<br>5th-95th: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=3
            )

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p50,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Median: $%{y:,.0f}<extra></extra>'
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
                    marker=dict(size=15, color=color),
                    text=[label],
                    textposition='top center',
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )

        # 8. Statistics table
        table_data = []
        for results, label in zip(results_list, labels):
            stats = results['stats']
            table_data.append([
                label,
                f"${stats['median_final_value']:,.0f}",
                f"{stats['median_cagr']*100:.1f}%",
                f"{stats['median_max_drawdown']*100:.1f}%",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['sortino_ratio']:.2f}"
            ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Portfolio', 'Median Value', 'Median CAGR', 'Median DD', 'Sharpe', 'Sortino'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=10, color='black')
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=9)
                )
            ),
            row=3, col=2
        )

        # 9. Probability outcomes
        prob_loss = [r['stats']['probability_loss'] * 100 for r in results_list]
        prob_double = [r['stats']['probability_double'] * 100 for r in results_list]

        # Use same legendgroup for bars
        for idx, label in enumerate(labels):
            legendgroup = f"group{idx}"
            color = self.colors[idx % len(self.colors)]

            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[prob_loss[idx]],
                    name=f'{label} - Loss',
                    marker_color=color,
                    opacity=0.5,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f'{label}<br>P(Loss): %{{y:.1f}}%<extra></extra>'
                ),
                row=3, col=3
            )

            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[prob_double[idx]],
                    name=f'{label} - Double',
                    marker_color=color,
                    opacity=0.8,
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f'{label}<br>P(Double): %{{y:.1f}}%<extra></extra>'
                ),
                row=3, col=3
            )

        # Update axes labels
        fig.update_xaxes(title_text="Trading Days", row=1, col=1)
        fig.update_xaxes(title_text="Final Value ($)", row=1, col=2)
        fig.update_xaxes(title_text="CAGR (%)", row=1, col=3)
        fig.update_xaxes(title_text="Portfolio", row=2, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)
        fig.update_xaxes(title_text="Trading Days", row=2, col=3)
        fig.update_xaxes(title_text="CAGR Volatility (%)", row=3, col=1)
        fig.update_xaxes(title_text="", row=3, col=3)

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=3)
        fig.update_yaxes(title_text="Mean CAGR (%)", row=3, col=1)
        fig.update_yaxes(title_text="Probability (%)", row=3, col=3)

        fig.update_layout(
            height=1400,
            title_text="Monte Carlo Portfolio Simulation Dashboard - Click Legend to Toggle Portfolios",
            title_font_size=18,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='closest',
            barmode='group'
        )

        return fig
