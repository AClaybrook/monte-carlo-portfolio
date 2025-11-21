"""
Investment Timing and Dollar-Cost Averaging Analysis Utility.

This script helps analyze:
1. Lump sum vs. Dollar Cost Averaging (DCA)
2. Market timing scenarios
3. Entry point sensitivity
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TimingAnalyzer:
    """Analyzes investment timing strategies"""

    def __init__(self, data_manager, ticker='VOO', initial_capital=100000):
        self.data_manager = data_manager
        self.ticker = ticker
        self.initial_capital = initial_capital

    def analyze_entry_points(self, years_back=10):
        """
        Analyze returns from different entry points over historical data.

        This shows: What if I had invested at different times in the past?
        """
        print(f"\n{'='*60}")
        print(f"Entry Point Analysis: {self.ticker}")
        print(f"{'='*60}")

        # Get historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * (years_back + 5))

        df = self.data_manager.get_data(self.ticker, start_date, end_date)
        prices = df['Adj Close'].values
        dates = df.index

        # Test different entry points
        results = []
        holding_periods = [1, 3, 5, 10]  # years

        for holding_years in holding_periods:
            holding_days = holding_years * 252

            if holding_days >= len(prices):
                continue

            for entry_idx in range(0, len(prices) - holding_days, 21):  # Monthly intervals
                entry_price = prices[entry_idx]
                exit_price = prices[entry_idx + holding_days]

                total_return = (exit_price / entry_price - 1) * 100
                cagr = ((exit_price / entry_price) ** (1 / holding_years) - 1) * 100

                # Calculate max drawdown during period
                period_prices = prices[entry_idx:entry_idx + holding_days + 1]
                running_max = np.maximum.accumulate(period_prices)
                drawdowns = (period_prices - running_max) / running_max
                max_dd = np.min(drawdowns) * 100

                results.append({
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[entry_idx + holding_days],
                    'holding_years': holding_years,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'total_return': total_return,
                    'cagr': cagr,
                    'max_drawdown': max_dd
                })

        results_df = pd.DataFrame(results)

        # Print summary statistics
        print(f"\nEntry Point Statistics (by holding period):")
        print(f"{'='*60}")

        for holding_years in holding_periods:
            period_data = results_df[results_df['holding_years'] == holding_years]
            if len(period_data) == 0:
                continue

            print(f"\n{holding_years}-Year Holding Period:")
            print(f"  Number of entry points tested: {len(period_data)}")
            print(f"  CAGR Statistics:")
            print(f"    Best:   {period_data['cagr'].max():6.2f}%")
            print(f"    Worst:  {period_data['cagr'].min():6.2f}%")
            print(f"    Median: {period_data['cagr'].median():6.2f}%")
            print(f"    Mean:   {period_data['cagr'].mean():6.2f}%")
            print(f"  Max Drawdown Statistics:")
            print(f"    Best (smallest):  {period_data['max_drawdown'].max():6.2f}%")
            print(f"    Worst (largest):  {period_data['max_drawdown'].min():6.2f}%")
            print(f"    Median:           {period_data['max_drawdown'].median():6.2f}%")
            print(f"  Probability of profit: {(period_data['total_return'] > 0).mean() * 100:.1f}%")

        # Create visualization
        self._visualize_entry_analysis(results_df)

        return results_df

    def compare_lump_sum_vs_dca(self, investment_amount=100000, dca_months=12, years_back=10):
        """
        Compare lump sum investment vs. dollar cost averaging.

        Parameters:
        -----------
        investment_amount : float
            Total amount to invest
        dca_months : int
            Number of months to spread DCA over
        years_back : int
            Years of historical data to analyze
        """
        print(f"\n{'='*60}")
        print(f"Lump Sum vs. DCA Analysis: {self.ticker}")
        print(f"{'='*60}")
        print(f"Investment Amount: ${investment_amount:,.0f}")
        print(f"DCA Period: {dca_months} months")

        # Get historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * (years_back + 5))

        df = self.data_manager.get_data(self.ticker, start_date, end_date)
        prices = df['Adj Close'].values
        dates = df.index

        results = []
        holding_years = 5
        holding_days = holding_years * 252
        dca_days = dca_months * 21  # Approximate trading days per month

        # Test different start dates
        for start_idx in range(0, len(prices) - holding_days - dca_days, 21):
            # Lump sum: invest everything at start
            ls_shares = investment_amount / prices[start_idx]
            ls_final_value = ls_shares * prices[start_idx + holding_days]
            ls_return = (ls_final_value / investment_amount - 1) * 100

            # DCA: invest equal amounts over dca_months
            dca_shares = 0
            monthly_investment = investment_amount / dca_months

            for month in range(dca_months):
                invest_idx = start_idx + (month * 21)  # Approximate monthly
                if invest_idx < len(prices):
                    dca_shares += monthly_investment / prices[invest_idx]

            dca_final_value = dca_shares * prices[start_idx + holding_days]
            dca_return = (dca_final_value / investment_amount - 1) * 100

            results.append({
                'start_date': dates[start_idx],
                'end_date': dates[start_idx + holding_days],
                'lump_sum_return': ls_return,
                'dca_return': dca_return,
                'difference': ls_return - dca_return
            })

        results_df = pd.DataFrame(results)

        # Print statistics
        print(f"\nResults over {len(results)} historical scenarios:")
        print(f"\nLump Sum:")
        print(f"  Mean Return:   {results_df['lump_sum_return'].mean():6.2f}%")
        print(f"  Median Return: {results_df['lump_sum_return'].median():6.2f}%")
        print(f"  Best:          {results_df['lump_sum_return'].max():6.2f}%")
        print(f"  Worst:         {results_df['lump_sum_return'].min():6.2f}%")

        print(f"\nDollar Cost Averaging:")
        print(f"  Mean Return:   {results_df['dca_return'].mean():6.2f}%")
        print(f"  Median Return: {results_df['dca_return'].median():6.2f}%")
        print(f"  Best:          {results_df['dca_return'].max():6.2f}%")
        print(f"  Worst:         {results_df['dca_return'].min():6.2f}%")

        print(f"\nComparison:")
        print(f"  Lump sum outperformed DCA: {(results_df['difference'] > 0).mean() * 100:.1f}% of the time")
        print(f"  Average difference:         {results_df['difference'].mean():6.2f}%")
        print(f"  Median difference:          {results_df['difference'].median():6.2f}%")

        # Visualize
        self._visualize_ls_vs_dca(results_df)

        return results_df

    def _visualize_entry_analysis(self, results_df):
        """Create visualization for entry point analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'CAGR Distribution by Holding Period',
                'Max Drawdown Distribution',
                'CAGR Over Time',
                'Return vs. Max Drawdown'
            )
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        holding_periods = sorted(results_df['holding_years'].unique())

        for idx, holding_years in enumerate(holding_periods):
            data = results_df[results_df['holding_years'] == holding_years]
            color = colors[idx % len(colors)]

            # 1. CAGR distribution
            fig.add_trace(
                go.Histogram(
                    x=data['cagr'],
                    name=f'{holding_years}Y',
                    marker_color=color,
                    opacity=0.6,
                    nbinsx=30
                ),
                row=1, col=1
            )

            # 2. Max drawdown distribution
            fig.add_trace(
                go.Histogram(
                    x=data['max_drawdown'],
                    name=f'{holding_years}Y',
                    marker_color=color,
                    opacity=0.6,
                    showlegend=False,
                    nbinsx=30
                ),
                row=1, col=2
            )

            # 3. CAGR over time
            fig.add_trace(
                go.Scatter(
                    x=data['entry_date'],
                    y=data['cagr'],
                    mode='markers',
                    name=f'{holding_years}Y',
                    marker=dict(color=color, size=4),
                    showlegend=False
                ),
                row=2, col=1
            )

            # 4. Risk-return scatter
            fig.add_trace(
                go.Scatter(
                    x=data['max_drawdown'].abs(),
                    y=data['cagr'],
                    mode='markers',
                    name=f'{holding_years}Y',
                    marker=dict(color=color, size=4),
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_xaxes(title_text="CAGR (%)", row=1, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=1, col=2)
        fig.update_xaxes(title_text="Entry Date", row=2, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="CAGR (%)", row=2, col=1)
        fig.update_yaxes(title_text="CAGR (%)", row=2, col=2)

        fig.update_layout(
            height=900,
            title_text=f"<b>Entry Point Analysis: {self.ticker}</b>",
            template='plotly_white',
            barmode='overlay'
        )

        fig.write_html('timing_analysis.html')
        print(f"\n✓ Saved timing analysis to: timing_analysis.html")

    def _visualize_ls_vs_dca(self, results_df):
        """Create visualization for lump sum vs DCA comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution Comparison',
                'Lump Sum vs DCA: Who Wins?',
                'Returns Over Time',
                'Difference Distribution (LS - DCA)'
            )
        )

        # 1. Return distributions
        fig.add_trace(
            go.Histogram(
                x=results_df['lump_sum_return'],
                name='Lump Sum',
                marker_color='#1f77b4',
                opacity=0.6,
                nbinsx=30
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=results_df['dca_return'],
                name='DCA',
                marker_color='#ff7f0e',
                opacity=0.6,
                nbinsx=30
            ),
            row=1, col=1
        )

        # 2. Win rate
        ls_wins = (results_df['difference'] > 0).sum()
        dca_wins = (results_df['difference'] < 0).sum()

        fig.add_trace(
            go.Bar(
                x=['Lump Sum', 'DCA'],
                y=[ls_wins, dca_wins],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[f'{ls_wins}', f'{dca_wins}'],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Returns over time
        fig.add_trace(
            go.Scatter(
                x=results_df['start_date'],
                y=results_df['lump_sum_return'],
                mode='markers',
                name='Lump Sum',
                marker=dict(color='#1f77b4', size=4),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=results_df['start_date'],
                y=results_df['dca_return'],
                mode='markers',
                name='DCA',
                marker=dict(color='#ff7f0e', size=4),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Difference distribution
        fig.add_trace(
            go.Histogram(
                x=results_df['difference'],
                marker_color='#2ca02c',
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_xaxes(title_text="Start Date", row=2, col=1)
        fig.update_xaxes(title_text="Difference (%)", row=2, col=2)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        fig.update_layout(
            height=900,
            title_text=f"<b>Lump Sum vs Dollar Cost Averaging: {self.ticker}</b>",
            template='plotly_white',
            barmode='overlay'
        )

        fig.write_html('lump_sum_vs_dca.html')
        print(f"✓ Saved LS vs DCA analysis to: lump_sum_vs_dca.html")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Investment Timing Analysis')
    parser.add_argument('--ticker', default='VOO', help='Ticker symbol to analyze')
    parser.add_argument('--capital', type=float, default=100000, help='Investment amount')
    parser.add_argument('--dca-months', type=int, default=12, help='DCA period in months')
    args = parser.parse_args()

    # Initialize
    data_manager = DataManager()
    analyzer = TimingAnalyzer(data_manager, ticker=args.ticker, initial_capital=args.capital)

    # Run analyses
    print(f"\nAnalyzing {args.ticker}...")

    # Entry point analysis
    entry_results = analyzer.analyze_entry_points(years_back=10)

    # Lump sum vs DCA
    ls_dca_results = analyzer.compare_lump_sum_vs_dca(
        investment_amount=args.capital,
        dca_months=args.dca_months,
        years_back=10
    )

    # Clean up
    data_manager.close()

    print("\n✓ Analysis complete!")
    print(f"  • timing_analysis.html")
    print(f"  • lump_sum_vs_dca.html")


if __name__ == "__main__":
    main()