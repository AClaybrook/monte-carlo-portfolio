"""
Enhanced Quant Analytics Module

Inspired by Portfolio Visualizer, provides:
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar, Treynor)
- Drawdown analysis with recovery times
- Rolling statistics
- Factor regression (market beta, alpha)
- Capture ratios (upside/downside)
- Value at Risk (VaR) and Conditional VaR
- Return decomposition
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date


@dataclass
class DrawdownPeriod:
    """Represents a single drawdown period"""
    start_date: date
    trough_date: date
    recovery_date: Optional[date]
    drawdown: float  # Negative percentage
    length_days: int
    recovery_days: Optional[int]


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Basic
    cagr: float
    arithmetic_return: float
    volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: Optional[float]
    information_ratio: Optional[float]

    # Downside
    max_drawdown: float
    average_drawdown: float
    downside_deviation: float

    # VaR
    var_95: float
    var_99: float
    cvar_95: float

    # Capture ratios (vs benchmark)
    upside_capture: Optional[float]
    downside_capture: Optional[float]

    # Market exposure
    beta: Optional[float]
    alpha: Optional[float]
    r_squared: Optional[float]
    correlation: Optional[float]

    # Distribution
    skewness: float
    kurtosis: float
    positive_periods: float  # % of periods with positive returns


class QuantAnalytics:
    """
    Advanced quantitative analytics for portfolio analysis.
    """

    # Risk-free rate (annualized)
    RISK_FREE_RATE = 0.045  # 4.5%
    TRADING_DAYS = 252

    def __init__(self, risk_free_rate: float = None):
        if risk_free_rate is not None:
            self.RISK_FREE_RATE = risk_free_rate

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns from prices"""
        return prices.pct_change().dropna()

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from prices"""
        return np.log(prices / prices.shift(1)).dropna()

    def calculate_cagr(self, returns: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        total_return = (1 + returns).prod()
        years = len(returns) / self.TRADING_DAYS
        if years <= 0:
            return 0.0
        return total_return ** (1 / years) - 1

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate return volatility"""
        vol = returns.std()
        return vol * np.sqrt(self.TRADING_DAYS) if annualize else vol

    def calculate_downside_deviation(self, returns: pd.Series, threshold: float = 0,
                                      annualize: bool = True) -> float:
        """Calculate downside deviation (semi-deviation)"""
        downside = returns[returns < threshold]
        if len(downside) == 0:
            return 0.0
        dd = np.sqrt(np.mean(downside ** 2))
        return dd * np.sqrt(self.TRADING_DAYS) if annualize else dd

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio"""
        excess_return = self.calculate_cagr(returns) - self.RISK_FREE_RATE
        vol = self.calculate_volatility(returns)
        return excess_return / vol if vol > 0 else 0.0

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino Ratio (uses downside deviation)"""
        excess_return = self.calculate_cagr(returns) - self.RISK_FREE_RATE
        downside = self.calculate_downside_deviation(returns)
        return excess_return / downside if downside > 0 else 0.0

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar Ratio (return / max drawdown)"""
        cagr = self.calculate_cagr(returns)
        max_dd = self.calculate_max_drawdown(returns)
        return cagr / abs(max_dd) if max_dd != 0 else 0.0

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    def analyze_drawdowns(self, returns: pd.Series,
                          min_drawdown: float = 0.05) -> List[DrawdownPeriod]:
        """
        Analyze all drawdown periods.

        Parameters:
            returns: Daily returns series
            min_drawdown: Minimum drawdown to include (e.g., 0.05 = 5%)

        Returns list of DrawdownPeriod objects sorted by severity.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        periods = []
        in_drawdown = False
        start_idx = None
        trough_idx = None
        trough_value = 0

        for i, (dt, dd) in enumerate(drawdown.items()):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i - 1 if i > 0 else i
                trough_idx = i
                trough_value = dd
            elif in_drawdown:
                if dd < trough_value:
                    # New trough
                    trough_idx = i
                    trough_value = dd
                elif dd == 0:
                    # Recovery
                    if abs(trough_value) >= min_drawdown:
                        periods.append(DrawdownPeriod(
                            start_date=drawdown.index[start_idx].date() if hasattr(drawdown.index[start_idx], 'date') else drawdown.index[start_idx],
                            trough_date=drawdown.index[trough_idx].date() if hasattr(drawdown.index[trough_idx], 'date') else drawdown.index[trough_idx],
                            recovery_date=drawdown.index[i].date() if hasattr(drawdown.index[i], 'date') else drawdown.index[i],
                            drawdown=trough_value,
                            length_days=trough_idx - start_idx,
                            recovery_days=i - trough_idx
                        ))
                    in_drawdown = False

        # Handle ongoing drawdown
        if in_drawdown and abs(trough_value) >= min_drawdown:
            periods.append(DrawdownPeriod(
                start_date=drawdown.index[start_idx].date() if hasattr(drawdown.index[start_idx], 'date') else drawdown.index[start_idx],
                trough_date=drawdown.index[trough_idx].date() if hasattr(drawdown.index[trough_idx], 'date') else drawdown.index[trough_idx],
                recovery_date=None,
                drawdown=trough_value,
                length_days=trough_idx - start_idx,
                recovery_days=None
            ))

        return sorted(periods, key=lambda x: x.drawdown)

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (historical method)"""
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate beta and alpha vs benchmark.

        Returns: (beta, alpha, r_squared)
        """
        # Align the series
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(aligned) < 30:
            return (None, None, None)

        aligned.columns = ['portfolio', 'benchmark']

        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned['benchmark'], aligned['portfolio']
        )

        # Annualize alpha
        alpha = intercept * self.TRADING_DAYS

        return (slope, alpha, r_value ** 2)

    def calculate_capture_ratios(self, returns: pd.Series,
                                  benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate upside and downside capture ratios.

        Returns: (upside_capture, downside_capture)
        """
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(aligned) < 30:
            return (None, None)

        aligned.columns = ['portfolio', 'benchmark']

        # Upside periods
        up_mask = aligned['benchmark'] > 0
        up_portfolio = (1 + aligned.loc[up_mask, 'portfolio']).prod() ** (self.TRADING_DAYS / up_mask.sum()) - 1
        up_benchmark = (1 + aligned.loc[up_mask, 'benchmark']).prod() ** (self.TRADING_DAYS / up_mask.sum()) - 1

        upside_capture = (up_portfolio / up_benchmark) * 100 if up_benchmark != 0 else 0

        # Downside periods
        down_mask = aligned['benchmark'] < 0
        down_portfolio = (1 + aligned.loc[down_mask, 'portfolio']).prod() ** (self.TRADING_DAYS / down_mask.sum()) - 1
        down_benchmark = (1 + aligned.loc[down_mask, 'benchmark']).prod() ** (self.TRADING_DAYS / down_mask.sum()) - 1

        downside_capture = (down_portfolio / down_benchmark) * 100 if down_benchmark != 0 else 0

        return (upside_capture, downside_capture)

    def calculate_information_ratio(self, returns: pd.Series,
                                     benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(aligned) < 30:
            return None

        aligned.columns = ['portfolio', 'benchmark']
        active_returns = aligned['portfolio'] - aligned['benchmark']

        active_return_ann = active_returns.mean() * self.TRADING_DAYS
        tracking_error = active_returns.std() * np.sqrt(self.TRADING_DAYS)

        return active_return_ann / tracking_error if tracking_error > 0 else 0

    def calculate_treynor_ratio(self, returns: pd.Series,
                                 benchmark_returns: pd.Series) -> Optional[float]:
        """Calculate Treynor Ratio"""
        beta, _, _ = self.calculate_beta(returns, benchmark_returns)
        if beta is None or beta == 0:
            return None

        excess_return = self.calculate_cagr(returns) - self.RISK_FREE_RATE
        return excess_return / beta

    def get_full_metrics(self, returns: pd.Series,
                         benchmark_returns: pd.Series = None) -> RiskMetrics:
        """
        Calculate all risk metrics for a return series.

        Parameters:
            returns: Daily portfolio returns
            benchmark_returns: Daily benchmark returns (optional)
        """
        cagr = self.calculate_cagr(returns)
        vol = self.calculate_volatility(returns)
        max_dd = self.calculate_max_drawdown(returns)
        dd_series = self.calculate_drawdown_series(returns)

        # Benchmark-relative metrics
        beta, alpha, r_sq = (None, None, None)
        upside_cap, downside_cap = (None, None)
        info_ratio = None
        treynor = None
        correlation = None

        if benchmark_returns is not None:
            beta, alpha, r_sq = self.calculate_beta(returns, benchmark_returns)
            upside_cap, downside_cap = self.calculate_capture_ratios(returns, benchmark_returns)
            info_ratio = self.calculate_information_ratio(returns, benchmark_returns)
            treynor = self.calculate_treynor_ratio(returns, benchmark_returns)
            correlation = returns.corr(benchmark_returns)

        return RiskMetrics(
            cagr=cagr,
            arithmetic_return=returns.mean() * self.TRADING_DAYS,
            volatility=vol,
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            sortino_ratio=self.calculate_sortino_ratio(returns),
            calmar_ratio=self.calculate_calmar_ratio(returns),
            treynor_ratio=treynor,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            average_drawdown=dd_series[dd_series < 0].mean() if (dd_series < 0).any() else 0,
            downside_deviation=self.calculate_downside_deviation(returns),
            var_95=self.calculate_var(returns, 0.95),
            var_99=self.calculate_var(returns, 0.99),
            cvar_95=self.calculate_cvar(returns, 0.95),
            upside_capture=upside_cap,
            downside_capture=downside_cap,
            beta=beta,
            alpha=alpha,
            r_squared=r_sq,
            correlation=correlation,
            skewness=returns.skew(),
            kurtosis=returns.kurtosis(),
            positive_periods=(returns > 0).mean() * 100
        )

    def calculate_rolling_metrics(self, returns: pd.Series,
                                   window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling metrics.

        Returns DataFrame with:
        - rolling_return: Annualized return
        - rolling_vol: Annualized volatility
        - rolling_sharpe: Rolling Sharpe ratio
        - rolling_sortino: Rolling Sortino ratio
        - rolling_max_dd: Rolling max drawdown
        """
        results = pd.DataFrame(index=returns.index)

        # Rolling cumulative return
        rolling_cum = returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (self.TRADING_DAYS / len(x)) - 1
        )
        results['rolling_return'] = rolling_cum

        # Rolling volatility
        results['rolling_vol'] = returns.rolling(window).std() * np.sqrt(self.TRADING_DAYS)

        # Rolling Sharpe
        results['rolling_sharpe'] = (rolling_cum - self.RISK_FREE_RATE) / results['rolling_vol']

        # Rolling Sortino
        rolling_downside = returns.rolling(window).apply(
            lambda x: np.sqrt(np.mean(x[x < 0] ** 2)) * np.sqrt(self.TRADING_DAYS) if (x < 0).any() else 0.01
        )
        results['rolling_sortino'] = (rolling_cum - self.RISK_FREE_RATE) / rolling_downside

        # Rolling max drawdown
        def rolling_max_dd(r):
            cum = (1 + r).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
            return dd.min()

        results['rolling_max_dd'] = returns.rolling(window).apply(rolling_max_dd)

        return results

    def calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculate monthly returns table (like Portfolio Visualizer).

        Returns DataFrame with years as rows, months as columns.
        """
        # Ensure datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Resample to monthly
        monthly = (1 + returns).resample('M').prod() - 1

        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Add annual return
        annual = (1 + returns).resample('Y').prod() - 1
        pivot['Year'] = annual.values

        return pivot

    def calculate_annual_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate annual returns"""
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        return (1 + returns).resample('Y').prod() - 1


def format_metrics_table(metrics: RiskMetrics) -> str:
    """Format metrics as a readable table"""
    lines = [
        "=" * 50,
        "RISK METRICS",
        "=" * 50,
        "",
        "RETURNS",
        f"  CAGR:              {metrics.cagr * 100:>8.2f}%",
        f"  Arithmetic Mean:   {metrics.arithmetic_return * 100:>8.2f}%",
        "",
        "RISK",
        f"  Volatility:        {metrics.volatility * 100:>8.2f}%",
        f"  Max Drawdown:      {metrics.max_drawdown * 100:>8.2f}%",
        f"  Avg Drawdown:      {metrics.average_drawdown * 100:>8.2f}%",
        f"  Downside Dev:      {metrics.downside_deviation * 100:>8.2f}%",
        "",
        "RISK-ADJUSTED",
        f"  Sharpe Ratio:      {metrics.sharpe_ratio:>8.2f}",
        f"  Sortino Ratio:     {metrics.sortino_ratio:>8.2f}",
        f"  Calmar Ratio:      {metrics.calmar_ratio:>8.2f}",
    ]

    if metrics.treynor_ratio is not None:
        lines.append(f"  Treynor Ratio:     {metrics.treynor_ratio:>8.2f}")
    if metrics.information_ratio is not None:
        lines.append(f"  Information Ratio: {metrics.information_ratio:>8.2f}")

    lines.extend([
        "",
        "VALUE AT RISK",
        f"  VaR (95%):         {metrics.var_95 * 100:>8.2f}%",
        f"  VaR (99%):         {metrics.var_99 * 100:>8.2f}%",
        f"  CVaR (95%):        {metrics.cvar_95 * 100:>8.2f}%",
    ])

    if metrics.beta is not None:
        lines.extend([
            "",
            "BENCHMARK METRICS",
            f"  Beta:              {metrics.beta:>8.2f}",
            f"  Alpha:             {metrics.alpha * 100:>8.2f}%",
            f"  R-Squared:         {metrics.r_squared * 100:>8.2f}%",
            f"  Correlation:       {metrics.correlation:>8.2f}",
        ])

    if metrics.upside_capture is not None:
        lines.extend([
            "",
            "CAPTURE RATIOS",
            f"  Upside Capture:    {metrics.upside_capture:>8.1f}%",
            f"  Downside Capture:  {metrics.downside_capture:>8.1f}%",
        ])

    lines.extend([
        "",
        "DISTRIBUTION",
        f"  Skewness:          {metrics.skewness:>8.2f}",
        f"  Kurtosis:          {metrics.kurtosis:>8.2f}",
        f"  % Positive:        {metrics.positive_periods:>8.1f}%",
        "",
        "=" * 50,
    ])

    return "\n".join(lines)


# Quick analysis function
def analyze_portfolio(prices: pd.Series,
                      benchmark_prices: pd.Series = None,
                      name: str = "Portfolio") -> Dict:
    """
    Quick portfolio analysis.

    Parameters:
        prices: Price series for portfolio
        benchmark_prices: Price series for benchmark (optional)
        name: Portfolio name

    Returns dict with metrics and analysis.
    """
    qa = QuantAnalytics()

    returns = qa.calculate_returns(prices)
    benchmark_returns = qa.calculate_returns(benchmark_prices) if benchmark_prices is not None else None

    metrics = qa.get_full_metrics(returns, benchmark_returns)
    drawdowns = qa.analyze_drawdowns(returns)
    monthly = qa.calculate_monthly_returns(returns)
    rolling = qa.calculate_rolling_metrics(returns)

    return {
        'name': name,
        'returns': returns,
        'metrics': metrics,
        'drawdowns': drawdowns,
        'monthly_returns': monthly,
        'rolling_metrics': rolling,
        'formatted_report': format_metrics_table(metrics)
    }