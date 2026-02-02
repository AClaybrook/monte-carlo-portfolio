"""
Portfolio Visualizer compatibility utilities.

Provides import/export functions for Portfolio Visualizer CSV format and URL generation.
"""
import os

PORTFOLIO_OUTPUT_DIR = "output/portfolios"


def export_portfolio_csv(allocations: dict, name: str = "Portfolio") -> str:
    """
    Generate PV-compatible CSV content.

    Args:
        allocations: Dict of {ticker: weight} where weight is 0-1
        name: Portfolio description for header

    Returns:
        CSV string in PV import format
    """
    lines = [f"{name}", "", "Symbol,Weight"]

    for ticker, weight in sorted(allocations.items()):
        pv_ticker = to_pv_ticker(ticker)
        lines.append(f"{pv_ticker},{weight*100:.0f}%")

    lines.append("")
    return "\n".join(lines)


def save_portfolio_csv(allocations: dict, name: str, output_dir: str = PORTFOLIO_OUTPUT_DIR) -> str:
    """
    Save portfolio to CSV file in output/portfolios/.

    Args:
        allocations: Dict of {ticker: weight}
        name: Portfolio name (used for filename)
        output_dir: Output directory path

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in name)
    filepath = os.path.join(output_dir, f"{safe_name}.csv")

    csv_content = export_portfolio_csv(allocations, name)
    with open(filepath, 'w') as f:
        f.write(csv_content)

    return filepath


def parse_portfolio_csv(csv_content: str) -> dict:
    """
    Parse PV CSV format into allocations dict.

    Args:
        csv_content: CSV string in PV format

    Returns:
        Dict with 'name' and 'allocations' keys
    """
    lines = [l.strip() for l in csv_content.strip().split('\n')]

    result = {
        'name': lines[0] if lines else 'Portfolio',
        'allocations': {}
    }

    # Find Symbol,Weight header
    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith('symbol'):
            header_idx = i
            break

    if header_idx is None:
        return result

    # Parse data rows
    for line in lines[header_idx + 1:]:
        if ',' not in line or not line.strip():
            continue
        parts = line.split(',')
        ticker = from_pv_ticker(parts[0].strip())
        weight_str = parts[1].strip().rstrip('%')
        weight = float(weight_str) / 100
        result['allocations'][ticker] = weight

    return result


def to_pv_ticker(ticker: str) -> str:
    """
    Convert local ticker format to PV format.

    Examples:
        BTC-USD -> BTC
        VOO -> VOO
    """
    if ticker.endswith('-USD'):
        return ticker[:-4]
    return ticker


def from_pv_ticker(ticker: str) -> str:
    """
    Convert PV ticker to local format.

    Examples:
        BTC -> BTC-USD
        VOO -> VOO
    """
    crypto = {'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'DOT'}
    if ticker.upper() in crypto:
        return f"{ticker.upper()}-USD"
    return ticker.upper()


def generate_pv_url(allocations: dict, start_year: int = 2017, end_year: int = 2025) -> str:
    """
    Generate Portfolio Visualizer backtest URL.

    Args:
        allocations: Dict of {ticker: weight}
        start_year: Backtest start year
        end_year: Backtest end year

    Returns:
        URL string that opens PV with the portfolio pre-configured
    """
    base = "https://www.portfoliovisualizer.com/backtest-portfolio"

    params = [
        's=y',
        'timePeriod=4',
        f'startYear={start_year}',
        f'endYear={end_year}',
        'initialAmount=10000',
        'annualOperation=0',
        'inflationAdjusted=true',
        'frequency=4',
        'rebalanceType=1',  # Annual rebalancing
    ]

    for i, (ticker, weight) in enumerate(sorted(allocations.items()), 1):
        pv_ticker = to_pv_ticker(ticker)
        params.append(f'symbol{i}={pv_ticker}')
        params.append(f'allocation{i}_1={weight*100:.0f}')

    return f"{base}?{'&'.join(params)}"


def generate_mc_url(allocations: dict, initial_amount: int = 1000000,
                    years: int = 30, periodic_amount: int = 0) -> str:
    """
    Generate Portfolio Visualizer Monte Carlo simulation URL using tickers.

    Args:
        allocations: Dict of {ticker: weight}
        initial_amount: Starting portfolio value
        years: Simulation time horizon
        periodic_amount: Periodic contribution/withdrawal amount

    Returns:
        URL string that opens PV Monte Carlo with the portfolio pre-configured
    """
    base = "https://www.portfoliovisualizer.com/monte-carlo-simulation"

    params = [
        's=y',
        f'initialAmount={initial_amount}',
        f'years={years}',
        'inflationAdjusted=true',
        'simulationModel=1',  # Historical returns
        'frequency=4',  # Annual
    ]

    if periodic_amount != 0:
        params.append(f'periodicAmount={periodic_amount}')
        params.append('adjustmentType=2')  # Inflation-adjusted contributions

    for i, (ticker, weight) in enumerate(sorted(allocations.items()), 1):
        pv_ticker = to_pv_ticker(ticker)
        params.append(f'symbol{i}={pv_ticker}')
        params.append(f'allocation{i}={weight*100:.0f}')

    return f"{base}?{'&'.join(params)}"
