"""
Main script - With Strategy Support and Bulk Download
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

from run_config import load_config_from_file, create_strategy_from_config
from data_manager import DataManager
from portfolio_simulator import PortfolioSimulator
from portfolio_optimizer import PortfolioOptimizer
from visualizations import PortfolioVisualizer
from pv_compat import save_portfolio_csv
from sweeps import run_sweep
from backtester import Backtester


def find_config_file(specified_path: str = None) -> Path:
    if specified_path:
        path = Path(specified_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {specified_path}")
        return path
    if Path('config/my_portfolios.py').exists():
        return Path('config/my_portfolios.py')
    if Path('config/example_config.py').exists():
        return Path('config/example_config.py')
    raise FileNotFoundError("No config file found.")


def evaluate_portfolio(sim, backtester, sim_cfg, label, assets, weights, start=None, end=None,
                       strategy_conf=None, rebalance=None, benchmark=None, description=None) -> dict:
    """Monte Carlo + backtest with identical cash flows, rebalancing and benchmark."""
    strategy = create_strategy_from_config(strategy_conf) if strategy_conf else None
    apply_to = strategy_conf.apply_to if strategy_conf else 'contributions'
    if strategy is None:
        # Zero-weight assets never trade without a strategy; skipping them saves simulation work
        kept = [(a, w) for a, w in zip(assets, weights) if w > 1e-9]
        assets, weights = [a for a, _ in kept], [w for _, w in kept]
    check = strategy_conf.check_frequency if strategy_conf else 'monthly'
    rebalance = rebalance or sim_cfg.rebalance
    sim_res = sim.simulate_portfolio(
        assets, weights, start_date_override=start, strategy=strategy,
        rebalance=rebalance, apply_to=apply_to, check_frequency=check)
    bt_res = backtester.run_backtest(
        assets, weights, sim_cfg.initial_capital,
        start_date_override=start, end_date=end,
        strategy=strategy, contribution_amount=sim_cfg.contribution_amount,
        contribution_frequency=sim_cfg.contribution_frequency,
        rebalance=rebalance, apply_to=apply_to, check_frequency=check,
        risk_free_rate=sim_cfg.risk_free_rate, benchmark=benchmark)
    m = bt_res['metrics']
    irr = f" | IRR: {m['IRR']*100:.2f}%" if m['Total Contributions'] else ""
    print(f"  {bt_res['strategy']}")
    print(f"  CAGR: {m['CAGR']*100:.2f}%{irr} | Max DD: {m['Max Drawdown']*100:.2f}% | "
          f"Sharpe: {m['Sharpe']:.2f}")
    return {'label': label, 'description': description, 'results': sim_res, 'backtest': bt_res}


def describe_assumptions(config, args) -> dict:
    sim = config.simulation
    reb = sim.rebalance
    rebalance = reb.frequency.capitalize() if reb.frequency != 'none' else 'None (buy and hold)'
    if reb.threshold:
        rebalance += f", {reb.threshold:.0%} band"
    if reb.transaction_cost_bps:
        rebalance += f", {reb.transaction_cost_bps:g} bps cost"
    freq = sim.contribution_frequency
    period = {'daily': 'day', 'weekly': 'week', 'monthly': 'month', 'quarterly': 'quarter', 'annual': 'year'}
    contrib = (f"${sim.contribution_amount:,.0f} every "
               + (f"{freq} trading days" if isinstance(freq, int) else period[freq])
               if sim.contribution_amount else 'None')
    return {
        'Initial capital': f"${sim.initial_capital:,.0f}",
        'Contributions': contrib,
        'Default rebalancing': rebalance,
        'Risk-free rate': f"{sim.risk_free_rate:.2%}",
        'Benchmark': config.benchmark_ticker or 'First asset of each portfolio',
        'Simulation': f"{sim.simulations:,} paths x {sim.years} years, {sim.method}"
                      + (f" (block {sim.block_size}d)" if sim.method == 'block_bootstrap' else ''),
        'Inflation': f"{sim.inflation_rate:.2%} (results in today's dollars)" if sim.inflation_rate else 'Not adjusted',
        'Seed': str(sim.seed) if sim.seed is not None else 'Random',
        'Data': 'SYNTHETIC' if args.synthetic else ('cached only' if args.offline else args.data_source),
    }


def collect_all_tickers(config) -> set:
    """Collect all tickers needed from config"""
    all_tickers = set()

    for p in config.portfolios:
        all_tickers.update([t.upper() for t in p.allocations.keys()])
    for sw in config.sweeps:
        all_tickers.update([t.upper() for t in sw.allocations.keys()])

    if config.optimization:
        all_tickers.update([t.upper() for t in config.optimization.assets])
    if config.benchmark_ticker:
        all_tickers.add(config.benchmark_ticker.upper())

    return all_tickers


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Simulator')
    parser.add_argument('config', nargs='?')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization step')
    parser.add_argument('--force-download', action='store_true', help='Force re-download all data')
    parser.add_argument('--coverage-report', action='store_true', help='Show data coverage report')
    parser.add_argument('--offline', action='store_true', help='Use cached data only (no yfinance calls)')
    parser.add_argument('--data-source', choices=['yfinance', 'fmp'], default='yfinance',
                        help='Data source for market data (default: yfinance)')
    parser.add_argument('--embed-plotlyjs', action='store_true',
                        help='Embed plotly.js in the report so it opens offline')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use deterministic synthetic prices (no network, not real data)')
    args = parser.parse_args()

    config_path = find_config_file(args.config)
    print(f"✓ Loading configuration from: {config_path}")

    try:
        config = load_config_from_file(str(config_path))
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        return 1

    # Initialize data manager
    if args.synthetic:
        from synthetic_data import SyntheticDataManager
        data_manager = SyntheticDataManager()
        print("  Data source: SYNTHETIC (generated prices, not real market data)")
    else:
        data_source = args.data_source
        data_manager = DataManager(db_path=config.database.path, data_source=data_source)
        if data_source != 'yfinance':
            print(f"  Data source: {data_source.upper()}")

    # Coverage report mode
    if args.coverage_report:
        all_tickers = list(collect_all_tickers(config))
        report = data_manager.get_data_coverage_report(all_tickers)
        print("\n" + "="*80)
        print("DATA COVERAGE REPORT")
        print("="*80)
        print(report.to_string(index=False))
        data_manager.close()
        return 0

    # Initialize simulation components
    sim = PortfolioSimulator(data_manager, config.simulation)
    optimizer = PortfolioOptimizer(sim, data_manager)
    visualizer = PortfolioVisualizer(sim)
    backtester = Backtester(data_manager)

    print("\n" + "="*60)
    print("BULK DOWNLOADING DATA")
    print("="*60)

    # Collect all tickers and bulk download
    all_tickers = list(collect_all_tickers(config))
    print(f"Tickers to load: {', '.join(sorted(all_tickers))}")

    # Calculate date range from config
    start_date, end_date = config.simulation.get_date_range()
    print(f"Date range: {start_date} to {end_date}")

    # Bulk download all data at once (or use cache only in offline mode)
    if args.offline:
        print("OFFLINE MODE - using cached data only (no yfinance calls)")
        cached_end = data_manager.get_latest_cached_date(all_tickers)
        if cached_end is None:
            print("✗ No cached data found for any tickers. Cannot run offline.")
            data_manager.close()
            return 1
        end_date = cached_end
        print(f"  Using cached data through: {end_date}")
        bulk_data = {}
        for ticker in all_tickers:
            df = data_manager._get_from_db(ticker, start_date, end_date)
            if df is not None and not df.empty:
                bulk_data[ticker] = df
    else:
        bulk_data = data_manager.bulk_download(
            all_tickers,
            start_date=start_date,
            end_date=end_date,
            force_update=args.force_download
        )

    print(f"\n✓ Loaded data for {len(bulk_data)} tickers")

    # Build asset map from downloaded data
    print("\n" + "="*60)
    print("BUILDING ASSET MAP")
    print("="*60)

    asset_map = {}
    start_dates = []

    for ticker in all_tickers:
        asset_conf = config.assets.get(ticker, None)
        name = asset_conf.name if asset_conf else ticker

        if ticker in bulk_data:
            df = bulk_data[ticker]
            returns = df['Adj Close'].pct_change().dropna()

            asset = {
                'ticker': ticker,
                'name': name,
                'historical_returns': returns,
                'full_data': df,
                'daily_mean': returns.mean(),
                'daily_std': returns.std()
            }
            asset_map[ticker] = asset

            if not returns.empty:
                start_dates.append(returns.index.min())
                print(f"  ✓ {ticker}: {len(df)} days, {returns.index.min().date()} to {returns.index.max().date()}")
        else:
            print(f"  ⚠ {ticker}: No data available")

    if not start_dates:
        print("Error: No data found for any assets.")
        data_manager.close()
        return 1

    global_start_date = max(start_dates)
    global_end_date = min(a['full_data'].index.max() for a in asset_map.values())
    print(f"\n✓ GLOBAL ALIGNMENT: {global_start_date.date()} to {global_end_date.date()}")

    sim_cfg = config.simulation
    bench_ticker = config.benchmark_ticker.upper() if config.benchmark_ticker else None
    bench_asset = asset_map.get(bench_ticker) if bench_ticker else None

    def evaluate(label, assets, weights, strategy_conf=None, rebalance=None, description=None):
        return evaluate_portfolio(sim, backtester, sim_cfg, label, assets, weights,
                                  start=global_start_date, end=global_end_date,
                                  strategy_conf=strategy_conf, rebalance=rebalance,
                                  benchmark=bench_asset, description=description)

    portfolio_results = []

    if bench_asset is not None:
        print(f"\n→ Benchmark: {bench_ticker}")
        bench_item = evaluate(f"Benchmark ({bench_ticker})", [bench_asset], [1.0])
        bench_item['is_benchmark'] = True
        portfolio_results.append(bench_item)

    print("\n" + "="*60)
    print("PROCESSING PORTFOLIOS")
    print("="*60)

    for p_conf in config.portfolios:
        print(f"\n→ {p_conf.name}")
        missing = [t for t in p_conf.allocations.keys() if t.upper() not in asset_map]
        if missing:
            print(f"  ⚠ Skipping - missing assets: {missing}")
            continue
        assets = [asset_map[t.upper()] for t in p_conf.allocations.keys()]
        weights = list(p_conf.allocations.values())
        portfolio_results.append(evaluate(p_conf.name, assets, weights, p_conf.strategy,
                                          p_conf.rebalance, p_conf.description))

    if config.optimization and not args.no_optimize:
        print("\n" + "="*60)
        print("RUNNING OPTIMIZATIONS (in-sample: weights are fit to the same history they are tested on)")
        print("="*60)

        opt_assets = [asset_map[name.upper()] for name in config.optimization.assets
                      if name.upper() in asset_map]
        if len(opt_assets) < 2:
            print("⚠ Need at least 2 assets for optimization")
        else:
            strategy_map = {
                'max_sharpe': lambda: optimizer.optimize_sharpe_ratio(
                    opt_assets, risk_free_rate=sim_cfg.risk_free_rate,
                    start_date_override=global_start_date),
                'min_volatility': lambda: optimizer.optimize_min_volatility(
                    opt_assets, start_date_override=global_start_date),
                'risk_parity': lambda: optimizer.optimize_risk_parity(
                    opt_assets, start_date_override=global_start_date),
                'max_sortino': lambda: optimizer.optimize_sortino_ratio(
                    opt_assets, risk_free_rate=sim_cfg.risk_free_rate,
                    start_date_override=global_start_date),
                'custom_weighted': lambda: optimizer.optimize_custom_weighted(
                    opt_assets, weights_config=config.optimization.objective_weights,
                    risk_free_rate=sim_cfg.risk_free_rate,
                    start_date_override=global_start_date),
            }
            for strat_name in config.optimization.active_strategies:
                if strat_name not in strategy_map:
                    print(f"⚠ Unknown strategy: {strat_name}")
                    continue
                print(f"\n→ {strat_name}...")
                opt = strategy_map[strat_name]()
                portfolio_results.append(evaluate(opt['label'], opt_assets, opt['allocations'],
                                                  description='Optimized (in-sample)'))

    sweep_results = []
    if config.sweeps:
        print("\n" + "="*60)
        print("PARAMETER SWEEPS")
        print("="*60)
        for sweep in config.sweeps:
            missing = [t for t in sweep.allocations if t.upper() not in asset_map]
            if missing:
                print(f"  ⚠ Skipping sweep '{sweep.name}' - missing assets: {missing}")
                continue
            sweep_results.append(run_sweep(sweep, asset_map, sim_cfg, start=global_start_date,
                                           end=global_end_date, benchmark=bench_asset))

    # Generate Report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    out_dir = 'output'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.visualization.output_filename}"

    for item in portfolio_results:
        alloc = {a['ticker']: w for a, w in zip(item['results']['assets'], item['results']['allocations'])
                 if w > 0.001}
        try:
            save_portfolio_csv(alloc, item['label'])
        except OSError as e:
            print(f"  ⚠ Could not save PV CSV for {item['label']}: {e}")

    visualizer.generate_html_report(
        portfolio_results,
        str(output_path),
        start_date=global_start_date.date(),
        end_date=global_end_date.date(),
        title=config.name,
        assumptions=describe_assumptions(config, args),
        synthetic=args.synthetic,
        embed_plotlyjs=config.visualization.embed_plotlyjs or args.embed_plotlyjs,
        sweeps=sweep_results,
    )
    print(f"✓ Report saved to: {output_path}")

    data_manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())