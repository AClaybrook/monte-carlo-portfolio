"""
End-to-end smoke test: main.py on synthetic data writes a complete report.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = '''
from run_config import RunConfig, PortfolioConfig, SimulationConfig, OptimizationConfig, StrategyConfig, SweepConfig
config = RunConfig(
    name="Smoke <Test>",
    portfolios=[
        PortfolioConfig(name='60/40', allocations={'VOO': 0.6, 'BND': 0.4}),
        PortfolioConfig(name='Thirds', allocations={'VOO': 1/3, 'QQQ': 1/3, 'BTC-USD': 1/3},
                        rebalance='quarterly'),
        PortfolioConfig(name='Protected', allocations={'VOO': 0.8, 'BND': 0.2},
                        strategy=StrategyConfig('drawdown_protection', apply_to='rebalance',
                                                check_frequency='daily',
                                                params={'threshold': 0.15,
                                                        'risk_off_allocation': {'BND': 1.0}})),
    ],
    simulation=SimulationConfig(initial_capital=10000, years=3, simulations=200, seed=1,
                                start_date='2016-01-01', end_date='2024-12-31',
                                contribution_amount=250, contribution_frequency='monthly'),
    optimization=OptimizationConfig(assets=['VOO', 'BND'], active_strategies=['min_volatility'],
                                    benchmark_ticker='VOO'),
    sweeps=[SweepConfig('Dip grid', {'VOO': 0.7, 'QQQ': 0.3},
                        StrategyConfig('buy_the_dip', {'target_ticker': 'QQQ'}),
                        grid={'threshold': [0.1, 0.2], 'aggressive_weight': [0.5, 0.8]})],
)
'''


def test_main_synthetic_report(tmp_path):
    cfg = tmp_path / 'smoke_config.py'
    cfg.write_text(CONFIG)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, 'main.py', str(cfg), '--synthetic'], cwd=ROOT,
                          capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = Path(ROOT, re.search(r'Report saved to: (\S+)', proc.stdout).group(1))
    try:
        page = report.read_text()
        assert '<title>Smoke &lt;Test&gt;</title>' in page          # names are escaped
        assert 'Synthetic data.' in page
        for section in ('summary', 'growth', 'returns', 'drawdowns', 'risk', 'allocation',
                        'montecarlo', 'walkforward', 'sweeps', 'notes'):
            assert f'<section id="{section}">' in page
        for label in ('Benchmark (VOO)', '60/40', 'Thirds', 'Protected', 'Min Volatility'):
            assert label in page
        assert 'signal rebalances' in page
        figs = json.loads(re.search(r'const FIGS = (\{.*?\});\nconst CMAP', page, re.S).group(1))
        assert {'c-growth', 'c-mcfan', 'c-corr', 'c-allocation', 'c-sweep0', 'c-wfgrowth'} <= set(figs)
        # Every PV backtest link has whole-percent weights summing to exactly 100
        urls = re.findall(r'href="(https://www.portfoliovisualizer.com/backtest-portfolio[^"]+)"', page)
        assert len(urls) == 5
        for url in urls:
            weights = [int(w) for w in re.findall(r'allocation\d+_1=(\d+)', url.replace('&amp;', '&'))]
            assert sum(weights) == 100, url
    finally:
        report.unlink(missing_ok=True)
